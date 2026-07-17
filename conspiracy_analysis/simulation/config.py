"""
Configuration dataclasses for the ABM simulation framework.

Bridges fitted Cox model outputs from notebook 01 into a structured
configuration consumed by the simulation engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from lifelines import CoxTimeVaryingFitter

from conspiracy_analysis import EXPOSURE_WINDOW
from conspiracy_analysis.models.baseline_hazards import (
    extract_cumulative_baseline_hazard,
    fit_linear_baseline,
    fit_weibull_baseline,
)

logger = logging.getLogger(__name__)
DEFAULT_EXPOSURE_WINDOW = float(EXPOSURE_WINDOW)


@dataclass(frozen=True)
class BaselineParams:
    """Parametric baseline hazard — linear (Model 1) or Weibull (Models 2+).

    Linear:  h0(t) = slope                (constant)
    Weibull: h0(t) = (k/lam)*(t/lam)^(k-1)
    """

    type: str  # "linear" or "weibull"
    slope: Optional[float] = None  # only for linear
    shape: Optional[float] = None  # k, only for weibull
    scale: Optional[float] = None  # lambda, only for weibull
    reference: str = "zero_covariates"
    centering_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.reference != "zero_covariates":
            raise ValueError("Simulation baselines must use the zero covariate reference")
        if not np.isfinite(self.centering_factor) or self.centering_factor <= 0:
            raise ValueError("centering_factor must be finite and positive")
        if self.type == "linear":
            if self.slope is None or not np.isfinite(self.slope) or self.slope < 0:
                raise ValueError("Linear baseline requires a finite nonnegative slope")
        elif self.type == "weibull":
            if (
                self.shape is None
                or self.scale is None
                or not np.isfinite(self.shape)
                or not np.isfinite(self.scale)
                or self.shape <= 0
                or self.scale <= 0
            ):
                raise ValueError("Weibull baseline requires finite positive shape and scale")
        else:
            raise ValueError(f"Unknown baseline type: {self.type}")

    def cumulative_hazard(self, t: float) -> float:
        """Cumulative baseline hazard from zero through time ``t``."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("Baseline cumulative hazard requires finite t >= 0")
        if self.type == "linear":
            if self.slope is None:
                raise ValueError("Linear baseline requires slope")
            return float(self.slope * t)
        if self.type == "weibull":
            return float((t / self.scale) ** self.shape)
        raise AssertionError("Baseline type was validated during construction")

    def integrated_hazard(self, start: float, end: float) -> float:
        """Integrated baseline hazard over the half open interval [start, end)."""
        if not np.isfinite(start) or start < 0:
            raise ValueError("Integrated baseline hazard requires finite start >= 0")
        if not np.isfinite(end) or end < start:
            raise ValueError("Integrated baseline hazard requires finite end >= start")
        increment = self.cumulative_hazard(end) - self.cumulative_hazard(start)
        return float(max(0.0, increment))

    def h0(self, t: float) -> float:
        """Instantaneous baseline hazard at time t."""
        if self.type == "linear":
            return self.slope
        if self.type == "weibull":
            if t < 0:
                raise ValueError("Instantaneous baseline hazard requires t >= 0")
            if t == 0:
                if self.shape < 1:
                    return float("inf")
                if self.shape == 1:
                    return 1.0 / self.scale
                return 0.0
            return (self.shape / self.scale) * (t / self.scale) ** (self.shape - 1)
        raise AssertionError("Baseline type was validated during construction")


@dataclass(frozen=True)
class CoxModelSpec:
    """One fitted Cox model stored as lookup tables for fast simulation.

    Attributes:
        model_number: 1-5, identifies which adoption transition this models.
        baseline: Parametric baseline hazard.
        s7_scores: Pre-computed log-hazard contribution for each s7 dummy value.
        conspiracy_scores: Log-hazard contribution per conspiracy dummy.
        beta_degree: Coefficient for log(1+degree).
        beta_cross_cluster: Coefficient for the cross-cluster binary flag.
    """

    model_number: int
    baseline: BaselineParams
    s7_scores: Dict[int, float] = field(default_factory=dict)
    conspiracy_scores: Dict[str, float] = field(default_factory=dict)
    beta_degree: float = 0.0
    beta_cross_cluster: float = 0.0


@dataclass
class SimulationConfig:
    """Full configuration for running the ABM.

    Attributes:
        cox_models: Mapping from model number to CoxModelSpec.
        conspiracies: List of conspiracy column names.
        cluster_assignments: Mapping from conspiracy name to cluster id.
        entry_times: Mapping from conspiracy name to first-appearance time step.
        seed_fraction: Fraction of eligible nodes to seed at entry times.
        steps: Number of hourly time steps to simulate.
        max_model_number: Highest model number available (derived from fitted
            models; adoptions beyond this reuse the last model).
    """

    cox_models: Dict[int, CoxModelSpec]
    conspiracies: List[str]
    cluster_assignments: Dict[str, int]
    entry_times: Dict[str, float]
    seed_fraction: float = 0.01
    steps: int = 5000
    max_model_number: int = 5
    hawkes_params: Optional[Tuple[float, float, float]] = None
    exposure_window: float = DEFAULT_EXPOSURE_WINDOW
    target_total_adoptions: Optional[int] = None
    require_peer_exposure: bool = True

    def get_model_for_adoption(self, n: int) -> CoxModelSpec:
        """Return the CoxModelSpec for the n-th adoption (1-indexed).

        Clamps to max_model_number: adoptions beyond that reuse the last model.
        """
        model_num = min(n, self.max_model_number)
        # Fall back to highest available if exact model not present
        while model_num > 0 and model_num not in self.cox_models:
            model_num -= 1
        if model_num == 0:
            raise ValueError(f"No Cox model available for adoption {n}")
        return self.cox_models[model_num]


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario configuration for a simulation run.

    Attributes:
        name: Short identifier (e.g., "baseline", "quarantine_24h").
        description: Human-readable description.
    """

    name: str
    description: str = ""
    quarantine_hours: float = 0.0
    sharing_block_rate: float = 0.0
    nudge_rejection_rate: float = 0.0


def override_baselines_to_linear(sim_config: SimulationConfig) -> SimulationConfig:
    """Create a copy of sim_config where all models use Model 1's linear baseline.

    Used for the 'no temporal effects' counterfactual: removes the Weibull
    temporal acceleration in Models 2+ while keeping all covariates intact.
    """
    linear_baseline = sim_config.cox_models[1].baseline
    if linear_baseline.type != "linear":
        raise ValueError(
            f"Expected Model 1 to have a linear baseline, got {linear_baseline.type}"
        )

    new_models: Dict[int, CoxModelSpec] = {}
    for model_num, spec in sim_config.cox_models.items():
        if model_num == 1:
            new_models[model_num] = spec
        else:
            new_models[model_num] = CoxModelSpec(
                model_number=spec.model_number,
                baseline=linear_baseline,
                s7_scores=spec.s7_scores,
                conspiracy_scores=spec.conspiracy_scores,
                beta_degree=spec.beta_degree,
                beta_cross_cluster=spec.beta_cross_cluster,
            )

    return SimulationConfig(
        cox_models=new_models,
        conspiracies=sim_config.conspiracies,
        cluster_assignments=sim_config.cluster_assignments,
        entry_times=sim_config.entry_times,
        seed_fraction=sim_config.seed_fraction,
        steps=sim_config.steps,
        max_model_number=sim_config.max_model_number,
        hawkes_params=sim_config.hawkes_params,
        exposure_window=sim_config.exposure_window,
        target_total_adoptions=sim_config.target_total_adoptions,
        require_peer_exposure=sim_config.require_peer_exposure,
    )


def _extract_cox_model_spec(
    model_number: int,
    ctv: CoxTimeVaryingFitter,
) -> CoxModelSpec:
    """Extract a CoxModelSpec from a fitted CoxTimeVaryingFitter.

    Fits a parametric baseline (linear for Model 1, Weibull for 2+)
    and pre-computes lookup tables for s7 dummies and conspiracy dummies.
    """
    params = ctv.params_
    times, cum_haz = extract_cumulative_baseline_hazard(ctv)

    norm_mean = getattr(ctv, "_norm_mean", None)
    if norm_mean is None:
        raise ValueError(
            f"Model {model_number} does not expose Lifelines _norm_mean"
        )
    missing_means = params.index.difference(norm_mean.index)
    if len(missing_means) > 0:
        raise ValueError(
            f"Model {model_number} centering means missing for {list(missing_means)}"
        )
    aligned_means = norm_mean.reindex(params.index).astype(float)
    centering_log_offset = -float(np.dot(
        params.to_numpy(dtype=float), aligned_means.to_numpy(dtype=float)
    ))
    centering_factor = float(np.exp(centering_log_offset))
    if not np.isfinite(centering_factor) or centering_factor <= 0:
        raise ValueError(
            f"Model {model_number} has invalid centering factor {centering_factor}"
        )

    # Lifelines stores the baseline at its fitted mean covariate vector.
    # Convert it to the zero covariate reference before fitting the compact
    # parametric baseline used by the simulation.
    cum_haz = cum_haz * centering_factor

    if model_number == 1:
        fit = fit_linear_baseline(times, cum_haz)
        baseline = BaselineParams(
            type="linear",
            slope=fit["slope"],
            reference="zero_covariates",
            centering_factor=centering_factor,
        )
    else:
        fit = fit_weibull_baseline(times, cum_haz)
        baseline = BaselineParams(
            type="weibull",
            shape=fit["shape"],
            scale=fit["scale"],
            reference="zero_covariates",
            centering_factor=centering_factor,
        )

    # Build s7 dummy lookup (s7_d1, s7_d2, s7_d3, s7_d4+)
    s7_scores: Dict[int, float] = {0: 0.0}
    for dummy_val in [1, 2, 3, 4]:
        col = f"s7_d{dummy_val}"
        if col in params:
            s7_scores[dummy_val] = float(params[col])
        else:
            s7_scores[dummy_val] = 0.0

    # Build conspiracy dummy lookup (fc_<name>)
    conspiracy_scores: Dict[str, float] = {}
    for col in params.index:
        if col.startswith("fc_"):
            conspiracy_scores[col[3:]] = float(params[col])

    beta_degree = float(params.get("degree", 0.0))
    beta_cross_cluster = float(params.get("cross_cluster", 0.0))

    spec = CoxModelSpec(
        model_number=model_number,
        baseline=baseline,
        s7_scores=s7_scores,
        conspiracy_scores=conspiracy_scores,
        beta_degree=beta_degree,
        beta_cross_cluster=beta_cross_cluster,
    )

    logger.info(
        f"Model {model_number}: baseline={baseline.type}, "
        f"centering_factor={centering_factor:.6f}, "
        f"beta_degree={beta_degree:.4f}, beta_cross_cluster={beta_cross_cluster:.4f}, "
        f"{len(conspiracy_scores)} conspiracy dummies"
    )
    return spec


def build_simulation_config_from_fitted_models(
    cox_models: Dict[int, CoxTimeVaryingFitter],
    conspiracies: List[str],
    cluster_assignments: Dict[str, int],
    entry_times: Dict[str, float],
    seed_fraction: float = 0.01,
    steps: int = 5000,
    hawkes_params: Optional[Tuple[float, float, float]] = None,
    exposure_window: float = DEFAULT_EXPOSURE_WINDOW,
    require_peer_exposure: bool = True,
) -> SimulationConfig:
    """Factory: convert notebook 01 outputs into a SimulationConfig.

    Args:
        cox_models: Dict mapping model number (1, 2, 3, 4, 5) to fitted
            CoxTimeVaryingFitter objects from lifelines.
        conspiracies: List of conspiracy column names.
        cluster_assignments: Mapping from conspiracy name to cluster id
            (from semantic clustering in notebook 01).
        entry_times: Mapping from conspiracy name to first-appearance time step.
        seed_fraction: Fraction of eligible nodes to seed at entry times.
        steps: Number of hourly time steps to simulate.
        hawkes_params: Global Hawkes (mu, alpha, beta) for re-sharing.
        exposure_window: Rolling window in hours for s7 exposure.
            The default is 336 h or 14 days.
        require_peer_exposure: If true, candidates with no visible exposed
            neighbor are ineligible for organic adoption.

    Returns:
        Fully configured SimulationConfig ready for the engine.
    """
    specs: Dict[int, CoxModelSpec] = {}
    for model_num, ctv in cox_models.items():
        specs[model_num] = _extract_cox_model_spec(model_num, ctv)

    max_model = max(specs.keys())
    logger.info(
        f"Built SimulationConfig: {len(specs)} models (1..{max_model}), "
        f"{len(conspiracies)} conspiracies, {steps} steps"
    )

    return SimulationConfig(
        cox_models=specs,
        conspiracies=conspiracies,
        cluster_assignments=cluster_assignments,
        entry_times=entry_times,
        seed_fraction=seed_fraction,
        steps=steps,
        max_model_number=max_model,
        hawkes_params=hawkes_params,
        exposure_window=exposure_window,
        require_peer_exposure=require_peer_exposure,
    )
