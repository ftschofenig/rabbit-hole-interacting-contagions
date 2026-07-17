"""
Hazard computation dispatching to the correct Cox model (1-5) based
on the agent's adoption count.

Fixes from old ABM:
- Dispatches to Models 1-5 instead of collapsing to 2 hazard functions
- Correctly computes tau (time origin) per model
- Includes cross_cluster covariate
- Uses P = 1 - exp(-h) instead of P = min(1, h)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conspiracy_analysis.simulation.agent_state import AgentState
    from conspiracy_analysis.simulation.config import (
        CoxModelSpec,
        ScenarioConfig,
        SimulationConfig,
    )


def _model_and_time_origin(
    agent: AgentState,
    t: float,
    sim_config: SimulationConfig,
) -> tuple[CoxModelSpec, float] | tuple[None, None]:
    """Resolve the sequential Cox model and its elapsed time origin."""
    adoption_n = agent.adoption_number_for_next
    model = sim_config.get_model_for_adoption(adoption_n)

    if model.model_number == 1:
        return model, max(0.0, float(t))

    prev_adoption_time = agent.get_nth_adoption_time(adoption_n - 1)
    if prev_adoption_time is None:
        prev_adoption_time = agent.get_last_adoption_time()
    if prev_adoption_time is None:
        return None, None
    return model, max(0.0, float(t) - float(prev_adoption_time))


def _partial_log_hazard(
    agent: AgentState,
    conspiracy: str,
    s7: int,
    model: CoxModelSpec,
    sim_config: SimulationConfig,
) -> float:
    """Compute the Cox linear predictor for one candidate narrative."""
    s7_key = min(s7, 4)
    log_ph = model.s7_scores.get(s7_key, model.s7_scores.get(4, 0.0))
    log_ph += model.conspiracy_scores.get(conspiracy, 0.0)
    log_ph += model.beta_degree * agent.log_degree

    if model.model_number >= 2:
        cross = 1.0 if agent.is_cross_cluster(
            conspiracy, sim_config.cluster_assignments
        ) else 0.0
        log_ph += model.beta_cross_cluster * cross
    return float(log_ph)


def compute_raw_hazard(
    agent: AgentState,
    conspiracy: str,
    t: float,
    s7: int,
    sim_config: SimulationConfig,
    scenario: ScenarioConfig,
) -> float:
    """Compute the instantaneous hazard h for a single node-conspiracy pair.

    Returns the raw hazard h = h0(tau) * exp(sum_beta_x), NOT the probability.
    This is needed for the competing risks formulation where individual hazards
    are summed before converting to a single adoption probability.

    Args:
        agent: Current agent state.
        conspiracy: Conspiracy being evaluated for adoption.
        t: Current simulation time step.
        s7: Number of exposed neighbors (0 if peer influence disabled).
        sim_config: Full simulation configuration.
        scenario: Scenario configuration.

    Returns:
        Instantaneous hazard h >= 0.
    """
    model, tau = _model_and_time_origin(agent, t, sim_config)
    if model is None:
        return 0.0

    # Baseline hazard
    h0 = model.baseline.h0(tau)
    if h0 <= 0:
        return 0.0

    log_ph = _partial_log_hazard(agent, conspiracy, s7, model, sim_config)

    # Instantaneous hazard
    h = h0 * math.exp(log_ph)

    return h


def compute_interval_hazard(
    agent: AgentState,
    conspiracy: str,
    t: float,
    s7: int,
    sim_config: SimulationConfig,
    scenario: ScenarioConfig,
    interval_hours: float = 1.0,
) -> float:
    """Compute cumulative Cox hazard over the next simulation interval.

    Covariates are held fixed over the interval. The returned value is the
    exact parametric baseline increment multiplied by the Cox partial hazard.
    It is suitable for competing risks aggregation and is not a probability.
    """
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    model, tau_start = _model_and_time_origin(agent, t, sim_config)
    if model is None:
        return 0.0

    baseline_increment = model.baseline.integrated_hazard(
        tau_start, tau_start + interval_hours
    )
    if baseline_increment <= 0:
        return 0.0

    log_ph = _partial_log_hazard(agent, conspiracy, s7, model, sim_config)
    return float(baseline_increment * math.exp(log_ph))


def compute_adoption_hazard(
    agent: AgentState,
    conspiracy: str,
    t: float,
    s7: int,
    sim_config: SimulationConfig,
    scenario: ScenarioConfig,
) -> float:
    """Compute adoption probability for a single node-conspiracy pair.

    Wrapper around compute_interval_hazard that converts H to P = 1 - exp(-H).
    Kept for backwards compatibility.

    Args:
        agent: Current agent state.
        conspiracy: Conspiracy being evaluated for adoption.
        t: Current simulation time step.
        s7: Number of exposed neighbors (0 if peer influence disabled).
        sim_config: Full simulation configuration.
        scenario: Scenario configuration.

    Returns:
        Adoption probability in [0, 1].
    """
    h = compute_interval_hazard(
        agent, conspiracy, t, s7, sim_config, scenario, interval_hours=1.0
    )
    if h <= 0:
        return 0.0
    return 1.0 - math.exp(-h)
