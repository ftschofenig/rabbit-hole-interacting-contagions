"""
Baseline hazard parametrization for converting Cox models to simulation hazards.

Model 1 (first conspiracy): Linear/constant baseline hazard h0(t) = a
Models 2+ (subsequent):     Weibull baseline hazard h0(t) = (k/lambda) * (t/lambda)^(k-1)

These parametric forms are fitted to the empirical cumulative baseline hazards
extracted from lifelines CoxTimeVaryingFitter objects.
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from lifelines import CoxTimeVaryingFitter

logger = logging.getLogger(__name__)


def extract_cumulative_baseline_hazard(
    ctv: CoxTimeVaryingFitter,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract empirical cumulative baseline hazard from a fitted Cox model.

    Args:
        ctv: Fitted CoxTimeVaryingFitter from lifelines.

    Returns:
        Tuple of (times, cumulative_hazard_values) as numpy arrays.
    """
    cum_baseline = ctv.baseline_cumulative_hazard_
    times = cum_baseline.index.to_numpy()
    values = cum_baseline.iloc[:, 0].to_numpy()
    return times, values


def cumulative_hazard_at_time(
    baseline: pd.DataFrame,
    t_ref: float,
) -> float:
    """Interpolate the cumulative baseline hazard at a reference time.

    Args:
        baseline: DataFrame with a single column of cumulative hazard values
            indexed by event time (output of `baseline_cumulative_hazard_`).
        t_ref: Reference time (hours).

    Returns:
        Cumulative baseline hazard at t_ref. Returns 0 if t_ref is before the
        first event, the final value if t_ref is beyond the last event, and
        step-function interpolation otherwise.
    """
    times = baseline.index.to_numpy()
    values = baseline.iloc[:, 0].to_numpy()
    if t_ref < times[0]:
        return 0.0
    if t_ref >= times[-1]:
        return float(values[-1])
    # Step function: cumulative hazard is right-continuous at each event time
    idx = np.searchsorted(times, t_ref, side='right') - 1
    return float(values[idx])


def fit_linear_baseline(
    times: np.ndarray,
    cum_hazard: np.ndarray,
    r_squared_warn_threshold: float = 0.90,
) -> Dict:
    """Fit a linear function to the cumulative baseline hazard.

    Used for Model 1 (first conspiracy adoption).
    H0(t) = a * t  =>  h0(t) = a (constant hazard rate).

    Computes goodness-of-fit statistics (R², relative RMSE) and emits a
    WARNING-level log if the linear (constant-hazard) assumption appears
    inadequate. The downstream t* / decay-time computations all use this
    slope as a reference, so a poor fit means the t* values lose their
    anchor and should not be reported as-is. Validation added per audit
    finding H2.

    Args:
        times: Time points from the empirical baseline hazard.
        cum_hazard: Cumulative baseline hazard values.
        r_squared_warn_threshold: Emit a WARNING when R² falls below this
            value. Default 0.90.

    Returns:
        Dict with keys:
        - 'slope': The constant hazard rate parameter 'a'.
        - 'rmse': Root mean squared error.
        - 'r_squared': Coefficient of determination (1 - SS_res/SS_tot,
          standard centered form). Closer to 1 = better linear fit.
        - 'relative_rmse': RMSE divided by the mean of cum_hazard. Useful
          as a unit-free fit-quality measure.
        - 'fitted_values': Fitted cumulative hazard at each time point.
    """
    # Through-origin OLS: minimize sum((cum_hazard - slope*times)^2)
    slope = np.sum(times * cum_hazard) / np.sum(times ** 2)

    if slope < 0:
        logger.warning(f"Negative baseline slope ({slope:.6f}). Clipping to 0.")
        slope = 0.0

    fitted = slope * times
    residuals = cum_hazard - fitted
    rmse = np.sqrt(np.mean(residuals ** 2))

    # Compute diagnostics for the linear cumulative hazard fit.
    ss_res = np.sum(residuals ** 2)
    cum_mean = cum_hazard.mean()
    ss_tot = np.sum((cum_hazard - cum_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    relative_rmse = rmse / cum_mean if cum_mean > 0 else float("inf")

    logger.info(
        f"Linear baseline fit: a = {slope:.6e}, RMSE = {rmse:.6e}, "
        f"R² = {r_squared:.4f}, relative RMSE = {relative_rmse:.2%}"
    )

    if r_squared < r_squared_warn_threshold:
        logger.warning(
            "Linear baseline fit may be inadequate: R² = %.4f (< %.2f), "
            "relative RMSE = %.2f%%. The constant-hazard assumption appears to "
            "be violated. Downstream t* / decay-to-baseline values that use "
            "this slope as a reference may be unreliable.",
            r_squared, r_squared_warn_threshold, relative_rmse * 100,
        )

    return {
        "slope": slope,
        "rmse": rmse,
        "r_squared": r_squared,
        "relative_rmse": relative_rmse,
        "fitted_values": fitted,
    }


def fit_weibull_baseline(
    times: np.ndarray,
    cum_hazard: np.ndarray,
    initial_guess: Tuple[float, float] = (1.0, 1.0),
    r_squared_warn_threshold: float = 0.95,
) -> Dict:
    """Fit a Weibull distribution to the cumulative baseline hazard.

    Used for Models 2+ (subsequent conspiracy adoptions).
    H0(t) = (t / lambda)^k  =>  h0(t) = (k / lambda) * (t / lambda)^(k-1)

    Computes goodness-of-fit statistics (R², relative RMSE) and emits a
    WARNING-level log if the Weibull form fits poorly. The downstream t*
    computations use the fitted (k, lambda) so a poor fit means the
    decay-to-baseline values are unreliable. Validation added per audit
    finding H2.

    Args:
        times: Time points from the empirical baseline hazard.
        cum_hazard: Cumulative baseline hazard values.
        initial_guess: Starting values for (shape k, scale lambda).
        r_squared_warn_threshold: Emit a WARNING when R² falls below this
            value. Default 0.95.

    Returns:
        Dict with keys:
        - 'shape': Weibull shape parameter k.
        - 'scale': Weibull scale parameter lambda.
        - 'rmse': Root mean squared error.
        - 'r_squared': Coefficient of determination (1 - SS_res/SS_tot,
          standard centered form). Closer to 1 = better Weibull fit.
        - 'relative_rmse': RMSE divided by the mean of cum_hazard.
        - 'fitted_values': Fitted cumulative hazard at each time point.
    """
    def weibull_cum(t, k, lam):
        return (t / lam) ** k

    popt, _ = curve_fit(
        weibull_cum, times, cum_hazard,
        p0=initial_guess,
        bounds=(0, np.inf),
        maxfev=10000,
    )
    shape, scale = popt

    fitted = weibull_cum(times, shape, scale)
    residuals = cum_hazard - fitted
    rmse = np.sqrt(np.mean(residuals ** 2))

    # Compute diagnostics for the Weibull cumulative hazard fit.
    ss_res = np.sum(residuals ** 2)
    cum_mean = cum_hazard.mean()
    ss_tot = np.sum((cum_hazard - cum_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    relative_rmse = rmse / cum_mean if cum_mean > 0 else float("inf")

    logger.info(
        f"Weibull baseline fit: k = {shape:.4f}, lambda = {scale:.4f}, "
        f"RMSE = {rmse:.6e}, R² = {r_squared:.4f}, relative RMSE = {relative_rmse:.2%}"
    )

    if r_squared < r_squared_warn_threshold:
        logger.warning(
            "Weibull baseline fit may be inadequate: R² = %.4f (< %.2f), "
            "relative RMSE = %.2f%%. The two-parameter Weibull form does not "
            "describe the empirical cumulative hazard well. Decay-to-baseline "
            "t* values derived from these (k, lambda) parameters may be "
            "unreliable.",
            r_squared, r_squared_warn_threshold, relative_rmse * 100,
        )

    return {
        "shape": shape,
        "scale": scale,
        "rmse": rmse,
        "r_squared": r_squared,
        "relative_rmse": relative_rmse,
        "fitted_values": fitted,
    }


def parametrize_all_baselines(
    cox_models: Dict[str, CoxTimeVaryingFitter],
) -> Dict[str, Dict]:
    """Fit parametric baseline hazards for all Cox models.

    Model 1 gets a linear fit; Models 2+ get Weibull fits.

    Args:
        cox_models: Dict mapping model names ('model_1', 'model_2', 'model_3')
            to fitted CoxTimeVaryingFitter objects.

    Returns:
        Dict mapping model names to their baseline hazard parameters.
        Each entry includes 'type' ('linear' or 'weibull') and fitted params.
    """
    results = {}

    for name, ctv in cox_models.items():
        times, cum_haz = extract_cumulative_baseline_hazard(ctv)

        if name == "model_1":
            fit = fit_linear_baseline(times, cum_haz)
            fit["type"] = "linear"
            fit["times"] = times
            fit["empirical"] = cum_haz
        else:
            fit = fit_weibull_baseline(times, cum_haz)
            fit["type"] = "weibull"
            fit["times"] = times
            fit["empirical"] = cum_haz

        results[name] = fit
        logger.info(f"{name}: baseline type = {fit['type']}")

    return results


def calculate_baseline_hazard(
    t: float,
    baseline_params: Dict,
) -> float:
    """Calculate instantaneous baseline hazard h0(t) from fitted parameters.

    Args:
        t: Time since entry (hours). Must be > 0 for Weibull.
        baseline_params: Dict from fit_linear_baseline or fit_weibull_baseline.
            Must contain 'type' key ('linear' or 'weibull').

    Returns:
        Instantaneous baseline hazard rate at time t.
    """
    if baseline_params["type"] == "linear":
        return baseline_params["slope"]

    elif baseline_params["type"] == "weibull":
        k = baseline_params["shape"]
        lam = baseline_params["scale"]
        if t <= 0:
            return 0.0
        return (k / lam) * (t / lam) ** (k - 1)

    else:
        raise ValueError(f"Unknown baseline type: {baseline_params['type']}")


def compute_decay_to_baseline_time(
    weibull_params: Dict,
    linear_params: Dict,
) -> float:
    """Compute time at which Weibull hazard decays to Model 1's constant level.

    Solves h_0(t*) = slope for the Weibull instantaneous hazard:
    (k/lambda)(t*/lambda)^(k-1) = slope
    => t* = lambda * (slope * lambda / k)^(1/(k-1))

    Args:
        weibull_params: Dict with 'shape' (k) and 'scale' (lambda) keys.
        linear_params: Dict with 'slope' key (Model 1's constant hazard rate).

    Returns:
        t* in hours. Returns inf if k >= 1 (non-decaying hazard).
    """
    k = weibull_params["shape"]
    lam = weibull_params["scale"]
    slope = linear_params["slope"]

    if k >= 1.0:
        return float("inf")  # Weibull is increasing or constant, never decays

    if slope <= 0:
        return float("inf")  # No positive baseline to cross

    t_star = lam * (slope * lam / k) ** (1.0 / (k - 1))

    logger.info(
        f"Decay-to-baseline time: t* = {t_star:.1f} hours ({t_star / 24:.1f} days) "
        f"[k={k:.4f}, lambda={lam:.1f}, slope={slope:.6e}]"
    )

    return t_star


def compute_all_decay_times(
    baseline_params: Dict[str, Dict],
) -> Dict[str, float]:
    """Compute decay-to-baseline times for all Weibull models.

    For each model with a Weibull baseline, computes the time at which
    its instantaneous hazard drops to Model 1's constant baseline level.

    Args:
        baseline_params: Dict mapping model names to their baseline hazard
            parameters (from parametrize_all_baselines). Must include 'model_1'.

    Returns:
        Dict mapping model name to t* in hours.
    """
    if "model_1" not in baseline_params:
        raise ValueError("baseline_params must include 'model_1' for reference level")

    linear = baseline_params["model_1"]
    results = {}

    for name in sorted(baseline_params.keys()):
        params = baseline_params[name]
        if params["type"] == "weibull":
            t_star = compute_decay_to_baseline_time(params, linear)
            results[name] = t_star
            logger.info(f"{name}: decay-to-baseline t* = {t_star:.1f} hours ({t_star / 24:.1f} days)")

    return results


def create_hazard_calculator(
    ctv: CoxTimeVaryingFitter,
    baseline_params: Dict,
) -> callable:
    """Create an optimized hazard function for use in simulations.

    Pre-computes lookup tables for s7 exposure dummies and conspiracy
    dummy coefficients to avoid repeated dictionary lookups.

    Args:
        ctv: Fitted CoxTimeVaryingFitter.
        baseline_params: Fitted baseline hazard parameters.

    Returns:
        Callable that takes (s_7_val, log_degree_val, conspiracy_name, [time])
        and returns the instantaneous hazard rate.
    """
    params = ctv.params_

    # Pre-compute s7 risk scores
    s7_scores = {}
    has_d1 = "s7_d1" in params
    for s_7 in range(500):
        score = 0.0
        if has_d1 and s_7 == 1:
            score = params["s7_d1"]
        elif s_7 == 2 and "s7_d2" in params:
            score = params["s7_d2"]
        elif s_7 == 3 and "s7_d3" in params:
            score = params["s7_d3"]
        elif s_7 >= 4 and "s7_d4" in params:
            score = params["s7_d4"]
        s7_scores[s_7] = score

    # Pre-compute conspiracy risk scores
    consp_scores = {}
    for col in params.index:
        if col.startswith("fc_"):
            consp_scores[col[3:]] = params[col]

    beta_degree = params.get("degree", 0.0)
    beta_cross_cluster = params.get("cross_cluster", 0.0)
    is_linear = baseline_params["type"] == "linear"

    if is_linear:
        slope = baseline_params["slope"]

        def hazard_linear(s_7_val, log_degree_val, conspiracy_name):
            s7_part = s7_scores.get(s_7_val, s7_scores.get(4, 0.0))
            consp_part = consp_scores.get(conspiracy_name, 0.0)
            log_ph = s7_part + beta_degree * log_degree_val + consp_part
            return slope * np.exp(log_ph)

        return hazard_linear

    else:
        k = baseline_params["shape"]
        lam = baseline_params["scale"]

        def hazard_weibull(s_7_val, log_degree_val, conspiracy_name, time,
                           cross_cluster=0):
            s7_part = s7_scores.get(s_7_val, s7_scores.get(4, 0.0))
            consp_part = consp_scores.get(conspiracy_name, 0.0)
            if time <= 0:
                return 0.0
            baseline = (k / lam) * (time / lam) ** (k - 1)
            log_ph = (s7_part + beta_degree * log_degree_val
                       + consp_part + beta_cross_cluster * cross_cluster)
            return baseline * np.exp(log_ph)

        return hazard_weibull
