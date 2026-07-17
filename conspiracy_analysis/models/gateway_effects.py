"""
Gateway effect analysis from Cox model coefficients.

Extracts and interprets the first_conspiracy dummy coefficients from
Model 2b (gateway) to identify which conspiracies act as gateways
(entry points) that accelerate or inhibit adoption of subsequent
conspiracies.

Also provides a two-dimensional gateway identification approach combining
contagiousness (Model 1 fc_ HRs) with acceleration (Model 2b fc_ HRs).
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from conspiracy_analysis.config import get_baseline_conspiracy
from conspiracy_analysis.utils.fallback_logging import log_bootstrap_fallback

logger = logging.getLogger(__name__)

BASELINE_CONSPIRACY = get_baseline_conspiracy()


def extract_gateway_coefficients(
    ctv: CoxTimeVaryingFitter,
) -> pd.DataFrame:
    """Extract first_conspiracy coefficients from a fitted Model 2b (gateway).

    These coefficients represent how much having adopted conspiracy X first
    changes the hazard of adopting a second conspiracy, relative to the
    baseline (ConsProb_fakenews).

    Args:
        ctv: Fitted CoxTimeVaryingFitter from Model 2b (gateway).

    Returns:
        DataFrame with columns: conspiracy, coefficient, hazard_ratio,
        exp_coef_lower_95, exp_coef_upper_95, p_value.
    """
    summary = ctv.summary

    fc_rows = summary[summary.index.str.startswith("fc_")]
    if fc_rows.empty:
        logger.warning("No first_conspiracy (fc_) coefficients found in model.")
        return pd.DataFrame()

    records = []
    for idx, row in fc_rows.iterrows():
        consp_name = idx[3:]  # Remove 'fc_' prefix
        records.append({
            "conspiracy": consp_name,
            "coefficient": row["coef"],
            "hazard_ratio": row["exp(coef)"],
            "exp_coef_lower_95": row["exp(coef) lower 95%"],
            "exp_coef_upper_95": row["exp(coef) upper 95%"],
            "p_value": row["p"],
        })

    df = pd.DataFrame(records).sort_values("hazard_ratio", ascending=False)
    df = df.reset_index(drop=True)

    logger.info(f"Extracted {len(df)} gateway coefficients from Model 2b")
    return df


def identify_gateway_conspiracies(
    ctv: CoxTimeVaryingFitter,
    significance_level: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Classify conspiracies as accelerators or inhibitors of second adoption.

    Args:
        ctv: Fitted CoxTimeVaryingFitter from Model 2b (gateway).
        significance_level: P-value threshold for significance.

    Returns:
        Dict with keys:
        - 'all': All gateway coefficients.
        - 'accelerators': Significant positive effects (HR > 1).
        - 'inhibitors': Significant negative effects (HR < 1).
        - 'cross_cluster': Cross-cluster coefficient info (if present).
    """
    df = extract_gateway_coefficients(ctv)
    if df.empty:
        return {"all": df, "accelerators": df, "inhibitors": df}

    significant = df[df["p_value"] < significance_level]
    accelerators = significant[significant["hazard_ratio"] > 1.0]
    inhibitors = significant[significant["hazard_ratio"] < 1.0]

    result = {
        "all": df,
        "accelerators": accelerators,
        "inhibitors": inhibitors,
    }

    # Extract cross_cluster coefficient if present
    summary = ctv.summary
    if "cross_cluster" in summary.index:
        cc_row = summary.loc["cross_cluster"]
        result["cross_cluster"] = {
            "coefficient": cc_row["coef"],
            "hazard_ratio": cc_row["exp(coef)"],
            "p_value": cc_row["p"],
        }
        logger.info(
            f"Cross-cluster effect: HR={cc_row['exp(coef)']:.3f}, "
            f"p={cc_row['p']:.4f}"
        )

    return result


def _extract_fc_hazard_ratios(ctv: CoxTimeVaryingFitter) -> pd.DataFrame:
    """Extract fc_ (first_conspiracy) hazard ratios from any fitted Cox model.

    Args:
        ctv: Fitted CoxTimeVaryingFitter containing fc_ covariates.

    Returns:
        DataFrame with columns: conspiracy, hr, ci_lower, ci_upper, p.
    """
    summary = ctv.summary
    fc_rows = summary[summary.index.str.startswith("fc_")]
    if fc_rows.empty:
        return pd.DataFrame()

    records = []
    for idx, row in fc_rows.iterrows():
        records.append({
            "conspiracy": idx[3:].replace("ConsProb_", ""),  # strip 'fc_' and 'ConsProb_' prefixes
            "hr": row["exp(coef)"],
            "ci_lower": row["exp(coef) lower 95%"],
            "ci_upper": row["exp(coef) upper 95%"],
            "p": row["p"],
        })
    return pd.DataFrame(records)


def identify_gateway_2d(
    ctv_1: CoxTimeVaryingFitter,
    ctv_2b: CoxTimeVaryingFitter,
    reference_conspiracy: str = BASELINE_CONSPIRACY,
    bootstrap_intervals: Optional[object] = None,
) -> pd.DataFrame:
    """Two-dimensional gateway identification from Models 1 and 2b.

    Combines contagiousness (Model 1 fc_ HRs, x-axis) with acceleration
    of second adoption (Model 2b gateway fc_ HRs, y-axis) into a single
    DataFrame. The reference conspiracy is added as (1.0, 1.0).

    Args:
        ctv_1: Fitted CoxTimeVaryingFitter from Model 1.
        ctv_2b: Fitted CoxTimeVaryingFitter from Model 2b (gateway).
        reference_conspiracy: The reference (omitted) conspiracy name
            (e.g., 'ConsProb_fakenews'). Added at HR = 1.0 on both axes.
        bootstrap_intervals: Optional bootstrap gateway table or artifact.
            When provided, its percentile intervals replace model based CIs.

    Returns:
        DataFrame with columns: conspiracy, model1_hr, model1_ci_lower,
        model1_ci_upper, model2_hr, model2_ci_lower, model2_ci_upper,
        model2_p.
    """
    fallback_reason = "using model summary confidence interval fallback"
    if bootstrap_intervals is not None:
        if isinstance(bootstrap_intervals, dict):
            boot = bootstrap_intervals.get("gateway_intervals")
        else:
            boot = bootstrap_intervals
        if isinstance(boot, pd.DataFrame) and not boot.empty:
            out = boot.copy()
            if "model2_p" not in out.columns:
                out["model2_p"] = np.nan
            if "interval_source" not in out.columns:
                out["interval_source"] = "bootstrap"
            return out
        log_bootstrap_fallback(
            "identify_gateway_2d",
            "missing gateway interval table",
        )
        fallback_reason = (
            "using model summary confidence interval fallback after missing gateway interval table"
        )

    log_bootstrap_fallback(
        "identify_gateway_2d",
        fallback_reason,
        fallback_interval_source="model_summary",
    )

    m1 = _extract_fc_hazard_ratios(ctv_1)
    m2 = _extract_fc_hazard_ratios(ctv_2b)

    if m1.empty or m2.empty:
        logger.warning("Could not extract fc_ coefficients from one or both models.")
        return pd.DataFrame()

    m1 = m1.rename(columns={
        "hr": "model1_hr",
        "ci_lower": "model1_ci_lower",
        "ci_upper": "model1_ci_upper",
        "p": "model1_p",
    })
    m2 = m2.rename(columns={
        "hr": "model2_hr",
        "ci_lower": "model2_ci_lower",
        "ci_upper": "model2_ci_upper",
        "p": "model2_p",
    })

    merged = pd.merge(
        m1[["conspiracy", "model1_hr", "model1_ci_lower", "model1_ci_upper"]],
        m2[["conspiracy", "model2_hr", "model2_ci_lower", "model2_ci_upper", "model2_p"]],
        on="conspiracy",
        how="outer",
    )
    merged["interval_source"] = "model_summary"

    # Add reference conspiracy at (1.0, 1.0)
    ref_name = reference_conspiracy.replace("ConsProb_", "")
    ref_row = pd.DataFrame([{
        "conspiracy": ref_name,
        "model1_hr": 1.0,
        "model1_ci_lower": 1.0,
        "model1_ci_upper": 1.0,
        "model2_hr": 1.0,
        "model2_ci_lower": 1.0,
        "model2_ci_upper": 1.0,
        "model2_p": np.nan,
        "interval_source": "reference",
    }])
    merged = pd.concat([merged, ref_row], ignore_index=True)

    logger.info(
        f"Two-dimensional gateway identification: {len(merged)} conspiracies "
        f"(including reference '{ref_name}')"
    )
    return merged
