"""
Cox proportional hazards models for first-time conspiracy adoption.

Implements sequential Cox models:
- Model 1: First conspiracy adoption (no prior flags)
- Model 2: Second conspiracy adoption (first_conspiracy flag + cross_cluster)
- Model 3+: Third+ adoption (cross_cluster only, NO prior conspiracy flags)
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from conspiracy_analysis.config import get_baseline_conspiracy

logger = logging.getLogger(__name__)

BASELINE_CONSPIRACY = get_baseline_conspiracy()


def fit_cox_model(
    df_long: pd.DataFrame,
    model_number: int,
    penalizer: float = 0.0,
    correction_for_censored: bool = False,
    gateway: bool = False,
    baseline: str = BASELINE_CONSPIRACY,
) -> Tuple[CoxTimeVaryingFitter, pd.DataFrame]:
    """Fit a Cox time-varying model on long-form data.

    Applies the correct formula based on model number:
    - Model 1: exposure dummies + degree + conspiracy dummies
    - Model 2a: + cross_cluster + conspiracy dummies (which conspiracy is adopted second)
    - Model 2b: + cross_cluster + first_conspiracy dummies (gateway effect)
    - Model 3+: + cross_cluster + conspiracy dummies

    Note:
        Lifelines does not support cluster-robust standard errors for
        CoxTimeVaryingFitter (both ``robust=True`` and ``_compute_residuals``
        are NotImplementedError stubs in lifelines 0.30.0). The reported
        SEs and CIs treat each row as an independent observation, but the
        long-form dataset has multiple correlated rows per user.

    Args:
        df_long: Long-form DataFrame with time-varying covariates.
        model_number: Which sequential model (1, 2, or 3+).
        penalizer: L2 penalizer for regularization.
        correction_for_censored: Whether censored neighbor correction is active.
        gateway: If True and model_number==2, use first_conspiracy dummies (Model 2b).

    Returns:
        Tuple of (fitted CoxTimeVaryingFitter, prepared DataFrame).
    """
    df = df_long.copy()

    # Feature engineering
    df["s_7"] = df["s_7"].astype(int)
    df["s7_d1"] = (df["s_7"] == 1).astype(int)
    df["s7_d2"] = (df["s_7"] == 2).astype(int)
    df["s7_d3"] = (df["s_7"] == 3).astype(int)
    df["s7_d4"] = (df["s_7"] >= 4).astype(int)
    df["degree"] = np.log1p(df["degree"])

    # Tau-scale shift for Models 2+
    if model_number == 2 and "first_time" in df.columns:
        df["entry"] = (df["entry"] - df["first_time"]).clip(lower=0)
        df["exit"] = df["exit"] - df["first_time"]
        df = df.drop(columns=["first_time"], errors="ignore")
    elif model_number >= 3:
        time_col = "second_time" if "second_time" in df.columns else "prior_time"
        if time_col in df.columns:
            df["entry"] = (df["entry"] - df[time_col]).clip(lower=0)
            df["exit"] = df["exit"] - df[time_col]
            df = df.drop(columns=[time_col], errors="ignore")

    # Drop helper columns not used as covariates
    df = df.drop(columns=["n_prior"], errors="ignore")

    # Drop invalid intervals
    df = df[df["exit"] > df["entry"]].copy()

    # Build exposure terms
    if correction_for_censored:
        exposure_terms = "s7_d2 + s7_d3 + s7_d4"
    else:
        exposure_terms = "s7_d1 + s7_d2 + s7_d3 + s7_d4"

    # Build formula based on model number
    # Choose dummy column: Model 2b (gateway) uses first_conspiracy; all others use conspiracy
    if gateway and model_number == 2 and "first_conspiracy" in df.columns:
        dummy_column = "first_conspiracy"
    else:
        dummy_column = "conspiracy"

    df, dummy_cols = _create_conspiracy_dummies(df, dummy_column, baseline=baseline)

    # Models 2+ include cross_cluster when available
    if model_number >= 2 and "cross_cluster" in df.columns:
        formula = f"{exposure_terms} + degree + cross_cluster + {' + '.join(dummy_cols)}"
    else:
        formula = f"{exposure_terms} + degree + {' + '.join(dummy_cols)}"

    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(
        df,
        id_col="id",
        event_col="event",
        start_col="entry",
        stop_col="exit",
        formula=formula,
        show_progress=False,
    )

    return ctv, df


def _parse_model_name(name: str) -> Tuple[int, bool]:
    """Parse model name into number and gateway flag.

    Examples: 'model_1' -> (1, False), 'model_2a' -> (2, False),
              'model_2b' -> (2, True), 'model_3' -> (3, False).
    """
    suffix = name.split("_", 1)[1]  # '1', '2a', '2b', '3', etc.
    gateway = suffix.endswith("b")
    suffix = suffix.rstrip("ab")
    return int(suffix), gateway


def fit_all_cox_models(
    short_form_data,
    G,
    step: int = 8,
    penalizer: float = 0.0,
    correction_for_censored: bool = False,
    exposure_window: Optional[int] = None,
    baseline: str = BASELINE_CONSPIRACY,
) -> Dict[str, Tuple[CoxTimeVaryingFitter, pd.DataFrame]]:
    """Fit sequential Cox models for all provided short-form DataFrames.

    Accepts either the old 3-positional-argument signature (backward compatible)
    or a dict mapping model names to short-form DataFrames.

    Args:
        short_form_data: Either a dict {"model_1": df, "model_2": df, ...}
            or the first positional DataFrame (df_short_1) for backward compat.
        G: Network graph.
        step: Time discretization step in hours.
        penalizer: L2 penalizer.
        correction_for_censored: Apply censored neighbor correction.
        exposure_window: Lookback window in hours for counting active neighbors.
            If None, uses the create_long_form default of 336 h or 14 days.

    Returns:
        Dict mapping model names to (fitted model, long-form data) tuples.
    """
    from conspiracy_analysis.data.preprocessing import create_long_form

    # Require a mapping from model names to DataFrames.
    if isinstance(short_form_data, pd.DataFrame):
        df_short_1 = short_form_data
        df_short_2 = G
        raise TypeError(
            "fit_all_cox_models no longer accepts 3 positional DataFrames. "
            "Pass a dict: fit_all_cox_models({'model_1': df1, 'model_2': df2, ...}, G)"
        )

    results = {}

    for name, df_short in short_form_data.items():
        model_number, gateway = _parse_model_name(name)
        logger.info(f"Fitting {name} (model_number={model_number}, gateway={gateway})...")

        # Determine which time column to merge for tau-scale shift
        time_col = None
        if model_number == 2 and "first_time" in df_short.columns:
            time_col = "first_time"
        elif model_number >= 3:
            if "second_time" in df_short.columns:
                time_col = "second_time"
            elif "prior_time" in df_short.columns:
                time_col = "prior_time"

        kwargs = dict(step=step, correction_for_censored_neighbors=correction_for_censored)
        if exposure_window is not None:
            kwargs["exposure_window"] = exposure_window
        df_long = create_long_form(df_short, G, **kwargs)

        # Merge timing column for tau-scale shift
        if time_col and time_col in df_short.columns:
            merge_cols = ["id", time_col]
            # Also pass n_prior if available
            if "n_prior" in df_short.columns:
                merge_cols.append("n_prior")
            merge_times = df_short[merge_cols].drop_duplicates()
            df_long = df_long.merge(merge_times, on="id", how="left")

        ctv, df = fit_cox_model(
            df_long, model_number=model_number,
            penalizer=penalizer, correction_for_censored=correction_for_censored,
            gateway=gateway, baseline=baseline,
        )
        results[name] = (ctv, df)

    return results


def _create_conspiracy_dummies(
    df: pd.DataFrame,
    column: str,
    baseline: str = BASELINE_CONSPIRACY,
) -> Tuple[pd.DataFrame, list]:
    """Create one-hot dummies for a conspiracy column, dropping the baseline.

    Args:
        df: DataFrame with the column to encode.
        column: Column name to create dummies from.
        baseline: Conspiracy to use as the reference (omitted) category.
            Falls back to most-common-by-event-count if not found.

    Returns:
        Tuple of (DataFrame with dummies added, list of dummy column names).
    """
    df[column] = df[column].astype(str)
    unique_vals = df[column].unique()
    if baseline in unique_vals:
        most_common = baseline
    else:
        logger.warning(
            f"Baseline '{baseline}' not found in column '{column}'. "
            f"Falling back to most common by event count."
        )
        most_common = df[df["event"] == 1][column].value_counts().index[0]
    logger.info(f"Setting baseline (reference) to: {most_common}")

    other = sorted([c for c in df[column].unique() if c != most_common])
    ordered = [most_common] + other
    df[column] = pd.Categorical(df[column], categories=ordered, ordered=True)

    dummies = pd.get_dummies(df[column], prefix="fc", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)
