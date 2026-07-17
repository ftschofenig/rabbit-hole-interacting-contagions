"""User clustered bootstrap utilities for survival model inference."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from conspiracy_analysis.analysis.statistics import (
    compute_semantic_barrier_analysis,
    compute_settler_effect,
)
from conspiracy_analysis.models.baseline_hazards import (
    calculate_baseline_hazard,
    compute_all_decay_times,
    parametrize_all_baselines,
)
from conspiracy_analysis.utils.fallback_logging import log_bootstrap_fallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBootstrapSpec:
    """Specification needed to refit one Cox time varying model."""

    name: str
    df_long: pd.DataFrame
    formula: str
    penalizer: float = 0.0
    strata: Optional[Union[str, Sequence[str]]] = None
    baseline_role: Optional[str] = None
    id_col: str = "id"
    event_col: str = "event"
    start_col: str = "entry"
    stop_col: str = "exit"


def load_bootstrap_artifact(path: str, required: bool = True) -> Optional[dict]:
    """Load a saved bootstrap artifact."""
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Missing bootstrap artifact: {path}. Run 01_analysis.ipynb first."
            )
        return None
    return joblib.load(path)


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _sample_long_form_by_user(
    df: pd.DataFrame,
    sampled_ids: Sequence[object],
    draw_index: int,
    id_col: str = "id",
) -> pd.DataFrame:
    """Copy all rows for sampled users and give every copy a unique id."""
    pieces = []
    grouped = {user_id: frame for user_id, frame in df.groupby(id_col, sort=False)}
    for position, user_id in enumerate(sampled_ids):
        if user_id not in grouped:
            continue
        piece = grouped[user_id].copy()
        piece[id_col] = f"{user_id}__boot{draw_index}_{position}"
        pieces.append(piece)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _fit_spec(spec: ModelBootstrapSpec, df_boot: pd.DataFrame) -> CoxTimeVaryingFitter:
    ctv = CoxTimeVaryingFitter(penalizer=spec.penalizer)
    kwargs = {
        "id_col": spec.id_col,
        "event_col": spec.event_col,
        "start_col": spec.start_col,
        "stop_col": spec.stop_col,
        "formula": spec.formula,
        "show_progress": False,
    }
    strata = _as_list(spec.strata)
    if strata:
        kwargs["strata"] = strata
    ctv.fit(df_boot, **kwargs)
    return ctv


def _record_coefficients(draw: int, model: str, ctv: CoxTimeVaryingFitter) -> List[dict]:
    return [
        {
            "draw": draw,
            "model": model,
            "term": term,
            "coef": float(coef),
            "hr": float(np.exp(coef)),
        }
        for term, coef in ctv.params_.items()
    ]


def _record_baseline(draw: int, model: str, params: Mapping[str, float]) -> dict:
    row = {
        "draw": draw,
        "model": model,
        "type": params["type"],
        "slope": np.nan,
        "shape": np.nan,
        "scale": np.nan,
        "rmse": float(params.get("rmse", np.nan)),
        "r_squared": float(params.get("r_squared", np.nan)),
        "relative_rmse": float(params.get("relative_rmse", np.nan)),
    }
    if params["type"] == "linear":
        row["slope"] = float(params["slope"])
    if params["type"] == "weibull":
        row["shape"] = float(params["shape"])
        row["scale"] = float(params["scale"])
    return row


def _effective_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero.")
    return n_jobs


def _parallel_call(
    function,
    task_args,
    n_jobs: int,
    parallel_backend: Optional[str],
    parallel_verbose: int,
):
    if n_jobs == 1:
        return [function(*args) for args in task_args]
    kwargs = {"n_jobs": n_jobs, "verbose": parallel_verbose}
    if parallel_backend is not None:
        kwargs["backend"] = parallel_backend
    return joblib.Parallel(**kwargs)(
        joblib.delayed(function)(*args) for args in task_args
    )


def _run_cox_bootstrap_draw(
    draw: int,
    sampled_ids: Sequence[object],
    specs: Sequence[ModelBootstrapSpec],
    eval_times_hours: Sequence[float],
    model_1_name: str,
) -> dict:
    coef_rows: List[dict] = []
    baseline_rows: List[dict] = []
    temporal_rows: List[dict] = []
    decay_rows: List[dict] = []
    failure_rows: List[dict] = []
    successes = {spec.name: 0 for spec in specs}
    fitted_for_baseline: Dict[str, CoxTimeVaryingFitter] = {}

    for spec in specs:
        df_boot = _sample_long_form_by_user(
            spec.df_long,
            sampled_ids,
            draw,
            spec.id_col,
        )
        if df_boot.empty or df_boot[spec.event_col].sum() == 0:
            failure_rows.append({
                "draw": draw,
                "model": spec.name,
                "error": "No rows or no events in bootstrap sample.",
            })
            continue
        try:
            ctv = _fit_spec(spec, df_boot)
        except Exception as exc:  # pragma: no cover
            failure_rows.append({
                "draw": draw,
                "model": spec.name,
                "error": repr(exc),
            })
            continue

        successes[spec.name] += 1
        coef_rows.extend(_record_coefficients(draw, spec.name, ctv))
        if spec.baseline_role:
            fitted_for_baseline[spec.name] = ctv

    if model_1_name in fitted_for_baseline:
        try:
            baselines = parametrize_all_baselines(fitted_for_baseline)
            for model, params in baselines.items():
                baseline_rows.append(_record_baseline(draw, model, params))

            m1_h0 = calculate_baseline_hazard(1.0, baselines[model_1_name])
            for model, params in baselines.items():
                if params["type"] != "weibull":
                    continue
                for time_value in eval_times_hours:
                    h0 = calculate_baseline_hazard(float(time_value), params)
                    temporal_rows.append({
                        "draw": draw,
                        "model": model,
                        "time_hours": float(time_value),
                        "ratio": float(h0 / m1_h0) if m1_h0 > 0 else np.nan,
                    })

            decay_times = compute_all_decay_times(baselines)
            for model, t_star in decay_times.items():
                decay_rows.append({
                    "draw": draw,
                    "model": model,
                    "t_star_hours": float(t_star),
                    "finite": bool(np.isfinite(t_star)),
                })
        except Exception as exc:  # pragma: no cover
            failure_rows.append({
                "draw": draw,
                "model": "__baseline__",
                "error": repr(exc),
            })

    return {
        "coef_rows": coef_rows,
        "baseline_rows": baseline_rows,
        "temporal_rows": temporal_rows,
        "decay_rows": decay_rows,
        "failure_rows": failure_rows,
        "successes": successes,
    }


def run_cox_user_bootstrap(
    specs: Sequence[ModelBootstrapSpec],
    master_user_ids: Sequence[object],
    n_draws: int,
    seed: int = 42,
    eval_times_hours: Sequence[float] = (1.0, 24.0, 72.0),
    min_successes: Optional[int] = None,
    model_1_name: str = "model_1",
    n_jobs: Optional[int] = 1,
    parallel_backend: Optional[str] = None,
    parallel_verbose: int = 0,
) -> dict:
    """Run a user clustered bootstrap across Cox models."""
    rng = np.random.default_rng(seed)
    master_user_ids = np.asarray(list(master_user_ids), dtype=object)
    n_jobs = _effective_n_jobs(n_jobs)
    sampled_ids_by_draw = [
        rng.choice(
            master_user_ids,
            size=len(master_user_ids),
            replace=True,
        )
        for _ in range(n_draws)
    ]
    task_args = [
        (
            draw,
            sampled_ids,
            specs,
            eval_times_hours,
            model_1_name,
        )
        for draw, sampled_ids in enumerate(sampled_ids_by_draw)
    ]
    draw_results = _parallel_call(
        _run_cox_bootstrap_draw,
        task_args,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_verbose=parallel_verbose,
    )

    coef_rows: List[dict] = []
    baseline_rows: List[dict] = []
    temporal_rows: List[dict] = []
    decay_rows: List[dict] = []
    failure_rows: List[dict] = []
    successes = {spec.name: 0 for spec in specs}

    for result in draw_results:
        coef_rows.extend(result["coef_rows"])
        baseline_rows.extend(result["baseline_rows"])
        temporal_rows.extend(result["temporal_rows"])
        decay_rows.extend(result["decay_rows"])
        failure_rows.extend(result["failure_rows"])
        for name, count in result["successes"].items():
            successes[name] += count

    if min_successes is not None:
        low_success = {
            name: count
            for name, count in successes.items()
            if count < min_successes
        }
        if low_success:
            raise RuntimeError(
                f"Bootstrap success threshold not met: {low_success}"
            )

    return {
        "metadata": {
            "n_draws": int(n_draws),
            "seed": int(seed),
            "n_jobs": int(n_jobs),
            "parallel_backend": parallel_backend,
            "n_master_user_ids": int(len(master_user_ids)),
            "eval_times_hours": [float(t) for t in eval_times_hours],
        },
        "coefficients_raw": pd.DataFrame(coef_rows),
        "baselines_raw": pd.DataFrame(baseline_rows),
        "temporal_hazard_ratios_raw": pd.DataFrame(temporal_rows),
        "decay_times_raw": pd.DataFrame(decay_rows),
        "failures": pd.DataFrame(failure_rows),
        "n_success": successes,
    }


def _quantile_summary(
    values: Iterable[float],
    estimate: float,
    value_name: str,
    alpha: float,
) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        lower = np.nan
        upper = np.nan
    else:
        lower = _safe_quantile(arr, alpha / 2)
        upper = _safe_quantile(arr, 1.0 - alpha / 2)
    return {
        value_name: float(estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_boot": int(arr.size),
    }


def _safe_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float(np.nan)
    values = np.sort(values)
    position = q * (values.size - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    if lower_index == upper_index:
        return float(values[lower_index])

    lower = values[lower_index]
    upper = values[upper_index]
    weight = position - lower_index
    if not np.isfinite(lower) or not np.isfinite(upper):
        if lower == upper:
            return float(lower)
        if np.isposinf(upper) and weight > 0:
            return float(np.inf)
        if np.isneginf(lower) and weight < 1:
            return float(-np.inf)
        return float(np.nan)
    return float(lower + weight * (upper - lower))


def _coef_estimate(
    cox_models: Mapping[str, CoxTimeVaryingFitter],
    model: str,
    term: str,
    default: float,
) -> float:
    ctv = cox_models.get(model)
    if ctv is None:
        log_bootstrap_fallback(
            "cox_coefficient_summary",
            "missing full sample model",
            model=model,
            term=term,
        )
        return float(default)
    if term not in ctv.params_.index:
        log_bootstrap_fallback(
            "cox_coefficient_summary",
            "missing full sample coefficient",
            model=model,
            term=term,
        )
        return float(default)
    return float(ctv.params_.loc[term])


def _summarize_coefficients(
    coef_raw: pd.DataFrame,
    cox_models: Mapping[str, CoxTimeVaryingFitter],
    alpha: float,
) -> pd.DataFrame:
    if coef_raw.empty:
        return pd.DataFrame()
    rows = []
    grouped = coef_raw.groupby(["model", "term"], sort=False)
    for (model, term), group in grouped:
        estimate = _coef_estimate(
            cox_models,
            model,
            term,
            group["coef"].median(),
        )
        coef_summary = _quantile_summary(group["coef"], estimate, "coef", alpha)
        hr_summary = _quantile_summary(
            np.exp(group["coef"]),
            np.exp(estimate),
            "hr",
            alpha,
        )
        rows.append({
            "model": model,
            "term": term,
            "coef": coef_summary["coef"],
            "coef_ci_lower": coef_summary["ci_lower"],
            "coef_ci_upper": coef_summary["ci_upper"],
            "hr": hr_summary["hr"],
            "hr_ci_lower": hr_summary["ci_lower"],
            "hr_ci_upper": hr_summary["ci_upper"],
            "n_boot": coef_summary["n_boot"],
        })
    return pd.DataFrame(rows)


def _summarize_exposure(
    coef_raw: pd.DataFrame,
    cox_models: Mapping[str, CoxTimeVaryingFitter],
    model_names: Sequence[str],
    exposure_terms: Sequence[str],
    ref_term: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for model in model_names:
        model_raw = coef_raw[coef_raw["model"] == model]
        if model_raw.empty:
            continue
        wide = model_raw.pivot_table(
            index="draw",
            columns="term",
            values="coef",
            aggfunc="first",
        )
        for term in exposure_terms:
            if term not in wide.columns or ref_term not in wide.columns:
                continue
            draws = np.exp(wide[term] - wide[ref_term]).dropna()
            ctv = cox_models.get(model)
            if ctv is not None and term in ctv.params_.index and ref_term in ctv.params_.index:
                estimate = float(np.exp(ctv.params_.loc[term] - ctv.params_.loc[ref_term]))
            else:
                reason = (
                    "missing full sample model"
                    if ctv is None
                    else "missing full sample exposure term"
                )
                log_bootstrap_fallback(
                    "exposure_summary",
                    reason,
                    model=model,
                    term=term,
                    ref_term=ref_term,
                )
                estimate = float(draws.median())
            summary = _quantile_summary(draws, estimate, "hr", alpha)
            rows.append({
                "model": model,
                "term": term,
                "ref_term": ref_term,
                "hr": summary["hr"],
                "ci_lower": summary["ci_lower"],
                "ci_upper": summary["ci_upper"],
                "n_boot": summary["n_boot"],
            })
    return pd.DataFrame(rows)


def _fc_conspiracy_name(term: str) -> str:
    return term.replace("fc_", "").replace("ConsProb_", "")


def _summarize_gateway(
    coefficient_intervals: pd.DataFrame,
    reference_conspiracy: str,
) -> pd.DataFrame:
    if coefficient_intervals.empty:
        return pd.DataFrame()
    m1 = coefficient_intervals[
        (coefficient_intervals["model"] == "model_1")
        & coefficient_intervals["term"].str.startswith("fc_")
    ].copy()
    m2 = coefficient_intervals[
        (coefficient_intervals["model"] == "model_2b")
        & coefficient_intervals["term"].str.startswith("fc_")
    ].copy()
    if m1.empty or m2.empty:
        return pd.DataFrame()
    m1["conspiracy"] = m1["term"].map(_fc_conspiracy_name)
    m2["conspiracy"] = m2["term"].map(_fc_conspiracy_name)
    m1 = m1.rename(columns={
        "hr": "model1_hr",
        "hr_ci_lower": "model1_ci_lower",
        "hr_ci_upper": "model1_ci_upper",
    })
    m2 = m2.rename(columns={
        "hr": "model2_hr",
        "hr_ci_lower": "model2_ci_lower",
        "hr_ci_upper": "model2_ci_upper",
    })
    merged = pd.merge(
        m1[["conspiracy", "model1_hr", "model1_ci_lower", "model1_ci_upper"]],
        m2[["conspiracy", "model2_hr", "model2_ci_lower", "model2_ci_upper"]],
        on="conspiracy",
        how="outer",
    )
    merged["interval_source"] = "bootstrap"
    ref_short = reference_conspiracy.replace("ConsProb_", "")
    ref_row = pd.DataFrame([{
        "conspiracy": ref_short,
        "model1_hr": 1.0,
        "model1_ci_lower": 1.0,
        "model1_ci_upper": 1.0,
        "model2_hr": 1.0,
        "model2_ci_lower": 1.0,
        "model2_ci_upper": 1.0,
        "model2_p": np.nan,
        "interval_source": "reference",
    }])
    merged["model2_p"] = np.nan
    return pd.concat([merged, ref_row], ignore_index=True)


def _summarize_baselines(
    baseline_raw: pd.DataFrame,
    baseline_params: Mapping[str, Mapping],
    alpha: float,
) -> pd.DataFrame:
    if baseline_raw.empty:
        return pd.DataFrame()
    rows = []
    for (model, param), group in baseline_raw.melt(
        id_vars=["draw", "model", "type"],
        value_vars=["slope", "shape", "scale", "rmse", "r_squared", "relative_rmse"],
        var_name="parameter",
        value_name="value",
    ).dropna(subset=["value"]).groupby(["model", "parameter"], sort=False):
        full = baseline_params.get(model, {})
        if param in full:
            estimate = float(full[param])
        else:
            log_bootstrap_fallback(
                "baseline_parameter_summary",
                "missing full sample baseline parameter",
                model=model,
                parameter=param,
            )
            estimate = float(group["value"].median())
        summary = _quantile_summary(group["value"], estimate, "value", alpha)
        rows.append({
            "model": model,
            "parameter": param,
            "value": summary["value"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
            "n_boot": summary["n_boot"],
        })
    return pd.DataFrame(rows)


def _summarize_temporal_hazard_ratios(
    temporal_raw: pd.DataFrame,
    baseline_params: Mapping[str, Mapping],
    alpha: float,
) -> pd.DataFrame:
    if temporal_raw.empty or "model_1" not in baseline_params:
        return pd.DataFrame()
    rows = []
    m1_h0 = calculate_baseline_hazard(1.0, baseline_params["model_1"])
    for (model, time_value), group in temporal_raw.groupby(["model", "time_hours"], sort=False):
        params = baseline_params.get(model)
        if params is not None and m1_h0 > 0:
            estimate = calculate_baseline_hazard(float(time_value), params) / m1_h0
        else:
            reason = (
                "missing full sample baseline parameters"
                if params is None
                else "nonpositive model 1 baseline hazard"
            )
            log_bootstrap_fallback(
                "temporal_hazard_summary",
                reason,
                model=model,
                time_hours=float(time_value),
            )
            estimate = group["ratio"].median()
        summary = _quantile_summary(group["ratio"].dropna(), estimate, "ratio", alpha)
        rows.append({
            "model": model,
            "time_hours": float(time_value),
            "ratio": summary["ratio"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
            "n_boot": summary["n_boot"],
        })
    return pd.DataFrame(rows)


def _summarize_decay_times(
    decay_raw: pd.DataFrame,
    baseline_params: Mapping[str, Mapping],
    alpha: float,
) -> pd.DataFrame:
    if decay_raw.empty:
        return pd.DataFrame()
    point_decay = (
        compute_all_decay_times(baseline_params)
        if "model_1" in baseline_params
        else {}
    )
    rows = []
    for model, group in decay_raw.groupby("model", sort=False):
        draws = group["t_star_hours"].astype(float)
        if model in point_decay:
            estimate = float(point_decay[model])
        else:
            log_bootstrap_fallback(
                "decay_time_summary",
                "missing full sample decay estimate",
                model=model,
            )
            estimate = float(draws.median())
        summary = _quantile_summary(draws, estimate, "t_star_hours", alpha)
        rows.append({
            "model": model,
            "t_star_hours": summary["t_star_hours"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
            "n_boot": summary["n_boot"],
            "infinite_share": float((~np.isfinite(draws)).mean()),
        })
    return pd.DataFrame(rows)


def summarize_cox_bootstrap(
    bootstrap: Mapping[str, object],
    cox_models: Optional[Mapping[str, CoxTimeVaryingFitter]] = None,
    baseline_params: Optional[Mapping[str, Mapping]] = None,
    alpha: float = 0.05,
    reference_conspiracy: str = "ConsProb_fakenews",
    exposure_models: Optional[Sequence[str]] = None,
    exposure_terms: Sequence[str] = ("s7_d2", "s7_d3", "s7_d4"),
    exposure_ref: str = "s7_d1",
) -> dict:
    """Summarize raw Cox bootstrap draws into plotting tables."""
    cox_models = cox_models or {}
    baseline_params = baseline_params or {}
    coef_raw = bootstrap.get("coefficients_raw", pd.DataFrame())
    baseline_raw = bootstrap.get("baselines_raw", pd.DataFrame())
    temporal_raw = bootstrap.get("temporal_hazard_ratios_raw", pd.DataFrame())
    decay_raw = bootstrap.get("decay_times_raw", pd.DataFrame())
    if exposure_models is None and not coef_raw.empty:
        exposure_models = sorted(coef_raw["model"].unique())
    elif exposure_models is None:
        exposure_models = []

    coef_intervals = _summarize_coefficients(coef_raw, cox_models, alpha)
    result = dict(bootstrap)
    result["coefficient_intervals"] = coef_intervals
    result["exposure_intervals"] = _summarize_exposure(
        coef_raw,
        cox_models,
        exposure_models,
        exposure_terms,
        exposure_ref,
        alpha,
    )
    result["gateway_intervals"] = _summarize_gateway(
        coef_intervals,
        reference_conspiracy,
    )
    result["baseline_parameter_intervals"] = _summarize_baselines(
        baseline_raw,
        baseline_params,
        alpha,
    )
    result["temporal_hazard_ratio_intervals"] = _summarize_temporal_hazard_ratios(
        temporal_raw,
        baseline_params,
        alpha,
    )
    result["decay_time_intervals"] = _summarize_decay_times(
        decay_raw,
        baseline_params,
        alpha,
    )
    return result


def _median_or_nan(values: Sequence[float]) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def _summarize_timeline_raw(raw: pd.DataFrame, full_rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows = []
    for (family, statistic, key), group in raw.groupby(
        ["family", "statistic", "key"],
        sort=False,
    ):
        full = full_rows[
            (full_rows["family"] == family)
            & (full_rows["statistic"] == statistic)
            & (full_rows["key"] == key)
        ]
        if not full.empty:
            estimate = float(full["estimate"].iloc[0])
        else:
            log_bootstrap_fallback(
                "timeline_summary",
                "missing full sample timeline estimate",
                family=family,
                statistic=statistic,
                key=key,
            )
            estimate = group["value"].median()
        summary = _quantile_summary(group["value"].dropna(), estimate, "estimate", alpha)
        rows.append({
            "family": family,
            "statistic": statistic,
            "key": key,
            "estimate": summary["estimate"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
            "n_boot": summary["n_boot"],
        })
    return pd.DataFrame(rows)


def _timeline_full_rows(
    timeline_specs: Mapping[str, Tuple[Sequence[Mapping], Mapping[str, int]]],
) -> pd.DataFrame:
    rows = []
    for family, (timelines, clusters) in timeline_specs.items():
        barrier = compute_semantic_barrier_analysis(list(timelines), clusters)
        settler = compute_settler_effect(list(timelines), clusters)
        for key, values in barrier.items():
            rows.append({
                "family": family,
                "statistic": "barrier",
                "key": key,
                "estimate": _median_or_nan(values),
            })
        for key, values in settler.items():
            rows.append({
                "family": family,
                "statistic": "settler",
                "key": key,
                "estimate": _median_or_nan(values),
            })
    return pd.DataFrame(rows)


def _run_timeline_bootstrap_draw(
    family: str,
    timelines: Sequence[Mapping],
    clusters: Mapping[str, int],
    draw: int,
    indices: Sequence[int],
) -> List[dict]:
    sampled = [timelines[i] for i in indices]
    barrier = compute_semantic_barrier_analysis(sampled, clusters)
    settler = compute_settler_effect(sampled, clusters)
    raw_rows = []
    for key, values in barrier.items():
        raw_rows.append({
            "draw": draw,
            "family": family,
            "statistic": "barrier",
            "key": key,
            "value": _median_or_nan(values),
        })
    for key, values in settler.items():
        raw_rows.append({
            "draw": draw,
            "family": family,
            "statistic": "settler",
            "key": key,
            "value": _median_or_nan(values),
        })
    return raw_rows


def run_timeline_bootstrap(
    timeline_specs: Mapping[str, Tuple[Sequence[Mapping], Mapping[str, int]]],
    n_draws: int,
    seed: int = 42,
    alpha: float = 0.05,
    n_jobs: Optional[int] = 1,
    parallel_backend: Optional[str] = None,
    parallel_verbose: int = 0,
) -> dict:
    """Bootstrap median transition statistics by resampling users."""
    rng = np.random.default_rng(seed)
    n_jobs = _effective_n_jobs(n_jobs)
    full_rows = _timeline_full_rows(timeline_specs)
    task_args = []

    for family, (timelines, clusters) in timeline_specs.items():
        timelines = list(timelines)
        n_timelines = len(timelines)
        if n_timelines == 0:
            continue
        for draw in range(n_draws):
            idx = rng.integers(0, n_timelines, size=n_timelines)
            task_args.append(
                (family, timelines, clusters, draw, idx)
            )

    draw_results = _parallel_call(
        _run_timeline_bootstrap_draw,
        task_args,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_verbose=parallel_verbose,
    )
    raw_rows = [row for result in draw_results for row in result]
    raw = pd.DataFrame(raw_rows)
    intervals = _summarize_timeline_raw(raw, full_rows, alpha)
    if "statistic" in intervals.columns:
        barrier_intervals = intervals[intervals["statistic"] == "barrier"].copy()
        settler_intervals = intervals[intervals["statistic"] == "settler"].copy()
    else:
        barrier_intervals = pd.DataFrame()
        settler_intervals = pd.DataFrame()
    return {
        "metadata": {
            "n_draws": int(n_draws),
            "seed": int(seed),
            "alpha": float(alpha),
            "n_jobs": int(n_jobs),
            "parallel_backend": parallel_backend,
            "families": list(timeline_specs.keys()),
        },
        "full_sample": full_rows,
        "raw": raw,
        "intervals": intervals,
        "barrier_intervals": barrier_intervals,
        "settler_intervals": settler_intervals,
    }


__all__ = [
    "ModelBootstrapSpec",
    "load_bootstrap_artifact",
    "run_cox_user_bootstrap",
    "run_timeline_bootstrap",
    "summarize_cox_bootstrap",
    "_sample_long_form_by_user",
]
