"""
Evaluation utilities for comparing simulation outputs.

Computes diffusion curves, first-adoption distributions, and
metrics for comparing simulated vs. empirical data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from conspiracy_analysis.simulation.runner import ScenarioResults, SimulationResult


def compute_diffusion_curves(
    results: ScenarioResults,
    steps: Optional[int] = None,
) -> pd.DataFrame:
    """Compute fraction of nodes that have adopted each conspiracy over time.

    Args:
        results: Scenario results containing multiple runs.
        steps: Number of time steps (inferred from histories if not given).

    Returns:
        DataFrame with columns: RUN, t, conspiracy, adoption_fraction.
    """
    if results.n_runs == 0:
        return pd.DataFrame(columns=["RUN", "t", "conspiracy", "adoption_fraction"])

    first_run = results.runs[0]
    n_nodes = first_run.n_nodes

    # Infer conspiracies from histories
    all_conspiracies = set()
    for run in results.runs:
        for node_hist in run.adoption_histories.values():
            all_conspiracies.update(node_hist.keys())
    conspiracies = sorted(all_conspiracies)

    if not conspiracies:
        return pd.DataFrame(columns=["RUN", "t", "conspiracy", "adoption_fraction"])

    # Infer max time
    if steps is None:
        max_time = 0
        for run in results.runs:
            for node_hist in run.adoption_histories.values():
                for t in node_hist.values():
                    max_time = max(max_time, int(t))
        steps = max_time + 1

    time_range = np.arange(steps)
    records = []

    for run in results.runs:
        # Collect adoption times per conspiracy
        adoption_times_by_consp: Dict[str, List[float]] = {
            c: [] for c in conspiracies
        }
        for node_hist in run.adoption_histories.values():
            for c, t in node_hist.items():
                adoption_times_by_consp[c].append(t)

        for c in conspiracies:
            times = np.array(sorted(adoption_times_by_consp[c]))
            if len(times) > 0:
                counts = np.searchsorted(times, time_range, side="right")
                fractions = counts / n_nodes
            else:
                fractions = np.zeros(steps)

            for i, frac in enumerate(fractions):
                records.append({
                    "RUN": run.run_id,
                    "t": i,
                    "conspiracy": c,
                    "adoption_fraction": frac,
                })

    return pd.DataFrame(records)


def compute_first_adoption_distribution(
    results: ScenarioResults,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute which conspiracy users adopt first (entry point distribution).

    Args:
        results: Scenario results.

    Returns:
        Tuple of (summary_df, raw_df).
        summary_df: Mean and std of first-adoption frequencies across runs.
        raw_df: Per-run counts and frequencies.
    """
    run_records = []

    # Collect all conspiracies
    all_conspiracies = set()
    for run in results.runs:
        for node_hist in run.adoption_histories.values():
            all_conspiracies.update(node_hist.keys())
    conspiracies = sorted(all_conspiracies)

    for run in results.runs:
        counts = {c: 0 for c in conspiracies}
        total_active = 0

        for node_hist in run.adoption_histories.values():
            if not node_hist:
                continue
            # Find earliest adoption
            first_consp = min(node_hist, key=node_hist.get)
            counts[first_consp] += 1
            total_active += 1

        for c in conspiracies:
            freq = counts[c] / total_active if total_active > 0 else 0.0
            run_records.append({
                "run_id": run.run_id,
                "conspiracy": c,
                "count": counts[c],
                "frequency": freq,
            })

    df_raw = pd.DataFrame(run_records)

    if df_raw.empty:
        df_stats = pd.DataFrame(
            columns=["conspiracy", "mean_frequency", "std_frequency"]
        )
        return df_stats, df_raw

    df_stats = (
        df_raw.groupby("conspiracy")["frequency"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_frequency", "std": "std_frequency"})
        .sort_values("mean_frequency", ascending=False)
        .reset_index(drop=True)
    )

    return df_stats, df_raw


def compare_scenarios(
    all_results: Dict[str, ScenarioResults],
    metric: str = "total_adoptions",
) -> pd.DataFrame:
    """Compare scenarios by a summary metric.

    Args:
        all_results: Dict mapping scenario name to ScenarioResults.
        metric: One of "total_adoptions", "mean_conspiracies_per_node".

    Returns:
        DataFrame with scenario-level summary statistics.
    """
    records = []

    for scenario_name, results in all_results.items():
        for run in results.runs:
            total = sum(
                len(node_hist) for node_hist in run.adoption_histories.values()
            )
            nodes_with_any = len(run.adoption_histories)
            mean_per_node = total / run.n_nodes if run.n_nodes > 0 else 0.0

            records.append({
                "scenario": scenario_name,
                "run_id": run.run_id,
                "total_adoptions": total,
                "nodes_with_any_adoption": nodes_with_any,
                "mean_conspiracies_per_node": mean_per_node,
                "adoption_rate": nodes_with_any / run.n_nodes if run.n_nodes > 0 else 0.0,
            })

    return pd.DataFrame(records)


def compute_empirical_comparison(
    results: ScenarioResults,
    empirical_portions: Dict[str, float],
) -> Dict[str, float]:
    """Compare simulated first-adoption distribution against empirical data.

    Args:
        results: Scenario results.
        empirical_portions: Dict mapping conspiracy name to empirical
            fraction of users who adopted it first.

    Returns:
        Dict with 'r_squared', 'correlation', 'rmse' keys.
    """
    summary_df, _ = compute_first_adoption_distribution(results)

    if summary_df.empty:
        return {"r_squared": float("nan"), "correlation": float("nan"), "rmse": float("nan")}

    # Align on shared conspiracies
    shared = set(summary_df["conspiracy"]) & set(empirical_portions.keys())
    if not shared:
        return {"r_squared": float("nan"), "correlation": float("nan"), "rmse": float("nan")}

    sim_vals = []
    emp_vals = []
    for c in sorted(shared):
        row = summary_df[summary_df["conspiracy"] == c]
        if not row.empty:
            sim_vals.append(row["mean_frequency"].iloc[0])
            emp_vals.append(empirical_portions[c])

    sim_arr = np.array(sim_vals)
    emp_arr = np.array(emp_vals)

    # R-squared
    ss_res = np.sum((emp_arr - sim_arr) ** 2)
    ss_tot = np.sum((emp_arr - emp_arr.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Correlation
    if len(sim_arr) > 1:
        correlation = float(np.corrcoef(sim_arr, emp_arr)[0, 1])
    else:
        correlation = float("nan")

    # RMSE
    rmse = float(np.sqrt(np.mean((sim_arr - emp_arr) ** 2)))

    return {
        "r_squared": r_squared,
        "correlation": correlation,
        "rmse": rmse,
    }
