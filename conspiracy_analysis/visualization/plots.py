"""
Diagnostic and supplementary plots for model assessment.

Includes hazard ratio forest plots, baseline hazard fit diagnostics,
silhouette score heatmaps, and diffusion dynamics from ABM simulations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxTimeVaryingFitter

from conspiracy_analysis.config import (
    get_baseline_conspiracy,
    get_conspiracy_labels,
)
from conspiracy_analysis.analysis.semantic import (
    SEMANTIC_CLUSTER_COLOR_CYCLE,
    build_cluster_display_metadata,
)
from conspiracy_analysis.visualization.style import (
    NHB_COL_SINGLE, NHB_COL_ONE_HALF, NHB_COL_DOUBLE,
    nhb_annotation_fontsize,
    nhb_main_title_fontsize,
)
from conspiracy_analysis.utils.fallback_logging import log_bootstrap_fallback

logger = logging.getLogger(__name__)
_BASELINE_CONSPIRACY = get_baseline_conspiracy()


def _bootstrap_table(bootstrap_intervals, table_name: str) -> Optional[pd.DataFrame]:
    if bootstrap_intervals is None:
        return None
    if isinstance(bootstrap_intervals, dict):
        table = bootstrap_intervals.get(table_name)
    else:
        table = bootstrap_intervals
    if isinstance(table, pd.DataFrame) and not table.empty:
        return table
    return None


def plot_hazard_ratios(
    ctv: CoxTimeVaryingFitter,
    covariate_prefix: str = "s7_",
    title: str = "Hazard Ratios with 95% CI",
    figsize: Tuple[float, float] = NHB_COL_SINGLE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot hazard ratios with 95% CI for selected covariates.

    Args:
        ctv: Fitted CoxTimeVaryingFitter.
        covariate_prefix: Prefix to filter covariates (e.g., 's7_', 'fc_').
        title: Plot title.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    summary = ctv.summary
    mask = summary.index.str.startswith(covariate_prefix)
    subset = summary.loc[mask, ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]]

    if subset.empty:
        logger.warning(f"No covariates found with prefix '{covariate_prefix}'")
        return plt.figure()

    hr = subset["exp(coef)"].values
    lower = subset["exp(coef) lower 95%"].values
    upper = subset["exp(coef) upper 95%"].values
    yerr = np.vstack([hr - lower, upper - hr])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(subset))
    ax.errorbar(x, hr, yerr=yerr, fmt="o", capsize=4)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(subset.index, rotation=45, ha="right")
    ax.set_ylabel("Hazard Ratio (exp(coef))")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_baseline_hazard_fit(
    baseline_params: Dict,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot empirical vs. fitted cumulative baseline hazard.

    Args:
        baseline_params: Output from fit_linear_baseline or fit_weibull_baseline.
            Must contain 'times', 'empirical', 'fitted_values', and 'type'.
        title: Plot title. Auto-generated if None.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    times = baseline_params["times"]
    empirical = baseline_params["empirical"]
    fitted = baseline_params["fitted_values"]
    fit_type = baseline_params["type"]

    if title is None:
        rmse = baseline_params.get("rmse", 0)
        title = f"Baseline Hazard Fit ({fit_type.title()}, RMSE={rmse:.2e})"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, empirical, "b.", alpha=0.5, label="Empirical")
    ax.plot(times, fitted, "r-", linewidth=2, label=f"Fitted ({fit_type})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Baseline Hazard")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_silhouette_heatmap(
    scores_df: pd.DataFrame,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a heatmap of silhouette scores across methods and k values.

    Args:
        scores_df: DataFrame with columns 'method', 'k', 'silhouette_score'
            (from find_optimal_clustering).
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    pivot = scores_df.pivot(index="method", columns="k", values="silhouette_score")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("Silhouette Scores by Linkage Method and k")
    ax.set_ylabel("Linkage Method")
    ax.set_xlabel("Number of Clusters (k)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_diffusion_dynamics(
    df_results: pd.DataFrame,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    title: str = "Simulated Diffusion Dynamics",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot adoption curves from ABM simulation results.

    Shows mean adoption fraction over time with 95% CI bands.

    Args:
        df_results: DataFrame from compute_diffusion_curves with columns
            RUN, t, conspiracy, adoption_fraction.
        figsize: Figure dimensions.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    aggregated = (
        df_results.groupby(["conspiracy", "t"])
        .agg(
            mean_portion=("adoption_fraction", "mean"),
            std_portion=("adoption_fraction", "std"),
            count=("adoption_fraction", "count"),
        )
        .reset_index()
    )
    aggregated["sem"] = aggregated["std_portion"] / np.sqrt(aggregated["count"])
    aggregated["ci_lower"] = (aggregated["mean_portion"] - 1.96 * aggregated["sem"]).clip(lower=0)
    aggregated["ci_upper"] = aggregated["mean_portion"] + 1.96 * aggregated["sem"]

    fig, ax = plt.subplots(figsize=figsize)
    for conspiracy, group in aggregated.groupby("conspiracy"):
        group = group.sort_values("t")
        ax.plot(group["t"], group["mean_portion"], label=conspiracy)
        ax.fill_between(group["t"], group["ci_lower"], group["ci_upper"], alpha=0.2)

    ax.set_xlabel("Time Step (hours)")
    ax.set_ylabel("Adoption Fraction")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_first_adoption_comparison(
    empirical_portions: Dict[str, float],
    df_stats: pd.DataFrame,
    conspiracies: List[str],
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, float]:
    """Scatter plot comparing empirical vs. simulated first adoption frequencies.

    Args:
        empirical_portions: Dict mapping conspiracy -> empirical frequency.
        df_stats: Summary stats from evaluate_first_adoption_stats.
        conspiracies: List of conspiracy names to compare.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Tuple of (Figure, R² value).
    """
    simulated = {}
    for _, row in df_stats.iterrows():
        simulated[row["conspiracy"]] = row["mean_frequency"]

    emp_vals, sim_vals = [], []
    for c in conspiracies:
        if c in empirical_portions and c in simulated:
            emp_vals.append(empirical_portions[c])
            sim_vals.append(simulated[c])

    emp_arr = np.array(emp_vals)
    sim_arr = np.array(sim_vals)
    ss_res = np.sum((emp_arr - sim_arr) ** 2)
    ss_tot = np.sum((emp_arr - emp_arr.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emp_vals, sim_vals)
    max_val = max(max(emp_vals), max(sim_vals))
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", label=f"y=x (R²={r_squared:.3f})")
    ax.set_xlabel("Empirical Portions")
    ax.set_ylabel("Simulated Portions")
    ax.set_title("Empirical vs. Simulated First Adoption Portions")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, r_squared


def plot_instantaneous_baseline_hazards(
    baseline_params: Dict,
    t_max_hours: float = 2000,
    n_points: int = 500,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    decay_times: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot baseline hazard ratios relative to Model 1 for all models on one figure.

    All hazards are normalized by Model 1's constant baseline, so Model 1
    appears as a flat line at y=1 and Weibull models show multiples of that
    baseline.

    Args:
        baseline_params: Dict from parametrize_all_baselines, mapping model names
            to fitted baseline hazard parameters.
        t_max_hours: Maximum time in hours for the x-axis.
        n_points: Number of points to evaluate the hazard at.
        figsize: Figure dimensions.
        decay_times: Optional dict of decay-to-baseline times (from
            compute_all_decay_times). If provided, marks crossing points.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    t = np.linspace(1, t_max_hours, n_points)  # start at 1 to avoid Weibull singularity
    t_days = t / 24

    # Normalization: divide all hazards by Model 1's constant baseline
    m1_params = baseline_params.get("model_1")
    if m1_params is None or m1_params["type"] != "linear":
        raise ValueError("model_1 with type 'linear' required for normalization")
    baseline_ref = m1_params["slope"]

    fig, ax = plt.subplots(figsize=figsize)

    # Color cycle for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_params)))

    for idx, (name, params) in enumerate(sorted(baseline_params.items())):
        color = colors[idx]

        if params["type"] == "linear":
            slope = params["slope"]
            h_t = np.full_like(t, slope / baseline_ref)
            ax.plot(t_days, h_t, color=color, linewidth=2, linestyle="--",
                    label=f"{name}: constant (reference = 1.0)")
        elif params["type"] == "weibull":
            k = params["shape"]
            lam = params["scale"]
            h_t = (k / lam) * (t / lam) ** (k - 1) / baseline_ref
            rmse = params.get("rmse", 0)
            ax.plot(t_days, h_t, color=color, linewidth=2,
                    label=f"{name}: Weibull k={k:.3f}, λ={lam:.0f} (RMSE={rmse:.2e})")

            # Mark decay-to-baseline crossing point
            if decay_times and name in decay_times:
                t_star = decay_times[name]
                if np.isfinite(t_star) and t_star <= t_max_hours:
                    ax.axvline(t_star / 24, color=color, linestyle=":", alpha=0.6)
                    ax.annotate(
                        f"t*={t_star / 24:.1f}d",
                        xy=(t_star / 24, 1.0),
                        xytext=(5, 10), textcoords="offset points",
                        fontsize=nhb_annotation_fontsize(), color=color,
                    )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time since prior adoption (days)")
    ax.set_ylabel("Hazard ratio (relative to Model 1 baseline)")
    ax.set_title("Baseline Hazard Ratios Across Models")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _model_sort_key(name: str):
    """Sort key for model names like 'model_2a', 'model_10' by number then suffix."""
    s = name.replace("model_", "")
    num = ''.join(c for c in s if c.isdigit())
    suffix = ''.join(c for c in s if not c.isdigit())
    return (int(num), suffix)


def plot_decay_times(
    decay_times: Dict[str, float],
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart showing decay-to-baseline time t* for each Weibull model.

    Args:
        decay_times: Dict mapping model names to t* in hours
            (from compute_all_decay_times).
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    names = sorted(decay_times.keys(), key=_model_sort_key)
    values_days = [decay_times[n] / 24 for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values_days, color="steelblue", edgecolor="black")

    for bar, val in zip(bars, values_days):
        if np.isfinite(val):
            ax.annotate(
                f"{val:.1f}d",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=nhb_annotation_fontsize(),
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Decay-to-baseline time t* (days)")
    ax.set_title("Time for Weibull Hazard to Decay to Model 1 Baseline Level")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# Short display labels for conspiracy names
_CONSPIRACY_LABELS = get_conspiracy_labels(short=True)

_UNKNOWN_CLUSTER_COLOR = {"Unknown": "#999999"}


def _colors_for_cluster_labels(cluster_labels: Dict[str, str]) -> Dict[str, str]:
    names = sorted(set(cluster_labels.values()))
    return {
        name: SEMANTIC_CLUSTER_COLOR_CYCLE[
            i % len(SEMANTIC_CLUSTER_COLOR_CYCLE)
        ]
        for i, name in enumerate(names)
    }


def plot_gateway_scatter(
    gateway_2d_df: pd.DataFrame,
    cluster_assignments: Optional[Dict[str, int]] = None,
    cluster_labels: Optional[Dict[str, str]] = None,
    cluster_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
    reference_conspiracy: str = _BASELINE_CONSPIRACY,
    text_scale: float = 1.0,
    title: str = "Contagiousness and Downstream Acceleration by Conspiracy",
    bootstrap_intervals=None,
) -> plt.Figure:
    """Scatter plot of two-dimensional gateway characterization.

    X-axis: Model 1 HR (contagiousness — how likely to be adopted first).
    Y-axis: Model 2 HR (acceleration — how strongly first adoption
    accelerates second adoption).

    Args:
        gateway_2d_df: DataFrame from identify_gateway_2d() with columns
            conspiracy, model1_hr, model1_ci_lower, model1_ci_upper,
            model2_hr, model2_ci_lower, model2_ci_upper, model2_p.
        cluster_assignments: Optional dict mapping 'ConsProb_<name>' ->
            cluster_id. If provided, used to derive cluster labels; if
            None, uses default cluster mapping.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    boot_table = _bootstrap_table(bootstrap_intervals, "gateway_intervals")
    if boot_table is not None:
        df = boot_table.copy()
        if "model2_p" not in df.columns:
            df["model2_p"] = np.nan
        if "interval_source" not in df.columns:
            df["interval_source"] = "bootstrap"
    else:
        df = gateway_2d_df.copy()
        source = df.get("interval_source")
        if source is not None:
            interval_source = source[source != "reference"]
        else:
            interval_source = None
        if interval_source is not None and (interval_source != "bootstrap").any():
            log_bootstrap_fallback(
                "gateway_scatter",
                "using supplied model summary confidence interval fallback",
                fallback_interval_source="model_summary",
            )

    if cluster_labels is not None:
        consp_to_cluster_label = cluster_labels
        colors_map = cluster_colors if cluster_colors is not None else _colors_for_cluster_labels(
            cluster_labels
        )
    elif cluster_assignments is not None:
        consp_to_cluster_label, colors_map = build_cluster_display_metadata(
            cluster_assignments
        )
    else:
        consp_to_cluster_label = {}
        colors_map = _UNKNOWN_CLUSTER_COLOR

    # Add cluster and label columns
    df["cluster"] = df["conspiracy"].map(consp_to_cluster_label).fillna("Unknown")
    df["label"] = df["conspiracy"].map(_CONSPIRACY_LABELS).fillna(df["conspiracy"])

    # Remove reference point (it sits at 1,1 by definition)
    ref_short = reference_conspiracy.replace("ConsProb_", "")
    ref_label = _CONSPIRACY_LABELS.get(ref_short, ref_short)
    df = df[df["conspiracy"] != ref_short]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot by cluster for legend grouping
    for cluster_name, color in colors_map.items():
        mask = df["cluster"] == cluster_name
        sub = df[mask]
        if sub.empty:
            continue

        xerr = np.vstack([
            (sub["model1_hr"] - sub["model1_ci_lower"]).values,
            (sub["model1_ci_upper"] - sub["model1_hr"]).values,
        ])
        yerr = np.vstack([
            (sub["model2_hr"] - sub["model2_ci_lower"]).values,
            (sub["model2_ci_upper"] - sub["model2_hr"]).values,
        ])
        xerr = np.clip(xerr, 0, None)
        yerr = np.clip(yerr, 0, None)

        ax.errorbar(
            sub["model1_hr"], sub["model2_hr"],
            xerr=xerr, yerr=yerr,
            fmt="o", color=color, markersize=8,
            capsize=3, elinewidth=1, alpha=0.85,
            label=cluster_name,
        )

    # Reference lines at HR = 1.0 (reference point)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    annot_fs = nhb_annotation_fontsize() * text_scale
    ax.annotate(
        f"{ref_label}\n(reference = 1, 1)",
        xy=(1.0, 1.0),
        xytext=(0.9, 1.1),
        textcoords="data",
        fontsize=annot_fs,
        fontstyle="italic",
        color="grey",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
    )

    # Label each point
    for _, row in df.iterrows():
        ax.annotate(
            row["label"],
            xy=(row["model1_hr"], row["model2_hr"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=annot_fs,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(f"Contagiousness (HR of First Adoption, ref = {ref_label})")
    ax.set_ylabel(f"Impact on Consecutive Adoption (Gateway Effect, ref = {ref_label})")
    ax.set_title(title, fontsize=nhb_main_title_fontsize())
    legend = ax.legend(
        title="Semantic Cluster", loc="upper center", framealpha=0.9,
        fontsize=annot_fs,
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontsize(annot_fs)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Gateway scatter plot saved to {save_path}")

    return fig


def plot_fc_forest(
    ctv: CoxTimeVaryingFitter,
    title: Optional[str] = None,
    sort_by_hr: bool = True,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    cluster_assignments: Optional[Dict[str, int]] = None,
    cluster_labels: Optional[Dict[str, str]] = None,
    cluster_colors: Optional[Dict[str, str]] = None,
    significance_level: float = 0.05,
    save_path: Optional[str] = None,
    reference_conspiracy: str = _BASELINE_CONSPIRACY,
    bootstrap_intervals=None,
    model_name: Optional[str] = None,
) -> plt.Figure:
    """Horizontal forest plot of fc_* conspiracy dummy HRs from any model.

    Args:
        ctv: Fitted CoxTimeVaryingFitter containing fc_* covariates.
        title: Plot title.
        sort_by_hr: If True, sort conspiracies by hazard ratio.
        figsize: Figure dimensions.
        cluster_assignments: Optional dict mapping 'ConsProb_<name>' ->
            cluster_id. If None, uses default cluster mapping.
        significance_level: P-value threshold for filled vs. open markers.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    ref_short = reference_conspiracy.replace("ConsProb_", "")
    ref_label = _CONSPIRACY_LABELS.get(ref_short, ref_short)
    if title is None:
        title = f"Conspiracy Hazard Ratios (ref = {ref_label})"

    summary = ctv.summary
    fc_mask = summary.index.str.startswith("fc_")
    fc_rows = summary.loc[fc_mask].copy()

    if fc_rows.empty:
        logger.warning("No fc_* covariates found in model.")
        return plt.figure()

    # Extract short conspiracy names
    fc_rows["short_name"] = fc_rows.index.str.replace("fc_ConsProb_", "", regex=False)
    fc_rows["label"] = fc_rows["short_name"].map(_CONSPIRACY_LABELS).fillna(fc_rows["short_name"])
    fc_rows["hr"] = fc_rows["exp(coef)"]
    fc_rows["ci_lower"] = fc_rows["exp(coef) lower 95%"]
    fc_rows["ci_upper"] = fc_rows["exp(coef) upper 95%"]
    fc_rows["p"] = fc_rows["p"]
    fc_rows["significant"] = fc_rows["p"] < significance_level
    boot_table = _bootstrap_table(bootstrap_intervals, "coefficient_intervals")
    if boot_table is not None and model_name is not None:
        boot_rows = boot_table[
            (boot_table["model"] == model_name)
            & boot_table["term"].str.startswith("fc_")
        ].set_index("term")
        for term in fc_rows.index:
            if term not in boot_rows.index:
                log_bootstrap_fallback(
                    "fc_forest",
                    "missing coefficient interval row",
                    model=model_name,
                    term=term,
                    fallback_interval_source="model_summary",
                )
                continue
            fc_rows.loc[term, "hr"] = float(boot_rows.loc[term, "hr"])
            fc_rows.loc[term, "ci_lower"] = float(boot_rows.loc[term, "hr_ci_lower"])
            fc_rows.loc[term, "ci_upper"] = float(boot_rows.loc[term, "hr_ci_upper"])
    else:
        reason = (
            "missing coefficient interval table"
            if boot_table is None
            else "missing model name"
        )
        log_bootstrap_fallback(
            "fc_forest",
            reason,
            model=model_name,
            fallback_interval_source="model_summary",
        )

    # Map cluster colors
    if cluster_labels is not None:
        consp_to_cluster = cluster_labels
        colors_map = cluster_colors if cluster_colors is not None else _colors_for_cluster_labels(
            cluster_labels
        )
    elif cluster_assignments is not None:
        consp_to_cluster, colors_map = build_cluster_display_metadata(
            cluster_assignments
        )
    else:
        consp_to_cluster = {}
        colors_map = _UNKNOWN_CLUSTER_COLOR

    fc_rows["cluster"] = fc_rows["short_name"].map(consp_to_cluster).fillna("Unknown")
    fc_rows["color"] = fc_rows["cluster"].map(colors_map).fillna("#999999")

    # Add reference category at HR=1.0
    ref_row = pd.DataFrame([{
        "short_name": ref_short,
        "label": f"{ref_label} (ref)",
        "hr": 1.0,
        "ci_lower": 1.0,
        "ci_upper": 1.0,
        "p": np.nan,
        "significant": True,
        "cluster": consp_to_cluster.get(ref_short, "Unknown"),
        "color": colors_map.get(consp_to_cluster.get(ref_short, "Unknown"), "#999999"),
    }])
    fc_rows = pd.concat([fc_rows, ref_row], ignore_index=True)

    if sort_by_hr:
        fc_rows = fc_rows.sort_values("hr", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(fc_rows))

    for i, (_, row) in enumerate(fc_rows.iterrows()):
        xerr = np.array([[row["hr"] - row["ci_lower"]], [row["ci_upper"] - row["hr"]]])
        xerr = np.clip(xerr, 0, None)
        marker = "o" if row["significant"] else "o"
        facecolor = row["color"] if row["significant"] else "white"
        ax.errorbar(
            row["hr"], i, xerr=xerr,
            fmt=marker, color=row["color"], markerfacecolor=facecolor,
            markersize=8, capsize=3, elinewidth=1.2, markeredgewidth=1.5,
        )

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fc_rows["label"].values)
    ax.set_xlabel(f"Hazard Ratio (ref = {ref_label})")
    ax.set_title(title)

    # Legend for clusters
    from matplotlib.lines import Line2D
    legend_elements = []
    for cluster_name, color in colors_map.items():
        if cluster_name in fc_rows["cluster"].values:
            legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                          markerfacecolor=color, markersize=8,
                                          label=cluster_name))
    legend_elements.append(Line2D([0], [0], marker="o", color="grey",
                                  markerfacecolor="white", markeredgecolor="grey",
                                  markersize=8, label="Not significant"))
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Forest plot saved to {save_path}")

    return fig


def plot_decay_times_line(
    decay_times: Dict[str, float],
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
) -> plt.Figure:
    """Line chart of decay-to-baseline times t* for each Weibull model.

    Same input as plot_decay_times but rendered as connected line with markers.

    Args:
        decay_times: Dict mapping model names to t* in hours
            (from compute_all_decay_times).
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    names = sorted(decay_times.keys(), key=_model_sort_key)
    boot_table = _bootstrap_table(bootstrap_intervals, "decay_time_intervals")
    values_days = []
    lower_days = []
    upper_days = []
    infinite_shares = []
    for name in names:
        boot_row = pd.DataFrame()
        if boot_table is not None:
            boot_row = boot_table[boot_table["model"] == name]
        if not boot_row.empty:
            value = float(boot_row["t_star_hours"].iloc[0])
            lower = float(boot_row["ci_lower"].iloc[0])
            upper = float(boot_row["ci_upper"].iloc[0])
            infinite_share = float(boot_row.get("infinite_share", pd.Series([0.0])).iloc[0])
        else:
            reason = (
                "missing decay interval table"
                if boot_table is None
                else "missing decay interval row"
            )
            log_bootstrap_fallback(
                "decay_times_line",
                reason,
                model=name,
            )
            value = float(decay_times[name])
            lower = value
            upper = value
            infinite_share = 0.0
        values_days.append(value / 24)
        lower_days.append(lower / 24)
        upper_days.append(upper / 24)
        infinite_shares.append(infinite_share)
    # Map "model_2a" / "model_3" -> adoption number (2, 3, ...)
    labels = []
    for n in names:
        s = n.replace("model_", "")
        num = ''.join(c for c in s if c.isdigit())
        labels.append(num)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    ax.plot(x, values_days, "o-", color="steelblue", linewidth=2, markersize=8)
    yerr = np.vstack([
        np.maximum(np.asarray(values_days) - np.asarray(lower_days), 0.0),
        np.maximum(np.asarray(upper_days) - np.asarray(values_days), 0.0),
    ])
    finite_err = np.where(np.isfinite(yerr), yerr, 0.0)
    ax.errorbar(x, values_days, yerr=finite_err, fmt="none", capsize=4, color="steelblue")

    for i, (lbl, val, infinite_share) in enumerate(zip(labels, values_days, infinite_shares)):
        if np.isfinite(val):
            label_text = f"{val:.1f}d"
            if infinite_share > 0:
                label_text = f"{label_text}\n{infinite_share:.0%} inf"
            ax.annotate(
                label_text,
                xy=(i, val),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=nhb_annotation_fontsize(),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of conspiracies adopted")
    ax.set_ylabel("Time for hazard to recover to baseline (days)")
    ax.set_title(
        "Time for the Hazard to Recover to Baseline",
        fontsize=nhb_main_title_fontsize(),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Decay times line chart saved to {save_path}")

    return fig


def plot_hawkes_dynamics(
    mu: float,
    alpha: float,
    beta: float,
    t_max: float = 500.0,
    n_simulations: int = 20,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Two-panel visualization of Hawkes self-exciting sharing dynamics.

    Left: Intensity decay curve after a single adoption event.
    Right: Simulated sharing timelines (event raster plot).

    Args:
        mu: Fitted baseline sharing rate.
        alpha: Fitted excitation parameter.
        beta: Fitted decay parameter.
        t_max: Time horizon in hours for panels A and B.
        n_simulations: Number of simulated sequences for panel B.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    from ..models.hawkes_sharing import simulate_hawkes_sequence

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] // 2))

    # --- Panel A: Intensity Decay Curve ---
    t_arr = np.linspace(0.01, t_max, 1000)
    # Intensity after a single event at t=0
    intensity_single = mu + alpha * beta * np.exp(-beta * t_arr)
    # Integrated intensity over [t, t+1] for sharing probability
    kernel_integral = alpha * (1.0 - np.exp(-beta)) if beta > 0 else 0.0
    integrated_single = mu + kernel_integral * np.exp(-beta * t_arr)
    prob_single = 1.0 - np.exp(-integrated_single)

    ax_a.plot(t_arr, intensity_single, color="steelblue", linewidth=2, label=r"$\lambda(t)$")
    ax_a.axhline(mu, color="grey", linestyle="--", linewidth=1, label=rf"$\mu$ = {mu:.4f}")

    # Mark half-life
    half_life = np.log(2) / beta
    intensity_at_half = mu + alpha * beta * np.exp(-beta * half_life)
    ax_a.axvline(half_life, color="coral", linestyle=":", linewidth=1,
                 label=f"Half-life = {half_life:.1f}h")
    ax_a.plot(half_life, intensity_at_half, "o", color="coral", markersize=6)

    # Mark time to decay to 2×mu
    if alpha * beta > mu:
        t_2mu = -np.log(mu / (alpha * beta)) / beta
        if t_2mu <= t_max:
            ax_a.axvline(t_2mu, color="darkred", linestyle=":", linewidth=1, alpha=0.7,
                         label=rf"$\lambda \to 2\mu$ at {t_2mu:.0f}h")

    ax_a.set_xlabel("Hours since adoption")
    ax_a.set_ylabel(r"Intensity $\lambda(t)$ (events/hour)")
    ax_a.set_title("Intensity Decay After Single Adoption")
    ax_a.legend()

    # Secondary y-axis for sharing probability
    ax_a2 = ax_a.twinx()
    ax_a2.plot(t_arr, prob_single, color="steelblue", linewidth=1, alpha=0.3)
    ax_a2.set_ylabel("P(share in next hour)", color="steelblue", alpha=0.5)
    ax_a2.tick_params(axis="y", labelcolor="steelblue")

    # --- Panel B: Simulated Sharing Timelines ---
    # Seed with initial adoption event at t=0, which triggers the self-exciting
    # cascade (matches ABM where record_adoption calls record_share)
    rng = np.random.default_rng(42)
    for i in range(n_simulations):
        events = simulate_hawkes_sequence(mu, alpha, beta, t_max, rng, initial_history=[0.0])
        ax_b.eventplot([events], lineoffsets=i, linelengths=0.7, colors="steelblue", linewidths=0.8)

    ax_b.set_xlabel("Hours since adoption")
    ax_b.set_ylabel("Simulated sequence")
    ax_b.set_title("Simulated Sharing Timelines")
    ax_b.set_yticks(range(n_simulations))
    ax_b.set_yticklabels([f"#{i+1}" for i in range(n_simulations)])
    ax_b.set_xlim(0, t_max)

    fig.suptitle(
        rf"Hawkes Sharing Dynamics ($\mu$={mu:.4f}, $\alpha$={alpha:.3f}, $\beta$={beta:.4f})",
        fontsize=nhb_main_title_fontsize(), y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Hawkes dynamics plot saved to {save_path}")

    return fig


def plot_hawkes_goodness_of_fit(
    sequences: List[List[float]],
    fit_result,
    figsize: Tuple[float, float] = (
        NHB_COL_DOUBLE[0],
        NHB_COL_DOUBLE[1] * 0.65,
    ),
    save_path: Optional[str] = None,
    observation_ends: Optional[List[float]] = None,
) -> plt.Figure:
    """Two panel comparison and simulation diagnostics for the Hawkes process.

    Panel A: AIC model comparison (Hawkes vs Poisson alternatives).
    Panel B: Empirical versus simulated interevent time distributions.

    Args:
        sequences: Empirical sharing sequences (from extract_sharing_sequences).
        fit_result: Structured Hawkes fit result.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.
        observation_ends: Optional observation window end times per sequence.

    Returns:
        Matplotlib Figure object.
    """
    from ..models.hawkes_sharing import (
        hawkes_model_comparison,
        simulate_hawkes_sequence,
    )

    mu, alpha, beta = fit_result.params

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

    clean_seqs = [s for s in sequences if len(s) > 1]
    if observation_ends is None:
        raise ValueError("observation_ends are required for parametric diagnostics")

    # Panel A: model comparison
    comp = hawkes_model_comparison(
        sequences, fit_result, observation_ends=observation_ends
    )
    colors_a = ["#999999", "#E24A33", "steelblue"]
    bars = ax_a.bar(comp["model"], comp["AIC"], color=colors_a, edgecolor="black")

    min_aic = comp["AIC"].min()
    for bar, aic_val in zip(bars, comp["AIC"]):
        delta = aic_val - min_aic
        label = f"ΔAIC={delta:.0f}" if delta > 0 else "Best"
        ax_a.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=nhb_annotation_fontsize(),
        )
    ax_a.set_ylabel("AIC (lower is better)")
    ax_a.set_title("A. AIC Model Comparison")
    ax_a.tick_params(axis="x", rotation=15)

    # Panel B: empirical versus simulated interevent times
    emp_dt = []
    for seq in clean_seqs:
        emp_dt.extend(np.diff(seq).tolist())
    emp_dt = np.array(emp_dt)

    # Simulate sequences with matching T distribution, conditioned on
    # a triggering event at t=0 (matching empirical adoption sequences)
    rng = np.random.default_rng(42)
    t_maxes = list(observation_ends)
    sim_dt = []
    for t_max_i in t_maxes:
        events = simulate_hawkes_sequence(mu, alpha, beta, t_max_i, rng, initial_history=[0.0])
        if len(events) > 1:
            sim_dt.extend(np.diff(events).tolist())
    sim_dt = np.array(sim_dt)

    if len(emp_dt) > 0 and len(sim_dt) > 0:
        max_dt = np.percentile(emp_dt, 99)
        bins = np.linspace(0, max_dt, 60)
        ax_b.hist(emp_dt, bins=bins, density=True, alpha=0.6, color="steelblue",
                  label=f"Empirical (n={len(emp_dt):,})", edgecolor="white", linewidth=0.3)
        ax_b.hist(sim_dt, bins=bins, density=True, alpha=0.5, color="coral",
                  label=f"Simulated (n={len(sim_dt):,})", edgecolor="white", linewidth=0.3)
        ax_b.legend()
    ax_b.set_xlabel("Interevent time (hours)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("B. Interevent Time Distribution")

    fig.suptitle(
        rf"Hawkes Model Comparison and Simulation Check ($\mu$={mu:.4f}, $\alpha$={alpha:.3f}, $\beta$={beta:.4f})",
        fontsize=nhb_main_title_fontsize(), y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Hawkes comparison plot saved to {save_path}")

    return fig


def _get_cluster_label(consp_name: str, cluster_assignments: Optional[Dict] = None) -> str:
    """Get semantic cluster label for a conspiracy name."""
    short = consp_name.replace("ConsProb_", "")
    if cluster_assignments is not None:
        consp_to_cluster, _ = build_cluster_display_metadata(cluster_assignments)
        return consp_to_cluster.get(consp_name, consp_to_cluster.get(short, "Unknown"))
    return "Unknown"
