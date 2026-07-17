"""
Publication-quality figures for the conspiracy analysis paper.

Includes dendrogram, cognitive barrier, settler effect, and paper-specific
composite figures for the conspiracy adoption dynamics narrative.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from lifelines import CoxTimeVaryingFitter

from conspiracy_analysis.config import get_baseline_conspiracy
from conspiracy_analysis.visualization.style import (
    NHB_COL_SINGLE, NHB_COL_ONE_HALF, NHB_COL_DOUBLE,
    nhb_annotation_fontsize,
    nhb_main_title_fontsize,
    nhb_significance_fontsize,
)
from conspiracy_analysis.analysis.statistics import (
    bootstrap_median_ci,
    test_barrier_significance,
    test_settler_significance,
)
from conspiracy_analysis.models.baseline_hazards import calculate_baseline_hazard
from conspiracy_analysis.utils.fallback_logging import log_bootstrap_fallback
from conspiracy_analysis.visualization.plots import _CONSPIRACY_LABELS

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


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    title: str = "Hierarchical Clustering Dendrogram",
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a dendrogram from a linkage matrix.

    Args:
        linkage_matrix: Output from scipy.cluster.hierarchy.linkage.
        labels: Conspiracy labels for leaf nodes.
        title: Plot title.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    hierarchy.dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title(title, fontsize=nhb_main_title_fontsize())
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Dendrogram saved to {save_path}")

    return fig


def plot_cognitive_barrier(
    gaps: Dict[str, List[float]],
    n_boot: int = 2000,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot within-cluster vs. between-cluster transition times.

    Bar chart with bootstrapped 95% confidence intervals on the median.

    Args:
        gaps: Dict with 'within_cluster' and 'between_clusters' keys,
            each mapping to a list of transition times (hours).
        n_boot: Number of bootstrap resamples.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    labels = ["Within Cluster", "Between Clusters"]
    keys = ["within_cluster", "between_clusters"]
    colors = ["#4A90E2", "#D0021B"]

    medians = []
    errors = [[], []]  # [lower_errors, upper_errors]

    for key in keys:
        data = gaps.get(key, [])
        if len(data) > 0:
            med, err_l, err_u = bootstrap_median_ci(data, n_boot=n_boot)
            medians.append(med)
            errors[0].append(err_l)
            errors[1].append(err_u)
        else:
            medians.append(0)
            errors[0].append(0)
            errors[1].append(0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, medians, yerr=errors, capsize=10, color=colors, alpha=0.8)
    ax.set_title(
        "Jumping to a New Cluster is Slower\n(Median with 95% Bootstrap CI)",
        fontsize=nhb_main_title_fontsize(),
    )
    ax.set_ylabel("Median Hours to Adoption")

    for i, v in enumerate(medians):
        ax.text(i, v, f"{v:.1f}h", ha="center", va="bottom", fontsize=nhb_annotation_fontsize())

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Cognitive barrier plot saved to {save_path}")

    return fig


def plot_settler_effect(
    settler_gaps: Dict[str, List[float]],
    n_boot: int = 2000,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the settler effect: pre-jump, jump, and post-jump dynamics.

    Line chart with error bars showing the characteristic slow-down
    when crossing semantic cluster boundaries.

    Args:
        settler_gaps: Dict with 'pre_jump', 'the_jump', 'post_jump' keys.
        n_boot: Number of bootstrap resamples.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    phase_keys = ["pre_jump", "the_jump", "post_jump"]
    phase_labels = [
        "Within old Cluster",
        "Jump to new Cluster",
        "Sharing additional\nin new Cluster",
    ]

    medians = []
    errors = [[], []]

    for key in phase_keys:
        data = settler_gaps.get(key, [])
        med, err_l, err_u = bootstrap_median_ci(data, n_boot=n_boot)
        medians.append(med)
        errors[0].append(err_l)
        errors[1].append(err_u)

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = [0, 1, 2]
    marker_colors = ["#4A90E2", "#D0021B", "#4A90E2"]

    ax.plot(x_pos, medians, color="#333333", linewidth=3, zorder=2)
    ax.scatter(x_pos, medians, color=marker_colors, s=150, zorder=3)
    for x, median, err_l, err_u, color in zip(
        x_pos, medians, errors[0], errors[1], marker_colors
    ):
        ax.errorbar(
            x, median, yerr=[[err_l], [err_u]], fmt="none",
            ecolor=color, elinewidth=1.5, capsize=5, zorder=1,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("Median Hours to Adoption")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Settler effect plot saved to {save_path}")

    return fig


def _p_to_stars(p: float) -> str:
    """Map a p-value to a conventional star annotation."""
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _settler_on_ax(
    ax,
    settler_gaps: Dict[str, List[float]],
    n_boot: int = 2000,
    show_significance: bool = False,
    bootstrap_intervals=None,
    bootstrap_family: Optional[str] = None,
) -> None:
    """Render a single settler panel onto a pre-made Axes.

    When ``show_significance=True``, overlays Holm-corrected pairwise
    Wilcoxon significance brackets between adjacent phase comparisons.
    """
    phase_keys = ["pre_jump", "the_jump", "post_jump"]
    phase_labels = [
        "Within old Cluster",
        "Jump to new Cluster",
        "Sharing additional\nin new Cluster",
    ]
    medians = []
    errors = [[], []]
    boot_table = _bootstrap_table(bootstrap_intervals, "settler_intervals")
    if boot_table is None:
        raise ValueError(
            "plot_settler_effect_triptych requires timeline bootstrap settler_intervals."
        )
    if bootstrap_family is None:
        raise ValueError(
            "plot_settler_effect_triptych requires bootstrap_family_map for every panel."
        )
    for key in phase_keys:
        mask = (
            (boot_table["family"] == bootstrap_family)
            & (boot_table["key"] == key)
        )
        if "statistic" in boot_table.columns:
            mask = mask & (boot_table["statistic"] == "settler")
        boot_row = boot_table.loc[mask]
        if not boot_row.empty:
            med = float(boot_row["estimate"].iloc[0])
            lower = float(boot_row["ci_lower"].iloc[0])
            upper = float(boot_row["ci_upper"].iloc[0])
            err_l = max(med - lower, 0.0)
            err_u = max(upper - med, 0.0)
        else:
            raise ValueError(
                "Missing timeline bootstrap settler interval "
                f"for family={bootstrap_family}, key={key}."
            )
        medians.append(med)
        errors[0].append(err_l)
        errors[1].append(err_u)

    x_pos = [0, 1, 2]
    marker_colors = ["#4A90E2", "#D0021B", "#4A90E2"]
    ax.plot(x_pos, medians, color="#333333", linewidth=3, zorder=2)
    ax.scatter(x_pos, medians, color=marker_colors, s=150, zorder=3)
    for x, median, err_l, err_u, color in zip(
        x_pos, medians, errors[0], errors[1], marker_colors
    ):
        ax.errorbar(
            x, median, yerr=[[err_l], [err_u]], fmt="none",
            ecolor=color, elinewidth=1.5, capsize=5, zorder=1,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("Median Hours to Adoption")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if show_significance:
        try:
            pair_stats = test_settler_significance(settler_gaps)["pairwise"]
        except Exception as exc:  # pragma: no cover — too few settlers for Friedman
            logger.warning(f"Could not compute settler significance: {exc}")
            return

        upper_bounds = np.asarray(medians) + np.asarray(errors[1])
        y_top = float(np.max(upper_bounds))
        y_bot = float(np.min(np.asarray(medians) - np.asarray(errors[0])))
        span = max(y_top - y_bot, 1.0)
        step = span * 0.12

        brackets = [
            ("pre_jump_vs_jump",       0, 1, y_top + step * 1.0),
            ("jump_vs_post_jump",      1, 2, y_top + step * 1.0),
            ("pre_jump_vs_post_jump",  0, 2, y_top + step * 2.0),
        ]
        tick = step * 0.25
        for key, xi, xj, y in brackets:
            p = pair_stats.get(key, {}).get("p_value_corrected", np.nan)
            label = _p_to_stars(p)
            ax.plot(
                [xi, xi, xj, xj],
                [y - tick, y, y, y - tick],
                color="#333333", linewidth=1.0, clip_on=False,
            )
            ax.text(
                (xi + xj) / 2, y + tick * 0.3, label,
                ha="center", va="bottom",
                fontsize=nhb_significance_fontsize(),
            )

        # Extend y-limits so the top bracket is not clipped.
        current_top = ax.get_ylim()[1]
        ax.set_ylim(top=max(current_top, y_top + step * 2.8))


def plot_settler_effect_triptych(
    panels: List[Tuple[str, Dict[str, List[float]]]],
    n_boot: int = 2000,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    share_y: bool = True,
    show_significance: bool = False,
    bootstrap_intervals=None,
    bootstrap_family_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Render ``len(panels)`` settler-effect panels side by side on a single row.

    Args:
        panels: List of (subplot_title, settler_gaps) pairs.
        n_boot: Retained for compatibility. Settler intervals come from
            bootstrap_intervals.
        figsize: Figure dimensions. Defaults to NHB_COL_DOUBLE scaled to the
            number of panels.
        save_path: If provided, save figure to this path.
        share_y: If True, put all panels on a common y-axis.
        show_significance: If True, overlay Holm-corrected pairwise Wilcoxon
            significance brackets on each panel.

    Returns:
        Matplotlib Figure object.
    """
    n = len(panels)
    if figsize is None:
        figsize = (NHB_COL_DOUBLE[0] * (n / 3), NHB_COL_DOUBLE[1] * 0.65)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=share_y, squeeze=False)
    axes = axes[0]
    bootstrap_family_map = bootstrap_family_map or {}
    for ax, (title, settler) in zip(axes, panels):
        _settler_on_ax(
            ax,
            settler,
            n_boot=n_boot,
            show_significance=show_significance,
            bootstrap_intervals=bootstrap_intervals,
            bootstrap_family=bootstrap_family_map.get(title),
        )
        ax.set_title(title, fontsize=nhb_main_title_fontsize())

    # Keep one y-axis label per row on the leftmost panel.
    for ax in axes[1:]:
        ax.set_ylabel("")

    # If sharing y and significance brackets are drawn, the brackets can push the
    # top limit up on individual panels; propagate the max across panels.
    if share_y and show_significance:
        top = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(top=top)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Settler triptych saved to {save_path}")

    return fig


# Composite figure functions


_MODEL_COLORS = {
    "model_1": "#1f77b4",
    "model_2a": "#ff7f0e",
    "model_3": "#2ca02c",
    "model_4": "#d62728",
    "model_5": "#9467bd",
}

_MODEL_LABELS = {
    "model_1": "Model 1 (1st adoption)",
    "model_2a": "Model 2a (2nd adoption)",
    "model_3": "Model 3 (3rd adoption)",
    "model_4": "Model 4 (4th adoption)",
    "model_5": "Model 5 (5th adoption)",
}


def _reref_exposure_hrs(
    ctv,
    exposure_vars,
    ref_var="s7_d1",
    model_name=None,
    reason="using variance matrix confidence interval fallback",
):
    """Re-reference exposure HRs to ref_var using the variance-covariance matrix.

    Computes exp(beta_k - beta_ref) with correct CI propagation via the
    delta method: Var(beta_k - beta_ref) = Var(beta_k) + Var(beta_ref)
                                           - 2*Cov(beta_k, beta_ref).

    Returns (hrs, ci_lowers, ci_uppers) arrays of length len(exposure_vars),
    where the ref_var entry is exactly (1.0, 1.0, 1.0).
    """
    log_bootstrap_fallback(
        "exposure_hr",
        reason,
        model=model_name,
        ref_term=ref_var,
        terms=",".join(exposure_vars),
        fallback_interval_source="cox_variance_matrix",
    )

    summary = ctv.summary
    V = ctv.variance_matrix_
    var_names = list(V.index)

    hrs, ci_lowers, ci_uppers = [], [], []
    for var in exposure_vars:
        if var not in summary.index or ref_var not in summary.index:
            hrs.append(np.nan)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            continue

        beta_k = summary.loc[var, "coef"]
        beta_ref = summary.loc[ref_var, "coef"]
        delta = beta_k - beta_ref

        idx_k = var_names.index(var)
        idx_ref = var_names.index(ref_var)
        var_delta = (V.iloc[idx_k, idx_k]
                     + V.iloc[idx_ref, idx_ref]
                     - 2 * V.iloc[idx_k, idx_ref])
        se_delta = np.sqrt(max(var_delta, 0.0))

        hrs.append(np.exp(delta))
        ci_lowers.append(np.exp(delta - 1.96 * se_delta))
        ci_uppers.append(np.exp(delta + 1.96 * se_delta))

    return np.array(hrs), np.array(ci_lowers), np.array(ci_uppers)


def _bootstrap_exposure_hrs(model_name, exposure_vars, bootstrap_intervals):
    boot_table = _bootstrap_table(bootstrap_intervals, "exposure_intervals")
    if boot_table is None:
        log_bootstrap_fallback(
            "exposure_hr",
            "missing exposure interval table",
            model=model_name,
        )
        return None
    rows = boot_table[boot_table["model"] == model_name]
    if rows.empty:
        log_bootstrap_fallback(
            "exposure_hr",
            "missing exposure model row",
            model=model_name,
        )
        return None
    rows = rows.set_index("term")
    hrs, ci_lowers, ci_uppers = [], [], []
    for var in exposure_vars:
        if var not in rows.index:
            log_bootstrap_fallback(
                "exposure_hr",
                "missing exposure term row",
                model=model_name,
                term=var,
            )
            return None
        row = rows.loc[var]
        hrs.append(float(row["hr"]))
        ci_lowers.append(float(row["ci_lower"]))
        ci_uppers.append(float(row["ci_upper"]))
    return np.array(hrs), np.array(ci_lowers), np.array(ci_uppers)


def plot_exposure_hr_comparison(
    cox_results: Dict[str, Tuple[CoxTimeVaryingFitter, object]],
    models_to_include: Optional[List[str]] = None,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
    bootstrap_model_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Line plot comparing exposure HRs across models (re-ref to 1 neighbor).

    Re-references all HRs so that s7_d1 = 1.0 for each model. This divides
    all HRs and CI bounds by that model's s7_d1 HR.

    Args:
        cox_results: Dict mapping model names to (ctv, df) tuples.
        models_to_include: Which models to include. Defaults to
            model_1, model_2a, model_3, model_4, model_5.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    if models_to_include is None:
        models_to_include = ["model_1", "model_2a", "model_3", "model_4", "model_5"]

    exposure_vars = ["s7_d2", "s7_d3", "s7_d4"]
    exposure_labels = ["2", "3", "4+"]
    bootstrap_model_map = bootstrap_model_map or {}

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(exposure_vars))
    n_models = len(models_to_include)
    jitter_width = 0.06  # offset per model
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * jitter_width

    for m_idx, model_name in enumerate(models_to_include):
        if model_name not in cox_results:
            continue
        ctv = cox_results[model_name][0]

        interval_name = bootstrap_model_map.get(model_name, model_name)
        boot_values = _bootstrap_exposure_hrs(
            interval_name,
            exposure_vars,
            bootstrap_intervals,
        )
        if boot_values is None:
            hrs, ci_lowers, ci_uppers = _reref_exposure_hrs(
                ctv,
                exposure_vars,
                model_name=interval_name,
            )
        else:
            hrs, ci_lowers, ci_uppers = boot_values

        yerr = np.vstack([hrs - ci_lowers, ci_uppers - hrs])
        yerr = np.clip(yerr, 0, None)

        color = _MODEL_COLORS.get(model_name, f"C{m_idx}")
        label = _MODEL_LABELS.get(model_name, model_name)

        ax.errorbar(x + offsets[m_idx], hrs, yerr=yerr, fmt="o-", color=color,
                    markersize=8, linewidth=2, capsize=5, capthick=1.5,
                    label=label)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(exposure_labels)
    ax.set_xlabel("Number of Conspiracy Sharing Neighbors (336 h window)")
    ax.set_ylabel("Hazard Ratio (ref = 1 neighbor)")
    ax.set_title(
        "Exposure Effect: Hazard Ratios Re-Referenced to 1 Neighbor",
        fontsize=nhb_main_title_fontsize(),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Exposure HR comparison saved to {save_path}")

    return fig


_EXPOSURE_COLORS = {
    "s7_d1": "#4e79a7",   # 1 neighbor
    "s7_d2": "#f28e2b",   # 2 neighbors
    "s7_d3": "#e15759",   # 3 neighbors
    "s7_d4": "#76b7b2",   # 4+ neighbors
}

_EXPOSURE_LABELS = {
    "s7_d1": "1 neighbor",
    "s7_d2": "2 neighbors",
    "s7_d3": "3 neighbors",
    "s7_d4": "4+ neighbors",
}


def plot_exposure_hr_by_model(
    cox_results: Dict[str, Tuple[CoxTimeVaryingFitter, object]],
    models_to_include: Optional[List[str]] = None,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
) -> plt.Figure:
    """Line plot of exposure HRs with model number on x-axis and neighbor count as hue.

    Re-references all HRs so that s7_d1 = 1.0 for each model (same logic as
    plot_exposure_hr_comparison). Shows how the dose-response to network
    exposure evolves across adoption stages.

    Args:
        cox_results: Dict mapping model names to (ctv, df) tuples.
        models_to_include: Which models to include. Defaults to
            model_1, model_2a, model_3, model_4, model_5.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    if models_to_include is None:
        models_to_include = ["model_1", "model_2a", "model_3", "model_4", "model_5"]

    exposure_vars = ["s7_d2", "s7_d3", "s7_d4"]

    # Collect data: for each model, extract re-referenced HRs per exposure level
    model_order = [m for m in models_to_include if m in cox_results]
    if not model_order:
        logger.warning("No matching models found in cox_results.")
        return plt.figure()

    # Structure: {exposure_var: {"hrs": [...], "ci_lower": [...], "ci_upper": [...]}}
    exposure_data = {var: {"hrs": [], "ci_lower": [], "ci_upper": []} for var in exposure_vars}

    for model_name in model_order:
        ctv = cox_results[model_name][0]

        boot_values = _bootstrap_exposure_hrs(
            model_name,
            exposure_vars,
            bootstrap_intervals,
        )
        if boot_values is None:
            hrs, ci_lowers, ci_uppers = _reref_exposure_hrs(
                ctv,
                exposure_vars,
                model_name=model_name,
            )
        else:
            hrs, ci_lowers, ci_uppers = boot_values

        for i, var in enumerate(exposure_vars):
            exposure_data[var]["hrs"].append(hrs[i])
            exposure_data[var]["ci_lower"].append(ci_lowers[i])
            exposure_data[var]["ci_upper"].append(ci_uppers[i])

    # Build display labels for x-axis
    x_labels_map = {
        "model_1": "Model 1",
        "model_2a": "Model 2",
        "model_3": "Model 3",
        "model_4": "Model 4",
        "model_5": "Model 5",
        "model_6": "Model 6",
        "model_7": "Model 7",
        "model_8": "Model 8",
    }

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(model_order))
    n_levels = len(exposure_vars)
    jitter_width = 0.06
    offsets = (np.arange(n_levels) - (n_levels - 1) / 2) * jitter_width

    for lvl_idx, var in enumerate(exposure_vars):
        hrs = np.array(exposure_data[var]["hrs"])
        ci_lower = np.array(exposure_data[var]["ci_lower"])
        ci_upper = np.array(exposure_data[var]["ci_upper"])

        yerr = np.vstack([hrs - ci_lower, ci_upper - hrs])
        yerr = np.clip(yerr, 0, None)

        color = _EXPOSURE_COLORS[var]
        label = _EXPOSURE_LABELS[var]

        ax.errorbar(
            x + offsets[lvl_idx], hrs, yerr=yerr,
            fmt="o-", color=color, markersize=8, linewidth=2,
            capsize=5, capthick=1.5, label=label,
        )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([x_labels_map.get(m, m) for m in model_order])
    ax.set_xlabel("Adoption Stage")
    ax.set_ylabel("Hazard Ratio (ref = 1 neighbor)")
    ax.set_title(
        "Network Sensitivity Across Adoption Stages",
        fontsize=nhb_main_title_fontsize(),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Exposure HR by model saved to {save_path}")

    return fig


def plot_temporal_hazard_shift(
    all_baselines: Dict[str, Dict],
    eval_times_hours: Optional[List[float]] = None,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
) -> plt.Figure:
    """Compare Weibull h0(t) from Models 2a, 3, 4, and 5 vs Model 1 constant h0.

    Shows grouped ratio bars: how many times higher each Weibull model's
    hazard is at each timepoint compared to Model 1's constant baseline.

    Args:
        all_baselines: Dict from parametrize_all_baselines.
        eval_times_hours: Timepoints in hours to evaluate. Defaults to [1, 24, 72].
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    if eval_times_hours is None:
        eval_times_hours = [1.0, 24.0, 72.0]

    m1_params = all_baselines.get("model_1")
    if m1_params is None:
        logger.warning("Need model_1 baseline for reference.")
        return plt.figure()

    m1_h0 = calculate_baseline_hazard(1.0, m1_params)  # constant

    weibull_models = ["model_2a", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8"]
    boot_table = _bootstrap_table(bootstrap_intervals, "temporal_hazard_ratio_intervals")
    model_ratios = {}  # {model_name: [ratio_at_t1, ratio_at_t2, ...]}
    model_lowers = {}
    model_uppers = {}
    for model_name in weibull_models:
        params = all_baselines.get(model_name)
        if params is None:
            continue
        ratios = []
        lowers = []
        uppers = []
        for t in eval_times_hours:
            boot_row = pd.DataFrame()
            if boot_table is not None:
                boot_row = boot_table[
                    (boot_table["model"] == model_name)
                    & (np.isclose(boot_table["time_hours"].astype(float), float(t)))
                ]
            if not boot_row.empty:
                ratios.append(float(boot_row["ratio"].iloc[0]))
                lowers.append(float(boot_row["ci_lower"].iloc[0]))
                uppers.append(float(boot_row["ci_upper"].iloc[0]))
            else:
                reason = (
                    "missing temporal hazard interval table"
                    if boot_table is None
                    else "missing temporal hazard interval row"
                )
                log_bootstrap_fallback(
                    "temporal_hazard_shift",
                    reason,
                    model=model_name,
                    time_hours=float(t),
                )
                h0 = calculate_baseline_hazard(t, params)
                ratio = h0 / m1_h0 if m1_h0 > 0 else 0
                ratios.append(ratio)
                lowers.append(ratio)
                uppers.append(ratio)
        model_ratios[model_name] = ratios
        model_lowers[model_name] = lowers
        model_uppers[model_name] = uppers

    labels = [f"{t:.0f}h" if t >= 1 else f"{t*60:.0f}m" for t in eval_times_hours]
    x = np.arange(len(labels))
    n_models = len(model_ratios)
    width = 0.8 / max(n_models, 1)  # bar width so groups fit nicely

    model_colors = {
        "model_2a": "#E24A33",
        "model_3":  "#348ABD",
        "model_4":  "#988ED5",
        "model_5":  "#9467bd",
        "model_6":  "#8EBA42",
        "model_7":  "#FFB000",
        "model_8":  "#777777",
    }

    model_display_names = {
        "model_2a": "After 1st Conspiracy",
        "model_3":  "After 2nd Conspiracy",
        "model_4":  "After 3rd Conspiracy",
        "model_5":  "After 4th Conspiracy",
        "model_6":  "After 5th Conspiracy",
        "model_7":  "After 6th Conspiracy",
        "model_8":  "After 7th Conspiracy",
    }

    fig, ax = plt.subplots(figsize=figsize)

    for i, (model_name, ratios) in enumerate(model_ratios.items()):
        offset = (i - (n_models - 1) / 2) * width
        color = model_colors.get(model_name, f"C{i}")
        display_name = model_display_names.get(model_name, model_name.replace("_", " ").title())
        yerr = np.vstack([
            np.maximum(np.asarray(ratios) - np.asarray(model_lowers[model_name]), 0.0),
            np.maximum(np.asarray(model_uppers[model_name]) - np.asarray(ratios), 0.0),
        ])
        bars = ax.bar(
            x + offset, ratios, width, yerr=yerr, capsize=2,
            color=color, edgecolor="black", alpha=0.85, label=display_name,
        )
        for bar, ratio in zip(bars, ratios):
            ax.annotate(
                f"{ratio:.1f}×",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=nhb_annotation_fontsize(),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.2, label="Before 1st Conspiracy (Baseline)")
    ax.set_xlabel("Time Since Last Conspiracy Adoption (hours)")
    ax.set_ylabel("Baseline Hazard Ratio (vs. Before 1st Conspiracy)")
    ax.set_title(
        "Temporal Hazard Shift Relative to Before First Conspiracy Adoption",
        fontsize=nhb_main_title_fontsize(),
    )
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Temporal hazard shift plot saved to {save_path}")

    return fig


def plot_cross_cluster_progression(
    cox_results: Dict[str, Tuple[CoxTimeVaryingFitter, object]],
    models_to_include: Optional[List[str]] = None,
    figsize: Tuple[float, float] = NHB_COL_ONE_HALF,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
) -> plt.Figure:
    """Line chart of cross_cluster HR across sequential models.

    Extracts the cross_cluster coefficient from each model's summary.

    Args:
        cox_results: Dict mapping model names to (ctv, df) tuples.
        models_to_include: Which models to include. Defaults to
            model_2a, model_3, model_4, model_5.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    if models_to_include is None:
        models_to_include = ["model_2a", "model_3", "model_4", "model_5"]

    labels_map = {
        "model_2a": "2nd",
        "model_3": "3rd",
        "model_4": "4th",
        "model_5": "5th",
    }

    hrs, ci_lowers, ci_uppers, x_labels = [], [], [], []
    boot_table = _bootstrap_table(bootstrap_intervals, "coefficient_intervals")

    for model_name in models_to_include:
        if model_name not in cox_results:
            continue
        ctv = cox_results[model_name][0]
        summary = ctv.summary
        if "cross_cluster" not in summary.index:
            continue

        boot_row = pd.DataFrame()
        if boot_table is not None:
            boot_row = boot_table[
                (boot_table["model"] == model_name)
                & (boot_table["term"] == "cross_cluster")
            ]
        if not boot_row.empty:
            hrs.append(float(boot_row["hr"].iloc[0]))
            ci_lowers.append(float(boot_row["hr_ci_lower"].iloc[0]))
            ci_uppers.append(float(boot_row["hr_ci_upper"].iloc[0]))
        else:
            reason = (
                "missing coefficient interval table"
                if boot_table is None
                else "missing cross cluster interval row"
            )
            log_bootstrap_fallback(
                "cross_cluster_progression",
                reason,
                model=model_name,
                term="cross_cluster",
                fallback_interval_source="model_summary",
            )
            row = summary.loc["cross_cluster"]
            hrs.append(row["exp(coef)"])
            ci_lowers.append(row["exp(coef) lower 95%"])
            ci_uppers.append(row["exp(coef) upper 95%"])
        x_labels.append(labels_map.get(model_name, model_name))

    if not hrs:
        logger.warning("No cross_cluster coefficients found.")
        return plt.figure()

    hrs = np.array(hrs)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    yerr = np.vstack([hrs - ci_lowers, ci_uppers - hrs])
    yerr = np.clip(yerr, 0, None)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(hrs))

    ax.errorbar(x, hrs, yerr=yerr, fmt="o-", color="#d62728",
                markersize=10, linewidth=2, capsize=5, capthick=1.5)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Adoption Number")
    ax.set_ylabel("Cross-Cluster Hazard Ratio")
    ax.set_title(
        "Cross-Cluster Penalty Across Sequential Adoptions",
        fontsize=nhb_main_title_fontsize(),
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Cross-cluster progression saved to {save_path}")

    return fig


def plot_temporal_null_comparison(
    semantic_gaps: Dict[str, List[float]],
    temporal_gaps: Dict[str, List[float]],
    semantic_settler: Dict[str, List[float]],
    temporal_settler: Dict[str, List[float]],
    n_boot: int = 2000,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
) -> plt.Figure:
    """2x2 panel comparing semantic vs. temporal barrier and settler effects.

    Top-left: semantic barrier, Top-right: temporal barrier.
    Bottom-left: semantic settler, Bottom-right: temporal settler.

    Args:
        semantic_gaps: Semantic barrier gaps (within_cluster, between_clusters).
        temporal_gaps: Temporal barrier gaps (within_cluster, between_clusters).
        semantic_settler: Semantic settler gaps (pre_jump, the_jump, post_jump).
        temporal_settler: Temporal settler gaps (pre_jump, the_jump, post_jump).
        n_boot: Number of bootstrap resamples.
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    barrier_boot = _bootstrap_table(bootstrap_intervals, "barrier_intervals")
    settler_boot = _bootstrap_table(bootstrap_intervals, "settler_intervals")

    # ── Helper: barrier bar chart ─────────────────────────────────────
    def _plot_barrier(ax, gaps, title, family):
        labels = ["Within Cluster", "Between Clusters"]
        keys = ["within_cluster", "between_clusters"]
        colors = ["#4A90E2", "#D0021B"]
        medians, errors = [], [[], []]
        for key in keys:
            boot_row = pd.DataFrame()
            if barrier_boot is not None:
                boot_row = barrier_boot[
                    (barrier_boot["family"] == family)
                    & (barrier_boot["key"] == key)
                ]
            if not boot_row.empty:
                med = float(boot_row["estimate"].iloc[0])
                err_l = max(med - float(boot_row["ci_lower"].iloc[0]), 0.0)
                err_u = max(float(boot_row["ci_upper"].iloc[0]) - med, 0.0)
            else:
                reason = (
                    "missing barrier interval table"
                    if barrier_boot is None
                    else "missing barrier interval row"
                )
                log_bootstrap_fallback(
                    "temporal_null_barrier",
                    reason,
                    family=family,
                    key=key,
                )
                data = gaps.get(key, [])
                if len(data) > 0:
                    med, err_l, err_u = bootstrap_median_ci(data, n_boot=n_boot)
                else:
                    med, err_l, err_u = 0, 0, 0
            medians.append(med)
            errors[0].append(err_l)
            errors[1].append(err_u)
        ax.bar(labels, medians, yerr=errors, capsize=10, color=colors, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Median Hours to Adoption")
        for i, v in enumerate(medians):
            ax.text(i, v, f"{v:.1f}h", ha="center", va="bottom", fontsize=nhb_annotation_fontsize())

    # ── Helper: settler line chart ────────────────────────────────────
    def _plot_settler(ax, settler, title, family):
        if settler_boot is None:
            raise ValueError(
                "plot_temporal_null_comparison requires timeline bootstrap settler_intervals."
            )
        phase_keys = ["pre_jump", "the_jump", "post_jump"]
        phase_labels = ["Within old\nCluster", "Jump to\nnew Cluster", "Settle in\nnew Cluster"]
        medians, errors = [], [[], []]
        for key in phase_keys:
            boot_row = settler_boot[
                (settler_boot["family"] == family)
                & (settler_boot["key"] == key)
            ]
            if not boot_row.empty:
                med = float(boot_row["estimate"].iloc[0])
                err_l = max(med - float(boot_row["ci_lower"].iloc[0]), 0.0)
                err_u = max(float(boot_row["ci_upper"].iloc[0]) - med, 0.0)
            else:
                raise ValueError(
                    "Missing timeline bootstrap settler interval "
                    f"for family={family}, key={key}."
                )
            medians.append(med)
            errors[0].append(err_l)
            errors[1].append(err_u)
        x_pos = [0, 1, 2]
        marker_colors = ["#4A90E2", "#D0021B", "#4A90E2"]
        ax.plot(x_pos, medians, color="#333333", linewidth=3, zorder=2)
        ax.scatter(x_pos, medians, color=marker_colors, s=150, zorder=3)
        for x, median, err_l, err_u, color in zip(
            x_pos, medians, errors[0], errors[1], marker_colors
        ):
            ax.errorbar(
                x, median, yerr=[[err_l], [err_u]], fmt="none",
                ecolor=color, elinewidth=1.5, capsize=5, zorder=1,
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phase_labels)
        ax.set_ylabel("Median Hours to Adoption")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    _plot_barrier(axes[0, 0], semantic_gaps, "Semantic Clustering: Cognitive Barrier", "semantic")
    _plot_barrier(axes[0, 1], temporal_gaps, "Temporal Clustering: Cognitive Barrier", "temporal")
    _plot_settler(axes[1, 0], semantic_settler, "Semantic Clustering: Settler Effect", "semantic")
    _plot_settler(axes[1, 1], temporal_settler, "Temporal Clustering: Settler Effect", "temporal")

    # Print statistical tests for figure caption
    for label, gaps in [("Semantic", semantic_gaps), ("Temporal", temporal_gaps)]:
        try:
            result = test_barrier_significance(gaps)
            p = result["p_value"]
            n_w = len(gaps.get("within_cluster", []))
            n_b = len(gaps.get("between_clusters", []))
            print(f"{label} Barrier — Mann-Whitney U (one-sided): "
                  f"U={result['statistic']:.0f}, p={p:.2e}, "
                  f"n={n_w} within, {n_b} between")
        except (ValueError, TypeError):
            print(f"{label} Barrier — insufficient data for test")

    for label, settler in [("Semantic", semantic_settler), ("Temporal", temporal_settler)]:
        try:
            result = test_settler_significance(settler)
            fp = result["friedman"]["p_value"]
            n_seq = len(settler.get("pre_jump", []))
            pw = result["pairwise"]
            print(f"{label} Settler — Friedman: \u03c7\u00b2={result['friedman']['statistic']:.1f}, "
                  f"p={fp:.2e}, n={n_seq} triads")
            for name, res in pw.items():
                print(f"  {name}: p_raw={res['p_value_raw']:.2e}, "
                      f"p_corrected={res['p_value_corrected']:.2e}")
        except (ValueError, TypeError):
            print(f"{label} Settler — insufficient data for test")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Temporal null comparison saved to {save_path}")

    return fig


def plot_conspiracy_hr_progression(
    cox_results: Dict[str, Tuple[CoxTimeVaryingFitter, object]],
    models_to_include: Optional[List[str]] = None,
    reference_conspiracy: str = _BASELINE_CONSPIRACY,
    figsize: Tuple[float, float] = NHB_COL_DOUBLE,
    save_path: Optional[str] = None,
    bootstrap_intervals=None,
) -> plt.Figure:
    """Line plot showing each conspiracy's HR progression across sequential models.

    One connected line per conspiracy, models on x-axis, exp(coef) on y-axis.
    The reference conspiracy appears as a flat line at HR=1.0.

    Args:
        cox_results: Dict mapping model names to (ctv, df) tuples.
        models_to_include: Which models to include. Defaults to
            model_1, model_2a, model_3, model_4, model_5.
        reference_conspiracy: The omitted category (e.g. 'ConsProb_fakenews').
        figsize: Figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    if models_to_include is None:
        models_to_include = ["model_1", "model_2a", "model_3", "model_4", "model_5"]

    x_labels_map = {
        "model_1": "1st",
        "model_2a": "2nd",
        "model_3": "3rd",
        "model_4": "4th",
        "model_5": "5th",
    }

    ref_short = reference_conspiracy.replace("ConsProb_", "")
    boot_table = _bootstrap_table(bootstrap_intervals, "coefficient_intervals")

    # Collect data from each model
    rows = []
    for model_name in models_to_include:
        if model_name not in cox_results:
            continue
        ctv = cox_results[model_name][0]
        summary = ctv.summary
        fc_mask = summary.index.str.startswith("fc_")
        fc_rows = summary.loc[fc_mask]

        for idx_name in fc_rows.index:
            short_name = idx_name.replace("fc_ConsProb_", "")
            boot_row = pd.DataFrame()
            if boot_table is not None:
                boot_row = boot_table[
                    (boot_table["model"] == model_name)
                    & (boot_table["term"] == idx_name)
                ]
            if not boot_row.empty:
                hr = float(boot_row["hr"].iloc[0])
                ci_lower = float(boot_row["hr_ci_lower"].iloc[0])
                ci_upper = float(boot_row["hr_ci_upper"].iloc[0])
            else:
                reason = (
                    "missing coefficient interval table"
                    if boot_table is None
                    else "missing conspiracy interval row"
                )
                log_bootstrap_fallback(
                    "conspiracy_hr_progression",
                    reason,
                    model=model_name,
                    term=idx_name,
                    fallback_interval_source="model_summary",
                )
                hr = fc_rows.loc[idx_name, "exp(coef)"]
                ci_lower = fc_rows.loc[idx_name, "exp(coef) lower 95%"]
                ci_upper = fc_rows.loc[idx_name, "exp(coef) upper 95%"]
            rows.append({
                "conspiracy": short_name,
                "model": model_name,
                "hr": hr,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })

    if not rows:
        logger.warning("No fc_* covariates found in any model.")
        return plt.figure()

    # Add reference conspiracy as HR=1.0 for every model
    for model_name in models_to_include:
        if model_name in cox_results:
            rows.append({
                "conspiracy": ref_short,
                "model": model_name,
                "hr": 1.0,
                "ci_lower": 1.0,
                "ci_upper": 1.0,
            })
    df = pd.DataFrame(rows)

    # Map models to x positions
    model_order = [m for m in models_to_include if m in cox_results]
    model_to_x = {m: i for i, m in enumerate(model_order)}
    df["x"] = df["model"].map(model_to_x)

    # Unique conspiracies and color assignment
    conspiracies = sorted(df["conspiracy"].unique())
    n_consp = len(conspiracies)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_consp, 1)))
    consp_colors = {c: cmap[i] for i, c in enumerate(conspiracies)}

    # Jitter settings
    jitter_width = 0.03
    offsets = {c: (i - (n_consp - 1) / 2) * jitter_width
               for i, c in enumerate(conspiracies)}

    fig, ax = plt.subplots(figsize=figsize)

    for consp in conspiracies:
        sub = df[df["conspiracy"] == consp].sort_values("x")
        if sub.empty:
            continue

        label = _CONSPIRACY_LABELS.get(consp, consp)
        if consp == ref_short:
            label = f"{label} (ref)"
        color = consp_colors[consp]
        x_jittered = sub["x"].values + offsets[consp]

        yerr = np.vstack([
            (sub["hr"] - sub["ci_lower"]).values,
            (sub["ci_upper"] - sub["hr"]).values,
        ])
        yerr = np.clip(yerr, 0, None)

        ax.errorbar(
            x_jittered, sub["hr"].values, yerr=yerr,
            fmt="o-", color=color, markersize=6, linewidth=1.5,
            capsize=4, capthick=1.2, label=label,
        )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([x_labels_map.get(m, m) for m in model_order])
    ax.set_xlabel("Adoption Number")
    ax.set_ylabel("Hazard Ratio (exp(coef))")
    ref_display = _CONSPIRACY_LABELS.get(ref_short, ref_short)
    ax.set_title(
        f"Conspiracy HR Progression Across Models (ref = {ref_display})",
        fontsize=nhb_main_title_fontsize(),
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Conspiracy HR progression saved to {save_path}")

    return fig
