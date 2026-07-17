"""
Panel plotters for the formal (toy) model results, used to build the
3-column x 5-row composite figure in the main-text figures notebook.

Each plotter takes a ``results`` dict (produced by the save-to-pkl cell at
the end of each formal-model notebook) and a matplotlib ``Axes``. Shared
colour palettes and per-panel logic are lifted verbatim from the
the formal model notebooks so the composite stays visually
identical to the per-model renders.
"""

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


_CLUSTER_COLORS = {0: "#4C72B0", 1: "#DD8452"}
_USER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
_DECAY_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
_SETTLER_COLORS = ["#4C72B0", "#999999", "#DD8452"]


def panel_a_belief_space(
    ax: plt.Axes,
    r: Dict[str, Any],
    *,
    show_legend: bool = True,
    no_reshaping_note: bool = False,
    max_users: int = 5,
) -> None:
    """Belief-space trajectories (first ``max_users`` users)."""
    names = r["names"]
    clusters = r["clusters"]
    signatures = r["signatures"]
    baseline = r["baseline"]
    records = r["records"]
    adoption_events = r["adoption_events"]

    for name in names:
        col = _CLUSTER_COLORS[clusters[name]]
        ax.scatter(*signatures[name], s=60, c=col,
                   edgecolors="black", linewidth=0.6, zorder=5)
        ax.annotate(name, signatures[name],
                    textcoords="offset points", xytext=(5, 4), color=col)

    ax.scatter(*baseline, s=55, c="black", marker="*", zorder=5)
    ax.annotate(r"baseline $\theta_0$", baseline,
                textcoords="offset points", xytext=(5, -10))

    user_event_idx = 0
    for u in range(min(max_users, len(records))):
        n_adopted = len(records[u])
        if n_adopted == 0:
            continue
        user_events = adoption_events[user_event_idx:user_event_idx + n_adopted]
        user_event_idx += n_adopted
        points = [np.asarray(baseline).copy()]
        for evt in user_events:
            points.append(np.asarray(evt["theta_post"]).copy())
        order = " -> ".join([rec[0] for rec in records[u]])
        ucol = _USER_COLORS[u % len(_USER_COLORS)]
        for i in range(len(points) - 1):
            ax.annotate("", xy=points[i + 1], xytext=points[i],
                        arrowprops=dict(arrowstyle="->", color=ucol,
                                        lw=1.0, alpha=0.55))
        ax.plot([], [], color=ucol, lw=1.2, alpha=0.8,
                label=f"User {u + 1}: {order}")

    ax.set_xlabel("Evaluative dim. 1")
    ax.set_ylabel("Evaluative dim. 2")
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.9, borderaxespad=0.15)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-0.35, 1.5)

    if no_reshaping_note:
        ax.text(
            0.5, 0.5,
            "No reshaping ($\\alpha=0$):\nnarrative positions are fixed",
            transform=ax.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888", alpha=0.9),
        )


def panel_b_hazard_decay(ax: plt.Axes, r: Dict[str, Any]) -> None:
    """Hazard-ratio decay curves by adoption depth."""
    decay_curves = r["decay_curves"]
    decay_times = r["decay_times"]
    events_by_depth = r["events_by_depth"]

    for depth in sorted(decay_curves.keys()):
        curve = decay_curves[depth]
        n_events = len(events_by_depth[depth])
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(depth, "th")
        label = f"After {depth}{suffix} adoption (n={n_events})"
        ax.plot(decay_times, curve,
                color=_DECAY_COLORS[(depth - 1) % len(_DECAY_COLORS)],
                linewidth=1.5, label=label)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8,
               label="Never-adopted baseline")
    ax.set_xlabel("Hours since adoption")
    ax.set_ylabel("Mean hazard ratio")
    ax.set_xlim(0, 300)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def panel_c_interaction_windows(ax: plt.Axes, r: Dict[str, Any]) -> None:
    """Expanding interaction windows: adoption depth vs. window duration."""
    windows = r["windows"]
    depths = sorted(windows.keys())
    vals = [windows[d] for d in depths]

    ax.bar(depths, vals,
           color=[_DECAY_COLORS[(d - 1) % len(_DECAY_COLORS)] for d in depths],
           edgecolor="black", linewidth=0.6, width=0.6)
    for d, v in zip(depths, vals):
        ax.text(d, v + max(vals) * 0.02, f"{v:.0f}h", ha="center")

    suffix_map = {1: "st", 2: "nd", 3: "rd"}
    ax.set_xlabel("Adoption number")
    ax.set_ylabel("Time to baseline (hours)")
    ax.set_xticks(depths)
    ax.set_xticklabels([f"{d}{suffix_map.get(d, 'th')}" for d in depths])
    ax.grid(True, alpha=0.3, axis="y")


def panel_d_tradeoff(
    ax: plt.Axes,
    r: Dict[str, Any],
    *,
    ymin: float = None,
) -> None:
    """Contagiousness-potency tradeoff scatter."""
    names = r["names"]
    clusters = r["clusters"]
    first_counts = r["first_counts"]
    potency = r["potency"]
    ref = r["ref"]
    ref_cong = r["ref_cong"]
    ref_pot = r["ref_pot"]

    for name in names:
        rel_c = first_counts.get(name, 0) / ref_cong if ref_cong else 0
        rel_p = potency[name] / ref_pot if ref_pot else 0
        col = _CLUSTER_COLORS[clusters[name]]
        ax.scatter(rel_c, rel_p, s=60, c=col,
                   edgecolors="black", linewidth=0.6, zorder=5)
        ax.annotate(name, (rel_c, rel_p),
                    textcoords="offset points", xytext=(5, 4), color=col)

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.annotate(f"reference ({ref})", (1.0, 1.0),
                textcoords="offset points", xytext=(-50, -12),
                color="grey", style="italic")

    ax.set_xlabel(f"Contagiousness (rel. to {ref})")
    ax.set_ylabel(f"Potency (rel. to {ref})")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.grid(True, alpha=0.3)


def panel_e_settler(ax: plt.Axes, r: Dict[str, Any]) -> None:
    """Settler dynamics: median transition times across three phases."""
    pre_jump = r["pre_jump_times"]
    the_jump = r["the_jump_times"]
    post_jump = r["post_jump_times"]

    categories = ["Within old\ncluster", "Jump to\nnew cluster",
                  "Settling in\nnew cluster"]
    medians = [np.median(pre_jump), np.median(the_jump), np.median(post_jump)]
    counts = [len(pre_jump), len(the_jump), len(post_jump)]

    bars = ax.bar(range(3), medians, color=_SETTLER_COLORS,
                  edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Median transition time (hours)")
    ax.set_ylim(0, max(medians) * 1.4 if max(medians) > 0 else 1)
    for i, (bar, n) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(medians) * 0.02,
                f"{medians[i]:.0f}h\n(n={n})",
                ha="center", color="#555")
    ax.grid(True, alpha=0.3, axis="y")


PANELS = [
    ("a", "Belief space trajectories", panel_a_belief_space),
    ("b", "Hazard ratio decay", panel_b_hazard_decay),
    ("c", "Interaction windows", panel_c_interaction_windows),
    ("d", "Contagiousness-potency tradeoff", panel_d_tradeoff),
    ("e", "Settler dynamics", panel_e_settler),
]
