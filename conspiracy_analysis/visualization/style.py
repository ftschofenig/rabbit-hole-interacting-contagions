"""Publication style for matplotlib figures.

Usage::

    from conspiracy_analysis.visualization.style import set_nhb_style
    set_nhb_style()

Call this once at the top of a notebook (typically in the first code cell,
after imports). It installs rcParams that match the target publication
guidelines:

- Helvetica / Arial sans-serif, 5-7 pt for body text.
- Top and right spines hidden; remaining spines 0.5 pt.
- Tick marks pointing outward, 0.5 pt width, 2.5 pt length.
- Near-black text (#222-#333) instead of pure black.
- 600 dpi output with TrueType font embedding for vector editing.
- Legends without frames; tight spacing.

Figure-size constants (`NHB_COL_SINGLE`, `NHB_COL_ONE_HALF`, `NHB_COL_DOUBLE`)
convert from NHB's mandated column widths (89 mm, 120 mm, 183 mm) to
matplotlib's inch-based API. Use them as `figsize=NHB_COL_DOUBLE` instead of
arbitrary tuples.

Panel labels (the bold "a", "b", "c" markers on multi-panel figures) are the
ONLY bold text permitted. Use `apply_panel_label(ax, "a")` on each subplot.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Millimeter to inch conversion
_MM = 1.0 / 25.4

# Scale figure dimensions and text for readable previews.
# Typesetting at column width restores the intended print dimensions.
_NHB_SCALE = 1.8

# Figure dimensions use configured print widths and the preview scale.
NHB_COL_SINGLE: Tuple[float, float] = (89 * _MM * _NHB_SCALE, 75 * _MM * _NHB_SCALE)
NHB_COL_ONE_HALF: Tuple[float, float] = (120 * _MM * _NHB_SCALE, 90 * _MM * _NHB_SCALE)
NHB_COL_DOUBLE: Tuple[float, float] = (183 * _MM * _NHB_SCALE, 110 * _MM * _NHB_SCALE)

# Slightly softened black tones
_NEARLY_BLACK_TEXT = "#222222"
_NEARLY_BLACK_AXIS = "#333333"

# Annotation size scales with the preview factor.
_NHB_ANNOTATION_PT_PRINT = 5  # Minimum annotation size at print scale.

# Significance symbols remain legible against bracket lines.
_NHB_SIGNIFICANCE_PT_PRINT = 7

# Main titles use one size across axes and figure headings.
_NHB_MAIN_TITLE_PT_PRINT = 8

# Shared semantic cluster colors
CLUSTER_PALETTE: Dict[str, str] = {
    "Political": "#E24A33",
    "Media": "#348ABD",
    "Pandemic": "#988ED5",
    "Biomedical": "#8EBA42",
}

# Palette for inline matplotlib cells that still hardcode GREEN or RED.
NHB_PALETTE: Dict[str, str] = {
    "positive": "#348ABD",   # blue (cool = increase relative to baseline)
    "negative": "#E24A33",   # red (warm = decrease)
    "neutral": "#777777",    # grey
    "highlight": "#FFB000",  # amber accent
    "model_1": "#1f77b4",
    "model_pooled": "#8c564b",
}


def _check_helvetica_available() -> None:
    """Warn once if Helvetica is unavailable; Arial/Liberation Sans fallback."""
    try:
        # `findfont` with `fallback_to_default=False` raises ValueError if
        # the requested font is truly missing. Otherwise it returns the path
        # of the best match (possibly a fallback).
        path = fm.findfont("Helvetica", fallback_to_default=False)
        # Even if findfont returns a path, matplotlib may have silently
        # substituted a different family. Check whether the resolved font
        # advertises 'Helvetica' in its family name.
        prop = fm.FontProperties(fname=path)
        family_name = prop.get_name().lower()
        if "helvetica" not in family_name:
            logger.info(
                "NHB style: Helvetica not installed on this system; "
                "falling back through sans-serif chain (Arial, Liberation "
                "Sans, DejaVu Sans). For publication-ready output, install "
                "Helvetica or Arial. Resolved: %s", prop.get_name(),
            )
    except ValueError:
        logger.warning(
            "NHB style: Helvetica font not found. Falling back to Arial, "
            "Liberation Sans, or DejaVu Sans. On Linux, "
            "'apt-get install fonts-liberation' installs a near-equivalent "
            "metric-compatible Arial substitute."
        )


def set_nhb_style(context: str = "paper") -> None:
    """Apply the shared publication style to matplotlib rcParams.

    Idempotent — safe to call multiple times. Modifies `plt.rcParams` in
    place. Call once per notebook (or per script) before any figures are
    drawn.

    Args:
        context: Reserved for future extension (e.g. ``"poster"`` with
            larger fonts). Currently only ``"paper"`` is implemented; any
            other value prints a warning and applies paper defaults.
    """
    if context != "paper":
        logger.warning(
            "NHB style: context=%r not implemented; using 'paper'.", context
        )

    _check_helvetica_available()

    # Scale all size settings for preview and typeset output.
    S = _NHB_SCALE

    rc: Dict[str, object] = {
        # Use regular sans serif text.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans",
        ],
        "font.weight": "regular",
        "axes.titleweight": "regular",

        # Scale font sizes by the preview factor.
        "font.size": 7 * S,
        "axes.titlesize": 7 * S,
        "axes.labelsize": 7 * S,
        "xtick.labelsize": 6 * S,
        "ytick.labelsize": 6 * S,
        "legend.fontsize": 6 * S,
        "legend.title_fontsize": 6 * S,
        "figure.labelsize": 7 * S,
        "figure.titlesize": 8 * S,

        # Math / Greek letters rendered in sans-serif (matches Helvetica body).
        "mathtext.fontset": "stixsans",

        # Near-black text colors.
        "text.color": _NEARLY_BLACK_TEXT,
        "axes.labelcolor": _NEARLY_BLACK_AXIS,
        "xtick.color": _NEARLY_BLACK_AXIS,
        "ytick.color": _NEARLY_BLACK_AXIS,

        # Hide upper spines and scale the remaining spine width.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5 * S,
        "axes.edgecolor": _NEARLY_BLACK_AXIS,

        # Ticks: thin, outward, short. All sizes scale with S.
        "xtick.major.width": 0.5 * S,
        "ytick.major.width": 0.5 * S,
        "xtick.minor.width": 0.4 * S,
        "ytick.minor.width": 0.4 * S,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5 * S,
        "ytick.major.size": 2.5 * S,
        "xtick.minor.size": 1.5 * S,
        "ytick.minor.size": 1.5 * S,
        "xtick.major.pad": 2 * S,
        "ytick.major.pad": 2 * S,

        # Lines and markers (scaled; at S=1.8: data lines 1.8pt, markers 7.2pt).
        "lines.linewidth": 1.0 * S,
        "lines.markersize": 4 * S,
        "lines.markeredgewidth": 0.5 * S,

        # Grid disabled by default; when re-enabled locally, use subtle dots.
        "axes.grid": False,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.4 * S,
        "grid.linestyle": ":",

        # Legend: no frame, tight spacing.
        "legend.frameon": False,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,

        # Use high resolution raster output.
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "savefig.format": "png",

        # Embed TrueType fonts in vector output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    mpl.rcParams.update(rc)
    logger.info("Applied NHB style (context=%s).", context)


def apply_nhb_spines(ax: "plt.Axes") -> None:
    """Hide top and right spines, thin remaining ones to 0.5 pt.

    Call after constructing a subplot if you bypassed `set_nhb_style` or
    need to enforce the spine treatment on a pre-existing figure.

    Args:
        ax: A matplotlib Axes object.
    """
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        if side in ax.spines:
            ax.spines[side].set_linewidth(0.5)
            ax.spines[side].set_color(_NEARLY_BLACK_AXIS)


def nhb_annotation_fontsize() -> float:
    """Return the current NHB-scaled in-plot annotation fontsize in points.

    Data labels that live INSIDE the plotting area (bar values, point
    labels, etc.) should use this size so they scale consistently with the
    rest of the figure when `_NHB_SCALE` changes. At strict NHB size
    (_NHB_SCALE = 1.0) this returns 5; at the default preview scale
    (_NHB_SCALE = 1.8) it returns 9.
    """
    return _NHB_ANNOTATION_PT_PRINT * _NHB_SCALE


def nhb_significance_fontsize() -> float:
    """Return the NHB-scaled significance-bracket fontsize in points.

    Slightly larger than `nhb_annotation_fontsize` so pairwise significance
    stars (* / ** / *** / n.s.) remain legible against bracket lines. At
    strict NHB size (_NHB_SCALE = 1.0) returns 7; at the default preview
    scale (_NHB_SCALE = 1.8) it returns 12.6.
    """
    return _NHB_SIGNIFICANCE_PT_PRINT * _NHB_SCALE


def nhb_main_title_fontsize() -> float:
    """Return the NHB-scaled main-title fontsize in points.

    Used for single-panel `ax.set_title` and `fig.suptitle` calls so every
    plot's headline matches. At strict NHB size returns 8; at the default
    preview scale (_NHB_SCALE = 1.8) it returns 14.4.
    """
    return _NHB_MAIN_TITLE_PT_PRINT * _NHB_SCALE


def apply_panel_label(
    ax: "plt.Axes",
    label: str,
    loc: str = "upper left",
    fontsize: float = None,
    dx: float = -0.08,
    dy: float = 0.02,
) -> None:
    """Add a bold panel label (``"a"``, ``"b"``, ``"c"``) to a subplot.

    Panel labels are the ONLY bold text permitted in NHB figures. The label
    is placed just outside the top-left axes corner by default (so it does
    not overlap tick labels).

    Args:
        ax: Matplotlib Axes object to annotate.
        label: Single character or short string (e.g. ``"a"``).
        loc: Currently only ``"upper left"`` is supported; any other value
            falls back to upper-left with a warning.
        fontsize: Point size for the label. Defaults to 8pt print * NHB_SCALE
            (i.e. 14.4pt on screen at default scale, 8pt after typesetting).
        dx: Horizontal offset in axes coordinates (negative = left of the
            axes box).
        dy: Vertical offset in axes coordinates (positive = above the axes
            box).
    """
    if loc != "upper left":
        logger.warning(
            "apply_panel_label: loc=%r not supported; using 'upper left'.", loc
        )
    if fontsize is None:
        fontsize = 8 * _NHB_SCALE
    ax.text(
        dx, 1.0 + dy, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom", ha="left",
        color=_NEARLY_BLACK_TEXT,
    )
