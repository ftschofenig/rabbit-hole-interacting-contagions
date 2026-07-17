"""Visualization functions for publication figures."""

from .style import (
    set_nhb_style,
    apply_nhb_spines,
    apply_panel_label,
    nhb_annotation_fontsize,
    NHB_COL_SINGLE,
    NHB_COL_ONE_HALF,
    NHB_COL_DOUBLE,
    CLUSTER_PALETTE,
    NHB_PALETTE,
)
from .figures import (
    plot_settler_effect,
    plot_settler_effect_triptych,
    plot_cognitive_barrier,
    plot_dendrogram,
    plot_exposure_hr_comparison,
    plot_temporal_hazard_shift,
    plot_cross_cluster_progression,
    plot_temporal_null_comparison,
)
from .plots import (
    plot_hazard_ratios,
    plot_baseline_hazard_fit,
    plot_silhouette_heatmap,
    plot_diffusion_dynamics,
    plot_first_adoption_comparison,
    plot_gateway_scatter,
    plot_fc_forest,
    plot_decay_times_line,
    plot_hawkes_dynamics,
    plot_hawkes_goodness_of_fit,
)
