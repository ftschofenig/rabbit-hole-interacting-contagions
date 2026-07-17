"""Statistical models for conspiracy adoption and sharing dynamics."""

from .cox_adoption import fit_cox_model, fit_all_cox_models, BASELINE_CONSPIRACY
from .baseline_hazards import (
    extract_cumulative_baseline_hazard,
    fit_linear_baseline,
    fit_weibull_baseline,
    parametrize_all_baselines,
    calculate_baseline_hazard,
    create_hazard_calculator,
)
from .hawkes_sharing import (
    HawkesFitResult,
    extract_sharing_sequences,
    fit_hawkes,
    fit_hawkes_with_ll,
    hawkes_intensity,
    hawkes_residual_acf,
    hawkes_residual_acf_envelope,
    hawkes_sharing_probability,
)
from .gateway_effects import (
    extract_gateway_coefficients,
    identify_gateway_conspiracies,
    identify_gateway_2d,
)
from .bootstrap_inference import (
    ModelBootstrapSpec,
    load_bootstrap_artifact,
    run_cox_user_bootstrap,
    run_timeline_bootstrap,
    summarize_cox_bootstrap,
)
