"""Agent-based model simulations combining Cox and Hawkes processes."""

from .counterfactuals import (
    compute_entry_times,
)

# Agent based simulation configuration
from .config import (
    BaselineParams,
    CoxModelSpec,
    SimulationConfig,
    ScenarioConfig,
    build_simulation_config_from_fitted_models,
    override_baselines_to_linear,
)
from .agent_state import AgentState
from .hazard_dispatch import (
    compute_adoption_hazard,
    compute_interval_hazard,
    compute_raw_hazard,
)
from .exposure import compute_neighbor_exposure
from .engine import (
    initialize_agents,
    seed_nodes,
    step,
    apply_adoptions,
    apply_immunities,
    apply_shares,
    is_quarantined,
)
from .scenarios import baseline, no_temporal_effects
from .runner import (
    SimulationResult,
    ScenarioResults,
    run_scenario,
    run_comparison,
)
from .evaluation import (
    compute_diffusion_curves,
    compute_first_adoption_distribution,
    compare_scenarios,
    compute_empirical_comparison,
)
from .provenance import (
    atomic_joblib_dump,
    atomic_json_dump,
    collect_package_versions,
    load_manifest_backed_simulation_bundle,
    sha256_file,
)
