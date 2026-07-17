"""
Parallel simulation runner.

Manages multiple replications of the ABM with deterministic seeding,
collects results into structured containers for downstream evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

from conspiracy_analysis.simulation.agent_state import AgentState
from conspiracy_analysis.simulation.config import SimulationConfig, ScenarioConfig
from conspiracy_analysis.simulation.engine import (
    initialize_agents,
    seed_nodes,
    step,
    apply_adoptions,
    apply_immunities,
    apply_shares,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a single ABM run.

    Attributes:
        run_id: Unique run identifier (also used as random seed).
        scenario_name: Name of the scenario that was run.
        adoption_histories: node_id -> conspiracy -> adoption_time.
            Only includes adopted conspiracies.
        n_nodes: Total number of nodes in the simulation.
    """

    run_id: int
    scenario_name: str
    adoption_histories: Dict[str, Dict[str, float]]
    n_nodes: int
    steps_completed: int = 0
    sharing_histories: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    visible_sharing_histories: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


@dataclass
class ScenarioResults:
    """Aggregated results for one scenario across multiple runs.

    Attributes:
        scenario: The scenario configuration used.
        runs: List of individual simulation results.
    """

    scenario: ScenarioConfig
    runs: List[SimulationResult] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def mean_steps_completed(self) -> int:
        """Mean number of steps completed across runs (rounded to int)."""
        if not self.runs:
            return 0
        return int(np.mean([r.steps_completed for r in self.runs]))


def _extract_adoption_histories(
    agents: Dict[str, AgentState],
) -> Dict[str, Dict[str, float]]:
    """Extract adoption histories from agent states.

    Returns:
        Dict mapping node_id -> {conspiracy: adoption_time} for each
        conspiracy the node adopted.
    """
    histories: Dict[str, Dict[str, float]] = {}
    for node_id, agent in agents.items():
        if agent.num_adopted > 0:
            node_hist = {}
            for time, conspiracy in agent._adoption_timeline:
                node_hist[conspiracy] = time
            histories[node_id] = node_hist
    return histories


def _extract_sharing_histories(
    agents: Dict[str, AgentState],
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """Extract sharing histories from agent states.

    Returns:
        Tuple of (all_shares, visible_shares):
            - all_shares: node_id -> {conspiracy: [share_times]} (includes blocked)
            - visible_shares: node_id -> {conspiracy: [share_times]} (visible only)
    """
    all_shares: Dict[str, Dict[str, List[float]]] = {}
    visible_shares: Dict[str, Dict[str, List[float]]] = {}
    for node_id, agent in agents.items():
        if agent._sharing_history:
            all_shares[node_id] = dict(agent._sharing_history)
        if agent._visible_sharing_history:
            visible_shares[node_id] = dict(agent._visible_sharing_history)
    return all_shares, visible_shares


def _worker_process(
    run_id: int,
    G: nx.Graph,
    sim_config: SimulationConfig,
    scenario: ScenarioConfig,
) -> SimulationResult:
    """Run a single ABM simulation (called by joblib).

    Args:
        run_id: Used as random seed for determinism.
        G: Network graph.
        sim_config: Simulation configuration.
        scenario: Scenario configuration.

    Returns:
        SimulationResult with adoption histories.
    """
    rng = np.random.default_rng(run_id)

    agents = initialize_agents(G, sim_config.conspiracies)
    n_nodes = len(agents)
    seeded: set = set()
    steps_completed = 0

    target = sim_config.target_total_adoptions

    for t in range(sim_config.steps):
        if t % 500 == 0:
            current_total = sum(a.num_adopted for a in agents.values())
            if target is not None:
                pct = current_total / target * 100
                print(
                    f"[{scenario.name}] Run {run_id}: step {t}/{sim_config.steps} | "
                    f"adoptions: {current_total}/{target} ({pct:.1f}%)",
                    flush=True,
                )
            else:
                print(
                    f"[{scenario.name}] Run {run_id}: step {t}/{sim_config.steps} | "
                    f"adoptions: {current_total}",
                    flush=True,
                )

        # Seed nodes at entry times
        for conspiracy in sim_config.conspiracies:
            seed_nodes(
                agents, conspiracy, sim_config.entry_times, t,
                sim_config.seed_fraction, sim_config.cluster_assignments,
                scenario, seeded, rng,
            )

        # Evaluate and apply synchronous updates
        new_adoptions, new_shares, new_immunities = step(t, G, agents, sim_config, scenario, rng=rng)
        apply_adoptions(
            new_adoptions, agents, t, sim_config.cluster_assignments, scenario, rng
        )
        apply_immunities(new_immunities, agents)
        apply_shares(new_shares, agents, t, scenario, rng)

        steps_completed = t + 1

        # Early stop if target total adoptions reached
        if sim_config.target_total_adoptions is not None:
            current_total = sum(a.num_adopted for a in agents.values())
            if current_total >= sim_config.target_total_adoptions:
                break

    histories = _extract_adoption_histories(agents)
    all_shares, visible_shares = _extract_sharing_histories(agents)

    return SimulationResult(
        run_id=run_id,
        scenario_name=scenario.name,
        adoption_histories=histories,
        n_nodes=n_nodes,
        steps_completed=steps_completed,
        sharing_histories=all_shares,
        visible_sharing_histories=visible_shares,
    )


def run_scenario(
    G: nx.Graph,
    sim_config: SimulationConfig,
    scenario: ScenarioConfig,
    runs: int = 6,
    n_jobs: int = -1,
) -> ScenarioResults:
    """Run multiple replications of a scenario in parallel.

    Args:
        G: Network graph.
        sim_config: Simulation configuration.
        scenario: Scenario configuration.
        runs: Number of independent replications.
        n_jobs: Number of parallel workers (-1 = all cores).

    Returns:
        ScenarioResults with all runs aggregated.
    """
    logger.info(
        f"Starting scenario '{scenario.name}': {runs} runs, "
        f"{sim_config.steps} steps, {len(G.nodes())} nodes"
    )

    results = ScenarioResults(scenario=scenario)

    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        tasks = (
            delayed(_worker_process)(
                run_id=i,
                G=G,
                sim_config=sim_config,
                scenario=scenario,
            )
            for i in range(1, runs + 1)
        )

        for result in tqdm(
            parallel(tasks), total=runs,
            desc=f"Simulating ({scenario.name})",
        ):
            results.runs.append(result)

    results.runs.sort(key=lambda r: r.run_id)
    return results


def run_comparison(
    G: nx.Graph,
    sim_config: SimulationConfig,
    scenarios: List[ScenarioConfig],
    runs: int = 6,
    n_jobs: int = -1,
) -> Dict[str, ScenarioResults]:
    """Run multiple scenarios and collect results for comparison.

    Args:
        G: Network graph.
        sim_config: Simulation configuration.
        scenarios: List of scenario configurations.
        runs: Number of replications per scenario.
        n_jobs: Number of parallel workers.

    Returns:
        Dict mapping scenario name to ScenarioResults.
    """
    all_results: Dict[str, ScenarioResults] = {}
    for scenario in scenarios:
        all_results[scenario.name] = run_scenario(
            G, sim_config, scenario, runs=runs, n_jobs=n_jobs
        )
    return all_results
