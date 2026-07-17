"""
Core simulation engine with synchronous updates.

Implements the main ABM loop: for each time step, evaluate all
node-conspiracy pairs for adoption (Cox hazards) and re-sharing
(Hawkes self-exciting process), applying updates synchronously.
"""

from __future__ import annotations

import math
import random
import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import networkx as nx

from conspiracy_analysis.simulation.agent_state import AgentState
from conspiracy_analysis.simulation.config import SimulationConfig, ScenarioConfig
from conspiracy_analysis.simulation.exposure import compute_neighbor_exposure
from conspiracy_analysis.simulation.hazard_dispatch import compute_interval_hazard

logger = logging.getLogger(__name__)


def initialize_agents(
    G: nx.Graph,
    conspiracies: List[str],
) -> Dict[str, AgentState]:
    """Create fresh AgentState for every node in the graph.

    Args:
        G: Human simulation graph with optional `first_active_time` node
            attributes.
        conspiracies: List of conspiracy column names.

    Returns:
        Dict mapping node id to AgentState.
    """
    agents: Dict[str, AgentState] = {}
    for node in G.nodes():
        fat = G.nodes[node].get("first_active_time", 0)
        log_deg = float(np.log1p(G.degree(node)))
        agents[node] = AgentState(log_degree=log_deg, first_active_time=fat)
    return agents


def is_quarantined(
    current_time: float,
    last_organic_adoption_time: float | None,
    quarantine_hours: float,
) -> bool:
    """Return whether the half open quarantine interval is active."""
    if quarantine_hours <= 0 or last_organic_adoption_time is None:
        return False
    return current_time < last_organic_adoption_time + quarantine_hours


def seed_nodes(
    agents: Dict[str, AgentState],
    conspiracy: str,
    entry_times: Dict[str, float],
    t: int,
    fraction: float,
    cluster_assignments: Dict[str, int],
    scenario: ScenarioConfig,
    seeded: Set[str] = None,
    rng: np.random.Generator = None,
) -> None:
    """Randomly activate a fraction of eligible nodes for a conspiracy.

    Seeds once at the first time step >= the conspiracy's entry time.

    Args:
        agents: Agent states (modified in-place).
        conspiracy: Conspiracy to seed.
        entry_times: Dict mapping conspiracy -> first appearance time.
        t: Current simulation time step.
        fraction: Fraction of eligible nodes to seed.
        cluster_assignments: For recording cluster visits on adoption.
        scenario: Current scenario config.
        seeded: Set tracking which conspiracies have already been seeded.
        rng: Per run random generator.
    """
    if seeded is not None and conspiracy in seeded:
        return
    if t < entry_times.get(conspiracy, float("inf")):
        return
    if seeded is not None:
        seeded.add(conspiracy)

    eligible = [
        node_id for node_id, state in agents.items()
        if state.first_active_time <= t and not state.has_adopted(conspiracy)
    ]
    if not eligible:
        return

    num_to_activate = int(len(eligible) * fraction)
    sample_size = min(num_to_activate, len(eligible))
    if rng is None:
        selected = random.sample(eligible, sample_size)
    else:
        selected = rng.choice(eligible, size=sample_size, replace=False).tolist()
    for node_id in selected:
        # Seeds are exempt from ALL counterfactual effects. The reputation
        # nudge, content moderation, and quarantine interventions only
        # apply to subsequent organic adoptions and Hawkes re-shares —
        # never to the act of seeding itself.
        agents[node_id].record_adoption(
            conspiracy, t, cluster_assignments, is_organic=False
        )


def step(
    t: int,
    G: nx.Graph,
    agents: Dict[str, AgentState],
    sim_config: SimulationConfig,
    scenario: ScenarioConfig,
    rng: np.random.Generator = None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]], Dict[str, Set[str]]]:
    """Execute one synchronous time step: adoptions (Cox) + sharing (Hawkes).

    Phase A — Adoption (competing risks):
    For each node, collect raw hazards h_c for all unadopted conspiracies c,
    compute total hazard H = sum(h_c), draw adoption with P = 1 - exp(-H),
    and if adopted, pick which conspiracy proportional to h_c / H.

    Phase B — Hawkes re-sharing:
    For each node and each adopted conspiracy, compute Hawkes intensity
    from sharing history and draw a Poisson count of sharing events.

    All decisions at time t are based on state at t,
    applied simultaneously after all evaluations.

    Args:
        t: Current time step.
        G: Network graph.
        agents: Current agent states.
        sim_config: Simulation configuration.
        scenario: Scenario configuration.

    Returns:
        Tuple of (new_adoptions, new_shares, new_immunities):
            - new_adoptions: node_id -> set of conspiracy names
            - new_shares: node_id -> {conspiracy: share_count}
            - new_immunities: node_id -> set of conspiracy names (rejected via nudge)
    """
    new_adoptions: Dict[str, Set[str]] = {}
    new_immunities: Dict[str, Set[str]] = {}
    if rng is None:
        rng = np.random.default_rng()

    # --- Phase A: Cox adoption (competing risks) ---
    for node_id, agent in agents.items():
        if t < agent.first_active_time:
            continue

        # Quarantine: skip node if it recently *organically* adopted.
        # Seed adoptions do not trigger the quarantine clock — only
        # adoptions made through this Phase A loop set the lock window.
        if scenario.quarantine_hours > 0:
            last_t = agent.get_last_organic_adoption_time()
            if is_quarantined(t, last_t, scenario.quarantine_hours):
                continue

        hazards: Dict[str, float] = {}

        for conspiracy in sim_config.conspiracies:
            if t < sim_config.entry_times.get(conspiracy, float("inf")):
                continue

            if agent.has_adopted(conspiracy):
                continue
            if agent.is_immune(conspiracy):
                continue

            # Compute neighbor exposure (time-windowed, sharing-based)
            s7 = compute_neighbor_exposure(
                node_id, conspiracy, agents, G, scenario,
                t=t, exposure_window=sim_config.exposure_window,
            )

            if sim_config.require_peer_exposure and s7 == 0:
                continue

            # Compute integrated hazard over the next hour, not an
            # instantaneous rate evaluated only at the interval start.
            h = compute_interval_hazard(
                agent, conspiracy, t, s7, sim_config, scenario
            )

            if h > 0:
                hazards[conspiracy] = h

        # Competing risks draw (at most one adoption)
        if hazards:
            total_hazard = sum(hazards.values())
            prob_any = 1.0 - math.exp(-total_hazard)

            if rng.random() < prob_any:
                conspiracies_list = list(hazards.keys())
                weights = np.asarray([hazards[c] for c in conspiracies_list])
                chosen = rng.choice(conspiracies_list, p=weights / weights.sum())
                if scenario.nudge_rejection_rate > 0 and rng.random() < scenario.nudge_rejection_rate:
                    new_immunities.setdefault(node_id, set()).add(chosen)
                else:
                    new_adoptions[node_id] = {chosen}

    # --- Phase B: Hawkes re-sharing ---
    new_shares: Dict[str, Dict[str, int]] = {}

    if sim_config.hawkes_params is not None:
        mu, alpha, beta = sim_config.hawkes_params
        cutoff = 21.0 / beta if beta > 0 else float("inf")
        kernel_integral = alpha * (1.0 - math.exp(-beta)) if beta > 0 else alpha * beta

        for node_id, agent in agents.items():
            if t < agent.first_active_time:
                continue

            # Quarantine: read-only lock — user cannot generate new shares
            # while locked. Uses the organic-only timestamp so seed
            # adoptions do not silence subsequent Hawkes re-sharing.
            if scenario.quarantine_hours > 0:
                last_t = agent.get_last_organic_adoption_time()
                if is_quarantined(t, last_t, scenario.quarantine_hours):
                    continue

            for conspiracy in sim_config.conspiracies:
                if not agent.has_adopted(conspiracy):
                    continue

                history = agent.get_sharing_history(conspiracy)
                if not history:
                    continue

                # Skip if adopted this exact step (initial share already recorded)
                if history[-1] == t:
                    continue

                # Compute Hawkes intensity inline for performance
                sum_excitation = 0.0
                for t_event in reversed(history):
                    dt = t - t_event
                    if dt > cutoff:
                        break
                    if dt > 0:
                        sum_excitation += math.exp(-beta * dt)

                intensity = mu + kernel_integral * sum_excitation

                # Repeat post counts use the integrated hourly Hawkes intensity
                # as the mean.
                n_shares = min(int(rng.poisson(intensity)), 50)
                if n_shares > 0:
                    if node_id not in new_shares:
                        new_shares[node_id] = {}
                    new_shares[node_id][conspiracy] = n_shares

    return new_adoptions, new_shares, new_immunities


def apply_adoptions(
    new_adoptions: Dict[str, Set[str]],
    agents: Dict[str, AgentState],
    t: int,
    cluster_assignments: Dict[str, int],
    scenario: ScenarioConfig = None,
    rng: np.random.Generator = None,
) -> int:
    """Apply collected adoptions synchronously.

    If scenario.sharing_block_rate > 0, the initial adoption tweet is subject
    to the same content moderation as re-shares: it may be hidden from
    neighbors while still contributing to the poster's Hawkes self-excitation.

    Args:
        new_adoptions: Output from step().
        agents: Agent states to update.
        t: Current time step.
        cluster_assignments: For recording cluster visits.
        scenario: Scenario config (for sharing_block_rate).
        rng: Per run random generator.

    Returns:
        Total number of new adoptions applied.
    """
    block_rate = scenario.sharing_block_rate if scenario else 0.0
    total = 0
    for node_id, conspiracies in new_adoptions.items():
        for conspiracy in conspiracies:
            agents[node_id].record_adoption(conspiracy, t, cluster_assignments)
            if block_rate > 0:
                draw = rng.random() if rng is not None else random.random()
                if draw < block_rate:
                    agent = agents[node_id]
                    vis_hist = agent._visible_sharing_history.get(conspiracy)
                    if vis_hist and vis_hist[-1] == t:
                        vis_hist.pop()
                        if not vis_hist:
                            del agent._visible_sharing_history[conspiracy]
            total += 1
    return total


def apply_immunities(
    new_immunities: Dict[str, Set[str]],
    agents: Dict[str, AgentState],
) -> int:
    """Apply collected immunities synchronously (from reputation nudge).

    Args:
        new_immunities: Output from step(). Maps node_id to set of
            conspiracy names the node rejected.
        agents: Agent states to update.

    Returns:
        Total number of immunities applied.
    """
    total = 0
    for node_id, conspiracies in new_immunities.items():
        for conspiracy in conspiracies:
            agents[node_id].record_immunity(conspiracy)
            total += 1
    return total


def apply_shares(
    new_shares: Dict[str, Dict[str, int]],
    agents: Dict[str, AgentState],
    t: int,
    scenario: ScenarioConfig = None,
    rng: np.random.Generator = None,
) -> int:
    """Apply collected sharing events synchronously.

    If scenario.sharing_block_rate > 0, each share has that probability of
    being blocked from the network (invisible to neighbors for s7 exposure)
    while still contributing to the poster's Hawkes self-excitation.

    Args:
        new_shares: Sharing output from step(). Maps node_id to
            {conspiracy: share_count}.
        agents: Agent states to update.
        t: Current time step.
        scenario: Scenario config (for sharing_block_rate).
        rng: Per run random generator.

    Returns:
        Total number of new shares applied.
    """
    block_rate = scenario.sharing_block_rate if scenario else 0.0
    total = 0
    for node_id, consp_counts in new_shares.items():
        for conspiracy, count in consp_counts.items():
            for _ in range(count):
                if block_rate > 0:
                    draw = rng.random() if rng is not None else random.random()
                    visible = draw >= block_rate
                else:
                    visible = True
                agents[node_id].record_share(conspiracy, t, visible=visible)
            total += count
    return total
