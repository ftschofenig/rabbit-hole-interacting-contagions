"""
Neighbor exposure computation for the ABM.

Exposure (s7) counts unique neighbors who shared the conspiracy within
a rolling time window, matching the Cox training data's 14 days or 336 h.
Sharing events are driven by the Hawkes self-exciting process.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from conspiracy_analysis.simulation.agent_state import AgentState
    from conspiracy_analysis.simulation.config import ScenarioConfig


def compute_neighbor_exposure(
    node_id: str,
    conspiracy: str,
    agents: Dict[str, AgentState],
    G: nx.Graph,
    scenario: ScenarioConfig,
    t: float,
    exposure_window: float,
) -> int:
    """Count unique neighbors who shared a conspiracy within the time window.

    Args:
        node_id: Target node.
        conspiracy: Conspiracy to check exposure for.
        agents: Mapping from node id to AgentState.
        G: Network graph.
        scenario: Scenario config (reserved for future use).
        t: Current simulation time step.
        exposure_window: Rolling window in hours. The production value is
            336 h or 14 days.

    Returns:
        Number of unique neighbors who shared within [t - exposure_window, t].
    """
    window_start = t - exposure_window
    count = 0
    for neighbor in G.adj[node_id]:
        neighbor_state = agents.get(neighbor)
        if neighbor_state is not None and neighbor_state.has_shared_recently(
            conspiracy, window_start
        ):
            count += 1
    return count
