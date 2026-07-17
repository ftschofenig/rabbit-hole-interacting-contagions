"""
Utility functions for exposure calculations, time conversions, and graph queries.

These helpers are used across the data, models, and simulation modules.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional

from conspiracy_analysis import EXPOSURE_WINDOW


def coerce_bot_score(value) -> Optional[float]:
    """Return a finite bot score, or None when the score is not usable."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return score


def passes_bot_filter(
    G: nx.Graph,
    node: str,
    threshold: float,
    mode: str,
) -> bool:
    """Return whether a node belongs in the requested bot score subset."""
    score = coerce_bot_score(G.nodes[node].get("bot_score"))
    if score is None:
        return False
    if mode == "HUMAN":
        return score <= threshold
    if mode == "BOT":
        return score > threshold
    raise ValueError(f"mode must be 'HUMAN' or 'BOT', got {mode!r}")


def compute_neighbor_exposure(
    G: nx.Graph,
    node: str,
    conspiracy: str,
    t: float,
    window: int = EXPOSURE_WINDOW,
) -> int:
    """Count unique neighbors who shared a conspiracy within a rolling time window.

    Args:
        G: Network graph with conspiracy sharing times stored as node attributes.
        node: The focal node ID.
        conspiracy: Conspiracy column name (e.g., 'ConsProb_fakenews').
        t: Current time step (hours).
        window: Lookback window in hours. The package default is 336 h or 14 days.

    Returns:
        Number of distinct neighbors who shared the conspiracy in [t - window, t].
    """
    count = 0
    t_start = t - window
    for neighbor in G.neighbors(node):
        activations = G.nodes[neighbor].get(conspiracy, [])
        if any(t_start <= act_t <= t for act_t in activations):
            count += 1
    return count



def find_max_time_in_graph(G: nx.Graph) -> float:
    """Find the latest time step across all conspiracies in the graph.

    Args:
        G: Network graph with conspiracy columns stored in G.graph['conspiracy_cols'].

    Returns:
        Maximum time step found in any node's conspiracy activation list.
    """
    conspiracies = G.graph.get("conspiracy_cols", [])
    max_time = -np.inf
    for node in G.nodes:
        for consp in conspiracies:
            activations = G.nodes[node].get(consp, [])
            if activations:
                node_max = max(activations)
                if node_max > max_time:
                    max_time = node_max
    return max_time


def get_first_appearance_times(
    G: nx.Graph,
    conspiracies: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Get the earliest time each conspiracy appeared in the graph.

    Args:
        G: Network graph.
        conspiracies: List of conspiracy column names. If None, uses
            G.graph['conspiracy_cols'].

    Returns:
        Dictionary mapping conspiracy name to its first appearance time.
    """
    if conspiracies is None:
        conspiracies = G.graph.get("conspiracy_cols", [])

    first_appearances = {}
    for consp in conspiracies:
        first_time = float("inf")
        for node, data in G.nodes(data=True):
            activations = data.get(consp, [])
            if activations:
                first_time = min(first_time, min(activations))
        first_appearances[consp] = first_time
    return first_appearances


def get_peak_frequency_times(
    G: nx.Graph,
    conspiracies: Optional[List[str]] = None,
    window: int = 24,
) -> Dict[str, float]:
    """Get the hour of peak sharing frequency for each conspiracy.

    For each conspiracy, aggregates all sharing events across all nodes
    into an hourly time series, applies a rolling sum over ``window`` hours,
    and returns the center of the window with the maximum count.

    Args:
        G: Network graph.
        conspiracies: List of conspiracy column names. If None, uses
            G.graph['conspiracy_cols'].
        window: Rolling window size in hours (default: 24).

    Returns:
        Dictionary mapping conspiracy name to the hour of peak frequency.
    """
    if conspiracies is None:
        conspiracies = G.graph.get("conspiracy_cols", [])

    peak_times = {}
    for consp in conspiracies:
        all_times = []
        for node, data in G.nodes(data=True):
            activations = data.get(consp, [])
            if activations:
                all_times.extend(activations)

        if not all_times:
            peak_times[consp] = float("inf")
            continue

        t_min = int(np.floor(min(all_times)))
        t_max = int(np.ceil(max(all_times)))
        n_bins = t_max - t_min + 1
        counts = np.zeros(n_bins)
        for t in all_times:
            idx = int(np.floor(t)) - t_min
            idx = min(idx, n_bins - 1)
            counts[idx] += 1

        rolling = np.convolve(counts, np.ones(window), mode="same")
        peak_times[consp] = t_min + int(np.argmax(rolling))

    return peak_times


def get_conspiracy_columns(G: nx.Graph) -> List[str]:
    """Retrieve the list of conspiracy column names from the graph.

    Args:
        G: Network graph.

    Returns:
        List of conspiracy column name strings.
    """
    return list(G.graph.get("conspiracy_cols", []))


def get_min_time_for_conspiracy(G: nx.Graph, conspiracy: str) -> float:
    """Find the earliest activation time for a specific conspiracy across all nodes.

    Args:
        G: Network graph.
        conspiracy: Conspiracy column name.

    Returns:
        Minimum activation time, or inf if no activations exist.
    """
    min_time = np.inf
    for node in G.nodes:
        activations = G.nodes[node].get(conspiracy, [])
        if activations:
            node_min = min(activations)
            if node_min < min_time:
                min_time = node_min
    return min_time
