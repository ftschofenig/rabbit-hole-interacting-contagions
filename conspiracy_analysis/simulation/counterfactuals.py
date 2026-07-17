"""
Counterfactual utilities for simulation scenarios.
"""

import logging
from typing import Dict, List

import networkx as nx

logger = logging.getLogger(__name__)


def compute_entry_times(
    G: nx.Graph,
    conspiracies: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute first appearance time for each conspiracy in the graph.

    Args:
        G: Network graph with conspiracy sharing times.
        conspiracies: List of conspiracy column names.
        normalize: If True, set all entry times to 1 (simultaneous start).

    Returns:
        Dict mapping conspiracy -> first appearance time (in hours).
    """
    if normalize:
        return {c: 1 for c in conspiracies}

    entry_times = {}
    for consp in conspiracies:
        first_time = float("inf")
        for node in G.nodes():
            times = G.nodes[node].get(consp, [])
            if times:
                first_time = min(first_time, min(times))
        entry_times[consp] = first_time if first_time != float("inf") else 0

    return entry_times
