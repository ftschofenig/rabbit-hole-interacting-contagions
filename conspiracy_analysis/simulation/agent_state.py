"""
Per-node agent state for the ABM simulation.

Tracks which conspiracies a node has adopted, the timeline of adoptions,
visited semantic clusters (for the cross-cluster covariate), and
per-conspiracy sharing histories (for Hawkes re-sharing dynamics).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


class AgentState:
    """State of a single agent (network node) in the simulation.

    After adoption, agents re-share according to a Hawkes self-exciting
    process. Exposure (s7) is computed from neighbors who shared within
    a rolling time window, not from adoption status alone.

    Attributes:
        adopted: Set of adopted conspiracy names.
        log_degree: Pre-computed log(1 + degree) for Cox covariate.
        first_active_time: Earliest time this node is eligible to participate.
    """

    __slots__ = (
        "adopted",
        "_adoption_timeline",
        "_visited_clusters",
        "_sharing_history",
        "_visible_sharing_history",
        "_immune",
        "_last_organic_adoption_time",
        "log_degree",
        "first_active_time",
    )

    def __init__(self, log_degree: float, first_active_time: float = 0.0):
        self.adopted: Set[str] = set()
        self._adoption_timeline: List[Tuple[float, str]] = []
        self._visited_clusters: Set[int] = set()
        self._sharing_history: Dict[str, List[float]] = {}
        self._visible_sharing_history: Dict[str, List[float]] = {}
        self._immune: Set[str] = set()
        self._last_organic_adoption_time: Optional[float] = None
        self.log_degree = log_degree
        self.first_active_time = first_active_time

    @property
    def num_adopted(self) -> int:
        """Number of distinct conspiracies adopted so far."""
        return len(self.adopted)

    @property
    def adoption_number_for_next(self) -> int:
        """The adoption number for the next conspiracy (1-indexed).

        If the agent has adopted 0 conspiracies, the next is adoption #1.
        """
        return self.num_adopted + 1

    def has_adopted(self, conspiracy: str) -> bool:
        return conspiracy in self.adopted

    def is_immune(self, conspiracy: str) -> bool:
        """Check if the agent is permanently immune to this conspiracy."""
        return conspiracy in self._immune

    def record_immunity(self, conspiracy: str) -> None:
        """Mark the agent as permanently immune to a conspiracy."""
        self._immune.add(conspiracy)

    def get_nth_adoption_time(self, n: int) -> Optional[float]:
        """Return the time of the n-th adoption (1-indexed), or None."""
        if n < 1 or n > len(self._adoption_timeline):
            return None
        return self._adoption_timeline[n - 1][0]

    def get_last_adoption_time(self) -> Optional[float]:
        """Return the time of the most recent adoption, or None."""
        if not self._adoption_timeline:
            return None
        return self._adoption_timeline[-1][0]

    def get_last_organic_adoption_time(self) -> Optional[float]:
        """Return the time of the most recent NON-seed adoption, or None.

        Used by the quarantine counterfactual so that seed adoptions do
        not trigger the lock window. Seeds still appear in
        _adoption_timeline (for has_adopted and Cox tau),
        just not in this organic-only timestamp.
        """
        return self._last_organic_adoption_time

    def get_nth_adoption_name(self, n: int) -> Optional[str]:
        """Return the conspiracy name of the n-th adoption (1-indexed), or None."""
        if n < 1 or n > len(self._adoption_timeline):
            return None
        return self._adoption_timeline[n - 1][1]

    def is_cross_cluster(
        self, conspiracy: str, cluster_assignments: Dict[str, int]
    ) -> bool:
        """Check if adopting this conspiracy crosses into a new semantic cluster.

        Returns True if the conspiracy's cluster has NOT been visited
        by any prior adoption. Returns False if no cluster info is available
        or if this is the agent's first adoption.
        """
        if not self._visited_clusters:
            return False
        target_cluster = cluster_assignments.get(conspiracy)
        if target_cluster is None:
            return False
        return target_cluster not in self._visited_clusters

    def record_adoption(
        self,
        conspiracy: str,
        time: float,
        cluster_assignments: Dict[str, int],
        is_organic: bool = True,
    ) -> None:
        """Record a new conspiracy adoption.

        Also records the initial sharing event (adoption = first tweet).

        Args:
            conspiracy: Name of the adopted conspiracy.
            time: Simulation time of adoption.
            cluster_assignments: Mapping from conspiracy to cluster id.
            is_organic: True for adoptions made through Phase A's
                competing-risks draw (the default); False for seed
                adoptions placed by seed_nodes. Only organic adoptions
                update _last_organic_adoption_time, which is the field
                the quarantine counterfactual reads.
        """
        if conspiracy in self.adopted:
            return
        self.adopted.add(conspiracy)
        self._adoption_timeline.append((time, conspiracy))
        if is_organic:
            self._last_organic_adoption_time = time
        cluster = cluster_assignments.get(conspiracy)
        if cluster is not None:
            self._visited_clusters.add(cluster)
        self.record_share(conspiracy, time)

    def record_share(self, conspiracy: str, time: float, visible: bool = True) -> None:
        """Record a sharing (re-tweet) event for a conspiracy.

        Args:
            conspiracy: Conspiracy being shared.
            time: Simulation time of share.
            visible: If True, the share is visible to neighbors (counted in s7).
                If False, the share is blocked by content moderation but still
                contributes to the poster's own Hawkes self-excitation.
        """
        if conspiracy not in self._sharing_history:
            self._sharing_history[conspiracy] = []
        self._sharing_history[conspiracy].append(time)
        if visible:
            if conspiracy not in self._visible_sharing_history:
                self._visible_sharing_history[conspiracy] = []
            self._visible_sharing_history[conspiracy].append(time)

    def has_shared_recently(
        self, conspiracy: str, window_start: float
    ) -> bool:
        """Check if the agent shared the conspiracy at or after window_start.

        Iterates history in reverse for efficiency (most recent first).
        """
        history = self._visible_sharing_history.get(conspiracy)
        if not history:
            return False
        # History is chronological; check most recent first
        for t_event in reversed(history):
            if t_event >= window_start:
                return True
            else:
                return False  # older events are even earlier, no need to check
        return False

    def get_sharing_history(self, conspiracy: str) -> List[float]:
        """Return the sharing history for a conspiracy (for Hawkes intensity)."""
        return self._sharing_history.get(conspiracy, [])
