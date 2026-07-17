"""
Semantic clustering with Silhouette Score optimization.

Determines the optimal number of clusters and linkage method by testing
all combinations and selecting the one that maximizes the Silhouette Score.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from conspiracy_analysis.config import get_included_conspiracies

logger = logging.getLogger(__name__)

SEMANTIC_CLUSTERING_FILENAME = "semantic_clustering.pkl"
_INTERMEDIATE_DIR = Path(__file__).resolve().parents[2] / "intermediate_files"
_SEMANTIC_CLUSTERING_PATH = _INTERMEDIATE_DIR / SEMANTIC_CLUSTERING_FILENAME

SEMANTIC_CLUSTER_COLOR_CYCLE = (
    "#E24A33",
    "#348ABD",
    "#988ED5",
    "#8EBA42",
    "#FFB000",
    "#009E73",
    "#56B4E9",
    "#CC79A7",
    "#777777",
)


def _configured_graph_conspiracies(G: nx.Graph) -> List[str]:
    graph_cols = G.graph.get("conspiracy_cols")
    if graph_cols:
        graph_set = set(graph_cols)
        return [c for c in get_included_conspiracies() if c in graph_set]
    return get_included_conspiracies()


def find_optimal_clustering(
    distance_matrix: pd.DataFrame,
    conspiracies: List[str],
    methods: List[str] = ("ward", "average", "complete", "single"),
    k_range: range = range(2, 9),
) -> Dict:
    """Find the optimal hierarchical clustering by maximizing Silhouette Score.

    Tests all combinations of linkage methods and cluster counts (k),
    then returns the configuration with the highest Silhouette Score.

    Args:
        distance_matrix: Symmetric DataFrame of pairwise semantic distances.
            Index and columns should be conspiracy column names.
        conspiracies: List of conspiracy names to include in clustering.
        methods: Linkage methods to test.
        k_range: Range of cluster counts to evaluate (default: 3-8).

    Returns:
        Dictionary containing:
        - 'best_method': Optimal linkage method name.
        - 'best_k': Optimal number of clusters.
        - 'best_score': Maximum Silhouette Score achieved.
        - 'cluster_assignments': Dict mapping conspiracy -> cluster_id.
        - 'cluster_map': Dict mapping cluster_id -> list of conspiracies.
        - 'linkage_matrix': Linkage matrix for the best method.
        - 'all_scores': DataFrame of all (method, k, score) tested.
        - 'conspiracies': List of conspiracies used.
    """
    missing_index = sorted(c for c in conspiracies if c not in distance_matrix.index)
    missing_columns = sorted(c for c in conspiracies if c not in distance_matrix.columns)
    if missing_index or missing_columns:
        parts = []
        if missing_index:
            parts.append(f"missing from distance matrix index: {missing_index}")
        if missing_columns:
            parts.append(f"missing from distance matrix columns: {missing_columns}")
        raise ValueError("Cannot cluster with incomplete distance matrix; " + "; ".join(parts))

    common = list(conspiracies)
    sem_matrix = distance_matrix.loc[common, common].values.copy()
    np.fill_diagonal(sem_matrix, 0.0)

    if np.isnan(sem_matrix).any():
        logger.warning("NaNs found in distance matrix; filling with max distance.")
        sem_matrix[np.isnan(sem_matrix)] = np.nanmax(sem_matrix)

    dist_condensed = squareform(sem_matrix)

    all_scores = []
    best_score = -1.0
    best_config = None

    for method in methods:
        linkage_matrix = hierarchy.linkage(dist_condensed, method=method)

        for k in k_range:
            if k >= len(common):
                continue
            labels = fcluster(linkage_matrix, k, criterion="maxclust")
            score = silhouette_score(sem_matrix, labels, metric="precomputed")
            all_scores.append({"method": method, "k": k, "silhouette_score": score})

            if score > best_score:
                best_score = score
                best_config = {
                    "method": method,
                    "k": k,
                    "labels": labels,
                    "linkage_matrix": linkage_matrix,
                }

    df_scores = pd.DataFrame(all_scores)

    cluster_assignments = {
        name: int(cid) for name, cid in zip(common, best_config["labels"])
    }

    cluster_map = {}
    for consp, cid in cluster_assignments.items():
        cluster_map.setdefault(cid, []).append(consp)

    logger.info(
        f"Optimal clustering: {best_config['method']} linkage with k={best_config['k']}, "
        f"Silhouette Score = {best_score:.4f}"
    )
    for cid, members in sorted(cluster_map.items()):
        logger.info(f"  Cluster {cid}: {', '.join(members)}")

    return {
        "best_method": best_config["method"],
        "best_k": best_config["k"],
        "best_score": best_score,
        "cluster_assignments": cluster_assignments,
        "cluster_map": cluster_map,
        "linkage_matrix": best_config["linkage_matrix"],
        "all_scores": df_scores,
        "conspiracies": common,
    }


def compute_temporal_distance_matrix(
    G: nx.Graph,
    conspiracies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute pairwise temporal distance matrix from first-appearance times.

    For each pair of conspiracies, the distance is the absolute difference
    in the hour each conspiracy first appeared anywhere in the network.

    Args:
        G: Network graph with conspiracy sharing times on nodes.
        conspiracies: Conspiracy column names. If None, uses
            G.graph['conspiracy_cols'].

    Returns:
        Symmetric DataFrame (n x n) with conspiracy names as index/columns,
        same format as the semantic distance matrix.
    """
    from conspiracy_analysis.utils.helpers import get_first_appearance_times

    if conspiracies is None:
        conspiracies = _configured_graph_conspiracies(G)

    first_times = get_first_appearance_times(G, conspiracies)
    n = len(conspiracies)
    matrix = np.zeros((n, n))

    for i, c1 in enumerate(conspiracies):
        for j, c2 in enumerate(conspiracies):
            matrix[i, j] = abs(first_times[c1] - first_times[c2])

    return pd.DataFrame(matrix, index=conspiracies, columns=conspiracies)


def compute_peak_frequency_distance_matrix(
    G: nx.Graph,
    conspiracies: Optional[List[str]] = None,
    window: int = 24,
) -> pd.DataFrame:
    """Compute pairwise distance matrix from peak-frequency times.

    For each pair of conspiracies, the distance is the absolute difference
    in the hour each conspiracy reached its peak sharing frequency
    (identified via a rolling sum over ``window`` hours).

    Args:
        G: Network graph with conspiracy sharing times on nodes.
        conspiracies: Conspiracy column names. If None, uses
            G.graph['conspiracy_cols'].
        window: Rolling window size in hours for peak detection.

    Returns:
        Symmetric DataFrame (n x n) with conspiracy names as index/columns.
    """
    from conspiracy_analysis.utils.helpers import get_peak_frequency_times

    if conspiracies is None:
        conspiracies = _configured_graph_conspiracies(G)

    peak_times = get_peak_frequency_times(G, conspiracies, window)
    n = len(conspiracies)
    matrix = np.zeros((n, n))

    for i, c1 in enumerate(conspiracies):
        for j, c2 in enumerate(conspiracies):
            matrix[i, j] = abs(peak_times[c1] - peak_times[c2])

    return pd.DataFrame(matrix, index=conspiracies, columns=conspiracies)


def assign_clusters(
    clustering_result: Dict,
) -> Dict[str, int]:
    """Extract the conspiracy -> cluster_id mapping from a clustering result.

    Args:
        clustering_result: Output from find_optimal_clustering().

    Returns:
        Dictionary mapping conspiracy name to cluster ID.
    """
    return clustering_result["cluster_assignments"]


def save_clustering_result(
    clustering_result: Dict,
    path: Optional[str] = None,
) -> None:
    """Save the selected semantic clustering for downstream notebooks."""
    import joblib

    output_path = Path(path) if path is not None else _SEMANTIC_CLUSTERING_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clustering_result, output_path)


def load_clustering_result(
    path: Optional[str] = None,
) -> Dict:
    """Load the selected semantic clustering saved by the analysis notebook."""
    import joblib

    input_path = Path(path) if path is not None else _SEMANTIC_CLUSTERING_PATH
    try:
        return joblib.load(input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{input_path} not found. Run 01_analysis.ipynb through the semantic "
            "clustering cell before generating downstream figures."
        ) from exc


def build_cluster_display_metadata(
    cluster_assignments: Dict[str, int],
    *,
    label_prefix: str = "Semantic cluster",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build plot labels and colors from computed cluster assignments.

    Returns a tuple of conspiracy to cluster label and cluster label to color.
    The conspiracy lookup contains both full ids and short ids.
    """
    cluster_ids = sorted(set(cluster_assignments.values()))
    cluster_names = {
        cid: f"{label_prefix} {cid}"
        for cid in cluster_ids
    }
    cluster_colors = {
        cluster_names[cid]: SEMANTIC_CLUSTER_COLOR_CYCLE[
            i % len(SEMANTIC_CLUSTER_COLOR_CYCLE)
        ]
        for i, cid in enumerate(cluster_ids)
    }

    conspiracy_to_cluster = {}
    for conspiracy, cid in cluster_assignments.items():
        cluster_name = cluster_names[cid]
        conspiracy_to_cluster[conspiracy] = cluster_name
        conspiracy_to_cluster[conspiracy.replace("ConsProb_", "")] = cluster_name

    return conspiracy_to_cluster, cluster_colors


def compute_cross_cluster_flag(
    df_short: pd.DataFrame,
    G: nx.Graph,
    cluster_assignments: Dict[str, int],
    model_number: int,
) -> pd.DataFrame:
    """Add cross_cluster binary flag to a short-form DataFrame.

    For Model 2: 1 if the new conspiracy is from a different cluster than
    the first conspiracy.

    For Model 3+: 1 if the new conspiracy is from a cluster the user hasn't
    shared from before (neither first nor second conspiracy's cluster).

    Args:
        df_short: Short-form DataFrame.
        G: Network graph (needed for Model 3 to look up first conspiracy).
        cluster_assignments: Conspiracy -> cluster_id mapping from optimal clustering.
        model_number: 2 for second conspiracy, 3 for third+ conspiracy.

    Returns:
        DataFrame with 'cross_cluster' column added.
    """
    conspiracies = _configured_graph_conspiracies(G)
    required = set(conspiracies)

    if "conspiracy" in df_short.columns:
        required.update(df_short["conspiracy"].dropna().unique())
    if model_number == 2 and "first_conspiracy" in df_short.columns:
        required.update(df_short["first_conspiracy"].dropna().unique())

    missing_clusters = sorted(
        conspiracy for conspiracy in required
        if conspiracy not in cluster_assignments
    )
    if missing_clusters:
        raise ValueError(
            f"Missing cluster assignments for model_number={model_number}: "
            f"{missing_clusters}"
        )

    def _is_cross_cluster(row):
        if model_number == 2:
            first_c = row.get("first_conspiracy")
            current_c = row.get("conspiracy")
            return int(cluster_assignments[first_c] != cluster_assignments[current_c])

        elif model_number >= 3:
            node = row["id"]
            first_times = {}
            for consp in conspiracies:
                activations = G.nodes[node].get(consp, [])
                if activations:
                    first_times[consp] = min(activations)
            if not first_times:
                return 0

            sorted_consps = sorted(first_times, key=first_times.get)
            current_consp = row.get("conspiracy")

            n_prior = int(row.get("n_prior", 2))

            visited_clusters = set()
            for i in range(min(n_prior, len(sorted_consps))):
                consp = sorted_consps[i]
                visited_clusters.add(cluster_assignments[consp])

            current_cluster = cluster_assignments[current_consp]
            return int(current_cluster not in visited_clusters)

        return 0

    df_short = df_short.copy()
    df_short["cross_cluster"] = df_short.apply(_is_cross_cluster, axis=1)
    return df_short
