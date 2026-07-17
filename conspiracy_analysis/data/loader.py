"""Load the public anonymized graph and semantic distance data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import networkx as nx
import pandas as pd

from conspiracy_analysis import CONSPIRACY_PROB_THRESHOLD
from conspiracy_analysis.config import (
    apply_conspiracy_config_to_graph,
    filter_semantic_matrix,
    get_semantic_label_map,
)


logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANONYMIZED_DATA_DIR = _PROJECT_ROOT / "anonymized_data"


def anonymized_data_dir() -> Path:
    """Return the public dataset directory."""
    return _ANONYMIZED_DATA_DIR


def anonymized_graph_path() -> Path:
    """Return the public graph path."""
    return _ANONYMIZED_DATA_DIR / "G_MC.pkl"


def _validate_cached_multiple_allowed(
    graph: nx.Graph,
    requested_multiple_allowed: bool,
    graph_path: Path,
) -> None:
    """Ensure graph metadata matches the requested assignment policy."""
    cached_multiple_allowed = graph.graph.get("multiple_allowed")

    if cached_multiple_allowed is None:
        raise ValueError(
            f"Public graph at {graph_path} does not record multiple_allowed. "
            "Use the published public data archive."
        )

    if bool(cached_multiple_allowed) != bool(requested_multiple_allowed):
        raise ValueError(
            f"Public graph at {graph_path} was built with "
            f"multiple_allowed={cached_multiple_allowed}, but "
            f"load_graph_and_tweets requested multiple_allowed="
            f"{requested_multiple_allowed}. Use a public data archive built "
            "with the requested policy."
        )


def _validate_cached_threshold(graph: nx.Graph, graph_path: Path) -> None:
    """Ensure graph metadata matches the package threshold policy."""
    cached_threshold = graph.graph.get("conspiracy_prob_threshold")

    if cached_threshold is None:
        raise ValueError(
            f"Public graph at {graph_path} does not record "
            "conspiracy_prob_threshold. Use the published public data archive."
        )

    if float(cached_threshold) != float(CONSPIRACY_PROB_THRESHOLD):
        raise ValueError(
            f"Public graph at {graph_path} was built with "
            f"conspiracy_prob_threshold={cached_threshold}, but the current "
            f"package threshold is {CONSPIRACY_PROB_THRESHOLD}. Use a public "
            "data archive built with the current package threshold."
        )


def _validate_cached_time_resolution(
    graph: nx.Graph,
    requested_time_resolution: str,
    graph_path: Path,
) -> None:
    """Ensure graph metadata matches the requested time resolution."""
    cached_time_resolution = graph.graph.get("time_resolution")

    if cached_time_resolution is None:
        raise ValueError(
            f"Public graph at {graph_path} does not record time_resolution. "
            "Use the published public data archive."
        )

    if str(cached_time_resolution) != str(requested_time_resolution):
        raise ValueError(
            f"Public graph at {graph_path} was built with "
            f"time_resolution={cached_time_resolution}, but "
            f"load_graph_and_tweets requested time_resolution="
            f"{requested_time_resolution}. Use a public data archive built "
            "with the requested time resolution."
        )


def _validate_bot_scores(graph: nx.Graph, graph_path: Path) -> None:
    missing = [
        node for node, data in graph.nodes(data=True)
        if "bot_score" not in data
    ]
    if missing:
        raise ValueError(
            f"Public graph at {graph_path} lacks bot_score for "
            f"{len(missing)} nodes. Use the published public data archive."
        )


def _validate_no_self_loops(graph: nx.Graph, graph_path: Path) -> None:
    loop_count = nx.number_of_selfloops(graph)
    if loop_count:
        raise ValueError(
            f"Public graph at {graph_path} contains {loop_count} self loops. "
            "Use the published public data archive."
        )


def load_graph_and_tweets(
    from_joblib: bool = True,
    time_resolution: str = "Hour",
    data_dir: Optional[str] = None,
    joblib_graph_path: Optional[str] = None,
    multiple_allowed: bool = True,
) -> Tuple[nx.Graph, None]:
    """Load the anonymized graph without loading any tweet table."""
    if not from_joblib:
        raise ValueError(
            "Raw graph construction is not part of this public repository. "
            "Extract the published public data archive, then call "
            "load_graph_and_tweets with from_joblib=True."
        )

    if joblib_graph_path is not None:
        graph_path = Path(joblib_graph_path)
    elif data_dir is not None:
        graph_path = Path(data_dir) / "G_MC.pkl"
    else:
        graph_path = anonymized_graph_path()

    logger.info("Loading anonymized graph from %s", graph_path)
    graph = joblib.load(graph_path)
    _validate_cached_multiple_allowed(graph, multiple_allowed, graph_path)
    _validate_cached_threshold(graph, graph_path)
    _validate_cached_time_resolution(graph, time_resolution, graph_path)
    _validate_no_self_loops(graph, graph_path)
    _validate_bot_scores(graph, graph_path)
    graph = apply_conspiracy_config_to_graph(graph)
    return graph, None


def load_semantic_distance_matrix(
    data_dir: Optional[str] = None,
    csv_filename: str = "conspiracy_semantic_distance.csv",
    apply_config: bool = True,
) -> pd.DataFrame:
    """Load and clean the public semantic distance matrix."""
    root = Path(data_dir) if data_dir is not None else anonymized_data_dir()
    dataframe = pd.read_csv(root / csv_filename, low_memory=False)

    rename_dict = get_semantic_label_map()
    dataframe = dataframe.rename(columns=rename_dict)
    dataframe = dataframe.set_index("Unnamed: 0")
    dataframe.index.name = None
    dataframe = dataframe.rename(index=rename_dict)

    if apply_config:
        dataframe = filter_semantic_matrix(dataframe)

    return dataframe


__all__ = [
    "anonymized_data_dir",
    "anonymized_graph_path",
    "load_graph_and_tweets",
    "load_semantic_distance_matrix",
]
