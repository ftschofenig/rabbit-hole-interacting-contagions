"""Shared conspiracy category configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "conspiracies.yaml"
)


def get_config_path() -> Path:
    """Return the YAML configuration path."""
    return _CONFIG_PATH


@lru_cache(maxsize=1)
def load_conspiracy_config() -> Dict[str, Any]:
    """Load and validate the conspiracy category configuration."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid conspiracy config at {_CONFIG_PATH}")

    included = config.get("included_conspiracies")
    baseline = config.get("baseline_conspiracy")
    merges = config.get("merges", {})
    labels = config.get("labels", {})
    semantic_map = config.get("semantic_label_map", {})

    if not isinstance(included, list) or not included:
        raise ValueError("included_conspiracies must be a nonempty list")
    if len(included) != len(set(included)):
        raise ValueError("included_conspiracies contains duplicate entries")
    if baseline not in included:
        raise ValueError("baseline_conspiracy must be in included_conspiracies")
    if not isinstance(merges, dict):
        raise ValueError("merges must be a mapping")
    if not isinstance(labels, dict):
        raise ValueError("labels must be a mapping")
    if not isinstance(semantic_map, dict):
        raise ValueError("semantic_label_map must be a mapping")

    missing_labels = [c for c in included if c not in labels]
    if missing_labels:
        raise ValueError(f"labels missing entries for {missing_labels}")

    for target, sources in merges.items():
        if target not in included:
            raise ValueError(f"merge target {target} is not included")
        if not isinstance(sources, list) or target not in sources:
            raise ValueError(f"merge sources for {target} must include the target")

    return config


def get_included_conspiracies() -> List[str]:
    """Return the configured analytic category order."""
    return list(load_conspiracy_config()["included_conspiracies"])


def get_baseline_conspiracy() -> str:
    """Return the configured omitted category for model dummies."""
    return str(load_conspiracy_config()["baseline_conspiracy"])


def get_category_merges() -> Dict[str, List[str]]:
    """Return canonical target to source probability columns."""
    merges = load_conspiracy_config().get("merges", {})
    return {str(k): list(v) for k, v in merges.items()}


def get_conspiracy_labels(short: bool = False) -> Dict[str, str]:
    """Return display labels keyed by full ids or short ids."""
    labels = {
        str(k): str(v)
        for k, v in load_conspiracy_config().get("labels", {}).items()
    }
    if not short:
        return labels
    return {k.replace("ConsProb_", ""): v for k, v in labels.items()}


def get_semantic_label_map() -> Dict[str, str]:
    """Return semantic matrix labels mapped to canonical ids."""
    return {
        str(k): str(v)
        for k, v in load_conspiracy_config().get("semantic_label_map", {}).items()
    }


def get_conspiracy_sources(conspiracy: str) -> List[str]:
    """Return probability columns that feed one analytic category."""
    return get_category_merges().get(conspiracy, [conspiracy])


def get_source_to_canonical_map() -> Dict[str, str]:
    """Return source probability column to canonical category."""
    source_map = {c: c for c in get_included_conspiracies()}
    for target, sources in get_category_merges().items():
        for source in sources:
            source_map[source] = target
    return source_map


def get_present_included_conspiracies(columns: Iterable[str]) -> List[str]:
    """Return included categories whose source columns are available."""
    available = set(columns)
    present = []
    for conspiracy in get_included_conspiracies():
        if any(source in available for source in get_conspiracy_sources(conspiracy)):
            present.append(conspiracy)
    return present


def filter_semantic_matrix(df):
    """Filter a renamed semantic matrix to the configured category order."""
    included = get_included_conspiracies()
    present = [c for c in included if c in df.index and c in df.columns]
    return df.loc[present, present]


def apply_conspiracy_config_to_graph(G):
    """Attach configured category metadata to an analytic graph."""
    included = get_included_conspiracies()
    included_set = set(included)

    for _, data in G.nodes(data=True):
        for attr in list(data.keys()):
            if attr.startswith("ConsProb_") and attr not in included_set:
                del data[attr]
        censored = data.get("censored_neibs_consp")
        if isinstance(censored, dict):
            for attr in list(censored.keys()):
                if attr not in included_set:
                    del censored[attr]
        for attr in included:
            data.setdefault(attr, [])

    G.graph["conspiracy_cols"] = included
    G.graph["category_merges"] = get_category_merges()
    return G


def canonicalize_conspiracy_values(
    values: Iterable[str],
    *,
    known_columns: Optional[Iterable[str]] = None,
) -> List[str]:
    """Map source category ids to configured analytic ids."""
    source_map = get_source_to_canonical_map()
    known = set(known_columns) if known_columns is not None else None
    canonical = []
    seen = set()
    for value in values:
        target = source_map.get(value)
        if target is None:
            continue
        if known is not None and not any(
            source in known for source in get_conspiracy_sources(target)
        ):
            continue
        if target not in seen:
            canonical.append(target)
            seen.add(target)
    return canonical


def _passes_threshold(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    try:
        if bool(value != value):
            return False
    except TypeError:
        return False
    try:
        return bool(value >= threshold)
    except TypeError:
        return False


def build_qualified_conspiracy_list(
    row: Mapping[str, Any],
    threshold: float,
    conspiracies: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return analytic categories whose source probabilities pass threshold."""
    result = []
    for conspiracy in conspiracies or get_included_conspiracies():
        for source in get_conspiracy_sources(conspiracy):
            if _passes_threshold(row.get(source), threshold):
                result.append(conspiracy)
                break
    return result
