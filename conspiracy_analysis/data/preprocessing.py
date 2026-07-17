"""
Event history preparation for Cox proportional hazard models.

Creates short-form (one row per subject-conspiracy) and long-form
(time-varying covariate) DataFrames at hourly resolution.

Short-form columns: id, time, entry, conspiracy, event, degree, [first_conspiracy], [cross_cluster]
Long-form columns: id, conspiracy, entry, exit, event, s_7, degree, [dummies]
"""

import math
import logging
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from conspiracy_analysis import BOT_SCORE_THRESHOLD, EXPOSURE_WINDOW, SIMULTANEOUS_NUDGE
from conspiracy_analysis.utils.helpers import (
    find_max_time_in_graph,
    get_min_time_for_conspiracy,
    compute_neighbor_exposure,
    passes_bot_filter,
)

logger = logging.getLogger(__name__)

BACKTRACK_EXPOSURE = EXPOSURE_WINDOW  # 14 days or 336 h for s_7 exposure count


def get_first_activation_data(
    G: nx.Graph,
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
    active_user_correction: bool = False,
    simultaneous_nudge: float = SIMULTANEOUS_NUDGE,
    attrition_correction: bool = False,
) -> pd.DataFrame:
    """Build short-form DataFrame for Model 1: first conspiracy adoption.

    Each user contributes one event row (the conspiracy they adopted first)
    and censored rows for all other conspiracies they were at risk for.

    Args:
        G: Network graph with conspiracy activation times on nodes.
        bot_score_threshold: Bot score cutoff for filtering.
        mode: 'HUMAN' to keep humans, 'BOT' to keep bots.
        active_user_correction: If True, use first_active_time as entry.
        attrition_correction: If True, censor non-adopters at last_active_time + 1
            instead of max_time.

    Returns:
        DataFrame with columns: id, time, entry, conspiracy, event, degree.
    """
    max_time = max(
        (G.nodes[n].get("last_active_time") for n in G.nodes
         if G.nodes[n].get("last_active_time") is not None),
        default=find_max_time_in_graph(G),
    )
    conspiracies = G.graph["conspiracy_cols"]
    results = []
    n_fallback_correction = 0  # users where first_active_time > first_adoption

    for node in tqdm(G.nodes, desc="Building first activation data"):
        if not passes_bot_filter(G, node, bot_score_threshold, mode):
            continue

        first, name, entry_first = np.inf, None, 0

        for consp in conspiracies:
            activations = G.nodes[node].get(consp, [])
            if activations:
                min_time = min(activations)
                if min_time < first:
                    first = min_time
                    name = consp
                    entry_first = get_min_time_for_conspiracy(G, consp)
                    if active_user_correction and G.nodes[node]["first_active_time"] is not None:
                        fat = G.nodes[node]["first_active_time"]
                        candidate = max(entry_first, fat)
                        if candidate < first:
                            entry_first = candidate
                        elif candidate == first:
                            entry_first = candidate
                            first += simultaneous_nudge
                        else:
                            # first_active_time > first_adoption: correction would
                            # invalidate the interval (entry > exit). Fall back to
                            # the raw conspiracy first-appearance; nudge if needed.
                            n_fallback_correction += 1
                            if entry_first == first:
                                first += simultaneous_nudge

        if first != np.inf and entry_first == first:
            first += simultaneous_nudge

        if first != np.inf:
            results.append({
                "id": node, "time": first, "entry": entry_first,
                "conspiracy": name, "event": 1, "degree": G.degree[node],
            })

        if first != np.inf:
            first_time = first
        elif attrition_correction:
            lat = G.nodes[node].get("last_active_time")
            first_time = (lat + 1) if lat is not None else max_time
        else:
            first_time = max_time
        consp_others = [c for c in conspiracies if c != name or first == np.inf]
        for consp in consp_others:
            entry = get_min_time_for_conspiracy(G, consp)
            if active_user_correction and G.nodes[node]["first_active_time"] is not None:
                entry = max(entry, G.nodes[node]["first_active_time"])
            if entry < first_time:
                results.append({
                    "id": node, "time": first_time, "entry": entry,
                    "conspiracy": consp, "event": 0, "degree": G.degree[node],
                })

    if active_user_correction and n_fallback_correction > 0:
        logger.info(
            f"active_user_correction: skipped correction on event row for "
            f"{n_fallback_correction} users whose first conspiracy share predates "
            f"their first observed activity (event retained with raw entry)."
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["time"] = df["time"].astype(float)
    simultaneous = df["entry"] == df["time"]
    df.loc[simultaneous, "time"] += simultaneous_nudge
    df = df[df["entry"] < df["time"]]
    return df


def get_second_activation_data(
    G: nx.Graph,
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
    simultaneous_nudge: float = SIMULTANEOUS_NUDGE,
    attrition_correction: bool = False,
) -> pd.DataFrame:
    """Build short-form DataFrame for Model 2: second conspiracy adoption.

    Includes `first_conspiracy` column (categorical) identifying which
    conspiracy was adopted first. This is the ONLY model with this flag.

    Args:
        G: Network graph.
        bot_score_threshold: Bot score cutoff.
        mode: 'HUMAN' or 'BOT'.
        attrition_correction: If True, censor users with only 1 adoption at
            last_active_time + 1 instead of max_time.

    Returns:
        DataFrame with columns: id, time, entry, conspiracy, event,
        first_conspiracy, first_time, degree.
    """
    max_time = max(
        (G.nodes[n].get("last_active_time") for n in G.nodes
         if G.nodes[n].get("last_active_time") is not None),
        default=find_max_time_in_graph(G),
    )
    conspiracies = G.graph["conspiracy_cols"]
    results = []

    for node in tqdm(G.nodes):
        if not passes_bot_filter(G, node, bot_score_threshold, mode):
            continue

        first_times = {}
        for consp in conspiracies:
            activations = G.nodes[node][consp]
            if activations:
                first_times[consp] = min(activations)

        ft_df = pd.DataFrame(list(first_times.items()), columns=["conspiracy", "time"])
        ft_df = ft_df.sort_values(by="time")

        if len(ft_df) >= 2:
            first_row = ft_df.iloc[0]
            second_row = ft_df.iloc[1]
            entry = max(first_row["time"], get_min_time_for_conspiracy(G, second_row["conspiracy"]))
            results.append({
                "id": node, "time": second_row["time"], "entry": entry,
                "conspiracy": second_row["conspiracy"], "event": 1,
                "first_conspiracy": first_row["conspiracy"],
                "first_time": first_row["time"], "degree": G.degree[node],
            })
            for consp in conspiracies:
                if consp in (first_row["conspiracy"], second_row["conspiracy"]):
                    continue
                entry = max(first_row["time"], get_min_time_for_conspiracy(G, consp))
                results.append({
                    "id": node, "time": second_row["time"], "entry": entry,
                    "conspiracy": consp, "event": 0,
                    "first_conspiracy": first_row["conspiracy"],
                    "first_time": first_row["time"], "degree": G.degree[node],
                })

        elif len(ft_df) == 1:
            first_row = ft_df.iloc[0]
            if attrition_correction:
                lat = G.nodes[node].get("last_active_time")
                censor_time = (lat + 1) if lat is not None else max_time
            else:
                censor_time = max_time
            for consp in conspiracies:
                if consp == first_row["conspiracy"]:
                    continue
                entry = max(first_row["time"], get_min_time_for_conspiracy(G, consp))
                results.append({
                    "id": node, "time": censor_time, "entry": entry,
                    "conspiracy": consp, "event": 0,
                    "first_conspiracy": first_row["conspiracy"],
                    "first_time": first_row["time"], "degree": G.degree[node],
                })

    df = pd.DataFrame(results)
    df["time"] = df["time"].astype(float)
    simultaneous = df["entry"] == df["time"]
    df.loc[simultaneous, "time"] += simultaneous_nudge
    df = df[df["entry"] < df["time"]]
    return df


def get_nth_activation_data(
    G: nx.Graph,
    n: int,
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
    simultaneous_nudge: float = SIMULTANEOUS_NUDGE,
    attrition_correction: bool = False,
) -> pd.DataFrame:
    """Build short-form DataFrame for Model N (n >= 3): n-th conspiracy adoption.

    Generalized version of get_third_activation_data. For the n-th adoption:
    - Event row: n-th conspiracy in each user's sorted adoption timeline
    - Entry time: max(time of (n-1)-th adoption, first appearance of target conspiracy)
    - prior_time: time of the (n-1)-th adoption (used for tau-scale shift)
    - n_prior: n-1, the number of prior adoptions (used for cross_cluster computation)

    Args:
        G: Network graph.
        n: Which adoption to model (3 = third, 4 = fourth, etc.). Must be >= 3.
        bot_score_threshold: Bot score cutoff.
        mode: 'HUMAN' or 'BOT'.
        attrition_correction: If True, censor users with exactly n-1 adoptions at
            last_active_time + 1 instead of max_time.

    Returns:
        DataFrame with columns: id, time, entry, conspiracy, event,
        prior_time, n_prior, degree.
    """
    if n < 3:
        raise ValueError(f"get_nth_activation_data requires n >= 3, got {n}")

    max_time = max(
        (G.nodes[n].get("last_active_time") for n in G.nodes
         if G.nodes[n].get("last_active_time") is not None),
        default=find_max_time_in_graph(G),
    )
    conspiracies = G.graph["conspiracy_cols"]
    results = []

    for node in G.nodes:
        if not passes_bot_filter(G, node, bot_score_threshold, mode):
            continue

        first_times = {}
        for consp in conspiracies:
            activations = G.nodes[node][consp]
            if activations:
                first_times[consp] = min(activations)

        ft_df = pd.DataFrame(list(first_times.items()), columns=["conspiracy", "time"])
        ft_df = ft_df.sort_values(by="time")

        if len(ft_df) >= n:
            prior_row = ft_df.iloc[n - 2]  # (n-1)-th adoption
            nth_row = ft_df.iloc[n - 1]    # n-th adoption
            entry = max(prior_row["time"], get_min_time_for_conspiracy(G, nth_row["conspiracy"]))
            results.append({
                "id": node, "time": nth_row["time"], "entry": entry,
                "conspiracy": nth_row["conspiracy"], "event": 1,
                "prior_time": prior_row["time"], "n_prior": n - 1,
                "degree": G.degree[node],
            })
            already_adopted = {ft_df.iloc[i]["conspiracy"] for i in range(n)}
            for consp in conspiracies:
                if consp in already_adopted:
                    continue
                entry = max(prior_row["time"], get_min_time_for_conspiracy(G, consp))
                results.append({
                    "id": node, "time": nth_row["time"], "entry": entry,
                    "conspiracy": consp, "event": 0,
                    "prior_time": prior_row["time"], "n_prior": n - 1,
                    "degree": G.degree[node],
                })

        elif len(ft_df) == n - 1:
            prior_row = ft_df.iloc[n - 2]  # (n-1)-th adoption
            already_adopted = {ft_df.iloc[i]["conspiracy"] for i in range(n - 1)}
            if attrition_correction:
                lat = G.nodes[node].get("last_active_time")
                censor_time = (lat + 1) if lat is not None else max_time
            else:
                censor_time = max_time
            for consp in conspiracies:
                if consp in already_adopted:
                    continue
                entry = max(prior_row["time"], get_min_time_for_conspiracy(G, consp))
                results.append({
                    "id": node, "time": censor_time, "entry": entry,
                    "conspiracy": consp, "event": 0,
                    "prior_time": prior_row["time"], "n_prior": n - 1,
                    "degree": G.degree[node],
                })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["time"] = df["time"].astype(float)
    simultaneous = df["entry"] == df["time"]
    df.loc[simultaneous, "time"] += simultaneous_nudge
    df = df[df["entry"] < df["time"]]
    return df


def get_third_activation_data(
    G: nx.Graph,
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
    simultaneous_nudge: float = SIMULTANEOUS_NUDGE,
    attrition_correction: bool = False,
) -> pd.DataFrame:
    """Build short-form DataFrame for Model 3: third conspiracy adoption.

    Thin wrapper around get_nth_activation_data(G, n=3) that renames
    `prior_time` to `second_time` for backward compatibility.

    Args:
        G: Network graph.
        bot_score_threshold: Bot score cutoff.
        mode: 'HUMAN' or 'BOT'.
        attrition_correction: If True, censor users with exactly 2 adoptions at
            last_active_time + 1 instead of max_time.

    Returns:
        DataFrame with columns: id, time, entry, conspiracy, event,
        second_time, n_prior, degree.
    """
    df = get_nth_activation_data(G, n=3, bot_score_threshold=bot_score_threshold, mode=mode, simultaneous_nudge=simultaneous_nudge, attrition_correction=attrition_correction)
    if not df.empty:
        df = df.rename(columns={"prior_time": "second_time"})
    return df


def get_max_model_number(
    G: nx.Graph,
    min_events: int = 100,
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
) -> Tuple[int, Dict[int, int]]:
    """Determine the maximum model number with sufficient events.

    For each user, counts how many distinct conspiracies they adopted.
    Model N requires users with N+ adoptions; the event count is the
    number of users with exactly N adoptions who contribute an event row.

    Args:
        G: Network graph.
        min_events: Minimum number of events required for a model.
        bot_score_threshold: Bot score cutoff.
        mode: 'HUMAN' or 'BOT'.

    Returns:
        Tuple of (max_n, event_counts_dict) where event_counts_dict maps
        model number to the number of events (users with >= N adoptions).
    """
    conspiracies = G.graph["conspiracy_cols"]
    adoption_counts = []

    for node in G.nodes:
        if not passes_bot_filter(G, node, bot_score_threshold, mode):
            continue
        count = 0
        for consp in conspiracies:
            activations = G.nodes[node].get(consp, [])
            if activations:
                count += 1
        if count > 0:
            adoption_counts.append(count)

    event_counts = {}
    max_n = 1
    for n in range(2, max(adoption_counts) + 1 if adoption_counts else 2):
        events = sum(1 for c in adoption_counts if c >= n)
        event_counts[n] = events
        if events >= min_events:
            max_n = n
        else:
            logger.info(
                f"Model {n}: {events} events (below threshold of {min_events}). "
                f"Stopping at Model {max_n}."
            )
            break

    logger.info(f"Auto-detected max model number: {max_n}")
    for n, count in event_counts.items():
        logger.info(f"  Model {n}: {count} events")

    return max_n, event_counts


def create_long_form(
    df_short: pd.DataFrame,
    G: nx.Graph,
    step: int = 8,
    correction_for_censored_neighbors: bool = False,
    exposure_window: int = BACKTRACK_EXPOSURE,
) -> pd.DataFrame:
    """Expand short-form data into time-varying covariate long-form.

    Each subject-conspiracy pair is split into intervals of `step` hours.
    At each interval, time-varying covariates (neighbor exposure, sharing
    counts) are computed.

    Args:
        df_short: Short-form DataFrame from get_*_activation_data().
        G: Network graph.
        step: Interval length in hours for discretization.
        correction_for_censored_neighbors: Apply censored neighbor correction.
        exposure_window: Lookback window in hours for counting active neighbors.
            The default is 14 days or 336 h.

    Returns:
        Long-form DataFrame with time-varying covariates.
    """
    rows = []
    has_first_conspiracy = "first_conspiracy" in df_short.columns
    if has_first_conspiracy:
        model_number = 2
    elif "prior_time" in df_short.columns or "second_time" in df_short.columns:
        model_number = 3  # same treatment for all N >= 3
    else:
        model_number = 1

    for _, r in df_short.iterrows():
        node = r["id"]
        consp = r["conspiracy"]
        entry = float(r["entry"])
        exit_ = float(r["time"])
        ev = int(r["event"])
        cross_cluster = r.get("cross_cluster", None)

        if consp is None or not (exit_ > entry):
            continue

        t = math.floor(entry / step) * step
        while t < exit_:
            start = max(entry, t)
            stop = min(exit_, t + step)
            if stop > start:
                s_val_7 = _compute_s_sum(G, node, consp, start, exposure_window)

                if correction_for_censored_neighbors and int(ev and stop == exit_) == 1:
                    censored = G.nodes[node].get("censored_neibs_consp", {})
                    if consp in censored and censored[consp] == 1 and s_val_7 == 0:
                        s_val_7 += 1

                row = {
                    "id": node, "conspiracy": consp,
                    "entry": start, "exit": stop,
                    "event": int(ev and stop == exit_),
                    "s_7": s_val_7,
                    "degree": G.degree[node],
                }

                if model_number == 2 and has_first_conspiracy:
                    row["first_conspiracy"] = r["first_conspiracy"]
                    row["cross_cluster"] = cross_cluster

                if model_number == 3:
                    row["cross_cluster"] = cross_cluster

                rows.append(row)
            t += step

    return pd.DataFrame(rows)


def _compute_s_sum(
    G: nx.Graph, node: str, consp: str, t: float, backtrack_time: int
) -> int:
    """Count unique neighbors who shared conspiracy within backtrack window."""
    count = 0
    for neighbor in G.neighbors(node):
        activations = G.nodes[neighbor].get(consp, [])
        activations = [time for time in activations if time <= t and time >= (t - backtrack_time)]
        if len(activations) > 0:
            count += 1
    return count
