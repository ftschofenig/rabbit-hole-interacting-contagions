"""
Statistical analysis for semantic barrier effects and transition dynamics.

Implements the cognitive barrier analysis (within-cluster vs. cross-cluster
transition times) and the settler effect analysis (pre-jump, jump, post-jump).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform
from tqdm import tqdm

from conspiracy_analysis import BOT_SCORE_THRESHOLD
from conspiracy_analysis.utils.helpers import passes_bot_filter

logger = logging.getLogger(__name__)


def bootstrap_median_ci(
    data: List[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrapped median with asymmetric confidence intervals.

    Args:
        data: List of observed values.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (median, lower_error, upper_error) where errors are
        relative to the median (for matplotlib errorbar format).
    """
    arr = np.array(data)
    n = len(arr)

    if n < 2:
        val = np.median(arr) if n > 0 else 0
        return val, 0, 0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, (n_boot, n))
    boot_medians = np.median(arr[indices], axis=1)

    lower = np.percentile(boot_medians, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_medians, (1 + ci) / 2 * 100)
    median = np.median(arr)

    return median, median - lower, upper - median


def extract_user_timelines(
    G,
    cluster_assignments: Dict[str, int],
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
    min_conspiracies: int = 2,
) -> List[Dict]:
    """Extract adoption timelines for users with multiple conspiracy adoptions.

    Args:
        G: Network graph with conspiracy activation times.
        cluster_assignments: Conspiracy -> cluster_id mapping.
        bot_score_threshold: Bot score threshold for filtering.
        mode: 'HUMAN' or 'BOT'.
        min_conspiracies: Minimum distinct conspiracies for inclusion.

    Returns:
        List of dicts, each with 'id' and conspiracy -> first_adoption_time pairs.
    """
    conspiracies = list(cluster_assignments.keys())
    timelines = []

    for node in tqdm(G.nodes(), desc="Extracting timelines"):
        if not passes_bot_filter(G, node, bot_score_threshold, mode):
            continue

        row = {"id": node}
        valid_count = 0
        for c in conspiracies:
            acts = G.nodes[node].get(c, [])
            if acts:
                val = np.min(acts)
                if not np.isnan(val):
                    row[c] = val
                    valid_count += 1

        if valid_count >= min_conspiracies:
            timelines.append(row)

    return timelines


def compute_semantic_barrier_analysis(
    user_timelines: List[Dict],
    cluster_assignments: Dict[str, int],
) -> Dict[str, List[float]]:
    """Compute within-cluster vs. cross-cluster transition times.

    Tests the hypothesis that users transition faster within semantic
    clusters than across them.

    Args:
        user_timelines: Output from extract_user_timelines().
        cluster_assignments: Conspiracy -> cluster_id mapping.

    Returns:
        Dict with 'within_cluster' and 'between_clusters' keys,
        each mapping to a list of transition times (hours).
    """
    gaps = {"within_cluster": [], "between_clusters": []}

    for row in user_timelines:
        timeline = [(c, t) for c, t in row.items() if c != "id"]
        timeline.sort(key=lambda x: x[1])

        if not timeline:
            continue

        seen_clusters = {cluster_assignments[timeline[0][0]]}

        for i in range(len(timeline) - 1):
            curr_c = timeline[i][0]
            next_c = timeline[i + 1][0]
            gap = abs(timeline[i + 1][1] - timeline[i][1])

            curr_cluster = cluster_assignments[curr_c]
            next_cluster = cluster_assignments[next_c]

            if curr_cluster == next_cluster:
                gaps["within_cluster"].append(gap)
            elif next_cluster not in seen_clusters:
                gaps["between_clusters"].append(gap)

            seen_clusters.add(next_cluster)

    logger.info(
        f"Barrier analysis: {len(gaps['within_cluster'])} within-cluster gaps, "
        f"{len(gaps['between_clusters'])} between-cluster gaps"
    )
    return gaps


def compute_settler_effect(
    user_timelines: List[Dict],
    cluster_assignments: Dict[str, int],
    new_clusters_only: bool = True,
) -> Dict[str, List[float]]:
    """Compute the settler effect: pre-jump, jump, and post-jump transition times.

    Looks for sequences where a user transitions:
    1. Within cluster A (pre-jump)
    2. From cluster A to cluster B (the jump)
    3. Within cluster B (post-jump / settling)

    Args:
        user_timelines: Output from extract_user_timelines().
        cluster_assignments: Conspiracy -> cluster_id mapping.
        new_clusters_only: If True, only include A,A,B,B patterns where
            cluster B has not been previously visited by that user.
            If False (default), all A,A,B,B patterns are included
            regardless of prior cluster visits.

    Returns:
        Dict with keys 'pre_jump', 'the_jump', 'post_jump',
        each mapping to a list of transition times (hours).
    """
    settler_gaps = {"pre_jump": [], "the_jump": [], "post_jump": []}

    for row in user_timelines:
        timeline = [(c, t) for c, t in row.items() if c != "id"]
        timeline.sort(key=lambda x: x[1])

        if len(timeline) < 4:
            continue

        if new_clusters_only:
            seen_clusters = {cluster_assignments[timeline[0][0]]}

        for i in range(1, len(timeline) - 2):
            if new_clusters_only:
                seen_clusters.add(cluster_assignments[timeline[i][0]])

            clusters = [
                cluster_assignments[timeline[j][0]] for j in range(i - 1, i + 3)
            ]

            is_stable_before = clusters[0] == clusters[1]
            is_switching = clusters[1] != clusters[2]
            is_stable_after = clusters[2] == clusters[3]

            if is_stable_before and is_switching and is_stable_after:
                if new_clusters_only and clusters[2] in seen_clusters:
                    continue
                settler_gaps["pre_jump"].append(timeline[i][1] - timeline[i - 1][1])
                settler_gaps["the_jump"].append(timeline[i + 1][1] - timeline[i][1])
                settler_gaps["post_jump"].append(timeline[i + 2][1] - timeline[i + 1][1])

    mode_label = "new clusters only" if new_clusters_only else "all patterns"
    logger.info(
        f"Settler effect ({mode_label}): {len(settler_gaps['pre_jump'])} sequences found"
    )
    return settler_gaps


def test_barrier_significance(
    gaps: Dict[str, List[float]],
) -> Dict[str, float]:
    """Run Mann-Whitney U test comparing within-cluster and between-cluster gaps.

    Args:
        gaps: Output from compute_semantic_barrier_analysis().

    Returns:
        Dict with 'statistic' and 'p_value' from the test.
    """
    stat, p = stats.mannwhitneyu(
        gaps["within_cluster"],
        gaps["between_clusters"],
        alternative="less",
    )
    return {"statistic": stat, "p_value": p}


def test_settler_significance(
    settler_gaps: Dict[str, List[float]],
) -> Dict:
    """Run Friedman test and pairwise Wilcoxon signed rank tests on settler gaps.

    Uses Friedman test (nonparametric repeated measures) because the three
    groups are matched triads from the same user sequences, not independent
    samples. Pairwise comparisons use Wilcoxon signed rank tests with
    Holm Bonferroni correction.

    Args:
        settler_gaps: Output from compute_settler_effect().

    Returns:
        Dict with 'friedman' (statistic + p_value) and 'pairwise' test
        results (each with statistic, p_value_raw, p_value_corrected).
    """
    groups = [settler_gaps["pre_jump"], settler_gaps["the_jump"], settler_gaps["post_jump"]]
    friedman_stat, friedman_p = stats.friedmanchisquare(*groups)

    pair_names = [
        ("pre_jump_vs_jump", 0, 1),
        ("pre_jump_vs_post_jump", 0, 2),
        ("jump_vs_post_jump", 1, 2),
    ]

    # Collect raw pairwise results
    raw_results = []
    for name, i, j in pair_names:
        stat, p = stats.wilcoxon(groups[i], groups[j], alternative="two-sided")
        raw_results.append((name, stat, p))

    # Apply Holm Bonferroni correction with monotonic adjusted p values.
    n_tests = len(raw_results)
    sorted_by_p = sorted(range(n_tests), key=lambda idx: raw_results[idx][2])
    corrected_p = [0.0] * n_tests
    running_max = 0.0
    for rank, idx in enumerate(sorted_by_p):
        adjusted = min(raw_results[idx][2] * (n_tests - rank), 1.0)
        running_max = max(running_max, adjusted)
        corrected_p[idx] = running_max

    pair_results = {}
    for idx, (name, stat, p_raw) in enumerate(raw_results):
        pair_results[name] = {
            "statistic": stat,
            "p_value_raw": p_raw,
            "p_value_corrected": corrected_p[idx],
        }

    return {
        "friedman": {"statistic": friedman_stat, "p_value": friedman_p},
        "pairwise": pair_results,
    }


def compute_coadoption_matrix(
    G,
    conspiracies: List[str],
    bot_score_threshold: float = BOT_SCORE_THRESHOLD,
    mode: str = "HUMAN",
) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity of user adoption sets.

    For each pair of conspiracies (i, j), Jaccard similarity is defined as
    |users_i ∩ users_j| / |users_i ∪ users_j|.

    Args:
        G: Network graph with conspiracy activation times on nodes.
        conspiracies: List of conspiracy column names.
        bot_score_threshold: Bot score threshold for filtering.
        mode: 'HUMAN' or 'BOT'.

    Returns:
        Symmetric DataFrame (len(conspiracies) × len(conspiracies)) of
        Jaccard similarities. Diagonal is 1.0.
    """
    # Build set of adopting users per conspiracy
    adopters = {c: set() for c in conspiracies}
    for node in G.nodes():
        if not passes_bot_filter(G, node, bot_score_threshold, mode):
            continue
        for c in conspiracies:
            acts = G.nodes[node].get(c, [])
            if acts and not all(np.isnan(a) for a in acts):
                adopters[c].add(node)

    n = len(conspiracies)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            set_i = adopters[conspiracies[i]]
            set_j = adopters[conspiracies[j]]
            union = len(set_i | set_j)
            if union == 0:
                jaccard = 0.0
            else:
                jaccard = len(set_i & set_j) / union
            mat[i, j] = jaccard
            mat[j, i] = jaccard

    logger.info(
        f"Co-adoption matrix: {n} conspiracies, "
        f"mean Jaccard = {mat[np.triu_indices(n, k=1)].mean():.4f}"
    )
    return pd.DataFrame(mat, index=conspiracies, columns=conspiracies)


def mantel_test(
    dist_matrix_1: pd.DataFrame,
    dist_matrix_2: pd.DataFrame,
    n_permutations: int = 9999,
    method: str = "spearman",
    seed: int = 42,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Permutation-based Mantel test between two distance matrices.

    Computes the correlation between the upper-triangle entries of two
    symmetric distance matrices and assesses significance by permuting
    rows/columns of one matrix.

    Args:
        dist_matrix_1: First (dis)similarity matrix (symmetric DataFrame).
            Diagonals need not be zero — they are ignored. The Mantel
            statistic is computed over strictly upper-triangle entries
            (``i < j``) only.
        dist_matrix_2: Second (dis)similarity matrix (symmetric DataFrame).
            Must share the same index/column labels as dist_matrix_1.
            Diagonals are likewise ignored.
        n_permutations: Number of random permutations for the null.
        method: Correlation method ('spearman' or 'pearson').
        seed: Random seed for reproducibility.
        alternative: Direction of the test. One of:
            - 'two-sided' (default): test H1 |r| > |r_observed|. Use when
              direction is not pre-specified.
            - 'greater': test H1 r > r_observed. Use when a positive
              correlation is expected a priori (e.g. both matrices are
              distance matrices and pairs close in one are expected to be
              close in the other).
            - 'less': test H1 r < r_observed. Use when a negative
              correlation is expected a priori.

    Returns:
        Dict with 'correlation' (observed), 'p_value', 'n_permutations',
        and 'alternative'.

    Raises:
        ValueError: If fewer than 3 shared labels, or an unknown
            'alternative' value.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less'; "
            f"got {alternative!r}"
        )

    # Align matrices to shared label order
    shared = dist_matrix_1.index.intersection(dist_matrix_2.index)
    if len(shared) < 3:
        raise ValueError(
            f"Need at least 3 shared labels, got {len(shared)}"
        )
    m1 = dist_matrix_1.loc[shared, shared].values.copy()
    m2 = dist_matrix_2.loc[shared, shared].values.copy()

    # Set diagonal values to zero because the Mantel statistic uses only
    # strictly upper triangle entries.
    np.fill_diagonal(m1, 0)
    np.fill_diagonal(m2, 0)

    # Extract upper triangle (condensed form)
    v1 = squareform(m1, checks=False)
    v2 = squareform(m2, checks=False)

    if method == "spearman":
        corr_func = lambda a, b: stats.spearmanr(a, b).statistic
    else:
        corr_func = lambda a, b: stats.pearsonr(a, b).statistic

    observed = corr_func(v1, v2)

    # Permutation test: shuffle rows/columns of m2
    rng = np.random.default_rng(seed)
    n = len(shared)
    count_extreme = 0

    for _ in range(n_permutations):
        perm = rng.permutation(n)
        m2_perm = m2[np.ix_(perm, perm)]
        v2_perm = squareform(m2_perm, checks=False)
        perm_corr = corr_func(v1, v2_perm)
        if alternative == "greater":
            if perm_corr >= observed:
                count_extreme += 1
        elif alternative == "less":
            if perm_corr <= observed:
                count_extreme += 1
        else:  # two-sided
            if abs(perm_corr) >= abs(observed):
                count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    logger.info(
        f"Mantel test ({method}, {alternative}): r = {observed:.4f}, "
        f"p = {p_value:.4f} ({n_permutations} permutations)"
    )
    return {
        "correlation": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "alternative": alternative,
    }
