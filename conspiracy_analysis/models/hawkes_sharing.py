"""
Hawkes self-exciting point process for repeated sharing behaviour.

After a user first adopts a conspiracy, their subsequent sharing follows
a univariate Hawkes process:

    lambda(t) = mu + alpha * beta * sum_{t_i < t} exp(-beta * (t - t_i))

Parameters:
    mu    - spontaneous (baseline) sharing rate
    alpha - excitation strength (avg. extra shares triggered per event)
    beta  - decay rate (1/beta ~ mean duration of excitement in hours)

The process is fitted via maximum likelihood on empirical sharing sequences.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HawkesFitResult:
    """Auditable result of the pooled conditioned Hawkes fit."""

    mu: float
    alpha: float
    beta: float
    log_likelihood: float
    n_sequences: int
    n_repeat_sequences: int
    n_repeat_events: int
    converged: bool
    selected_start: Tuple[float, float, float]
    n_starts: int
    optimizer_message: str

    @property
    def params(self) -> Tuple[float, float, float]:
        return self.mu, self.alpha, self.beta


def _prepare_conditioned_sequences(
    sequences: List[List[float]],
    observation_ends: Optional[List[float]],
) -> List[Tuple[np.ndarray, float]]:
    """Validate and pair every adopter history with its observation end."""
    if observation_ends is not None and len(observation_ends) != len(sequences):
        raise ValueError("observation_ends must match sequences length")

    paired: List[Tuple[np.ndarray, float]] = []
    for idx, sequence in enumerate(sequences):
        if len(sequence) < 1:
            continue
        seq = np.asarray(sequence, dtype=float)
        if not np.all(np.isfinite(seq)) or np.any(np.diff(seq) < 0):
            raise ValueError(f"Sequence {idx} must contain finite sorted times")
        if not np.isclose(seq[0], 0.0):
            raise ValueError(f"Sequence {idx} must be relative to a first event at zero")
        T = float(observation_ends[idx]) if observation_ends is not None else float(seq[-1])
        if not np.isfinite(T) or T < seq[-1]:
            raise ValueError(
                f"Observation end for sequence {idx} must be finite and no earlier than its last event"
            )
        paired.append((seq, T))
    return paired


def _hawkes_log_likelihood_from_pairs(
    paired: List[Tuple[np.ndarray, float]],
    mu: float,
    alpha: float,
    beta: float,
) -> float:
    """Conditioned Hawkes log likelihood for validated histories."""
    if mu <= 0 or alpha <= 0 or alpha >= 1 or beta <= 0:
        return -np.inf

    total_ll = 0.0
    for seq, T in paired:
        integral = mu * T + np.sum(alpha * (1.0 - np.exp(-beta * (T - seq))))
        log_int = 0.0
        recurrence = 0.0
        for delta in np.diff(seq):
            recurrence = math.exp(-beta * delta) * (1.0 + recurrence)
            intensity = mu + alpha * beta * recurrence
            if intensity <= 0 or not np.isfinite(intensity):
                return -np.inf
            log_int += math.log(intensity)
        total_ll += log_int - integral
    return float(total_ll)


def _default_hawkes_starts(
    paired: List[Tuple[np.ndarray, float]],
    initial_guess: Tuple[float, float, float],
) -> List[Tuple[float, float, float]]:
    """Build a compact deterministic set of data scaled optimizer starts."""
    total_time = sum(T for _, T in paired)
    repeat_events = sum(len(seq) - 1 for seq, _ in paired)
    empirical_rate = repeat_events / total_time if total_time > 0 else initial_guess[0]
    empirical_rate = float(np.clip(empirical_rate, 1e-6, 0.5))

    candidates = [
        initial_guess,
        (empirical_rate, 0.2, 1.0 / 24.0),
        (empirical_rate, 0.5, 1.0 / 72.0),
        (empirical_rate, 0.8, 1.0 / 168.0),
        (empirical_rate * 0.25, 0.5, 1.0 / 24.0),
        (empirical_rate * 0.25, 0.8, 1.0 / 72.0),
        (empirical_rate * 4.0, 0.2, 1.0 / 72.0),
        (empirical_rate * 4.0, 0.5, 1.0 / 168.0),
    ]
    unique: List[Tuple[float, float, float]] = []
    for mu, alpha, beta in candidates:
        start = (
            float(np.clip(mu, 1e-6, 1.0)),
            float(np.clip(alpha, 1e-6, 0.99)),
            float(np.clip(beta, 1e-6, 5.0)),
        )
        if start not in unique:
            unique.append(start)
    return unique


def extract_sharing_sequences(
    G,
    conspiracies: Optional[List[str]] = None,
    study_end: Optional[float] = None,
) -> Tuple[List[List[float]], Optional[List[float]]]:
    """Extract relative sharing timestamps for each user-conspiracy pair.

    For each (user, conspiracy) where the user has shared at least once,
    creates a list of sharing times relative to the first share (t=0).

    Args:
        G: Network graph with conspiracy sharing times on nodes.
        conspiracies: List of conspiracy column names. If None, uses
            G.graph['conspiracy_cols'].
        study_end: Absolute time of study end. If provided, returns
            per-sequence observation window end times (study_end - adoption_time)
            so that the integral in MLE accounts for post-last-event silence.

    Returns:
        Tuple of (sequences, observation_ends):
            - sequences: List of sorted relative timestamp lists.
            - observation_ends: List of observation window durations (one per
              sequence), or None if study_end was not provided.
    """
    if conspiracies is None:
        conspiracies = G.graph["conspiracy_cols"]

    sequences = []
    observation_ends = [] if study_end is not None else None
    for node in G.nodes():
        for consp in conspiracies:
            sharings = G.nodes[node].get(consp, [])
            if len(sharings) > 0:
                sorted_times = sorted(sharings)
                start = sorted_times[0]
                relative = sorted(t - start for t in sorted_times)
                sequences.append(relative)
                if observation_ends is not None:
                    observation_ends.append(study_end - start)

    logger.info(f"Extracted {len(sequences)} sharing sequences")
    return sequences, observation_ends


def fit_hawkes(
    sequences: List[List[float]],
    initial_guess: Tuple[float, float, float] = (0.001, 0.5, 0.1),
    bounds: Optional[List[Tuple[float, float]]] = None,
    observation_ends: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Compatibility wrapper returning only the fitted parameter tuple."""
    return fit_hawkes_with_ll(
        sequences,
        initial_guess=initial_guess,
        bounds=bounds,
        observation_ends=observation_ends,
    ).params


def hawkes_intensity(
    t: float,
    history: List[float],
    mu: float,
    alpha: float,
    beta: float,
    memory_cutoff: Optional[float] = None,
) -> float:
    """Compute Hawkes intensity at time t given event history.

    Args:
        t: Current time.
        history: Sorted list of past event times (all < t).
        mu: Baseline rate.
        alpha: Excitation parameter.
        beta: Decay parameter.
        memory_cutoff: Only consider events within this time window.
            Default: 21/beta (captures >99.99% of kernel mass).

    Returns:
        Instantaneous intensity lambda(t).
    """
    if memory_cutoff is None:
        memory_cutoff = 21.0 / beta if beta > 0 else float("inf")

    sum_excitation = 0.0
    for t_event in reversed(history):
        dt = t - t_event
        if dt > memory_cutoff:
            break
        if dt > 0:
            sum_excitation += math.exp(-beta * dt)

    return mu + alpha * beta * sum_excitation


def hawkes_sharing_probability(
    t: float,
    history: List[float],
    mu: float,
    alpha: float,
    beta: float,
    memory_cutoff: Optional[float] = None,
) -> float:
    """Convert Hawkes intensity to sharing probability for a 1-hour step.

    Integrates the intensity over [t, t+1) and applies the complementary
    exponential: P(share) = 1 - exp(-integral).

    Args:
        t: Current time step.
        history: Sorted list of past sharing times.
        mu: Baseline rate.
        alpha: Excitation parameter.
        beta: Decay parameter.
        memory_cutoff: Hawkes memory window.

    Returns:
        Probability of sharing in [t, t+1).
    """
    if memory_cutoff is None:
        memory_cutoff = 21.0 / beta if beta > 0 else float("inf")

    # Integrate intensity over [t, t+1]:
    # integral = mu + sum_i alpha * (exp(-beta*(t-t_i)) - exp(-beta*(t+1-t_i)))
    #          = mu + alpha*(1-exp(-beta)) * sum_i exp(-beta*(t-t_i))
    kernel_integral = alpha * (1.0 - math.exp(-beta)) if beta > 0 else 0.0
    sum_excitation = 0.0
    for t_event in reversed(history):
        dt = t - t_event
        if dt > memory_cutoff:
            break
        if dt > 0:
            sum_excitation += math.exp(-beta * dt)

    integrated_intensity = mu + kernel_integral * sum_excitation
    return 1.0 - math.exp(-integrated_intensity)


def fit_hawkes_with_ll(
    sequences: List[List[float]],
    initial_guess: Tuple[float, float, float] = (0.001, 0.5, 0.1),
    bounds: Optional[List[Tuple[float, float]]] = None,
    observation_ends: Optional[List[float]] = None,
) -> HawkesFitResult:
    """Fit the conditioned Hawkes model from deterministic optimizer starts.

    Every adopter history contributes to the compensator. Histories with one
    event therefore inform the fitted event rate even though they contribute
    no repeat event log intensity term.
    """
    if bounds is None:
        bounds = [(1e-6, 1.0), (1e-6, 0.99), (1e-6, 5.0)]
    paired = _prepare_conditioned_sequences(sequences, observation_ends)
    if not paired:
        raise ValueError("At least one adopter history is required for Hawkes fitting")

    def neg_log_likelihood(params):
        ll = _hawkes_log_likelihood_from_pairs(paired, *params)
        return -ll if np.isfinite(ll) else 1e100

    starts = _default_hawkes_starts(paired, initial_guess)
    successful = []
    failures = []
    for start in starts:
        result = minimize(
            neg_log_likelihood,
            start,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            successful.append((float(-result.fun), start, result))
        else:
            failures.append(f"{start}: {result.message}")

    if not successful:
        details = "; ".join(failures)
        raise RuntimeError(f"No Hawkes optimizer start converged. {details}")

    ll, selected_start, best = max(successful, key=lambda item: item[0])
    mu, alpha, beta = (float(value) for value in best.x)
    n_repeat_sequences = sum(len(seq) > 1 for seq, _ in paired)
    n_repeat_events = sum(len(seq) - 1 for seq, _ in paired)
    fit = HawkesFitResult(
        mu=mu,
        alpha=alpha,
        beta=beta,
        log_likelihood=float(ll),
        n_sequences=len(paired),
        n_repeat_sequences=n_repeat_sequences,
        n_repeat_events=n_repeat_events,
        converged=True,
        selected_start=selected_start,
        n_starts=len(starts),
        optimizer_message=str(best.message),
    )
    logger.info(
        "Hawkes fit converged on %d histories from %d starts: "
        "mu=%.5f, alpha=%.5f, beta=%.5f, LL=%.1f",
        fit.n_sequences,
        fit.n_starts,
        fit.mu,
        fit.alpha,
        fit.beta,
        fit.log_likelihood,
    )
    return fit


def simulate_hawkes_sequence(
    mu: float,
    alpha: float,
    beta: float,
    t_max: float,
    rng: Optional[np.random.Generator] = None,
    initial_history: Optional[List[float]] = None,
) -> List[float]:
    """Simulate a Hawkes process realization via Ogata's thinning algorithm.

    Args:
        mu: Baseline rate.
        alpha: Excitation parameter (must be < 1 for stationarity).
        beta: Decay parameter.
        t_max: Maximum simulation time.
        rng: NumPy random generator. If None, creates a new one.
        initial_history: Optional list of event times to seed the history.
            These events influence the intensity for subsequent draws.
            Use [0.0] to condition on an adoption event at t=0.

    Returns:
        Sorted list of event times in [0, t_max], including initial_history.
    """
    if rng is None:
        rng = np.random.default_rng()

    events: List[float] = list(initial_history) if initial_history else []
    t = 0.0

    while t < t_max:
        # Upper bound: intensity right after the last event (post-jump).
        # Since the kernel decays, the intensity only decreases from here.
        if events:
            lam_star = mu + alpha * beta * sum(
                math.exp(-beta * (t - ti)) for ti in events if t - ti >= 0
            )
        else:
            lam_star = mu
        lam_star = max(lam_star, mu)

        # Draw candidate inter-arrival time
        u = rng.random()
        dt = -math.log(u) / lam_star
        t = t + dt

        if t >= t_max:
            break

        # Accept/reject
        lam_t = hawkes_intensity(t, events, mu, alpha, beta)
        if rng.random() < lam_t / lam_star:
            events.append(t)

    return events


# Goodness-of-fit diagnostics


def _hawkes_compensator_increments(
    seq: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Compute compensator increments Λ(tᵢ) - Λ(tᵢ₋₁) for one sequence.

    The compensator of the Hawkes process is:
        Λ(t) = μt + α Σ_{tᵢ<t} (1 - exp(-β(t - tᵢ)))

    By the time-rescaling theorem, if the model is correct the increments
    τᵢ = Λ(tᵢ) - Λ(tᵢ₋₁) should be i.i.d. Exp(1).
    """
    n = len(seq)
    if n < 2:
        return np.array([])

    dt = np.diff(seq)
    increments = np.zeros(n - 1)

    for i in range(n - 1):
        # Compensator increment between seq[i] and seq[i+1]
        delta = dt[i]
        # Baseline contribution
        inc = mu * delta
        # Excitation contribution from all events at or before seq[i]
        for j in range(i + 1):
            t_j = seq[j]
            # ∫_{seq[i]}^{seq[i+1]} α·β·exp(-β(s - t_j)) ds
            # = α [exp(-β(seq[i] - t_j)) - exp(-β(seq[i+1] - t_j))]
            inc += alpha * (
                math.exp(-beta * (seq[i] - t_j))
                - math.exp(-beta * (seq[i + 1] - t_j))
            )
        increments[i] = inc

    return increments


def _ks_distance_exp1(values: np.ndarray) -> float:
    """Compute Kolmogorov Smirnov distance from an Exp(1) distribution."""
    if len(values) == 0:
        return np.nan

    sorted_values = np.sort(values)
    n = len(sorted_values)
    expected_cdf = 1.0 - np.exp(-sorted_values)
    empirical_upper = np.arange(1, n + 1, dtype=float) / n
    empirical_lower = np.arange(0, n, dtype=float) / n
    return float(np.max(np.maximum(
        empirical_upper - expected_cdf,
        expected_cdf - empirical_lower,
    )))


def hawkes_residual_analysis(
    sequences: List[List[float]],
    mu: float,
    alpha: float,
    beta: float,
) -> Dict:
    """Time-rescaling residual analysis for Hawkes process.

    Computes compensator increments across all sequences and reports the
    Kolmogorov Smirnov distance from Exp(1) as a descriptive diagnostic.

    Args:
        sequences: List of event time sequences (2+ events each).
        mu: Fitted baseline rate.
        alpha: Fitted excitation parameter.
        beta: Fitted decay parameter.

    Returns:
        Dict with keys:
            - 'residuals': np.ndarray of compensator increments
            - 'ks_statistic': KS distance from Exp(1)
            - 'n_sequences': number of sequences used
            - 'n_residuals': total number of residuals
    """
    all_increments = []
    clean_seqs = [np.array(s, dtype=float) for s in sequences if len(s) > 1]

    for seq in clean_seqs:
        incs = _hawkes_compensator_increments(seq, mu, alpha, beta)
        if len(incs) > 0:
            all_increments.append(incs)

    residuals = np.concatenate(all_increments) if all_increments else np.array([])

    ks_stat = _ks_distance_exp1(residuals)

    logger.info(
        f"Hawkes residual analysis: {len(residuals)} residuals from "
        f"{len(clean_seqs)} sequences, KS distance={ks_stat:.4f}"
    )

    return {
        "residuals": residuals,
        "ks_statistic": ks_stat,
        "n_sequences": len(clean_seqs),
        "n_residuals": len(residuals),
    }


def _poisson_log_likelihood(
    sequences: List[np.ndarray],
    observation_ends: Optional[List[float]] = None,
) -> Tuple[float, int]:
    """Log-likelihood of a homogeneous Poisson process, conditioned on first event.

    Uses a single global MLE rate: λ̂ = N_total / T_total.
    Only events after the first (t>0) contribute to the log-intensity sum.
    """
    n_total = sum(len(seq) - 1 for seq in sequences)  # exclude first events
    t_total = 0.0
    for idx, seq in enumerate(sequences):
        T = observation_ends[idx] if observation_ends is not None else seq[-1]
        if T > 0:
            t_total += T
    if t_total <= 0 or n_total <= 0:
        return -1e10, 1
    lam = n_total / t_total
    # LL = Σ_{i>0} log(λ) - λ·T = N·log(λ) - λ·T_total
    total_ll = n_total * math.log(lam) - lam * t_total
    return total_ll, 1  # 1 parameter


def _inhomogeneous_poisson_log_likelihood(
    sequences: List[np.ndarray],
    observation_ends: Optional[List[float]] = None,
) -> Tuple[float, int]:
    """Log-likelihood of inhomogeneous Poisson: λ(t) = a·exp(-b·t) + c.

    Conditioned on first event: only events at t>0 contribute to the
    log-intensity sum. Fitted via MLE with L-BFGS-B.
    """
    def neg_ll(params):
        a, b, c = params
        if a <= 0 or b <= 0 or c <= 1e-10:
            return 1e10
        total = 0.0
        for idx, seq in enumerate(sequences):
            T = observation_ends[idx] if observation_ends is not None else seq[-1]
            # Integral from 0 to T
            integral = (a / b) * (1 - math.exp(-b * T)) + c * T
            # Log-intensity sum: skip first event (t=0), only events at t>0
            if len(seq) > 1:
                subsequent = seq[1:]
                log_int = np.sum(np.log(a * np.exp(-b * subsequent) + c))
            else:
                log_int = 0.0
            total += log_int - integral
        return -total

    total_time = sum(
        float(observation_ends[idx]) if observation_ends is not None else float(seq[-1])
        for idx, seq in enumerate(sequences)
    )
    repeat_events = sum(len(seq) - 1 for seq in sequences)
    rate = float(np.clip(repeat_events / total_time if total_time > 0 else 0.001, 1e-6, 1.0))
    starts = [
        (rate, 1.0 / 24.0, rate),
        (rate, 1.0 / 72.0, rate * 0.25),
        (rate * 4.0, 1.0 / 168.0, rate * 0.25),
        (rate * 0.25, 1.0 / 12.0, rate),
    ]
    successful = []
    failures = []
    for start in starts:
        result = minimize(
            neg_ll,
            x0=start,
            bounds=[(1e-6, 50), (1e-6, 10), (1e-6, 5)],
            method="L-BFGS-B",
        )
        if (
            result.success
            and np.isfinite(result.fun)
            and np.all(np.isfinite(result.x))
        ):
            successful.append(result)
        else:
            failures.append(f"{start}: {result.message}")
    if not successful:
        raise RuntimeError(
            "No inhomogeneous Poisson optimizer start converged. " + "; ".join(failures)
        )
    best = min(successful, key=lambda result: result.fun)
    return float(-best.fun), 3


def _hawkes_log_likelihood(
    sequences: List[np.ndarray],
    mu: float,
    alpha: float,
    beta: float,
    observation_ends: Optional[List[float]] = None,
) -> Tuple[float, int]:
    """Log-likelihood of fitted Hawkes process, conditioned on first event.

    Skips log(λ(t_0)) = log(μ) for the first event at t=0, since all
    sequences are defined relative to the triggering adoption event.
    The integral still starts from 0.
    """
    paired = _prepare_conditioned_sequences(sequences, observation_ends)
    return _hawkes_log_likelihood_from_pairs(paired, mu, alpha, beta), 3


def hawkes_model_comparison(
    sequences: List[List[float]],
    fit_or_mu,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    observation_ends: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compare Hawkes vs. simpler point process models via AIC/BIC.

    Models compared:
        1. Homogeneous Poisson: λ(t) = const (1 param)
        2. Inhomogeneous Poisson: λ(t) = a·exp(-bt) + c (3 params)
        3. Hawkes: λ(t) = μ + αβΣexp(-β(t-tᵢ)) (3 params)

    Args:
        sequences: List of event time sequences.
        fit_or_mu: A HawkesFitResult or a fitted Hawkes baseline rate.
        alpha, beta: Fitted Hawkes parameters for compatibility when fit_or_mu
            is a numeric baseline rate.
        observation_ends: Optional observation window end times for each
            sequence. If None, uses last event time as T.

    Returns:
        DataFrame with columns: model, log_likelihood, n_params, AIC, BIC.
    """
    if isinstance(fit_or_mu, HawkesFitResult):
        mu, fitted_alpha, fitted_beta = fit_or_mu.params
    else:
        if alpha is None or beta is None:
            raise ValueError("alpha and beta are required with a numeric mu")
        mu, fitted_alpha, fitted_beta = float(fit_or_mu), float(alpha), float(beta)

    paired = _prepare_conditioned_sequences(sequences, observation_ends)
    clean_seqs = [seq for seq, _ in paired]
    clean_obs_ends = [T for _, T in paired]
    n_sequences = len(clean_seqs)
    n_repeat_events = sum(len(seq) - 1 for seq in clean_seqs)

    results = []
    for name, func in [
        ("Homogeneous Poisson", lambda: _poisson_log_likelihood(clean_seqs, clean_obs_ends)),
        ("Inhomogeneous Poisson", lambda: _inhomogeneous_poisson_log_likelihood(clean_seqs, clean_obs_ends)),
        (
            "Hawkes",
            lambda: _hawkes_log_likelihood(
                clean_seqs,
                mu,
                fitted_alpha,
                fitted_beta,
                clean_obs_ends,
            ),
        ),
    ]:
        ll, k = func()
        aic = 2 * k - 2 * ll
        bic = k * math.log(n_sequences) - 2 * ll if n_sequences >= 2 else np.nan
        results.append({
            "model": name,
            "log_likelihood": ll,
            "n_params": k,
            "n_sequences": n_sequences,
            "n_repeat_events": n_repeat_events,
            "AIC": aic,
            "BIC": bic,
        })

    df = pd.DataFrame(results)
    logger.info(f"Model comparison:\n{df.to_string(index=False)}")
    return df


def hawkes_interevent_acf(
    sequences: List[List[float]],
    n_lags: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation of inter-event times across all sequences.

    Computes within-sequence ACFs and averages them (weighted by the number
    of valid lag pairs), avoiding artificial cross-sequence boundary effects.

    Positive autocorrelation at short lags indicates clustering (self-excitation),
    which a Poisson process cannot produce.

    Args:
        sequences: List of event time sequences.
        n_lags: Number of lags to compute.

    Returns:
        Tuple of (lags array, acf values array).
    """
    # Collect within-sequence inter-event times, centered per-sequence
    per_seq_dt = []
    for seq in sequences:
        arr = np.array(seq, dtype=float)
        if len(arr) > 2:
            per_seq_dt.append(np.diff(arr))

    if not per_seq_dt:
        return np.arange(1, n_lags + 1), np.full(n_lags, np.nan)

    total_dt = sum(len(dt) for dt in per_seq_dt)
    if total_dt < n_lags + 1:
        return np.arange(1, n_lags + 1), np.full(n_lags, np.nan)

    # Pooled within-sequence variance (center each sequence by its own mean
    # to avoid spurious positive ACF from between-sequence heterogeneity)
    ss_within = 0.0
    n_within = 0
    centered_seqs = []
    for dt in per_seq_dt:
        seq_mean = np.mean(dt)
        centered = dt - seq_mean
        centered_seqs.append(centered)
        ss_within += np.sum(centered ** 2)
        n_within += len(centered)

    var_dt = ss_within / n_within if n_within > 0 else 0.0
    if var_dt == 0:
        return np.arange(1, n_lags + 1), np.zeros(n_lags)

    # Compute within-sequence ACF, weighted average across sequences
    acf = np.zeros(n_lags)
    counts = np.zeros(n_lags)

    for centered in centered_seqs:
        n = len(centered)
        for lag in range(1, n_lags + 1):
            if lag < n:
                pairs = centered[:-lag] * centered[lag:]
                acf[lag - 1] += np.sum(pairs)
                counts[lag - 1] += len(pairs)

    # Normalize by count and within-sequence variance
    valid = counts > 0
    acf[valid] = acf[valid] / (counts[valid] * var_dt)
    acf[~valid] = np.nan

    return np.arange(1, n_lags + 1), acf


def hawkes_residual_acf(
    sequences: List[List[float]],
    mu: float,
    alpha: float,
    beta: float,
    n_lags: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autocorrelation of rescaled gaps centered on their Exp(1) mean."""
    centered_sequences = []
    for sequence in sequences:
        increments = _hawkes_compensator_increments(
            np.asarray(sequence, dtype=float), mu, alpha, beta
        )
        if len(increments) > 0:
            centered_sequences.append(increments - 1.0)

    lags = np.arange(1, n_lags + 1)
    if not centered_sequences:
        return lags, np.full(n_lags, np.nan)

    pooled = np.concatenate(centered_sequences)
    variance = float(np.mean(pooled ** 2))
    if variance == 0:
        return lags, np.zeros(n_lags)

    acf = np.full(n_lags, np.nan)
    for lag in lags:
        products = [
            values[:-lag] * values[lag:]
            for values in centered_sequences
            if len(values) > lag
        ]
        if products:
            acf[lag - 1] = float(np.mean(np.concatenate(products)) / variance)
    return lags, acf


def hawkes_residual_acf_envelope(
    observation_ends: List[float],
    mu: float,
    alpha: float,
    beta: float,
    n_lags: int = 20,
    n_simulations: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric uncertainty envelope using all empirical observation windows."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    windows = np.asarray(observation_ends, dtype=float)
    if len(windows) == 0 or np.any(~np.isfinite(windows)) or np.any(windows < 0):
        raise ValueError("observation_ends must contain finite nonnegative values")

    seed_sequence = np.random.SeedSequence(seed)

    def simulate_one(child_seed):
        rng = np.random.default_rng(child_seed)
        simulated_sequences = [
            simulate_hawkes_sequence(
                mu,
                alpha,
                beta,
                t_max=float(window),
                rng=rng,
                initial_history=[0.0],
            )
            for window in windows
        ]
        return hawkes_residual_acf(
            simulated_sequences, mu, alpha, beta, n_lags=n_lags
        )[1]

    simulated_acfs = np.asarray(Parallel(n_jobs=n_jobs)(
        delayed(simulate_one)(child_seed)
        for child_seed in seed_sequence.spawn(n_simulations)
    ))
    lower = np.nanpercentile(simulated_acfs, 2.5, axis=0)
    upper = np.nanpercentile(simulated_acfs, 97.5, axis=0)
    return np.arange(1, n_lags + 1), lower, upper
