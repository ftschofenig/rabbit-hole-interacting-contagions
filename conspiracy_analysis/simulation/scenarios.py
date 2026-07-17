"""
Pre-defined simulation scenarios.

Each factory returns a ScenarioConfig that modifies ABM behavior
for counterfactual analysis.
"""

from conspiracy_analysis.simulation.config import ScenarioConfig


def baseline() -> ScenarioConfig:
    """Standard simulation with all mechanisms active."""
    return ScenarioConfig(
        name="baseline",
        description="Full model with peer influence and interacting contagions.",
    )


def no_temporal_effects() -> ScenarioConfig:
    """Counterfactual: remove temporal acceleration in baseline hazards.

    All models use Model 1's constant baseline hazard instead of the
    Weibull baselines fitted for Models 2+.  Must be paired with
    ``override_baselines_to_linear()`` on the SimulationConfig.
    """
    return ScenarioConfig(
        name="no_temporal_effects",
        description="All models use Model 1's constant baseline (no Weibull decay).",
    )


def quarantine(hours: float = 24.0) -> ScenarioConfig:
    """Counterfactual: read-only lockout for `hours` after first adoption.

    After adopting any conspiracy at time T, the user is placed in a
    read-only quarantine while current time is less than T + hours. Activity
    resumes exactly at T + hours. During the lockout:

    1. They cannot adopt any new conspiracy (Phase A in engine.py).
    2. They cannot generate new sharing events for any conspiracy
       they hold (Phase B in engine.py — Hawkes re-sharing is suppressed
       for all of their conspiracies, not just the one that triggered
       the lock).

    Their existing shares — including the initial adoption share at
    time T — remain visible to neighbours throughout the quarantine.
    Other users' exposure (s7) calculations still count posts the
    locked user made before the lock started; only new posts are
    blocked.

    This mirrors platform read-only lockouts deployed during the
    COVID-19 information environment: Twitter's 12h first-strike lock,
    YouTube's 7-day strike freeze, Reddit and Discord temporary posting
    restrictions. The user is muted, but their existing content stays
    up.

    Seed adoptions placed by `seed_nodes` do not trigger the quarantine
    clock. Only organic adoptions made through Phase A's competing-risks
    draw set the lock window — a user who is purely a seed for some
    conspiracy can continue to generate Hawkes re-shares and adopt other
    conspiracies through Phase A normally until they make their first
    organic adoption.
    """
    return ScenarioConfig(
        name=f"quarantine_{int(hours)}h",
        description=f"Read-only lockout for {int(hours)}h after first adoption.",
        quarantine_hours=hours,
    )


def reputation_nudge(rejection_rate: float = 0.5) -> ScenarioConfig:
    """Counterfactual: users are warned before adopting a conspiracy.

    When a user is about to adopt a conspiracy through Phase A's
    competing-risks draw (peer influence), an independent Bernoulli
    draw with success probability `rejection_rate` decides whether they
    "heed the warning". On a hit, the adoption is replaced with
    permanent immunity to that specific conspiracy: the user is added
    to the agent's `_immune` set and will never adopt it again,
    regardless of how many neighbours later do. Immunity is per-(user,
    conspiracy) and monotonic — there is no decay and no second chance
    for the conspiracy that was rejected, but unrelated conspiracies
    remain eligible.

    Only organic (Cox-driven) adoptions face the nudge — `seed_nodes`
    does not consult `nudge_rejection_rate`. A seed user can still be
    nudged on a *different* conspiracy if they later reach Phase A's
    competing-risks draw for it.
    """
    return ScenarioConfig(
        name=f"nudge_{int(rejection_rate * 100)}pct",
        description=f"Reputation nudge: {rejection_rate * 100:.0f}% of organic adoptions become permanent immunity (seeds exempt).",
        nudge_rejection_rate=rejection_rate,
    )


def content_moderation(block_rate: float = 0.9) -> ScenarioConfig:
    """Counterfactual: content moderation blocks a fraction of conspiracy shares.

    Each re-share event has probability block_rate of being flagged and removed
    from the network. Blocked shares still contribute to the poster's own
    Hawkes self-excitation (they see their own post) but are invisible to
    neighbors (not counted in s7 exposure).

    Each tweet from an organic adoption, including its initial adoption
    share, is subject to the same detection probability. Seed users'
    initial adoption shares are exempt from moderation; only their
    subsequent Hawkes re-shares (via `apply_shares`) face the block rate.
    """
    return ScenarioConfig(
        name=f"moderation_{int(block_rate * 100)}pct",
        description=f"Content moderation: {block_rate * 100:.0f}% of conspiracy re-shares blocked.",
        sharing_block_rate=block_rate,
    )
