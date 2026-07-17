"""Log bootstrap interval fallback events."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any


_DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[2] / "logs" / "bootstrap_fallbacks.log"
)
_LOG_ENV_VAR = "CONSPIRACY_BOOTSTRAP_FALLBACK_LOG"


def _log_path() -> Path:
    override = os.environ.get(_LOG_ENV_VAR)
    if override:
        return Path(override)
    return _DEFAULT_LOG_PATH


def log_bootstrap_fallback(context: str, reason: str, **details: Any) -> None:
    """Append one bootstrap fallback event to the audit log."""
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        f"timestamp={datetime.now(timezone.utc).isoformat()}",
        f"context={context}",
        f"reason={reason}",
    ]
    fields.extend(f"{key}={value}" for key, value in details.items())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("Bootstrap interval fallback triggered | " + " | ".join(fields) + "\n")


__all__ = ["log_bootstrap_fallback"]
