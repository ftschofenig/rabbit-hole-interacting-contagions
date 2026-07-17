"""Reproducibility helpers for simulation artifacts."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib


def canonicalize_json_value(value: Any) -> Any:
    """Return the strict JSON representation used by simulation manifests."""
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)
    return json.loads(serialized)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_package_versions(distributions: Iterable[str]) -> Dict[str, str]:
    """Collect installed distribution versions for an explicit package list."""
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def atomic_joblib_dump(value: Any, path: Path, **dump_kwargs) -> Path:
    """Write a joblib artifact to a temporary file and atomically replace it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary, **dump_kwargs)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_json_dump(value: Dict[str, Any], path: Path) -> Path:
    """Write a JSON document and atomically replace the destination."""
    canonical_value = canonicalize_json_value(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                canonical_value,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_manifest_backed_simulation_bundle(
    path: Path,
    required_version: str = "simulation_correction_v2",
) -> Dict[str, Any]:
    """Load a simulation bundle only when its embedded and sidecar manifests match."""
    target = Path(path)
    bundle = joblib.load(target)
    if not isinstance(bundle, dict) or "manifest" not in bundle:
        raise ValueError(f"Simulation bundle {target} has no embedded manifest")
    manifest = bundle["manifest"]
    if not isinstance(manifest, dict):
        raise ValueError(f"Simulation bundle {target} has an invalid manifest")
    manifest = canonicalize_json_value(manifest)
    if manifest.get("manifest_version") != required_version:
        raise ValueError(
            f"Simulation bundle {target} does not use {required_version}"
        )
    sidecar_path = target.with_name(f"{target.stem}_manifest.json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Simulation manifest is missing: {sidecar_path}")
    with sidecar_path.open("r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    sidecar = canonicalize_json_value(sidecar)
    if sidecar != manifest:
        raise ValueError("Embedded and sidecar simulation manifests do not match")
    bundle["manifest"] = manifest
    return bundle
