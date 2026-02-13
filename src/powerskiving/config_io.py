"""Config loading with strict JSON parsing and version contract checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_VERSION = "1-1-1"
GEOM_SPEC_VERSION = "1-1-1"


class ConfigError(ValueError):
    """Raised when config violates frozen input contracts."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ConfigError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load strict JSON config and return parsed object + raw-bytes sha256 hex."""
    raw = Path(path).read_bytes()
    config_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError("config must be UTF-8") from exc

    try:
        cfg = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(cfg, dict):
        raise ConfigError("config must be a JSON object")

    spec_version = cfg.get("spec_version")
    geom_spec_version = cfg.get("geom_spec_version")
    if spec_version is None:
        raise ConfigError("spec_version is required")
    if geom_spec_version is None:
        raise ConfigError("geom_spec_version is required")
    if spec_version != SPEC_VERSION:
        raise ConfigError(
            f"spec_version mismatch: got {spec_version!r}, expected {SPEC_VERSION!r}"
        )
    if geom_spec_version != GEOM_SPEC_VERSION:
        raise ConfigError(
            "geom_spec_version mismatch: "
            f"got {geom_spec_version!r}, expected {GEOM_SPEC_VERSION!r}"
        )

    return cfg, config_sha256
