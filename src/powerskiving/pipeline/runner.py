"""Pipeline runner."""

from __future__ import annotations

from pathlib import Path

from .step00_io import run_step00


def run(config_path: str | Path) -> Path:
    """Run the pipeline from step00 and return step00 report path."""
    return run_step00(config_path)
