"""Step00 I/O preparation."""

from __future__ import annotations

import shutil
from pathlib import Path

from powerskiving.config_io import ConfigError, load_config
from powerskiving.json_canon import write_json

STEP00_REPORT_NAME = "cad_report_step00.json"


def _clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_step00(config_path: str | Path) -> Path:
    """Load config, clean output dir, and always emit step00 CAD report."""
    cfg = None
    output_dir: Path | None = None
    status = "failed"
    error_code = "io_missing_or_corrupt"
    message = "step00 failed before config load"

    try:
        cfg, _config_sha256 = load_config(config_path)
        output_dir = Path(cfg["output_dir"])
        _clean_output_dir(output_dir)
        status = "ok"
        error_code = "ok"
        message = "step00 io preparation completed"
    except Exception as exc:  # keep broad: cad_report must be written even on failures
        if cfg is not None and isinstance(cfg.get("output_dir"), str):
            output_dir = Path(cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path("out") / "step00_failed"
            output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(exc, ConfigError):
            error_code = "config_error"
        else:
            error_code = "io_missing_or_corrupt"
        message = str(exc)

    cad_report = {
        "step_id": "step00_io",
        "status": status,
        "error_code": error_code,
        "message": message,
        "exception_stacktrace": None,
    }
    report_path = output_dir / STEP00_REPORT_NAME
    write_json(report_path, cad_report)
    return report_path
