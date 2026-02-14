"""Pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from powerskiving.config_io import load_config

from .step00_io import run_step00
from .step01_golden import run_step01
from .step02_grid import run_step02
from .step03A_boundary import run_step03A
from .step03B_view_mesh import run_step03B
from .step05_cutting_edge import run_step05
from .step07_sections_dxf import run_step07


@dataclass
class RunnerResult:
    status: str
    reason_code: str
    output_dir: Path
    ctx: dict[str, Any] = field(default_factory=dict)
    step00_report_path: Path | None = None
    step01_report_path: Path | None = None
    step02_report_path: Path | None = None
    step03A_report_path: Path | None = None
    step03B_report_path: Path | None = None
    step05_report_path: Path | None = None
    step07_report_path: Path | None = None

    def __getattr__(self, name: str) -> Any:
        """
        Backward-compatibility shim for callers that treated run() as Path.

        Delegate unknown attributes to step00 report path when available.
        """
        if self.step00_report_path is not None:
            return getattr(self.step00_report_path, name)
        raise AttributeError(name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_from_report(report: dict[str, Any]) -> tuple[str, str]:
    report_status = str(report.get("status", "failed"))
    reason_code = str(report.get("reason_code") or report.get("error_code") or "UNKNOWN")
    if report_status == "ok":
        return "ok", "ok"
    if report_status == "reject":
        return "reject", reason_code
    return "failed", reason_code


def run(config_path: str | Path) -> RunnerResult:
    """Run step00->step07 in order and stop on the first non-ok step."""
    step00_report_path = run_step00(config_path)
    step00_report = _read_json(step00_report_path)
    output_dir = step00_report_path.parent
    result = RunnerResult(
        status="failed",
        reason_code="STEP00_EXCEPTION",
        output_dir=output_dir,
        ctx={},
        step00_report_path=step00_report_path,
    )

    result.status, result.reason_code = _status_from_report(step00_report)
    if result.status != "ok":
        return result

    cfg, _ = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    result.output_dir = output_dir

    step01_report_path = run_step01(cfg, result.ctx)
    result.step01_report_path = step01_report_path
    step01_report = _read_json(step01_report_path)
    result.status, result.reason_code = _status_from_report(step01_report)
    if result.status != "ok":
        return result

    step02_report_path = run_step02(cfg, result.ctx)
    result.step02_report_path = step02_report_path
    step02_report = _read_json(step02_report_path)
    result.status, result.reason_code = _status_from_report(step02_report)
    if result.status != "ok":
        return result

    step03A_report_path = run_step03A(cfg, result.ctx)
    result.step03A_report_path = step03A_report_path
    step03A_report = _read_json(step03A_report_path)
    result.status, result.reason_code = _status_from_report(step03A_report)
    if result.status != "ok":
        return result

    step03B_report_path = run_step03B(cfg, result.ctx)
    result.step03B_report_path = step03B_report_path
    step03B_report = _read_json(step03B_report_path)
    result.status, result.reason_code = _status_from_report(step03B_report)
    if result.status != "ok":
        return result

    step05_report_path = run_step05(cfg, result.ctx)
    result.step05_report_path = step05_report_path
    step05_report = _read_json(step05_report_path)
    result.status, result.reason_code = _status_from_report(step05_report)
    if result.status != "ok":
        return result

    step07_report_path = run_step07(cfg, result.ctx)
    result.step07_report_path = step07_report_path
    step07_report = _read_json(step07_report_path)
    result.status, result.reason_code = _status_from_report(step07_report)
    return result
