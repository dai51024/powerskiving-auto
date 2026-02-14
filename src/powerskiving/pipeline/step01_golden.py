"""Step01 frozen golden selection gate."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from powerskiving.geom_kernel import GOLDEN_STATUS_OK, golden_select_s_rot
from powerskiving.json_canon import write_json

STEP01_REPORT_NAME = "cad_report_step01.json"


def _build_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_mm": float(cfg["module_mm"]),
        "z1": int(cfg["z1"]),
        "z2": int(cfg["z2"]),
        "pressure_angle_deg": float(cfg["pressure_angle_deg"]),
        "face_width_mm": float(cfg["face_width_mm"]),
        "center_distance_a_mm": float(cfg["center_distance_a_mm"]),
        "theta_tooth_center_rad": float(cfg["theta_tooth_center_rad"]),
        "dtheta_deadband_rad": float(cfg["dtheta_deadband_rad"]),
        "nu": int(cfg["Nu"]),
        "nv": int(cfg["Nv"]),
        "grid_u_min_mm": float(cfg["grid_u_min_mm"]),
        "grid_u_max_mm": float(cfg["grid_u_max_mm"]),
        "golden_tol_p95_mm": float(cfg["golden_tol_p95_mm"]),
        "golden_tol_max_mm": float(cfg["golden_tol_max_mm"]),
        "golden_dz_mm": float(cfg["golden_dz_mm"]),
        "golden_dz_max_mm": float(cfg["golden_dz_max_mm"]),
        "golden_min_points": int(cfg["golden_min_points"]),
        "golden_pitch_band_dr_mm": float(cfg["golden_pitch_band_dr_mm"]),
        "golden_ref_n": int(cfg["golden_ref_n"]),
    }


def run_step01(cfg: dict[str, Any], ctx: dict[str, Any]) -> Path:
    """Execute golden selection and always emit step01 CAD report."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STEP01_REPORT_NAME

    status = "failed"
    reason_code = "STEP01_EXCEPTION"
    message = "step01 failed before golden evaluation"
    payload: dict[str, Any] = {
        "s_rot_selected": None,
        "golden_p95_mm": None,
        "golden_max_mm": None,
        "z_mid_tool_mm": None,
        "golden_dz_used_mm": None,
        "kernel_status": None,
        "candidate_plus": None,
        "candidate_minus": None,
    }

    try:
        result = golden_select_s_rot(**_build_kwargs_from_config(cfg))
        status = "ok" if result.status == GOLDEN_STATUS_OK else "reject"
        reason_code = result.reason_code
        message = "step01 golden selection completed"
        payload = {
            "s_rot_selected": result.s_rot_selected,
            "golden_p95_mm": result.golden_p95_mm,
            "golden_max_mm": result.golden_max_mm,
            "z_mid_tool_mm": result.z_mid_tool_mm,
            "golden_dz_used_mm": result.golden_dz_used_mm,
            "kernel_status": result.status,
            "candidate_plus": asdict(result.candidate_plus),
            "candidate_minus": asdict(result.candidate_minus),
        }
        if status == "ok":
            ctx["s_rot_selected"] = result.s_rot_selected
            ctx["golden_p95_mm"] = result.golden_p95_mm
            ctx["golden_max_mm"] = result.golden_max_mm
    except Exception as exc:  # keep broad: cad_report must be written even on failures
        message = str(exc)

    cad_report = {
        "step_id": "step01_golden",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "exception_stacktrace": None,
        "ctx": dict(ctx),
        **payload,
    }
    write_json(report_path, cad_report)
    return report_path

