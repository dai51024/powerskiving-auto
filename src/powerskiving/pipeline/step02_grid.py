"""Step02 raw conjugate grid generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from powerskiving.deterministic import fixed, percentile
from powerskiving.geom_kernel import ToolConjugateGridRawPoint, tool_conjugate_grid_raw
from powerskiving.json_canon import write_json

STEP02_REPORT_NAME = "cad_report_step02.json"
PLUS_RAW_CSV_NAME = "tool_conjugate_grid_plus_raw.csv"
MINUS_RAW_CSV_NAME = "tool_conjugate_grid_minus_raw.csv"
_RAW_HEADER = (
    "iu,iv,u_mm,v_mm,x_mm,y_mm,z_mm,nx,ny,nz,theta1_rad,theta2_rad,residual_abs,valid,reason_code"
)


def _write_raw_csv(
    path: Path,
    points: Iterable[ToolConjugateGridRawPoint],
    *,
    mm_digits: int,
    rad_digits: int,
    unitless_digits: int,
) -> None:
    lines = [_RAW_HEADER]
    for p in points:
        lines.append(
            ",".join(
                [
                    str(p.iu),
                    str(p.iv),
                    fixed(p.u_mm, mm_digits),
                    fixed(p.v_mm, mm_digits),
                    fixed(p.x_mm, mm_digits),
                    fixed(p.y_mm, mm_digits),
                    fixed(p.z_mm, mm_digits),
                    fixed(p.nx, unitless_digits),
                    fixed(p.ny, unitless_digits),
                    fixed(p.nz, unitless_digits),
                    fixed(p.theta1_rad, rad_digits),
                    fixed(p.theta2_rad, rad_digits),
                    fixed(p.residual_abs, mm_digits),
                    str(p.valid),
                    p.reason_code,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _summary_stats(valid_points: list[ToolConjugateGridRawPoint], nu: int, nv: int) -> dict[str, Any]:
    total_per_side = nu * nv
    total_points = 2 * total_per_side
    residuals = [p.residual_abs for p in valid_points]
    if not residuals:
        return {
            "n_total_points": total_points,
            "n_valid_points": 0,
            "valid_ratio_total": 0.0,
            "residual_abs_min": None,
            "residual_abs_p50": None,
            "residual_abs_p95": None,
            "residual_abs_max": None,
        }
    return {
        "n_total_points": total_points,
        "n_valid_points": len(residuals),
        "valid_ratio_total": len(residuals) / total_points,
        "residual_abs_min": min(residuals),
        "residual_abs_p50": percentile(residuals, 0.5),
        "residual_abs_p95": percentile(residuals, 0.95),
        "residual_abs_max": max(residuals),
    }


def run_step02(cfg: dict[str, Any], ctx: dict[str, Any]) -> Path:
    """Generate raw grids from selected s_rot and always emit step02 report."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STEP02_REPORT_NAME

    status = "failed"
    reason_code = "STEP02_EXCEPTION"
    message = "step02 failed before grid generation"
    payload: dict[str, Any] = {
        "s_rot_selected": ctx.get("s_rot_selected"),
        "valid_ratio_total": None,
        "valid_ratio_plus": None,
        "valid_ratio_minus": None,
        "n_total_points": None,
        "n_valid_points": None,
        "n_valid_plus": None,
        "n_valid_minus": None,
        "residual_abs_min": None,
        "residual_abs_p50": None,
        "residual_abs_p95": None,
        "residual_abs_max": None,
        "theta1_jump_count": None,
        "theta1_jump_count_plus": None,
        "theta1_jump_count_minus": None,
        "plus_csv": None,
        "minus_csv": None,
    }

    try:
        if "s_rot_selected" not in ctx or ctx["s_rot_selected"] not in (-1, 1):
            raise ValueError("s_rot_selected is missing from ctx")
        s_rot_selected = int(ctx["s_rot_selected"])
        nu = int(cfg["Nu"])
        nv = int(cfg["Nv"])
        mm_digits = int(cfg["csv_float_digits_mm"])
        rad_digits = int(cfg["csv_float_digits_rad"])
        unitless_digits = int(cfg["csv_float_digits_unitless"])
        raw = tool_conjugate_grid_raw(
            module_mm=float(cfg["module_mm"]),
            z1=int(cfg["z1"]),
            z2=int(cfg["z2"]),
            pressure_angle_deg=float(cfg["pressure_angle_deg"]),
            face_width_mm=float(cfg["face_width_mm"]),
            center_distance_a_mm=float(cfg["center_distance_a_mm"]),
            sigma_rad=float(cfg["sigma_rad"]),
            theta_tooth_center_rad=float(cfg["theta_tooth_center_rad"]),
            dtheta_deadband_rad=float(cfg["dtheta_deadband_rad"]),
            nu=nu,
            nv=nv,
            grid_u_min_mm=float(cfg["grid_u_min_mm"]),
            grid_u_max_mm=float(cfg["grid_u_max_mm"]),
            s_rot=s_rot_selected,
        )

        plus_points = list(raw.plus_points)
        minus_points = list(raw.minus_points)
        valid_plus_points = [p for p in plus_points if p.valid == 1]
        valid_minus_points = [p for p in minus_points if p.valid == 1]
        valid_points = valid_plus_points + valid_minus_points

        plus_csv = output_dir / PLUS_RAW_CSV_NAME
        minus_csv = output_dir / MINUS_RAW_CSV_NAME
        _write_raw_csv(
            plus_csv,
            plus_points,
            mm_digits=mm_digits,
            rad_digits=rad_digits,
            unitless_digits=unitless_digits,
        )
        _write_raw_csv(
            minus_csv,
            minus_points,
            mm_digits=mm_digits,
            rad_digits=rad_digits,
            unitless_digits=unitless_digits,
        )

        total_per_side = nu * nv
        stats = _summary_stats(valid_points, nu, nv)
        status = "ok" if len(valid_points) > 0 else "reject"
        reason_code = "OK" if status == "ok" else "NO_VALID_POINTS"
        message = "step02 raw grid generation completed"
        payload = {
            "s_rot_selected": s_rot_selected,
            "valid_ratio_total": stats["valid_ratio_total"],
            "valid_ratio_plus": len(valid_plus_points) / total_per_side,
            "valid_ratio_minus": len(valid_minus_points) / total_per_side,
            "n_total_points": stats["n_total_points"],
            "n_valid_points": stats["n_valid_points"],
            "n_valid_plus": len(valid_plus_points),
            "n_valid_minus": len(valid_minus_points),
            "residual_abs_min": stats["residual_abs_min"],
            "residual_abs_p50": stats["residual_abs_p50"],
            "residual_abs_p95": stats["residual_abs_p95"],
            "residual_abs_max": stats["residual_abs_max"],
            "theta1_jump_count": raw.theta1_jump_count,
            "theta1_jump_count_plus": raw.theta1_jump_count_plus,
            "theta1_jump_count_minus": raw.theta1_jump_count_minus,
            "plus_csv": plus_csv.name,
            "minus_csv": minus_csv.name,
        }
        if status == "ok":
            ctx.update(
                {
                    "valid_ratio_total": stats["valid_ratio_total"],
                    "valid_ratio_plus": len(valid_plus_points) / total_per_side,
                    "valid_ratio_minus": len(valid_minus_points) / total_per_side,
                    "residual_abs_p95": stats["residual_abs_p95"],
                    "residual_abs_max": stats["residual_abs_max"],
                    "theta1_jump_count": raw.theta1_jump_count,
                }
            )
    except Exception as exc:  # keep broad: cad_report must be written even on failures
        message = str(exc)

    cad_report = {
        "step_id": "step02_grid",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "exception_stacktrace": None,
        "ctx": dict(ctx),
        **payload,
    }
    write_json(report_path, cad_report)
    return report_path
