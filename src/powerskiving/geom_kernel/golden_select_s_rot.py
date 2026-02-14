"""Frozen golden selection for s_rot at sigma=0 (GEOM_SPEC 7)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from powerskiving.deterministic import percentile, wrap_rad

from .tool_conjugate_grid_raw import ToolConjugateGridRawPoint, ToolConjugateGridRawResult, tool_conjugate_grid_raw

STATUS_OK = "OK"
STATUS_REJECT = "REJECT"

REASON_OK = "OK"
REASON_INVALID_INPUT = "INVALID_INPUT"
REASON_NO_VALID_POINTS = "NO_VALID_POINTS"
REASON_MIN_POINTS_FAIL = "MIN_POINTS_FAIL"
REASON_NO_EVAL_POINTS = "NO_EVAL_POINTS"
REASON_BRACKET_FAIL = "BRACKET_FAIL"
REASON_CONVERGENCE_FAIL = "CONVERGENCE_FAIL"
REASON_SOLVER_FAIL = "SOLVER_FAIL"
REASON_GOLDEN_FAILED = "GOLDEN_FAILED"

_SOLVER_FAIL_PRIORITIES = (
    REASON_BRACKET_FAIL,
    REASON_CONVERGENCE_FAIL,
    REASON_SOLVER_FAIL,
)


@dataclass(frozen=True)
class GoldenCandidateResult:
    s_rot: int
    status: str
    reason_code: str
    golden_p95_mm: float | None
    golden_max_mm: float | None
    z_mid_tool_mm: float | None
    golden_dz_used_mm: float | None
    plus_eval_count: int
    minus_eval_count: int


@dataclass(frozen=True)
class GoldenSelectSRotResult:
    status: str
    reason_code: str
    s_rot_selected: int | None
    golden_p95_mm: float | None
    golden_max_mm: float | None
    z_mid_tool_mm: float | None
    golden_dz_used_mm: float | None
    candidate_plus: GoldenCandidateResult
    candidate_minus: GoldenCandidateResult


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _rotate_z_2d(theta: float, x: float, y: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return _nz(c * x - s * y), _nz(s * x + c * y)


def _point_segment_distance_2d(
    *, qx: float, qy: float, ax: float, ay: float, bx: float, by: float
) -> float | None:
    vx = _nz(bx - ax)
    vy = _nz(by - ay)
    wx = _nz(qx - ax)
    wy = _nz(qy - ay)
    vv = _nz(vx * vx + vy * vy)
    if not _finite(vx, vy, wx, wy, vv):
        return None
    if vv <= 0.0:
        dx = _nz(qx - ax)
        dy = _nz(qy - ay)
        d2 = _nz(dx * dx + dy * dy)
        if d2 < 0.0:
            return None
        out = math.sqrt(d2)
        if not math.isfinite(out):
            return None
        return _nz(out)
    t = _nz((wx * vx + wy * vy) / vv)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    px = _nz(ax + t * vx)
    py = _nz(ay + t * vy)
    dx = _nz(qx - px)
    dy = _nz(qy - py)
    d2 = _nz(dx * dx + dy * dy)
    if d2 < 0.0:
        return None
    out = math.sqrt(d2)
    if not math.isfinite(out):
        return None
    return _nz(out)


def _polyline_min_distance_2d(
    *, qx: float, qy: float, polyline: tuple[tuple[float, float], ...]
) -> float | None:
    if len(polyline) < 2:
        return None
    best = math.inf
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        d = _point_segment_distance_2d(qx=qx, qy=qy, ax=ax, ay=ay, bx=bx, by=by)
        if d is None:
            return None
        if d < best:
            best = d
    if not math.isfinite(best):
        return None
    return _nz(best)


def _candidate_fail(
    *,
    s_rot: int,
    reason_code: str,
    z_mid_tool_mm: float | None = None,
    golden_dz_used_mm: float | None = None,
) -> GoldenCandidateResult:
    return GoldenCandidateResult(
        s_rot=s_rot,
        status=STATUS_REJECT,
        reason_code=reason_code,
        golden_p95_mm=None,
        golden_max_mm=None,
        z_mid_tool_mm=None if z_mid_tool_mm is None else _nz(z_mid_tool_mm),
        golden_dz_used_mm=None if golden_dz_used_mm is None else _nz(golden_dz_used_mm),
        plus_eval_count=0,
        minus_eval_count=0,
    )


def _pick_solver_fail_reason(points: tuple[ToolConjugateGridRawPoint, ...]) -> str:
    reasons = {p.reason_code for p in points if p.valid == 0}
    for code in _SOLVER_FAIL_PRIORITIES:
        if code in reasons:
            return code
    return REASON_NO_VALID_POINTS


def _build_reference_polylines(
    *,
    module_mm: float,
    z1: int,
    pressure_angle_deg: float,
    golden_pitch_band_dr_mm: float,
    golden_ref_n: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]] | None:
    alpha = math.radians(pressure_angle_deg)
    r_p = _nz(0.5 * module_mm * float(z1))
    r_b = _nz(r_p * math.cos(alpha))
    if (not _finite(alpha, r_p, r_b)) or r_b <= 0.0:
        return None
    t_p = _nz(math.tan(alpha))
    x_p = _nz(r_b * (math.cos(t_p) + t_p * math.sin(t_p)))
    y_p = _nz(r_b * (math.sin(t_p) - t_p * math.cos(t_p)))
    if not _finite(t_p, x_p, y_p):
        return None
    phi_p = wrap_rad(math.atan2(y_p, x_p))
    theta_half = _nz(math.pi / (2.0 * float(z1)))
    delta = wrap_rad(theta_half - phi_p)

    r_min = _nz(r_p - golden_pitch_band_dr_mm)
    r_max = _nz(r_p + golden_pitch_band_dr_mm)
    if r_max < r_min:
        return None
    if r_min < r_b:
        r_min = r_b

    tr_min = _nz(r_min / r_b)
    tr_max = _nz(r_max / r_b)
    sq_min = _nz(tr_min * tr_min - 1.0)
    sq_max = _nz(tr_max * tr_max - 1.0)
    if sq_min < 0.0:
        sq_min = 0.0
    if sq_max < 0.0:
        sq_max = 0.0
    t_min = _nz(math.sqrt(sq_min))
    t_max = _nz(math.sqrt(sq_max))
    if not _finite(t_min, t_max) or t_max < t_min:
        return None

    plus: list[tuple[float, float]] = []
    minus: list[tuple[float, float]] = []
    n_ref = float(golden_ref_n)
    span = _nz(t_max - t_min)
    for j in range(golden_ref_n):
        t_j = _nz(t_min + (float(j) + 0.5) * span / n_ref)
        x0 = _nz(r_b * (math.cos(t_j) + t_j * math.sin(t_j)))
        y0 = _nz(r_b * (math.sin(t_j) - t_j * math.cos(t_j)))
        if not _finite(t_j, x0, y0):
            return None
        plus.append(_rotate_z_2d(delta, x0, y0))
        minus.append(_rotate_z_2d(-delta, x0, -y0))

    return tuple(plus), tuple(minus)


def _side_bucket(
    *,
    p: ToolConjugateGridRawPoint,
    theta_tooth_center_rad: float,
    dtheta_deadband_rad: float,
) -> str | None:
    theta_tool = wrap_rad(math.atan2(p.y_mm, p.x_mm))
    dtheta = wrap_rad(theta_tool - theta_tooth_center_rad)
    if abs(dtheta) <= dtheta_deadband_rad:
        return None
    if dtheta > 0.0:
        return "plus"
    return "minus"


def _eval_candidate(
    *,
    s_rot: int,
    module_mm: float,
    z1: int,
    z2: int,
    pressure_angle_deg: float,
    face_width_mm: float,
    center_distance_a_mm: float,
    theta_tooth_center_rad: float,
    dtheta_deadband_rad: float,
    nu: int,
    nv: int,
    grid_u_min_mm: float,
    grid_u_max_mm: float,
    golden_dz_mm: float,
    golden_dz_max_mm: float,
    golden_min_points: int,
    golden_pitch_band_dr_mm: float,
    ref_plus: tuple[tuple[float, float], ...],
    ref_minus: tuple[tuple[float, float], ...],
    raw_generator: Callable[..., ToolConjugateGridRawResult],
) -> GoldenCandidateResult:
    raw = raw_generator(
        module_mm=module_mm,
        z1=z1,
        z2=z2,
        pressure_angle_deg=pressure_angle_deg,
        face_width_mm=face_width_mm,
        center_distance_a_mm=center_distance_a_mm,
        sigma_rad=0.0,
        theta_tooth_center_rad=theta_tooth_center_rad,
        dtheta_deadband_rad=dtheta_deadband_rad,
        nu=nu,
        nv=nv,
        grid_u_min_mm=grid_u_min_mm,
        grid_u_max_mm=grid_u_max_mm,
        s_rot=s_rot,
    )
    points = raw.plus_points + raw.minus_points
    valid_points = [p for p in points if p.valid == 1]
    if not valid_points:
        return _candidate_fail(s_rot=s_rot, reason_code=_pick_solver_fail_reason(points))

    z_values = [_nz(float(p.z_mm)) for p in valid_points]
    z_mid = percentile(z_values, 0.5)
    dz = _nz(golden_dz_mm)
    band_points: list[ToolConjugateGridRawPoint] = []
    while True:
        band_points = [p for p in valid_points if abs(_nz(p.z_mm - z_mid)) <= dz]
        if len(band_points) >= golden_min_points:
            break
        dz_next = _nz(dz * 2.0)
        if dz_next > golden_dz_max_mm:
            return _candidate_fail(
                s_rot=s_rot,
                reason_code=REASON_MIN_POINTS_FAIL,
                z_mid_tool_mm=z_mid,
                golden_dz_used_mm=dz,
            )
        dz = dz_next

    r_pitch_tool = _nz(0.5 * module_mm * float(z1))
    r_lo = _nz(r_pitch_tool - golden_pitch_band_dr_mm)
    r_hi = _nz(r_pitch_tool + golden_pitch_band_dr_mm)
    plus_dist: list[float] = []
    minus_dist: list[float] = []

    for p in band_points:
        r = _nz(math.hypot(p.x_mm, p.y_mm))
        if (r < r_lo) or (r > r_hi):
            continue
        side = _side_bucket(
            p=p,
            theta_tooth_center_rad=theta_tooth_center_rad,
            dtheta_deadband_rad=dtheta_deadband_rad,
        )
        if side is None:
            continue
        if side == "plus":
            d = _polyline_min_distance_2d(qx=p.x_mm, qy=p.y_mm, polyline=ref_plus)
            if d is None:
                return _candidate_fail(s_rot=s_rot, reason_code=REASON_INVALID_INPUT)
            plus_dist.append(d)
        else:
            d = _polyline_min_distance_2d(qx=p.x_mm, qy=p.y_mm, polyline=ref_minus)
            if d is None:
                return _candidate_fail(s_rot=s_rot, reason_code=REASON_INVALID_INPUT)
            minus_dist.append(d)

    if (not plus_dist) or (not minus_dist):
        return _candidate_fail(
            s_rot=s_rot,
            reason_code=REASON_NO_EVAL_POINTS,
            z_mid_tool_mm=z_mid,
            golden_dz_used_mm=dz,
        )

    plus_p95 = percentile(plus_dist, 0.95)
    minus_p95 = percentile(minus_dist, 0.95)
    plus_max = max(plus_dist)
    minus_max = max(minus_dist)
    golden_p95 = _nz(max(plus_p95, minus_p95))
    golden_max = _nz(max(plus_max, minus_max))

    return GoldenCandidateResult(
        s_rot=s_rot,
        status=STATUS_OK,
        reason_code=REASON_OK,
        golden_p95_mm=golden_p95,
        golden_max_mm=golden_max,
        z_mid_tool_mm=_nz(z_mid),
        golden_dz_used_mm=_nz(dz),
        plus_eval_count=len(plus_dist),
        minus_eval_count=len(minus_dist),
    )


def _pick_candidate(
    first: GoldenCandidateResult,
    second: GoldenCandidateResult,
) -> GoldenCandidateResult | None:
    first_ok = first.status == STATUS_OK
    second_ok = second.status == STATUS_OK
    if first_ok and not second_ok:
        return first
    if second_ok and not first_ok:
        return second
    if not first_ok and not second_ok:
        return None
    assert first.golden_p95_mm is not None
    assert second.golden_p95_mm is not None
    assert first.golden_max_mm is not None
    assert second.golden_max_mm is not None
    if second.golden_p95_mm < first.golden_p95_mm:
        return second
    if first.golden_p95_mm < second.golden_p95_mm:
        return first
    if second.golden_max_mm < first.golden_max_mm:
        return second
    if first.golden_max_mm < second.golden_max_mm:
        return first
    return first


def _pick_reject_reason(plus: GoldenCandidateResult, minus: GoldenCandidateResult) -> str:
    for code in _SOLVER_FAIL_PRIORITIES:
        if plus.reason_code == code or minus.reason_code == code:
            return code
    if plus.reason_code == REASON_MIN_POINTS_FAIL or minus.reason_code == REASON_MIN_POINTS_FAIL:
        return REASON_MIN_POINTS_FAIL
    if plus.reason_code == REASON_NO_EVAL_POINTS or minus.reason_code == REASON_NO_EVAL_POINTS:
        return REASON_NO_EVAL_POINTS
    if plus.reason_code == REASON_NO_VALID_POINTS or minus.reason_code == REASON_NO_VALID_POINTS:
        return REASON_NO_VALID_POINTS
    if plus.reason_code == REASON_INVALID_INPUT or minus.reason_code == REASON_INVALID_INPUT:
        return REASON_INVALID_INPUT
    return REASON_GOLDEN_FAILED


def golden_select_s_rot(
    *,
    module_mm: float,
    z1: int,
    z2: int,
    pressure_angle_deg: float,
    face_width_mm: float,
    center_distance_a_mm: float,
    theta_tooth_center_rad: float,
    dtheta_deadband_rad: float,
    nu: int,
    nv: int,
    grid_u_min_mm: float,
    grid_u_max_mm: float,
    golden_tol_p95_mm: float,
    golden_tol_max_mm: float,
    golden_dz_mm: float,
    golden_dz_max_mm: float,
    golden_min_points: int,
    golden_pitch_band_dr_mm: float,
    golden_ref_n: int,
    raw_generator: Callable[..., ToolConjugateGridRawResult] = tool_conjugate_grid_raw,
) -> GoldenSelectSRotResult:
    if (
        (not _finite(
            module_mm,
            float(z1),
            float(z2),
            pressure_angle_deg,
            face_width_mm,
            center_distance_a_mm,
            theta_tooth_center_rad,
            dtheta_deadband_rad,
            grid_u_min_mm,
            grid_u_max_mm,
            golden_tol_p95_mm,
            golden_tol_max_mm,
            golden_dz_mm,
            golden_dz_max_mm,
            golden_pitch_band_dr_mm,
            float(golden_min_points),
            float(golden_ref_n),
            float(nu),
            float(nv),
        ))
        or z1 <= 0
        or z2 <= 0
        or nu <= 1
        or nv <= 1
        or module_mm <= 0.0
        or face_width_mm <= 0.0
        or dtheta_deadband_rad < 0.0
        or golden_tol_p95_mm < 0.0
        or golden_tol_max_mm < 0.0
        or golden_dz_mm <= 0.0
        or golden_dz_max_mm <= 0.0
        or golden_dz_mm > golden_dz_max_mm
        or golden_min_points <= 0
        or golden_pitch_band_dr_mm < 0.0
        or golden_ref_n < 20
    ):
        fail_plus = _candidate_fail(s_rot=1, reason_code=REASON_INVALID_INPUT)
        fail_minus = _candidate_fail(s_rot=-1, reason_code=REASON_INVALID_INPUT)
        return GoldenSelectSRotResult(
            status=STATUS_REJECT,
            reason_code=REASON_INVALID_INPUT,
            s_rot_selected=None,
            golden_p95_mm=None,
            golden_max_mm=None,
            z_mid_tool_mm=None,
            golden_dz_used_mm=None,
            candidate_plus=fail_plus,
            candidate_minus=fail_minus,
        )

    refs = _build_reference_polylines(
        module_mm=module_mm,
        z1=z1,
        pressure_angle_deg=pressure_angle_deg,
        golden_pitch_band_dr_mm=golden_pitch_band_dr_mm,
        golden_ref_n=golden_ref_n,
    )
    if refs is None:
        fail_plus = _candidate_fail(s_rot=1, reason_code=REASON_INVALID_INPUT)
        fail_minus = _candidate_fail(s_rot=-1, reason_code=REASON_INVALID_INPUT)
        return GoldenSelectSRotResult(
            status=STATUS_REJECT,
            reason_code=REASON_INVALID_INPUT,
            s_rot_selected=None,
            golden_p95_mm=None,
            golden_max_mm=None,
            z_mid_tool_mm=None,
            golden_dz_used_mm=None,
            candidate_plus=fail_plus,
            candidate_minus=fail_minus,
        )
    ref_plus, ref_minus = refs

    cand_plus = _eval_candidate(
        s_rot=1,
        module_mm=module_mm,
        z1=z1,
        z2=z2,
        pressure_angle_deg=pressure_angle_deg,
        face_width_mm=face_width_mm,
        center_distance_a_mm=center_distance_a_mm,
        theta_tooth_center_rad=theta_tooth_center_rad,
        dtheta_deadband_rad=dtheta_deadband_rad,
        nu=nu,
        nv=nv,
        grid_u_min_mm=grid_u_min_mm,
        grid_u_max_mm=grid_u_max_mm,
        golden_dz_mm=golden_dz_mm,
        golden_dz_max_mm=golden_dz_max_mm,
        golden_min_points=golden_min_points,
        golden_pitch_band_dr_mm=golden_pitch_band_dr_mm,
        ref_plus=ref_plus,
        ref_minus=ref_minus,
        raw_generator=raw_generator,
    )
    cand_minus = _eval_candidate(
        s_rot=-1,
        module_mm=module_mm,
        z1=z1,
        z2=z2,
        pressure_angle_deg=pressure_angle_deg,
        face_width_mm=face_width_mm,
        center_distance_a_mm=center_distance_a_mm,
        theta_tooth_center_rad=theta_tooth_center_rad,
        dtheta_deadband_rad=dtheta_deadband_rad,
        nu=nu,
        nv=nv,
        grid_u_min_mm=grid_u_min_mm,
        grid_u_max_mm=grid_u_max_mm,
        golden_dz_mm=golden_dz_mm,
        golden_dz_max_mm=golden_dz_max_mm,
        golden_min_points=golden_min_points,
        golden_pitch_band_dr_mm=golden_pitch_band_dr_mm,
        ref_plus=ref_plus,
        ref_minus=ref_minus,
        raw_generator=raw_generator,
    )
    winner = _pick_candidate(cand_plus, cand_minus)
    if winner is None:
        return GoldenSelectSRotResult(
            status=STATUS_REJECT,
            reason_code=_pick_reject_reason(cand_plus, cand_minus),
            s_rot_selected=None,
            golden_p95_mm=None,
            golden_max_mm=None,
            z_mid_tool_mm=None,
            golden_dz_used_mm=None,
            candidate_plus=cand_plus,
            candidate_minus=cand_minus,
        )

    assert winner.golden_p95_mm is not None
    assert winner.golden_max_mm is not None
    assert winner.z_mid_tool_mm is not None
    assert winner.golden_dz_used_mm is not None

    if (winner.golden_p95_mm > golden_tol_p95_mm) or (winner.golden_max_mm > golden_tol_max_mm):
        return GoldenSelectSRotResult(
            status=STATUS_REJECT,
            reason_code=REASON_GOLDEN_FAILED,
            s_rot_selected=winner.s_rot,
            golden_p95_mm=_nz(winner.golden_p95_mm),
            golden_max_mm=_nz(winner.golden_max_mm),
            z_mid_tool_mm=_nz(winner.z_mid_tool_mm),
            golden_dz_used_mm=_nz(winner.golden_dz_used_mm),
            candidate_plus=cand_plus,
            candidate_minus=cand_minus,
        )

    return GoldenSelectSRotResult(
        status=STATUS_OK,
        reason_code=REASON_OK,
        s_rot_selected=winner.s_rot,
        golden_p95_mm=_nz(winner.golden_p95_mm),
        golden_max_mm=_nz(winner.golden_max_mm),
        z_mid_tool_mm=_nz(winner.z_mid_tool_mm),
        golden_dz_used_mm=_nz(winner.golden_dz_used_mm),
        candidate_plus=cand_plus,
        candidate_minus=cand_minus,
    )


__all__ = [
    "GoldenCandidateResult",
    "GoldenSelectSRotResult",
    "golden_select_s_rot",
    "STATUS_OK",
    "STATUS_REJECT",
    "REASON_OK",
    "REASON_INVALID_INPUT",
    "REASON_NO_VALID_POINTS",
    "REASON_MIN_POINTS_FAIL",
    "REASON_NO_EVAL_POINTS",
    "REASON_BRACKET_FAIL",
    "REASON_CONVERGENCE_FAIL",
    "REASON_SOLVER_FAIL",
    "REASON_GOLDEN_FAILED",
]
