"""Frozen raw tool conjugate grid generation (GEOM_SPEC 6)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from powerskiving.deterministic import wrap_rad

from .solve_theta1_bisect_scan import (
    STATUS_OK as SOLVER_STATUS_OK,
    solve_theta1_bisect_scan,
)
from .work_target_surface import (
    STATUS_INVALID_NORMAL as WTS_STATUS_INVALID_NORMAL,
    STATUS_OK as WTS_STATUS_OK,
    STATUS_OUTSIDE_DOMAIN as WTS_STATUS_OUTSIDE_DOMAIN,
    work_target_surface,
)

EPS_NORM = 1.0e-12
STATUS_OK = "OK"
REASON_OK = "OK"
REASON_SOLVER_FAIL = "SOLVER_FAIL"
REASON_OUTSIDE_DOMAIN = "OUTSIDE_DOMAIN"
REASON_INVALID_NORMAL = "INVALID_NORMAL"
REASON_NAN_FORBIDDEN_REPLACED_WITH_ZERO = "NAN_FORBIDDEN_REPLACED_WITH_ZERO"


@dataclass(frozen=True)
class ToolConjugateGridRawPoint:
    iu: int
    iv: int
    u_mm: float
    v_mm: float
    x_mm: float
    y_mm: float
    z_mm: float
    nx: float
    ny: float
    nz: float
    theta1_rad: float
    theta2_rad: float
    residual_abs: float
    valid: int
    reason_code: str


@dataclass(frozen=True)
class ToolConjugateGridRawResult:
    plus_points: tuple[ToolConjugateGridRawPoint, ...]
    minus_points: tuple[ToolConjugateGridRawPoint, ...]
    theta1_jump_count: int
    theta1_jump_count_plus: int
    theta1_jump_count_minus: int


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _rotate_z(theta: float, p: tuple[float, float, float]) -> tuple[float, float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = p
    return _nz(c * x - s * y), _nz(s * x + c * y), _nz(z)


def _rotate_y(theta: float, p: tuple[float, float, float]) -> tuple[float, float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = p
    return _nz(c * x + s * z), _nz(y), _nz(-s * x + c * z)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return (
        _nz(ay * bz - az * by),
        _nz(az * bx - ax * bz),
        _nz(ax * by - ay * bx),
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return _nz(ax * bx + ay * by + az * bz)


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return _nz(ax - bx), _nz(ay - by), _nz(az - bz)


def _normalize3(x: float, y: float, z: float) -> tuple[float, float, float] | None:
    n = math.sqrt(x * x + y * y + z * z)
    if (not math.isfinite(n)) or n < EPS_NORM:
        return None
    inv = 1.0 / n
    nx = _nz(x * inv)
    ny = _nz(y * inv)
    nz = _nz(z * inv)
    if not _finite(nx, ny, nz):
        return None
    return nx, ny, nz


def _point_invalid(*, iu: int, iv: int, u_mm: float, v_mm: float, reason_code: str) -> ToolConjugateGridRawPoint:
    return ToolConjugateGridRawPoint(
        iu=iu,
        iv=iv,
        u_mm=_nz(u_mm),
        v_mm=_nz(v_mm),
        x_mm=0.0,
        y_mm=0.0,
        z_mm=0.0,
        nx=0.0,
        ny=0.0,
        nz=0.0,
        theta1_rad=0.0,
        theta2_rad=0.0,
        residual_abs=0.0,
        valid=0,
        reason_code=reason_code,
    )


def _point_valid(
    *,
    iu: int,
    iv: int,
    u_mm: float,
    v_mm: float,
    p_t: tuple[float, float, float],
    n_t: tuple[float, float, float],
    theta1_rad: float,
    theta2_rad: float,
    residual_abs: float,
) -> ToolConjugateGridRawPoint:
    return ToolConjugateGridRawPoint(
        iu=iu,
        iv=iv,
        u_mm=_nz(u_mm),
        v_mm=_nz(v_mm),
        x_mm=_nz(p_t[0]),
        y_mm=_nz(p_t[1]),
        z_mm=_nz(p_t[2]),
        nx=_nz(n_t[0]),
        ny=_nz(n_t[1]),
        nz=_nz(n_t[2]),
        theta1_rad=_nz(theta1_rad),
        theta2_rad=_nz(theta2_rad),
        residual_abs=_nz(residual_abs),
        valid=1,
        reason_code=REASON_OK,
    )


def _sample_lin(min_v: float, max_v: float, i: int, n: int) -> float:
    if n <= 1:
        raise ValueError("n must be >= 2")
    t = float(i) / float(n - 1)
    return _nz(min_v + t * (max_v - min_v))


def _compute_theta1_jump_count(points: tuple[ToolConjugateGridRawPoint, ...], nu: int, nv: int) -> int:
    count = 0
    for iv in range(nv):
        for iu in range(nu):
            idx = iv * nu + iu
            p = points[idx]
            if p.valid != 1:
                continue
            if iu + 1 < nu:
                q = points[iv * nu + (iu + 1)]
                if q.valid == 1:
                    d = abs(wrap_rad(p.theta1_rad - q.theta1_rad))
                    if d > (0.5 * math.pi):
                        count += 1
            if iv + 1 < nv:
                q = points[(iv + 1) * nu + iu]
                if q.valid == 1:
                    d = abs(wrap_rad(p.theta1_rad - q.theta1_rad))
                    if d > (0.5 * math.pi):
                        count += 1
    return count


def tool_conjugate_grid_raw(
    *,
    module_mm: float,
    z1: int,
    z2: int,
    pressure_angle_deg: float,
    face_width_mm: float,
    center_distance_a_mm: float,
    sigma_rad: float,
    theta_tooth_center_rad: float,
    dtheta_deadband_rad: float,
    nu: int,
    nv: int,
    grid_u_min_mm: float,
    grid_u_max_mm: float,
    s_rot: int,
) -> ToolConjugateGridRawResult:
    """
    Generate frozen raw conjugate grids for both sides and compute jump counts.

    Points are ordered by iv-major then iu-major for each side.
    """
    if not _finite(
        module_mm,
        float(z1),
        float(z2),
        pressure_angle_deg,
        face_width_mm,
        center_distance_a_mm,
        sigma_rad,
        theta_tooth_center_rad,
        dtheta_deadband_rad,
        grid_u_min_mm,
        grid_u_max_mm,
        float(nu),
        float(nv),
        float(s_rot),
    ):
        raise ValueError("NaN/Inf input is not allowed")
    if nu <= 1 or nv <= 1:
        raise ValueError("nu and nv must be >= 2")
    if z1 <= 0 or z2 <= 0:
        raise ValueError("z1 and z2 must be > 0")
    if module_mm <= 0.0 or face_width_mm <= 0.0:
        raise ValueError("module_mm and face_width_mm must be > 0")
    if dtheta_deadband_rad < 0.0:
        raise ValueError("dtheta_deadband_rad must be >= 0")
    if s_rot not in (-1, 1):
        raise ValueError("s_rot must be +/-1")

    z_min = _nz(-0.5 * face_width_mm)
    ratio = _nz(float(s_rot) * float(z1) / float(z2))
    c1_w = (0.0, _nz(center_distance_a_mm), 0.0)
    k1 = (_nz(math.sin(sigma_rad)), 0.0, _nz(math.cos(sigma_rad)))
    k2 = (0.0, 0.0, 1.0)

    def make_side(side: str) -> tuple[tuple[ToolConjugateGridRawPoint, ...], float]:
        points: list[ToolConjugateGridRawPoint] = []
        theta1_seed = 0.0
        for iv in range(nv):
            v_mm = _sample_lin(z_min, -z_min, iv, nv)
            for iu in range(nu):
                u_mm = _sample_lin(grid_u_min_mm, grid_u_max_mm, iu, nu)
                w = work_target_surface(
                    side=side,
                    u_mm=u_mm,
                    v_mm=v_mm,
                    module_mm=module_mm,
                    z2=z2,
                    pressure_angle_deg=pressure_angle_deg,
                )
                if w.status == WTS_STATUS_OUTSIDE_DOMAIN:
                    points.append(
                        _point_invalid(
                            iu=iu,
                            iv=iv,
                            u_mm=u_mm,
                            v_mm=v_mm,
                            reason_code=REASON_OUTSIDE_DOMAIN,
                        )
                    )
                    continue
                if w.status == WTS_STATUS_INVALID_NORMAL:
                    points.append(
                        _point_invalid(
                            iu=iu,
                            iv=iv,
                            u_mm=u_mm,
                            v_mm=v_mm,
                            reason_code=REASON_INVALID_NORMAL,
                        )
                    )
                    continue
                if w.status != WTS_STATUS_OK or w.p_W0 is None or w.n_W0 is None:
                    points.append(
                        _point_invalid(
                            iu=iu,
                            iv=iv,
                            u_mm=u_mm,
                            v_mm=v_mm,
                            reason_code=REASON_SOLVER_FAIL,
                        )
                    )
                    continue
                p_w0 = w.p_W0
                n_w0 = w.n_W0

                def f(theta1: float) -> float:
                    theta2 = _nz(ratio * theta1)
                    p_w = _rotate_z(theta2, p_w0)
                    n_w = _rotate_z(theta2, n_w0)
                    term1 = _cross(k1, _sub(p_w, c1_w))
                    term2 = _cross(k2, p_w)
                    v_rel = (
                        _nz(term1[0] - ratio * term2[0]),
                        _nz(term1[1] - ratio * term2[1]),
                        _nz(term1[2] - ratio * term2[2]),
                    )
                    return _dot(v_rel, n_w)

                def dtheta_fn(theta1: float) -> float:
                    theta2 = _nz(ratio * theta1)
                    p_w = _rotate_z(theta2, p_w0)
                    p_shift = _sub(p_w, c1_w)
                    p_t = _rotate_z(-theta1, _rotate_y(-sigma_rad, p_shift))
                    theta_tool = wrap_rad(math.atan2(p_t[1], p_t[0]))
                    return wrap_rad(theta_tool - theta_tooth_center_rad)

                solve = solve_theta1_bisect_scan(
                    f=f,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                    theta1_seed=theta1_seed,
                )
                if solve.status != SOLVER_STATUS_OK or solve.theta1_rad is None or solve.residual_abs is None:
                    points.append(
                        _point_invalid(
                            iu=iu,
                            iv=iv,
                            u_mm=u_mm,
                            v_mm=v_mm,
                            reason_code=REASON_SOLVER_FAIL,
                        )
                    )
                    continue

                theta1 = _nz(solve.theta1_rad)
                theta2 = _nz(ratio * theta1)
                p_w = _rotate_z(theta2, p_w0)
                n_w = _rotate_z(theta2, n_w0)
                p_shift = _sub(p_w, c1_w)
                p_t = _rotate_z(-theta1, _rotate_y(-sigma_rad, p_shift))
                n_t_raw = _rotate_z(-theta1, _rotate_y(-sigma_rad, n_w))
                n_t = _normalize3(n_t_raw[0], n_t_raw[1], n_t_raw[2])
                if n_t is None:
                    points.append(
                        _point_invalid(
                            iu=iu,
                            iv=iv,
                            u_mm=u_mm,
                            v_mm=v_mm,
                            reason_code=REASON_INVALID_NORMAL,
                        )
                    )
                    continue
                residual_abs = _nz(abs(solve.residual_abs))
                if not _finite(
                    p_t[0],
                    p_t[1],
                    p_t[2],
                    n_t[0],
                    n_t[1],
                    n_t[2],
                    theta1,
                    theta2,
                    residual_abs,
                ):
                    points.append(
                        _point_invalid(
                            iu=iu,
                            iv=iv,
                            u_mm=u_mm,
                            v_mm=v_mm,
                            reason_code=REASON_NAN_FORBIDDEN_REPLACED_WITH_ZERO,
                        )
                    )
                    continue

                points.append(
                    _point_valid(
                        iu=iu,
                        iv=iv,
                        u_mm=u_mm,
                        v_mm=v_mm,
                        p_t=p_t,
                        n_t=n_t,
                        theta1_rad=theta1,
                        theta2_rad=theta2,
                        residual_abs=residual_abs,
                    )
                )
                theta1_seed = wrap_rad(theta1)
        return tuple(points), theta1_seed

    plus_points, _ = make_side("plus")
    minus_points, _ = make_side("minus")
    jumps_plus = _compute_theta1_jump_count(plus_points, nu, nv)
    jumps_minus = _compute_theta1_jump_count(minus_points, nu, nv)
    return ToolConjugateGridRawResult(
        plus_points=plus_points,
        minus_points=minus_points,
        theta1_jump_count=jumps_plus + jumps_minus,
        theta1_jump_count_plus=jumps_plus,
        theta1_jump_count_minus=jumps_minus,
    )


__all__ = [
    "tool_conjugate_grid_raw",
    "ToolConjugateGridRawPoint",
    "ToolConjugateGridRawResult",
    "STATUS_OK",
    "REASON_OK",
    "REASON_SOLVER_FAIL",
    "REASON_OUTSIDE_DOMAIN",
    "REASON_INVALID_NORMAL",
    "REASON_NAN_FORBIDDEN_REPLACED_WITH_ZERO",
]
