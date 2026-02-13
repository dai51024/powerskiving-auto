"""Ideal involute work target surface kernel (GEOM_SPEC 5 frozen formulas)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from powerskiving.deterministic import wrap_rad

EPS_NORM = 1.0e-12
STATUS_OK = "OK"
STATUS_OUTSIDE_DOMAIN = "OUTSIDE_DOMAIN"
STATUS_INVALID_INPUT = "INVALID_INPUT"
STATUS_INVALID_NORMAL = "INVALID_NORMAL"


@dataclass(frozen=True)
class WorkTargetSurfaceResult:
    """Result of ideal involute work target surface evaluation."""

    status: str
    p_W0: tuple[float, float, float] | None
    n_W0: tuple[float, float, float] | None


def _nz(x: float) -> float:
    """Normalize signed zero to +0.0."""
    if x == 0.0:
        return 0.0
    return x


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _normalize2(x: float, y: float) -> tuple[float, float] | None:
    n = math.sqrt(x * x + y * y)
    if not math.isfinite(n) or n < EPS_NORM:
        return None
    inv = 1.0 / n
    nx = _nz(x * inv)
    ny = _nz(y * inv)
    if not _finite(nx, ny):
        return None
    return nx, ny


def _normalize3(x: float, y: float, z: float) -> tuple[float, float, float] | None:
    n = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(n) or n < EPS_NORM:
        return None
    inv = 1.0 / n
    nx = _nz(x * inv)
    ny = _nz(y * inv)
    nz = _nz(z * inv)
    if not _finite(nx, ny, nz):
        return None
    return nx, ny, nz


def _rotate2(theta: float, x: float, y: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return _nz(c * x - s * y), _nz(s * x + c * y)


def work_target_surface(
    *,
    side: str,
    u_mm: float,
    v_mm: float,
    module_mm: float,
    z2: int,
    pressure_angle_deg: float,
) -> WorkTargetSurfaceResult:
    """
    Evaluate frozen ideal involute work target surface in W0 coordinates.

    Domain:
      - side in {"plus", "minus"}
      - u_mm is radius r (must satisfy r >= r_base_work)

    Returns status-only failures instead of raising for out-of-domain/input errors.
    """
    if side not in ("plus", "minus"):
        return WorkTargetSurfaceResult(status=STATUS_INVALID_INPUT, p_W0=None, n_W0=None)

    if not _finite(
        u_mm,
        v_mm,
        module_mm,
        float(z2),
        pressure_angle_deg,
    ):
        return WorkTargetSurfaceResult(status=STATUS_INVALID_INPUT, p_W0=None, n_W0=None)

    if z2 <= 0 or module_mm <= 0.0:
        return WorkTargetSurfaceResult(status=STATUS_INVALID_INPUT, p_W0=None, n_W0=None)

    alpha = pressure_angle_deg * math.pi / 180.0
    if not math.isfinite(alpha):
        return WorkTargetSurfaceResult(status=STATUS_INVALID_INPUT, p_W0=None, n_W0=None)

    r_pitch_work = 0.5 * module_mm * float(z2)
    r_base_work = r_pitch_work * math.cos(alpha)

    if not math.isfinite(r_base_work) or r_base_work <= EPS_NORM:
        return WorkTargetSurfaceResult(status=STATUS_INVALID_INPUT, p_W0=None, n_W0=None)

    r = u_mm
    if r < r_base_work:
        return WorkTargetSurfaceResult(status=STATUS_OUTSIDE_DOMAIN, p_W0=None, n_W0=None)

    rr = r / r_base_work
    t_sq = rr * rr - 1.0
    if t_sq < 0.0:
        t_sq = 0.0
    t = math.sqrt(t_sq)

    ct = math.cos(t)
    st = math.sin(t)

    x0 = r_base_work * (ct + t * st)
    y0 = r_base_work * (st - t * ct)

    tx0 = ct
    ty0 = st

    nx0 = -st
    ny0 = ct

    t_p = math.tan(alpha)
    cp = math.cos(t_p)
    sp = math.sin(t_p)
    x0_p = r_base_work * (cp + t_p * sp)
    y0_p = r_base_work * (sp - t_p * cp)

    phi_p = wrap_rad(math.atan2(y0_p, x0_p))
    theta_half = math.pi / (2.0 * float(z2))
    delta = wrap_rad(theta_half - phi_p)

    if side == "plus":
        p2x, p2y = _rotate2(delta, x0, y0)
        n2x, n2y = _rotate2(delta, nx0, ny0)
    else:
        p2x, p2y = _rotate2(-delta, x0, -y0)
        n2x, n2y = _rotate2(-delta, nx0, -ny0)

    r_hat = _normalize2(p2x, p2y)
    if r_hat is None:
        return WorkTargetSurfaceResult(status=STATUS_INVALID_NORMAL, p_W0=None, n_W0=None)

    if n2x * r_hat[0] + n2y * r_hat[1] > 0.0:
        n2x = -n2x
        n2y = -n2y

    n3 = _normalize3(n2x, n2y, 0.0)
    if n3 is None:
        return WorkTargetSurfaceResult(status=STATUS_INVALID_NORMAL, p_W0=None, n_W0=None)

    p3 = (_nz(p2x), _nz(p2y), _nz(v_mm))
    if not _finite(*p3, *n3):
        return WorkTargetSurfaceResult(status=STATUS_INVALID_INPUT, p_W0=None, n_W0=None)

    return WorkTargetSurfaceResult(status=STATUS_OK, p_W0=p3, n_W0=n3)


__all__ = [
    "work_target_surface",
    "WorkTargetSurfaceResult",
    "STATUS_OK",
    "STATUS_OUTSIDE_DOMAIN",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_NORMAL",
]
