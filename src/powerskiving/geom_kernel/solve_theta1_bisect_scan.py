"""Frozen theta1 solver: bracket scan + bisection (GEOM_SPEC 6.4)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from powerskiving.deterministic import wrap_rad

THETA1_SEED_INIT = 0.0
THETA1_STEP_SCAN_RAD = math.pi / 180.0
THETA1_SCAN_MAX_STEPS = 360
THETA1_BISECT_MAX_ITER = 80
THETA1_TOL_RAD = 1.0e-12
F_ZERO_EPS = 1.0e-12
THETA1_REFINE_DIV = 10
THETA1_REFINE_MAX_STEPS = 10

STATUS_OK = "OK"
STATUS_REJECT = "REJECT"

REASON_OK = "OK"
REASON_SOLVER_FAIL = "SOLVER_FAIL"
REASON_BRACKET_FAIL = "BRACKET_FAIL"
REASON_DEADBAND = "DEADBAND"
REASON_SIDE_MISMATCH = "SIDE_MISMATCH"
REASON_INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class SolveTheta1Result:
    status: str
    reason_code: str
    theta1_rad: float | None
    residual_abs: float | None


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _f_clamp(fv: float) -> float:
    if abs(fv) <= F_ZERO_EPS:
        return 0.0
    return fv


def _evaluate_f(f: Callable[[float], float], x: float) -> tuple[float | None, str | None]:
    fv = f(x)
    if not math.isfinite(fv):
        return None, REASON_INVALID_INPUT
    return _nz(float(fv)), None


def _accept_root(
    *,
    x: float,
    dtheta_fn: Callable[[float], float],
    side: str,
    dtheta_deadband_rad: float,
) -> tuple[bool, str]:
    dtheta = dtheta_fn(x)
    if not math.isfinite(dtheta):
        return False, REASON_INVALID_INPUT
    dtheta_w = wrap_rad(float(dtheta))
    if abs(dtheta_w) <= dtheta_deadband_rad:
        return False, REASON_DEADBAND
    if side == "plus":
        if dtheta_w > 0.0:
            return True, REASON_OK
        return False, REASON_SIDE_MISMATCH
    if side == "minus":
        if dtheta_w < 0.0:
            return True, REASON_OK
        return False, REASON_SIDE_MISMATCH
    return False, REASON_INVALID_INPUT


def _bisect(
    *,
    f: Callable[[float], float],
    a: float,
    b: float,
) -> tuple[float | None, str | None]:
    if a >= b:
        return None, REASON_BRACKET_FAIL

    fa, err_a = _evaluate_f(f, a)
    fb, err_b = _evaluate_f(f, b)
    if err_a is not None:
        return None, err_a
    if err_b is not None:
        return None, err_b
    assert fa is not None
    assert fb is not None

    if abs(fa) <= F_ZERO_EPS:
        return _nz(a), None
    if abs(fb) <= F_ZERO_EPS:
        return _nz(b), None

    fca = _f_clamp(fa)
    fcb = _f_clamp(fb)
    if fca * fcb > 0.0:
        return None, REASON_BRACKET_FAIL

    left = a
    right = b
    f_left = fa
    fca_left = fca

    for _ in range(THETA1_BISECT_MAX_ITER):
        m = 0.5 * (left + right)
        fm, err_m = _evaluate_f(f, m)
        if err_m is not None:
            return None, err_m
        assert fm is not None
        if abs(right - left) <= THETA1_TOL_RAD or abs(fm) <= F_ZERO_EPS:
            return _nz(m), None
        fcm = _f_clamp(fm)
        if fca_left * fcm <= 0.0:
            right = m
        else:
            left = m
            f_left = fm
            fca_left = _f_clamp(f_left)
    return _nz(0.5 * (left + right)), None


def _scan_window(
    *,
    f: Callable[[float], float],
    dtheta_fn: Callable[[float], float],
    side: str,
    dtheta_deadband_rad: float,
    x0: float,
    step: float,
    max_steps: int,
    best_abs: float,
    best_x: float,
    first_reject_reason: str | None,
) -> tuple[SolveTheta1Result | None, float, float, str | None]:
    f0, err0 = _evaluate_f(f, x0)
    if err0 is not None:
        return (
            SolveTheta1Result(
                status=STATUS_REJECT,
                reason_code=err0,
                theta1_rad=None,
                residual_abs=None,
            ),
            best_abs,
            best_x,
            first_reject_reason,
        )
    assert f0 is not None
    best_abs_out = best_abs
    best_x_out = best_x
    if abs(f0) < best_abs_out:
        best_abs_out = abs(f0)
        best_x_out = _nz(x0)

    if abs(f0) <= F_ZERO_EPS:
        ok, reason = _accept_root(
            x=x0,
            dtheta_fn=dtheta_fn,
            side=side,
            dtheta_deadband_rad=dtheta_deadband_rad,
        )
        if ok:
            return (
                SolveTheta1Result(
                    status=STATUS_OK,
                    reason_code=REASON_OK,
                    theta1_rad=_nz(x0),
                    residual_abs=_nz(abs(f0)),
                ),
                best_abs_out,
                best_x_out,
                first_reject_reason,
            )
        if first_reject_reason is None:
            first_reject_reason = reason

    prev_plus_x = _nz(x0)
    prev_plus_f = f0
    prev_minus_x = _nz(x0)
    prev_minus_f = f0

    for i in range(1, max_steps + 1):
        x1 = _nz(x0 + float(i) * step)
        f1, err1 = _evaluate_f(f, x1)
        if err1 is not None:
            return (
                SolveTheta1Result(
                    status=STATUS_REJECT,
                    reason_code=err1,
                    theta1_rad=None,
                    residual_abs=None,
                ),
                best_abs_out,
                best_x_out,
                first_reject_reason,
            )
        assert f1 is not None
        if abs(f1) < best_abs_out:
            best_abs_out = abs(f1)
            best_x_out = x1
        if abs(f1) <= F_ZERO_EPS:
            ok, reason = _accept_root(
                x=x1,
                dtheta_fn=dtheta_fn,
                side=side,
                dtheta_deadband_rad=dtheta_deadband_rad,
            )
            if ok:
                return (
                    SolveTheta1Result(
                        status=STATUS_OK,
                        reason_code=REASON_OK,
                        theta1_rad=x1,
                        residual_abs=_nz(abs(f1)),
                    ),
                    best_abs_out,
                    best_x_out,
                    first_reject_reason,
                )
            if first_reject_reason is None:
                first_reject_reason = reason
        else:
            a = prev_plus_x
            fa = prev_plus_f
            b = x1
            fb = f1
            fca = _f_clamp(fa)
            fcb = _f_clamp(fb)
            if fca == 0.0:
                ok, reason = _accept_root(
                    x=a,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                )
                if ok:
                    return (
                        SolveTheta1Result(
                            status=STATUS_OK,
                            reason_code=REASON_OK,
                            theta1_rad=_nz(a),
                            residual_abs=_nz(abs(fa)),
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                if first_reject_reason is None:
                    first_reject_reason = reason
            elif fcb == 0.0:
                ok, reason = _accept_root(
                    x=b,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                )
                if ok:
                    return (
                        SolveTheta1Result(
                            status=STATUS_OK,
                            reason_code=REASON_OK,
                            theta1_rad=_nz(b),
                            residual_abs=_nz(abs(fb)),
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                if first_reject_reason is None:
                    first_reject_reason = reason
            elif fca * fcb < 0.0:
                root, br_err = _bisect(f=f, a=a, b=b)
                if br_err is not None:
                    return (
                        SolveTheta1Result(
                            status=STATUS_REJECT,
                            reason_code=br_err,
                            theta1_rad=None,
                            residual_abs=None,
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                assert root is not None
                fr, err_r = _evaluate_f(f, root)
                if err_r is not None:
                    return (
                        SolveTheta1Result(
                            status=STATUS_REJECT,
                            reason_code=err_r,
                            theta1_rad=None,
                            residual_abs=None,
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                assert fr is not None
                ok, reason = _accept_root(
                    x=root,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                )
                if ok:
                    return (
                        SolveTheta1Result(
                            status=STATUS_OK,
                            reason_code=REASON_OK,
                            theta1_rad=_nz(root),
                            residual_abs=_nz(abs(fr)),
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                if first_reject_reason is None:
                    first_reject_reason = reason
        prev_plus_x = x1
        prev_plus_f = f1

        x2 = _nz(x0 - float(i) * step)
        f2, err2 = _evaluate_f(f, x2)
        if err2 is not None:
            return (
                SolveTheta1Result(
                    status=STATUS_REJECT,
                    reason_code=err2,
                    theta1_rad=None,
                    residual_abs=None,
                ),
                best_abs_out,
                best_x_out,
                first_reject_reason,
            )
        assert f2 is not None
        if abs(f2) < best_abs_out:
            best_abs_out = abs(f2)
            best_x_out = x2
        if abs(f2) <= F_ZERO_EPS:
            ok, reason = _accept_root(
                x=x2,
                dtheta_fn=dtheta_fn,
                side=side,
                dtheta_deadband_rad=dtheta_deadband_rad,
            )
            if ok:
                return (
                    SolveTheta1Result(
                        status=STATUS_OK,
                        reason_code=REASON_OK,
                        theta1_rad=x2,
                        residual_abs=_nz(abs(f2)),
                    ),
                    best_abs_out,
                    best_x_out,
                    first_reject_reason,
                )
            if first_reject_reason is None:
                first_reject_reason = reason
        else:
            a = x2
            fa = f2
            b = prev_minus_x
            fb = prev_minus_f
            if a > b:
                a, b = b, a
                fa, fb = fb, fa
            fca = _f_clamp(fa)
            fcb = _f_clamp(fb)
            if fca == 0.0:
                ok, reason = _accept_root(
                    x=a,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                )
                if ok:
                    return (
                        SolveTheta1Result(
                            status=STATUS_OK,
                            reason_code=REASON_OK,
                            theta1_rad=_nz(a),
                            residual_abs=_nz(abs(fa)),
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                if first_reject_reason is None:
                    first_reject_reason = reason
            elif fcb == 0.0:
                ok, reason = _accept_root(
                    x=b,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                )
                if ok:
                    return (
                        SolveTheta1Result(
                            status=STATUS_OK,
                            reason_code=REASON_OK,
                            theta1_rad=_nz(b),
                            residual_abs=_nz(abs(fb)),
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                if first_reject_reason is None:
                    first_reject_reason = reason
            elif fca * fcb < 0.0:
                root, br_err = _bisect(f=f, a=a, b=b)
                if br_err is not None:
                    return (
                        SolveTheta1Result(
                            status=STATUS_REJECT,
                            reason_code=br_err,
                            theta1_rad=None,
                            residual_abs=None,
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                assert root is not None
                fr, err_r = _evaluate_f(f, root)
                if err_r is not None:
                    return (
                        SolveTheta1Result(
                            status=STATUS_REJECT,
                            reason_code=err_r,
                            theta1_rad=None,
                            residual_abs=None,
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                assert fr is not None
                ok, reason = _accept_root(
                    x=root,
                    dtheta_fn=dtheta_fn,
                    side=side,
                    dtheta_deadband_rad=dtheta_deadband_rad,
                )
                if ok:
                    return (
                        SolveTheta1Result(
                            status=STATUS_OK,
                            reason_code=REASON_OK,
                            theta1_rad=_nz(root),
                            residual_abs=_nz(abs(fr)),
                        ),
                        best_abs_out,
                        best_x_out,
                        first_reject_reason,
                    )
                if first_reject_reason is None:
                    first_reject_reason = reason
        prev_minus_x = x2
        prev_minus_f = f2

    return None, best_abs_out, best_x_out, first_reject_reason


def solve_theta1_bisect_scan(
    *,
    f: Callable[[float], float],
    dtheta_fn: Callable[[float], float],
    side: str,
    dtheta_deadband_rad: float,
    theta1_seed: float = THETA1_SEED_INIT,
) -> SolveTheta1Result:
    """
    Solve f(theta1)=0 using frozen scan+bisect, then apply side/deadband acceptance.
    """
    if side not in ("plus", "minus"):
        return SolveTheta1Result(
            status=STATUS_REJECT,
            reason_code=REASON_INVALID_INPUT,
            theta1_rad=None,
            residual_abs=None,
        )
    if (not math.isfinite(theta1_seed)) or (not math.isfinite(dtheta_deadband_rad)):
        return SolveTheta1Result(
            status=STATUS_REJECT,
            reason_code=REASON_INVALID_INPUT,
            theta1_rad=None,
            residual_abs=None,
        )
    if dtheta_deadband_rad < 0.0:
        return SolveTheta1Result(
            status=STATUS_REJECT,
            reason_code=REASON_INVALID_INPUT,
            theta1_rad=None,
            residual_abs=None,
        )

    best_abs = math.inf
    best_x = _nz(theta1_seed)
    first_reject_reason: str | None = None

    coarse, best_abs, best_x, first_reject_reason = _scan_window(
        f=f,
        dtheta_fn=dtheta_fn,
        side=side,
        dtheta_deadband_rad=dtheta_deadband_rad,
        x0=_nz(theta1_seed),
        step=THETA1_STEP_SCAN_RAD,
        max_steps=THETA1_SCAN_MAX_STEPS,
        best_abs=best_abs,
        best_x=best_x,
        first_reject_reason=first_reject_reason,
    )
    if coarse is not None:
        return coarse

    refine_step = THETA1_STEP_SCAN_RAD / float(THETA1_REFINE_DIV)
    refined, _, _, first_reject_reason = _scan_window(
        f=f,
        dtheta_fn=dtheta_fn,
        side=side,
        dtheta_deadband_rad=dtheta_deadband_rad,
        x0=best_x,
        step=refine_step,
        max_steps=THETA1_REFINE_MAX_STEPS,
        best_abs=best_abs,
        best_x=best_x,
        first_reject_reason=first_reject_reason,
    )
    if refined is not None:
        return refined

    reason = REASON_SOLVER_FAIL if first_reject_reason is None else first_reject_reason
    return SolveTheta1Result(
        status=STATUS_REJECT,
        reason_code=reason,
        theta1_rad=None,
        residual_abs=None,
    )


__all__ = [
    "SolveTheta1Result",
    "solve_theta1_bisect_scan",
    "STATUS_OK",
    "STATUS_REJECT",
    "REASON_OK",
    "REASON_SOLVER_FAIL",
    "REASON_BRACKET_FAIL",
    "REASON_DEADBAND",
    "REASON_SIDE_MISMATCH",
    "REASON_INVALID_INPUT",
]
