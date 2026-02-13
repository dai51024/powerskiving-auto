"""Geometry kernels frozen by GEOM_SPEC."""

from .work_target_surface import (
    STATUS_INVALID_INPUT,
    STATUS_INVALID_NORMAL,
    STATUS_OK,
    STATUS_OUTSIDE_DOMAIN,
    WorkTargetSurfaceResult,
    work_target_surface,
)
from .solve_theta1_bisect_scan import (
    REASON_BRACKET_FAIL,
    REASON_DEADBAND,
    REASON_INVALID_INPUT as SOLVER_REASON_INVALID_INPUT,
    REASON_OK as SOLVER_REASON_OK,
    REASON_SIDE_MISMATCH,
    REASON_SOLVER_FAIL,
    STATUS_OK as SOLVER_STATUS_OK,
    STATUS_REJECT,
    SolveTheta1Result,
    solve_theta1_bisect_scan,
)

__all__ = [
    "work_target_surface",
    "WorkTargetSurfaceResult",
    "STATUS_OK",
    "STATUS_OUTSIDE_DOMAIN",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_NORMAL",
    "solve_theta1_bisect_scan",
    "SolveTheta1Result",
    "SOLVER_STATUS_OK",
    "STATUS_REJECT",
    "SOLVER_REASON_OK",
    "REASON_SOLVER_FAIL",
    "REASON_BRACKET_FAIL",
    "REASON_DEADBAND",
    "REASON_SIDE_MISMATCH",
    "SOLVER_REASON_INVALID_INPUT",
]
