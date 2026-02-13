"""Geometry kernels frozen by GEOM_SPEC."""

from .work_target_surface import (
    STATUS_INVALID_INPUT,
    STATUS_INVALID_NORMAL,
    STATUS_OK,
    STATUS_OUTSIDE_DOMAIN,
    WorkTargetSurfaceResult,
    work_target_surface,
)

__all__ = [
    "work_target_surface",
    "WorkTargetSurfaceResult",
    "STATUS_OK",
    "STATUS_OUTSIDE_DOMAIN",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_NORMAL",
]
