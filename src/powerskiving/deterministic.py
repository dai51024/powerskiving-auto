"""Deterministic numeric helpers for powerskiving (frozen by GEOM_SPEC)."""

from __future__ import annotations

import math
from typing import Iterable

PI = math.pi
TAU = 2.0 * math.pi


def wrap_rad(x: float) -> float:
    """
    Wrap radians to (-pi, +pi], with the convention wrap_rad(+pi) == -pi.
    Implementation follows the frozen normalization formula.
    """
    if not math.isfinite(x):
        raise ValueError("NaN/Inf is not allowed")
    y = x - TAU * math.floor((x + PI) / TAU)  # in [-pi, +pi)
    # Enforce +pi -> -pi convention if it ever appears (safety)
    if y == PI:
        y = -PI
    # Normalize -0 -> +0
    if y == 0.0:
        return 0.0
    return y


def q(x: float, d: int = 6) -> float:
    """
    Quantize exactly as GEOM_SPEC:
      q(x,d) = sign(x) * floor(|x|*10^d + 0.5) / 10^d
    Constraints:
      - sign(0) = +1 (no -0)
      - no NaN/Inf
      - MUST NOT use built-in round()
    """
    if d < 0:
        raise ValueError("d must be >= 0")
    if not math.isfinite(x):
        raise ValueError("NaN/Inf is not allowed")

    sgn = 1.0
    if x < 0.0:
        sgn = -1.0

    ax = abs(x)
    scale = 10.0 ** d
    y = sgn * math.floor(ax * scale + 0.5) / scale

    # normalize -0 -> +0
    if y == 0.0:
        return 0.0
    return y


def fixed(x: float, d: int = 6) -> str:
    """
    Fixed decimal formatting (no scientific notation), after quantization.
    """
    y = q(x, d)
    # Python fixed-point format never uses scientific notation.
    return f"{y:.{d}f}"


def percentile(values: Iterable[float], p: float) -> float:
    """
    higher_order_stat percentile (no interpolation):
      p in (0,1]
      k = ceil(p*N) - 1
      return sorted(values)[k]
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    arr = list(values)
    if not arr:
        raise ValueError("values must not be empty")
    for v in arr:
        if not math.isfinite(float(v)):
            raise ValueError("NaN/Inf is not allowed in values")

    arr.sort()
    n = len(arr)
    k = math.ceil(p * n) - 1
    if k < 0:
        k = 0
    if k >= n:
        k = n - 1
    y = float(arr[k])
    if y == 0.0:
        return 0.0
    return y
