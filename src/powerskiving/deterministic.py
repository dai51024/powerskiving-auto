"""Deterministic numeric helpers for powerskiving."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
from typing import Iterable

_PI = math.pi
_TAU = 2.0 * math.pi


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert numeric input to Decimal deterministically."""
    if isinstance(value, Decimal):
        d = value
    elif isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN/Inf is not allowed")
        d = Decimal(str(value))
    elif isinstance(value, str):
        try:
            d = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid numeric string: {value}") from exc
    else:
        raise TypeError(f"unsupported type: {type(value)!r}")

    if not d.is_finite():
        raise ValueError("NaN/Inf is not allowed")
    return d


def _normalize_neg_zero_decimal(d: Decimal) -> Decimal:
    if d.is_zero():
        return Decimal("0")
    return d


def wrap_rad(rad: float) -> float:
    """Wrap radians into [-pi, pi) and force wrap_rad(pi) == -pi."""
    if not math.isfinite(rad):
        raise ValueError("NaN/Inf is not allowed")
    wrapped = rad - _TAU * math.floor((rad + _PI) / _TAU)
    if wrapped == 0.0:
        return 0.0
    return wrapped


def q(value: float | int | str | Decimal, digits: int = 6) -> Decimal:
    """Deterministic fixed-point quantization (ROUND_HALF_UP)."""
    if digits < 0:
        raise ValueError("digits must be >= 0")

    d = _to_decimal(value)
    quantum = Decimal("1").scaleb(-digits)
    qd = d.quantize(quantum, rounding=ROUND_HALF_UP)
    return _normalize_neg_zero_decimal(qd)


def fixed(value: float | int | str | Decimal, digits: int = 6) -> str:
    """Format value in non-scientific fixed notation."""
    qd = q(value, digits)
    s = format(qd, "f")
    if digits == 0:
        if "." in s:
            return s.split(".", 1)[0]
        return s

    if "." not in s:
        return f"{s}.{'0' * digits}"

    head, tail = s.split(".", 1)
    if len(tail) < digits:
        tail = tail + ("0" * (digits - len(tail)))
    elif len(tail) > digits:
        tail = tail[:digits]
    return f"{head}.{tail}"


def percentile(values: Iterable[float | int | str | Decimal], p: float) -> Decimal:
    """higher_order_stat percentile with no interpolation."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    arr = [_to_decimal(v) for v in values]
    if not arr:
        raise ValueError("values must not be empty")

    arr.sort()
    n = len(arr)
    k = math.ceil(p * n) - 1
    if k < 0:
        k = 0
    if k >= n:
        k = n - 1
    return _normalize_neg_zero_decimal(arr[k])
