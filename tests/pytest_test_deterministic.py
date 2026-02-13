import math

import pytest

from powerskiving.deterministic import fixed, percentile, q, wrap_rad


def test_wrap_rad_pi_is_negative_pi() -> None:
    assert wrap_rad(math.pi) == -math.pi


def test_wrap_rad_range() -> None:
    values = [
        -1000.0,
        -7.0,
        -math.pi,
        -0.1,
        0.0,
        0.1,
        math.pi,
        7.0,
        1000.0,
    ]
    for v in values:
        out = wrap_rad(v)
        assert -math.pi <= out < math.pi


def test_q_normalizes_negative_zero() -> None:
    assert str(q("-0", 6)) == "0"
    assert str(q(-0.0, 6)) == "0"


def test_fixed_no_scientific_notation() -> None:
    s = fixed("12345678901234567890.123456", 6)
    assert "e" not in s.lower()
    assert s == "12345678901234567890.123456"


def test_fixed_forbids_negative_zero() -> None:
    assert fixed("-0", 6) == "0.000000"
    assert fixed(-0.0, 3) == "0.000"


def test_nan_inf_forbidden() -> None:
    with pytest.raises(ValueError):
        q(float("nan"), 6)
    with pytest.raises(ValueError):
        q(float("inf"), 6)
    with pytest.raises(ValueError):
        wrap_rad(float("nan"))
    with pytest.raises(ValueError):
        wrap_rad(float("inf"))


def test_percentile_higher_order_stat() -> None:
    arr = [1.0, 2.0, 3.0, 4.0]
    assert str(percentile(arr, 0.0)) == "1"
    assert str(percentile(arr, 0.5)) == "2"
    assert str(percentile(arr, 0.75)) == "3"
    assert str(percentile(arr, 1.0)) == "4"
