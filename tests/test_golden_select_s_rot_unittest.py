import hashlib
import importlib
import json
import math
import unittest
from dataclasses import asdict

from powerskiving.geom_kernel import (
    GOLDEN_REASON_BRACKET_FAIL,
    GOLDEN_REASON_CONVERGENCE_FAIL,
    GOLDEN_REASON_OK,
    GOLDEN_STATUS_OK,
    GOLDEN_STATUS_REJECT,
    golden_select_s_rot,
)
from powerskiving.geom_kernel.tool_conjugate_grid_raw import (
    REASON_OK as RAW_REASON_OK,
    ToolConjugateGridRawPoint,
    ToolConjugateGridRawResult,
)
from powerskiving.deterministic import wrap_rad

golden_mod = importlib.import_module("powerskiving.geom_kernel.golden_select_s_rot")


def _point(*, x: float, y: float, z: float, valid: int, reason_code: str) -> ToolConjugateGridRawPoint:
    return ToolConjugateGridRawPoint(
        iu=0,
        iv=0,
        u_mm=0.0,
        v_mm=0.0,
        x_mm=x,
        y_mm=y,
        z_mm=z,
        nx=0.0,
        ny=0.0,
        nz=1.0,
        theta1_rad=0.0,
        theta2_rad=0.0,
        residual_abs=0.0,
        valid=valid,
        reason_code=reason_code,
    )


def _rz(theta: float, x: float, y: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x - s * y, s * x + c * y


def _ref_pitch_points(*, module_mm: float, z1: int, pressure_angle_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    alpha = math.radians(pressure_angle_deg)
    r_p = 0.5 * module_mm * float(z1)
    r_b = r_p * math.cos(alpha)
    t_p = math.tan(alpha)
    x0 = r_b * (math.cos(t_p) + t_p * math.sin(t_p))
    y0 = r_b * (math.sin(t_p) - t_p * math.cos(t_p))
    phi_p = wrap_rad(math.atan2(y0, x0))
    theta_half = math.pi / (2.0 * float(z1))
    delta = wrap_rad(theta_half - phi_p)
    plus = _rz(delta, x0, y0)
    minus = _rz(-delta, x0, -y0)
    return plus, minus


def _build_legacy_reference_polylines(
    *,
    module_mm: float,
    z1: int,
    pressure_angle_deg: float,
    golden_pitch_band_dr_mm: float,
    golden_ref_n: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]] | None:
    alpha = math.radians(pressure_angle_deg)
    r_p = 0.5 * module_mm * float(z1)
    r_b = r_p * math.cos(alpha)
    if r_b <= 0.0:
        return None
    t_p = math.tan(alpha)
    x_p = r_b * (math.cos(t_p) + t_p * math.sin(t_p))
    y_p = r_b * (math.sin(t_p) - t_p * math.cos(t_p))
    phi_p = wrap_rad(math.atan2(y_p, x_p))
    theta_half = math.pi / (2.0 * float(z1))
    delta = wrap_rad(theta_half - phi_p)

    r_min = r_p - golden_pitch_band_dr_mm
    r_max = r_p + golden_pitch_band_dr_mm
    if r_max < r_min:
        return None
    if r_min < r_b:
        r_min = r_b
    t_min = math.sqrt(max((r_min / r_b) * (r_min / r_b) - 1.0, 0.0))
    t_max = math.sqrt(max((r_max / r_b) * (r_max / r_b) - 1.0, 0.0))
    if t_max < t_min:
        return None

    plus: list[tuple[float, float]] = []
    minus: list[tuple[float, float]] = []
    span = t_max - t_min
    n_ref = float(golden_ref_n)
    for j in range(golden_ref_n):
        t_j = t_min + (float(j) + 0.5) * span / n_ref
        x0 = r_b * (math.cos(t_j) + t_j * math.sin(t_j))
        y0 = r_b * (math.sin(t_j) - t_j * math.cos(t_j))
        plus.append(_rz(delta, x0, y0))
        minus.append(_rz(-delta, x0, -y0))
    return tuple(plus), tuple(minus)


def _rotate_polyline(theta: float, polyline: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    return tuple(_rz(theta, x, y) for x, y in polyline)


def _median_dtheta(polyline: tuple[tuple[float, float], ...], theta_tooth_center_rad: float) -> float:
    values = [wrap_rad(math.atan2(y, x) - theta_tooth_center_rad) for x, y in polyline]
    values.sort()
    return values[(len(values) - 1) // 2]


def _mean_polyline_distance(
    points: tuple[tuple[float, float], ...],
    ref: tuple[tuple[float, float], ...],
) -> float:
    dists = [golden_mod._polyline_min_distance_2d(qx=x, qy=y, polyline=ref) for x, y in points]
    assert all(d is not None for d in dists)
    return sum(float(d) for d in dists) / float(len(dists))


class TestGoldenSelectSRot(unittest.TestCase):
    def _base_kwargs(self):
        return dict(
            module_mm=2.0,
            z1=20,
            z2=60,
            pressure_angle_deg=20.0,
            face_width_mm=10.0,
            center_distance_a_mm=40.0,
            theta_tooth_center_rad=0.0,
            dtheta_deadband_rad=1.0e-6,
            nu=2,
            nv=2,
            grid_u_min_mm=56.5,
            grid_u_max_mm=65.0,
            golden_tol_p95_mm=10.0,
            golden_tol_max_mm=10.0,
            golden_dz_mm=0.1,
            golden_dz_max_mm=0.8,
            golden_min_points=6,
            golden_pitch_band_dr_mm=1.0,
            golden_ref_n=20,
        )

    def test_min_grid_selects_deterministic_s_rot(self):
        def raw_generator(**kwargs):
            plus_ref, minus_ref = _ref_pitch_points(
                module_mm=kwargs["module_mm"],
                z1=kwargs["z1"],
                pressure_angle_deg=kwargs["pressure_angle_deg"],
            )
            offset = 0.0 if kwargs["s_rot"] == 1 else 0.4
            zs = (-0.15, -0.05, 0.0, 0.05)
            plus_points = tuple(
                _point(x=plus_ref[0] + offset, y=plus_ref[1], z=zv, valid=1, reason_code=RAW_REASON_OK) for zv in zs
            )
            minus_points = tuple(
                _point(x=minus_ref[0] + offset, y=minus_ref[1], z=zv, valid=1, reason_code=RAW_REASON_OK) for zv in zs
            )
            return ToolConjugateGridRawResult(
                plus_points=plus_points,
                minus_points=minus_points,
                theta1_jump_count=0,
                theta1_jump_count_plus=0,
                theta1_jump_count_minus=0,
            )

        kwargs = self._base_kwargs()
        res = golden_select_s_rot(raw_generator=raw_generator, **kwargs)
        self.assertEqual(res.status, GOLDEN_STATUS_OK)
        self.assertEqual(res.reason_code, GOLDEN_REASON_OK)
        self.assertEqual(res.s_rot_selected, 1)
        self.assertIsNotNone(res.golden_p95_mm)
        self.assertIsNotNone(res.golden_max_mm)
        self.assertIsNotNone(res.z_mid_tool_mm)
        self.assertIsNotNone(res.golden_dz_used_mm)
        self.assertIsNotNone(res.debug)
        assert res.debug is not None
        self.assertEqual(res.golden_p95_mm, res.debug["raw_worst_p95_mm"])
        self.assertIn("fitted_worst_p95_mm", res.debug)

    def test_rejects_bracketing_failure(self):
        def raw_generator(**kwargs):
            _ = kwargs
            invalid = tuple(_point(x=0.0, y=0.0, z=0.0, valid=0, reason_code="BRACKET_FAIL") for _ in range(2))
            return ToolConjugateGridRawResult(
                plus_points=invalid,
                minus_points=invalid,
                theta1_jump_count=0,
                theta1_jump_count_plus=0,
                theta1_jump_count_minus=0,
            )

        res = golden_select_s_rot(raw_generator=raw_generator, **self._base_kwargs())
        self.assertEqual(res.status, GOLDEN_STATUS_REJECT)
        self.assertEqual(res.reason_code, GOLDEN_REASON_BRACKET_FAIL)
        self.assertIsNone(res.s_rot_selected)

    def test_rejects_convergence_failure(self):
        def raw_generator(**kwargs):
            _ = kwargs
            invalid = tuple(_point(x=0.0, y=0.0, z=0.0, valid=0, reason_code="CONVERGENCE_FAIL") for _ in range(2))
            return ToolConjugateGridRawResult(
                plus_points=invalid,
                minus_points=invalid,
                theta1_jump_count=0,
                theta1_jump_count_plus=0,
                theta1_jump_count_minus=0,
            )

        res = golden_select_s_rot(raw_generator=raw_generator, **self._base_kwargs())
        self.assertEqual(res.status, GOLDEN_STATUS_REJECT)
        self.assertEqual(res.reason_code, GOLDEN_REASON_CONVERGENCE_FAIL)
        self.assertIsNone(res.s_rot_selected)

    def test_deterministic_and_sha256_fixed(self):
        def raw_generator(**kwargs):
            plus_ref, minus_ref = _ref_pitch_points(
                module_mm=kwargs["module_mm"],
                z1=kwargs["z1"],
                pressure_angle_deg=kwargs["pressure_angle_deg"],
            )
            offset = 0.0 if kwargs["s_rot"] == 1 else 0.4
            zs = (-0.15, -0.05, 0.0, 0.05)
            plus_points = tuple(
                _point(x=plus_ref[0] + offset, y=plus_ref[1], z=zv, valid=1, reason_code=RAW_REASON_OK) for zv in zs
            )
            minus_points = tuple(
                _point(x=minus_ref[0] + offset, y=minus_ref[1], z=zv, valid=1, reason_code=RAW_REASON_OK) for zv in zs
            )
            return ToolConjugateGridRawResult(
                plus_points=plus_points,
                minus_points=minus_points,
                theta1_jump_count=0,
                theta1_jump_count_plus=0,
                theta1_jump_count_minus=0,
            )

        kwargs = self._base_kwargs()
        res1 = golden_select_s_rot(raw_generator=raw_generator, **kwargs)
        res2 = golden_select_s_rot(raw_generator=raw_generator, **kwargs)
        self.assertEqual(res1, res2)

        payload = asdict(res1)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        got = hashlib.sha256(blob).hexdigest()
        self.assertEqual(got, "e34fd30ed5b58bd902d63c33ac66870da2e5ac2291c3442efa93c88958a802ec")

    def test_tie_keeps_first_candidate(self):
        def raw_generator(**kwargs):
            plus_ref, minus_ref = _ref_pitch_points(
                module_mm=kwargs["module_mm"],
                z1=kwargs["z1"],
                pressure_angle_deg=kwargs["pressure_angle_deg"],
            )
            zs = (-0.15, -0.05, 0.0, 0.05)
            plus_points = tuple(
                _point(x=plus_ref[0], y=plus_ref[1], z=zv, valid=1, reason_code=RAW_REASON_OK) for zv in zs
            )
            minus_points = tuple(
                _point(x=minus_ref[0], y=minus_ref[1], z=zv, valid=1, reason_code=RAW_REASON_OK) for zv in zs
            )
            return ToolConjugateGridRawResult(
                plus_points=plus_points,
                minus_points=minus_points,
                theta1_jump_count=0,
                theta1_jump_count_plus=0,
                theta1_jump_count_minus=0,
            )

        res = golden_select_s_rot(raw_generator=raw_generator, **self._base_kwargs())
        self.assertEqual(res.status, GOLDEN_STATUS_OK)
        self.assertEqual(res.reason_code, GOLDEN_REASON_OK)
        self.assertEqual(res.s_rot_selected, res.candidate_plus.s_rot)
        self.assertEqual(res.candidate_plus.golden_p95_mm, res.candidate_minus.golden_p95_mm)
        self.assertEqual(res.candidate_plus.golden_max_mm, res.candidate_minus.golden_max_mm)

    def test_reference_labels_follow_dtheta_sign_when_theta_center_nonzero(self):
        kwargs = self._base_kwargs()
        for theta in (-2.356194490192345, 1.200000000000000, -1.700000000000000):
            refs = golden_mod._build_reference_polylines(
                module_mm=kwargs["module_mm"],
                z1=kwargs["z1"],
                pressure_angle_deg=kwargs["pressure_angle_deg"],
                golden_pitch_band_dr_mm=kwargs["golden_pitch_band_dr_mm"],
                golden_ref_n=kwargs["golden_ref_n"],
                theta_tooth_center_rad=theta,
            )
            self.assertIsNotNone(refs)
            assert refs is not None
            ref_plus, ref_minus = refs
            med_plus = _median_dtheta(ref_plus, theta)
            med_minus = _median_dtheta(ref_minus, theta)
            self.assertGreater(med_plus, 0.0)
            self.assertLess(med_minus, 0.0)

    def test_reference_relabel_changes_distance_order_to_same_side(self):
        kwargs = self._base_kwargs()
        theta = -2.356194490192345
        legacy_refs = _build_legacy_reference_polylines(
            module_mm=kwargs["module_mm"],
            z1=kwargs["z1"],
            pressure_angle_deg=kwargs["pressure_angle_deg"],
            golden_pitch_band_dr_mm=kwargs["golden_pitch_band_dr_mm"],
            golden_ref_n=80,
        )
        self.assertIsNotNone(legacy_refs)
        assert legacy_refs is not None
        legacy_plus, legacy_minus = legacy_refs

        current_refs = golden_mod._build_reference_polylines(
            module_mm=kwargs["module_mm"],
            z1=kwargs["z1"],
            pressure_angle_deg=kwargs["pressure_angle_deg"],
            golden_pitch_band_dr_mm=kwargs["golden_pitch_band_dr_mm"],
            golden_ref_n=80,
            theta_tooth_center_rad=theta,
        )
        self.assertIsNotNone(current_refs)
        assert current_refs is not None
        ref_plus, ref_minus = current_refs

        true_a = _rotate_polyline(theta, legacy_plus)
        true_b = _rotate_polyline(theta, legacy_minus)
        med_a = _median_dtheta(true_a, theta)
        med_b = _median_dtheta(true_b, theta)
        self.assertTrue((med_a > 0.0 and med_b < 0.0) or (med_a < 0.0 and med_b > 0.0))
        if med_a > 0.0 and med_b < 0.0:
            points_plus = true_a[::4]
            points_minus = true_b[::4]
        else:
            points_plus = true_b[::4]
            points_minus = true_a[::4]

        legacy_pp = _mean_polyline_distance(points_plus, legacy_plus)
        legacy_pm = _mean_polyline_distance(points_plus, legacy_minus)
        legacy_mm = _mean_polyline_distance(points_minus, legacy_minus)
        legacy_mp = _mean_polyline_distance(points_minus, legacy_plus)

        new_pp = _mean_polyline_distance(points_plus, ref_plus)
        new_pm = _mean_polyline_distance(points_plus, ref_minus)
        new_mm = _mean_polyline_distance(points_minus, ref_minus)
        new_mp = _mean_polyline_distance(points_minus, ref_plus)

        self.assertLess(legacy_pm, legacy_pp)
        self.assertLess(new_pp, new_pm)
        self.assertLess(new_mm, new_mp)


if __name__ == "__main__":
    unittest.main()
