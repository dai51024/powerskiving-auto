import hashlib
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
        self.assertEqual(got, "6bc0004eca967a893893c14fc7e8cabe8ddb83d6c1776a6c3cf9bb900a0d4f8b")

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


if __name__ == "__main__":
    unittest.main()
