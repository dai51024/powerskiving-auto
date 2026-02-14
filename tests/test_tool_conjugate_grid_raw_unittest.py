import hashlib
import json
import math
import unittest
from dataclasses import asdict, fields

from powerskiving.geom_kernel import (
    RAW_REASON_INVALID_NORMAL,
    RAW_REASON_OK,
    RAW_REASON_OUTSIDE_DOMAIN,
    RAW_REASON_SOLVER_FAIL,
    REASON_NAN_FORBIDDEN_REPLACED_WITH_ZERO,
    tool_conjugate_grid_raw,
)


def _is_negative_zero(x: float) -> bool:
    return x == 0.0 and math.copysign(1.0, x) < 0.0


class TestToolConjugateGridRaw(unittest.TestCase):
    def _run_small_grid(self):
        return tool_conjugate_grid_raw(
            module_mm=2.0,
            z1=20,
            z2=60,
            pressure_angle_deg=20.0,
            face_width_mm=10.0,
            center_distance_a_mm=40.0,
            sigma_rad=0.2617993877991494,
            theta_tooth_center_rad=0.0,
            dtheta_deadband_rad=0.03490658503988659,
            nu=7,
            nv=5,
            grid_u_min_mm=56.5,
            grid_u_max_mm=65.0,
            s_rot=1,
        )

    def test_shape_and_required_fields(self):
        res = self._run_small_grid()
        self.assertEqual(len(res.plus_points), 35)
        self.assertEqual(len(res.minus_points), 35)

        expected_fields = {
            "iu",
            "iv",
            "u_mm",
            "v_mm",
            "x_mm",
            "y_mm",
            "z_mm",
            "nx",
            "ny",
            "nz",
            "theta1_rad",
            "theta2_rad",
            "residual_abs",
            "valid",
            "reason_code",
        }
        point_fields = {f.name for f in fields(res.plus_points[0])}
        self.assertSetEqual(point_fields, expected_fields)

    def test_valid_count_and_reason_set(self):
        res = self._run_small_grid()
        all_points = list(res.plus_points) + list(res.minus_points)
        valid_count = sum(p.valid for p in all_points)
        self.assertGreater(valid_count, 0)

        expected_reasons = {
            RAW_REASON_OK,
            RAW_REASON_SOLVER_FAIL,
            RAW_REASON_OUTSIDE_DOMAIN,
            RAW_REASON_INVALID_NORMAL,
            REASON_NAN_FORBIDDEN_REPLACED_WITH_ZERO,
        }
        actual_reasons = {p.reason_code for p in all_points}
        self.assertTrue(actual_reasons.issubset(expected_reasons))

    def test_finite_and_no_negative_zero_outputs(self):
        res = self._run_small_grid()
        all_points = list(res.plus_points) + list(res.minus_points)
        for p in all_points:
            self.assertIsInstance(p.valid, int)
            self.assertIn(p.valid, (0, 1))
            values = (
                p.u_mm,
                p.v_mm,
                p.x_mm,
                p.y_mm,
                p.z_mm,
                p.nx,
                p.ny,
                p.nz,
                p.theta1_rad,
                p.theta2_rad,
                p.residual_abs,
            )
            for v in values:
                self.assertTrue(math.isfinite(v))
                self.assertFalse(_is_negative_zero(v))

    def test_deterministic_and_sha256_fixed(self):
        res1 = self._run_small_grid()
        res2 = self._run_small_grid()
        self.assertEqual(res1, res2)

        payload = {
            "plus_points": [asdict(p) for p in res1.plus_points],
            "minus_points": [asdict(p) for p in res1.minus_points],
            "theta1_jump_count": res1.theta1_jump_count,
            "theta1_jump_count_plus": res1.theta1_jump_count_plus,
            "theta1_jump_count_minus": res1.theta1_jump_count_minus,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        got = hashlib.sha256(blob).hexdigest()
        self.assertEqual(got, "51a10d990c323eb46edb22af94a537460a3650bb2b54a95caaac7e6c794f2481")


if __name__ == "__main__":
    unittest.main()
