import math
import unittest

from powerskiving.geom_kernel import (
    STATUS_INVALID_INPUT,
    STATUS_OK,
    STATUS_OUTSIDE_DOMAIN,
    work_target_surface,
)


class TestWorkTargetSurface(unittest.TestCase):
    def setUp(self):
        self.module_mm = 2.0
        self.z2 = 40
        self.pressure_angle_deg = 20.0

    def _r_base(self) -> float:
        r_pitch = 0.5 * self.module_mm * self.z2
        alpha = self.pressure_angle_deg * math.pi / 180.0
        return r_pitch * math.cos(alpha)

    def test_domain_points_return_finite_values(self):
        r_base = self._r_base()
        for side in ("plus", "minus"):
            res = work_target_surface(
                side=side,
                u_mm=r_base + 1.25,
                v_mm=3.0,
                module_mm=self.module_mm,
                z2=self.z2,
                pressure_angle_deg=self.pressure_angle_deg,
            )
            self.assertEqual(res.status, STATUS_OK)
            self.assertIsNotNone(res.p_W0)
            self.assertIsNotNone(res.n_W0)
            assert res.p_W0 is not None
            assert res.n_W0 is not None
            self.assertTrue(all(math.isfinite(v) for v in res.p_W0))
            self.assertTrue(all(math.isfinite(v) for v in res.n_W0))

    def test_outside_domain_returns_status(self):
        r_base = self._r_base()
        res = work_target_surface(
            side="plus",
            u_mm=r_base - 1.0e-6,
            v_mm=0.0,
            module_mm=self.module_mm,
            z2=self.z2,
            pressure_angle_deg=self.pressure_angle_deg,
        )
        self.assertEqual(res.status, STATUS_OUTSIDE_DOMAIN)
        self.assertIsNone(res.p_W0)
        self.assertIsNone(res.n_W0)

    def test_boundary_u_equal_r_base_is_not_outside_domain(self):
        r_base = self._r_base()
        res = work_target_surface(
            side="plus",
            u_mm=r_base,
            v_mm=0.0,
            module_mm=self.module_mm,
            z2=self.z2,
            pressure_angle_deg=self.pressure_angle_deg,
        )
        self.assertEqual(res.status, STATUS_OK)
        self.assertIsNotNone(res.p_W0)
        self.assertIsNotNone(res.n_W0)

    def test_u_less_than_r_base_is_outside_domain(self):
        r_base = self._r_base()
        res = work_target_surface(
            side="minus",
            u_mm=r_base - 0.1,
            v_mm=1.0,
            module_mm=self.module_mm,
            z2=self.z2,
            pressure_angle_deg=self.pressure_angle_deg,
        )
        self.assertEqual(res.status, STATUS_OUTSIDE_DOMAIN)
        self.assertIsNone(res.p_W0)
        self.assertIsNone(res.n_W0)

    def test_nan_inf_inputs_return_invalid_input_status(self):
        r_base = self._r_base()
        bad_u_values = (math.nan, math.inf, -math.inf)
        for bad_u in bad_u_values:
            with self.subTest(u_mm=bad_u):
                res = work_target_surface(
                    side="plus",
                    u_mm=bad_u,
                    v_mm=0.0,
                    module_mm=self.module_mm,
                    z2=self.z2,
                    pressure_angle_deg=self.pressure_angle_deg,
                )
                self.assertEqual(res.status, STATUS_INVALID_INPUT)
                self.assertIsNone(res.p_W0)
                self.assertIsNone(res.n_W0)

        bad_v_values = (math.nan, math.inf, -math.inf)
        for bad_v in bad_v_values:
            with self.subTest(v_mm=bad_v):
                res = work_target_surface(
                    side="plus",
                    u_mm=r_base + 1.0,
                    v_mm=bad_v,
                    module_mm=self.module_mm,
                    z2=self.z2,
                    pressure_angle_deg=self.pressure_angle_deg,
                )
                self.assertEqual(res.status, STATUS_INVALID_INPUT)
                self.assertIsNone(res.p_W0)
                self.assertIsNone(res.n_W0)

    def test_same_input_same_output(self):
        r_base = self._r_base()
        kwargs = dict(
            side="minus",
            u_mm=r_base + 2.0,
            v_mm=-4.0,
            module_mm=self.module_mm,
            z2=self.z2,
            pressure_angle_deg=self.pressure_angle_deg,
        )
        res1 = work_target_surface(**kwargs)
        res2 = work_target_surface(**kwargs)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
