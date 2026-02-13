import math
import unittest

from powerskiving.geom_kernel import (
    REASON_DEADBAND,
    REASON_SIDE_MISMATCH,
    SOLVER_REASON_OK,
    SOLVER_STATUS_OK,
    STATUS_REJECT,
    solve_theta1_bisect_scan,
)


class TestSolveTheta1BisectScan(unittest.TestCase):
    def test_converges_within_expected_tolerance(self):
        target = 0.7

        def f(theta1: float) -> float:
            return theta1 - target

        def dtheta(theta1: float) -> float:
            _ = theta1
            return 0.2

        res = solve_theta1_bisect_scan(
            f=f,
            dtheta_fn=dtheta,
            side="plus",
            dtheta_deadband_rad=1.0e-3,
            theta1_seed=0.0,
        )

        self.assertEqual(res.status, SOLVER_STATUS_OK)
        self.assertEqual(res.reason_code, SOLVER_REASON_OK)
        self.assertIsNotNone(res.theta1_rad)
        self.assertIsNotNone(res.residual_abs)
        assert res.theta1_rad is not None
        assert res.residual_abs is not None
        self.assertLessEqual(abs(res.theta1_rad - target), 1.0e-10)
        self.assertLessEqual(res.residual_abs, 1.0e-12)
        self.assertEqual(math.copysign(1.0, res.theta1_rad), 1.0)

    def test_rejects_deadband_and_side_mismatch(self):
        target = -0.4

        def f(theta1: float) -> float:
            return theta1 - target

        deadband_res = solve_theta1_bisect_scan(
            f=f,
            dtheta_fn=lambda _: 0.0,
            side="plus",
            dtheta_deadband_rad=1.0e-6,
            theta1_seed=0.0,
        )
        self.assertEqual(deadband_res.status, STATUS_REJECT)
        self.assertEqual(deadband_res.reason_code, REASON_DEADBAND)
        self.assertIsNone(deadband_res.theta1_rad)
        self.assertIsNone(deadband_res.residual_abs)

        side_res = solve_theta1_bisect_scan(
            f=f,
            dtheta_fn=lambda _: -0.2,
            side="plus",
            dtheta_deadband_rad=1.0e-6,
            theta1_seed=0.0,
        )
        self.assertEqual(side_res.status, STATUS_REJECT)
        self.assertEqual(side_res.reason_code, REASON_SIDE_MISMATCH)
        self.assertIsNone(side_res.theta1_rad)
        self.assertIsNone(side_res.residual_abs)

    def test_deterministic_same_input_same_output(self):
        def f(theta1: float) -> float:
            return math.cos(theta1) - 0.25

        def dtheta(theta1: float) -> float:
            return 0.15 + 0.0 * theta1

        kwargs = dict(
            f=f,
            dtheta_fn=dtheta,
            side="plus",
            dtheta_deadband_rad=1.0e-4,
            theta1_seed=0.123,
        )
        res1 = solve_theta1_bisect_scan(**kwargs)
        res2 = solve_theta1_bisect_scan(**kwargs)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
