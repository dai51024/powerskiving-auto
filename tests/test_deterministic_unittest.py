import math
import unittest

from powerskiving.deterministic import wrap_rad, q, fixed, percentile

class TestDeterministic(unittest.TestCase):
    def test_wrap_rad_pi(self):
        self.assertEqual(wrap_rad(math.pi), -math.pi)

    def test_q_no_negative_zero(self):
        self.assertEqual(q(-0.0, 8), 0.0)
        self.assertEqual(math.copysign(1.0, q(-0.0, 8)), 1.0)

    def test_q_reject_nan_inf(self):
        with self.assertRaises(Exception):
            q(float("nan"), 8)
        with self.assertRaises(Exception):
            q(float("inf"), 8)

    def test_fixed_no_scientific(self):
        s = fixed(1e-12, 12)
        self.assertNotIn("e", s.lower())
        self.assertNotIn("inf", s.lower())
        self.assertNotIn("nan", s.lower())

    def test_percentile_hos(self):
        xs = [1,2,3,4,5,6,7,8,9,10]
        # higher-order-stat: k = ceil(p*N)-1
        self.assertEqual(percentile(xs, 0.5), 5)
        self.assertEqual(percentile(xs, 1.0), 10)

if __name__ == "__main__":
    unittest.main()
