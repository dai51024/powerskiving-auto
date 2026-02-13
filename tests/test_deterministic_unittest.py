import math
import unittest

from powerskiving.deterministic import wrap_rad, q, fixed, percentile

class TestDeterministic(unittest.TestCase):
    def test_wrap_rad_pi(self):
        self.assertEqual(wrap_rad(math.pi), -math.pi)

    def test_q_no_negative_zero(self):
        self.assertEqual(q(-0.0, 8), 0.0)
        self.assertEqual(math.copysign(1.0, q(-0.0, 8)), 1.0)

    def test_q_formula_half_up(self):
        # boundary checks around .5
        self.assertEqual(q(1.2345649, 6), 1.234565)
        self.assertEqual(q(1.2345644, 6), 1.234564)
        self.assertEqual(q(-1.2345649, 6), -1.234565)

    def test_q_reject_nan_inf(self):
        with self.assertRaises(ValueError):
            q(float("nan"), 8)
        with self.assertRaises(ValueError):
            q(float("inf"), 8)

    def test_fixed_no_scientific(self):
        s = fixed(1e-12, 12)
        self.assertNotIn("e", s.lower())

    def test_percentile_hos(self):
        xs = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(percentile(xs, 0.5), 5.0)
        self.assertEqual(percentile(xs, 1.0), 10.0)

    def test_percentile_reject_zero(self):
        with self.assertRaises(ValueError):
            percentile([1,2,3], 0.0)

if __name__ == "__main__":
    unittest.main()
