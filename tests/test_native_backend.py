import unittest

import numpy as np

import categorical_samplers as cs


@unittest.skipUnless(cs.native_available(), "native backend is not built")
class NativeBackendTests(unittest.TestCase):
    def test_native_backend_loads(self):
        self.assertTrue(cs.native_available())

    def test_native_binomial_is_reproducible(self):
        a = cs.binomial(1000, 0.125, size=256, seed=123, method="centerout")
        b = cs.binomial(1000, 0.125, size=256, seed=123, method="centerout")
        self.assertTrue(np.array_equal(a, b))

    def test_native_binomial_mean_smoke(self):
        draws = cs.binomial(200, 0.37, size=20000, seed=99, method="centerout")
        self.assertLess(abs(float(draws.mean()) - 74.0), 0.5)

    def test_native_wait2_mean_smoke(self):
        draws = cs.binomial(5000, 0.002, size=1000000, seed=123, method="wait2")
        self.assertLess(abs(float(draws.mean()) - 10.0), 0.015)

    def test_native_auto_matches_wait2_for_sparse_case(self):
        auto = cs.binomial(5000, 0.002, size=256, seed=101, method="auto")
        wait2 = cs.binomial(5000, 0.002, size=256, seed=101, method="wait2")
        self.assertTrue(np.array_equal(auto, wait2))


if __name__ == "__main__":
    unittest.main()
