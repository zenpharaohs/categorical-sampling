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

    def test_native_multinomial_is_reproducible(self):
        p = np.array([0.05, 0.15, 0.30, 0.50])
        for method in ("pivot", "smallk-cdf", "rep"):
            with self.subTest(method=method):
                a = cs.multinomial(25, p, size=128, seed=123, method=method)
                b = cs.multinomial(25, p, size=128, seed=123, method=method)
                self.assertTrue(np.array_equal(a, b))
                self.assertEqual(a.dtype, np.int64)
                self.assertTrue(np.all(a.sum(axis=1) == 25))

    def test_native_multinomial_mean_smoke(self):
        p = np.array([0.05, 0.15, 0.30, 0.50])
        K = 40
        for method in ("pivot", "smallk-cdf", "rep"):
            with self.subTest(method=method):
                draws = cs.multinomial(K, p, size=200000, seed=321, method=method)
                means = draws.mean(axis=0)
                se = np.sqrt(K * p * (1.0 - p) / draws.shape[0])
                z = np.abs(means - K * p) / se
                self.assertTrue(np.all(z < 4.0), f"{method}: multinomial means too far: z={z}")


if __name__ == "__main__":
    unittest.main()
