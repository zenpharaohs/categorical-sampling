import unittest
import importlib.util

import numpy as np

import categorical_samplers as cs
from categorical_samplers.validation import (
    categorical_counts,
    categorical_exact_pmf,
    multinomial_counts,
    multinomial_exact_pmf,
    randomized_categorical_pit,
    validate_categorical_draws,
)


class ValidationSmokeTests(unittest.TestCase):
    def test_iid_bounds_are_ordered(self):
        out = cs.bhattacharyya_iid_bounds((0.99, 0.995), threshold=0.5)
        self.assertLessEqual(out["affinity_lower"], out["affinity_upper"])
        self.assertLessEqual(out["n_cross_lower"], out["n_cross_upper"])

    def test_empirical_categorical_affinity(self):
        p = [0.2, 0.3, 0.5]
        draws = cs.categorical(p, size=10000, seed=10)
        affinity = cs.empirical_affinity(categorical_exact_pmf(p), categorical_counts(draws))
        self.assertGreater(affinity, 0.99)

    def test_small_multinomial_affinity(self):
        p = [0.2, 0.3, 0.5]
        draws = cs.multinomial(4, p, size=20000, seed=11)
        affinity = cs.empirical_affinity(multinomial_exact_pmf(4, p), multinomial_counts(draws))
        self.assertGreater(affinity, 0.99)
        self.assertTrue(np.all(draws.sum(axis=1) == 4))

    def test_randomized_categorical_pit_uses_category_intervals(self):
        p = [0.2, 0.3, 0.5]
        draws = np.array([0, 1, 2, 2])
        u = randomized_categorical_pit(draws, p, rng=np.random.default_rng(123))
        self.assertEqual(u.shape, draws.shape)
        self.assertTrue(np.all((0.0 < u) & (u < 1.0)))
        self.assertTrue(0.0 <= u[0] <= 0.2)
        self.assertTrue(0.2 <= u[1] <= 0.5)
        self.assertTrue(np.all((0.5 <= u[2:]) & (u[2:] < 1.0)))

    def test_combined_validation_always_reports_discrete_affinity(self):
        p = [0.2, 0.3, 0.5]
        draws = cs.categorical(p, size=10000, seed=12)
        report = validate_categorical_draws(draws, p, rng=13)
        self.assertEqual(report["n"], draws.size)
        self.assertGreater(report["discrete_affinity"], 0.99)
        self.assertEqual(report["pit"].shape, draws.shape)

    @unittest.skipUnless(
        importlib.util.find_spec("hellinger_qualify") and importlib.util.find_spec("streaming_pit_validate"),
        "optional PIT validation packages are not installed",
    )
    def test_combined_validation_uses_optional_pit_packages(self):
        p = [0.2, 0.3, 0.5]
        draws = cs.categorical(p, size=20000, seed=14)
        report = validate_categorical_draws(draws, p, rng=15, d_max=32)
        self.assertIsNotNone(report["legendre"])
        self.assertIsNotNone(report["hellinger"])
        self.assertIsNotNone(report["hellinger_cv"])
        self.assertLess(report["hellinger_cv"].adjusted_h2, 1e-3)


if __name__ == "__main__":
    unittest.main()
