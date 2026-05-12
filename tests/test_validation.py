import unittest

import numpy as np

import categorical_samplers as cs
from categorical_samplers.validation import (
    categorical_counts,
    categorical_exact_pmf,
    multinomial_counts,
    multinomial_exact_pmf,
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


if __name__ == "__main__":
    unittest.main()
