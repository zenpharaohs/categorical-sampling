import unittest

import numpy as np

import categorical_samplers as cs


class SamplerSmokeTests(unittest.TestCase):
    def test_seeded_binomial_is_reproducible(self):
        a = cs.binomial(50, 0.3, size=100, seed=123)
        b = cs.binomial(50, 0.3, size=100, seed=123)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.dtype, np.int64)

    def test_categorical_uses_probability_vector(self):
        draws = cs.categorical([0.0, 1.0, 0.0], size=20, seed=1)
        self.assertTrue(np.array_equal(draws, np.ones(20, dtype=np.int64)))

    def test_multinomial_rows_sum_to_k(self):
        draws = cs.multinomial(17, [0.2, 0.3, 0.5], size=25, seed=7)
        self.assertEqual(draws.shape, (25, 3))
        self.assertTrue(np.all(draws.sum(axis=1) == 17))

    def test_streams_are_buffered(self):
        stream = cs.MultinomialStream(5, [0.25, 0.75], seed=9, buffer_size=3)
        draws = np.vstack([stream.draw(2), stream.draw(4)])
        self.assertEqual(draws.shape, (6, 2))
        self.assertTrue(np.all(draws.sum(axis=1) == 5))


if __name__ == "__main__":
    unittest.main()
