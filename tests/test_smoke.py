import unittest

import numpy as np

import categorical_samplers as cs


class SamplerSmokeTests(unittest.TestCase):
    def test_version_is_exposed(self):
        self.assertEqual(cs.__version__, "0.1.0a1")

    def test_seeded_binomial_is_reproducible(self):
        a = cs.binomial(50, 0.3, size=100, seed=123)
        b = cs.binomial(50, 0.3, size=100, seed=123)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.dtype, np.int64)

    def test_categorical_uses_probability_vector(self):
        draws = cs.categorical([0.0, 1.0, 0.0], size=20, seed=1)
        self.assertTrue(np.array_equal(draws, np.ones(20, dtype=np.int64)))

    def test_categorical_index_base_is_explicit(self):
        draws = cs.categorical([0.0, 1.0], size=4, seed=1, index_base=1)
        self.assertTrue(np.array_equal(draws, np.full(4, 2, dtype=np.int64)))
        with self.assertRaises(ValueError):
            cs.categorical([0.5, 0.5], index_base=2)
        with self.assertRaises(TypeError):
            cs.categorical([0.5, 0.5], index_base=0.5)

    def test_multinomial_rows_sum_to_k(self):
        draws = cs.multinomial(17, [0.2, 0.3, 0.5], size=25, seed=7)
        self.assertEqual(draws.shape, (25, 3))
        self.assertTrue(np.all(draws.sum(axis=1) == 17))

    def test_streams_are_buffered(self):
        stream = cs.MultinomialStream(5, [0.25, 0.75], seed=9, buffer_size=3)
        draws = np.vstack([stream.draw(2), stream.draw(4)])
        self.assertEqual(draws.shape, (6, 2))
        self.assertTrue(np.all(draws.sum(axis=1) == 5))

    def test_invalid_method_names_are_rejected(self):
        with self.assertRaises(ValueError):
            cs.binomial(5, 0.5, method="typo")
        with self.assertRaises(ValueError):
            cs.categorical([0.5, 0.5], method="typo")
        with self.assertRaises(ValueError):
            cs.multinomial(5, [0.5, 0.5], method="typo")
        with self.assertRaises(ValueError):
            cs.multinomial(5, [0.5, 0.5], method="adaptive")

    def test_counts_must_be_exact_integers(self):
        with self.assertRaises(TypeError):
            cs.binomial(5.5, 0.5)
        with self.assertRaises(TypeError):
            cs.categorical([0.5, 0.5], size=2.5)
        with self.assertRaises(TypeError):
            cs.multinomial(5.5, [0.5, 0.5])
        with self.assertRaises(TypeError):
            cs.multinomial(5, [0.5, 0.5], size=False)

    def test_stream_draws_do_not_depend_on_chunking(self):
        cases = (
            (
                cs.BinomialStream(20, 0.3, seed=17, buffer_size=5),
                cs.BinomialStream(20, 0.3, seed=17, buffer_size=5),
                np.concatenate,
            ),
            (
                cs.CategoricalStream([0.2, 0.3, 0.5], seed=17, buffer_size=5),
                cs.CategoricalStream([0.2, 0.3, 0.5], seed=17, buffer_size=5),
                np.concatenate,
            ),
            (
                cs.MultinomialStream(20, [0.2, 0.3, 0.5], seed=17, buffer_size=5),
                cs.MultinomialStream(20, [0.2, 0.3, 0.5], seed=17, buffer_size=5),
                np.vstack,
            ),
        )
        for one_call, chunked, combine in cases:
            with self.subTest(stream=type(one_call).__name__):
                expected = one_call.draw(13)
                actual = combine([chunked.draw(2), chunked.draw(4), chunked.draw(7)])
                self.assertTrue(np.array_equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
