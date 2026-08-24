import unittest
from collections import Counter

import numpy as np

import categorical_samplers as cs

try:
    import cb_sampler  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "cb_sampler":
        raise
    HAS_CB_SAMPLER = False
else:
    HAS_CB_SAMPLER = True


@unittest.skipUnless(HAS_CB_SAMPLER, "cb-sampler optional dependency is not installed")
class MultinomialTunerTests(unittest.TestCase):
    K = 100
    P = np.array([0.5, 0.3, 0.2])
    SIZE = 64

    def test_context_is_permutation_invariant(self):
        left = cs.MultinomialTuner.context(self.K, self.P, self.SIZE)
        right = cs.MultinomialTuner.context(self.K, self.P[::-1], self.SIZE)
        self.assertEqual(left, right)

    def test_each_method_is_warmed_once(self):
        with cs.MultinomialTuner(seed=11) as tuner:
            choices = []
            for elapsed in (900.0, 500.0, 100.0):
                method = tuner.choose(self.K, self.P, self.SIZE)
                choices.append(method)
                tuner.observe(self.K, self.P, self.SIZE, method, elapsed * self.SIZE)
            self.assertEqual(choices, ["pivot", "smallk-cdf", "rep"])
            report = tuner.diagnostics()["contexts"][0]
            self.assertTrue(report["ready"])
            self.assertTrue(
                all(arm["observations"] == 1 for arm in report["methods"].values())
            )

    def test_stationary_fast_arm_dominates(self):
        latency = {"pivot": 900.0, "smallk-cdf": 500.0, "rep": 100.0}
        with cs.MultinomialTuner(seed=11) as tuner:
            choices = []
            for _ in range(500):
                method = tuner.choose(self.K, self.P, self.SIZE)
                tuner.observe(
                    self.K, self.P, self.SIZE, method, latency[method] * self.SIZE
                )
                choices.append(method)
        self.assertGreater(Counter(choices)["rep"], 450)

    def test_policy_adapts_after_performance_reversal(self):
        before = {"pivot": 900.0, "smallk-cdf": 500.0, "rep": 100.0}
        after = {"pivot": 50.0, "smallk-cdf": 400.0, "rep": 1000.0}
        with cs.MultinomialTuner(seed=11) as tuner:
            choices = []
            for index in range(1200):
                method = tuner.choose(self.K, self.P, self.SIZE)
                latency = before if index < 150 else after
                tuner.observe(
                    self.K, self.P, self.SIZE, method, latency[method] * self.SIZE
                )
                choices.append(method)
        self.assertGreater(Counter(choices[:150])["rep"], 125)
        self.assertGreater(Counter(choices[-300:])["pivot"], 275)

    def test_reset_discards_contexts_and_close_stops_use(self):
        tuner = cs.MultinomialTuner(seed=11)
        method = tuner.choose(self.K, self.P, self.SIZE)
        tuner.observe(self.K, self.P, self.SIZE, method, 100.0)
        self.assertEqual(len(tuner.diagnostics()["contexts"]), 1)
        tuner.reset()
        self.assertEqual(tuner.diagnostics()["contexts"], [])
        tuner.close()
        with self.assertRaises(RuntimeError):
            tuner.choose(self.K, self.P, self.SIZE)


@unittest.skipUnless(
    HAS_CB_SAMPLER and cs.native_available(),
    "adaptive integration requires cb-sampler and the native backend",
)
class MultinomialTunerIntegrationTests(unittest.TestCase):
    def test_adaptive_api_warms_all_native_methods(self):
        p = np.array([0.2, 0.3, 0.5])
        with cs.MultinomialTuner(seed=29) as tuner:
            draws = [
                cs.multinomial(25, p, size=32, seed=index, tuner=tuner)
                for index in range(3)
            ]
            methods = tuner.diagnostics()["contexts"][0]["methods"]
        self.assertTrue(all(np.all(draw.sum(axis=1) == 25) for draw in draws))
        self.assertTrue(all(arm["observations"] == 1 for arm in methods.values()))

    def test_multinomial_stream_can_tune_refills(self):
        with cs.MultinomialTuner(seed=31) as tuner:
            stream = cs.MultinomialStream(
                25,
                [0.2, 0.3, 0.5],
                seed=7,
                buffer_size=8,
                tuner=tuner,
            )
            draws = stream.draw(25)
            methods = tuner.diagnostics()["contexts"][0]["methods"]
        self.assertTrue(np.all(draws.sum(axis=1) == 25))
        self.assertEqual(sum(arm["observations"] for arm in methods.values()), 4)


if __name__ == "__main__":
    unittest.main()
