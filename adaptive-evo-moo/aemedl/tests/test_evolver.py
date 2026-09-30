import unittest

import numpy as np

from adaptive_evo import (
    AdaptiveEvolutionaryClassifier, EvoConfig, OnlineMLP, StaticMLP, make_stream, prequential,
)
from adaptive_evo.evolver import OBJECTIVES


def _small_cfg(**kw) -> EvoConfig:
    base = dict(n_classes=3, pop_size=10, max_hidden=12, generations=1, drift_generations=3)
    base.update(kw)
    return EvoConfig(**base)


class TestConfig(unittest.TestCase):
    def test_validation(self) -> None:
        for bad in (dict(n_classes=1), dict(min_hidden=0), dict(min_hidden=9, max_hidden=4),
                    dict(pop_size=2), dict(recent_size=3), dict(immigrant_frac=1.0),
                    dict(lr_bounds=(0.5, 0.1))):
            with self.assertRaises(ValueError):
                EvoConfig(**{"n_classes": 3, **bad})


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = make_stream("sudden", n_samples=1800, n_drifts=2, seed=4)

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            AdaptiveEvolutionaryClassifier(_small_cfg()).predict(np.zeros((2, 8)))

    def test_input_validation(self) -> None:
        m = AdaptiveEvolutionaryClassifier(_small_cfg())
        with self.assertRaises(ValueError):
            m.partial_fit(np.zeros((5, 3)), np.zeros(4, dtype=int))
        with self.assertRaises(ValueError):
            m.partial_fit(np.full((5, 3), np.nan), np.zeros(5, dtype=int))
        with self.assertRaises(ValueError):
            m.partial_fit(np.zeros((5, 3)), np.full(5, 7))
        m.partial_fit(np.random.default_rng(0).normal(size=(20, 3)), np.zeros(20, dtype=int))
        with self.assertRaises(ValueError):
            m.partial_fit(np.zeros((5, 4)), np.zeros(5, dtype=int))

    def test_learns_above_chance_on_stationary_concept(self) -> None:
        s = make_stream("sudden", n_samples=1500, n_drifts=1, seed=7)
        m = AdaptiveEvolutionaryClassifier(_small_cfg(seed=7))
        for lo in range(0, 700, 100):
            m.partial_fit(s.X[lo:lo + 100], s.y[lo:lo + 100])
        acc = np.mean(m.predict(s.X[700:800]) == s.y[700:800])
        self.assertGreater(acc, 0.6)  # chance ~ 0.33

    def test_probabilities_valid(self) -> None:
        m = AdaptiveEvolutionaryClassifier(_small_cfg())
        m.partial_fit(self.stream.X[:100], self.stream.y[:100])
        p = m.predict_proba(self.stream.X[100:150])
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue(np.all(p >= 0))

    def test_reproducible_with_seed(self) -> None:
        outs = []
        for _ in range(2):
            m = AdaptiveEvolutionaryClassifier(_small_cfg(seed=11))
            for lo in range(0, 300, 100):
                m.partial_fit(self.stream.X[lo:lo + 100], self.stream.y[lo:lo + 100])
            outs.append(m.predict_proba(self.stream.X[300:340]))
        np.testing.assert_allclose(outs[0], outs[1])

    def test_pareto_front_is_non_dominated_and_shaped(self) -> None:
        m = AdaptiveEvolutionaryClassifier(_small_cfg())
        for lo in range(0, 300, 100):
            m.partial_fit(self.stream.X[lo:lo + 100], self.stream.y[lo:lo + 100])
        F = m.pareto_front()
        self.assertEqual(F.shape[1], len(OBJECTIVES))
        for i in range(len(F)):
            for j in range(len(F)):
                if i != j:
                    self.assertFalse(np.all(F[i] <= F[j]) and np.any(F[i] < F[j]))
        self.assertTrue(np.all((F >= 0) & (F <= 1.5)))

    def test_architecture_respects_bounds(self) -> None:
        cfg = _small_cfg(min_hidden=3, max_hidden=8)
        m = AdaptiveEvolutionaryClassifier(cfg)
        for lo in range(0, 500, 100):
            m.partial_fit(self.stream.X[lo:lo + 100], self.stream.y[lo:lo + 100])
        self.assertTrue(all(cfg.min_hidden <= g.n_active <= cfg.max_hidden for g in m._pop))

    def test_detects_sudden_drift_and_adapts_better_than_static(self) -> None:
        s = make_stream("sudden", n_samples=2400, n_drifts=1, seed=2)
        evo = AdaptiveEvolutionaryClassifier(_small_cfg(seed=2))
        r_evo = prequential(evo, s, "evo")
        r_static = prequential(StaticMLP(3), s, "static")
        self.assertTrue(any(11 <= d <= 15 for d in evo.drifts_detected), evo.drifts_detected)
        self.assertGreater(r_evo.accuracy, r_static.accuracy + 0.1)

    def test_memory_buffers_are_bounded(self) -> None:
        cfg = _small_cfg(recent_size=120, old_size=150)
        m = AdaptiveEvolutionaryClassifier(cfg)
        for lo in range(0, 1500, 100):
            m.partial_fit(self.stream.X[lo:lo + 100], self.stream.y[lo:lo + 100])
        self.assertLessEqual(len(m._yr), 120)
        self.assertLessEqual(len(m._yo), 150)


class TestEvaluation(unittest.TestCase):
    def test_prequential_metrics_in_range(self) -> None:
        s = make_stream("sudden", n_samples=1500, n_drifts=2, seed=1)
        r = prequential(OnlineMLP(3), s, "online", chunk_size=100)
        self.assertEqual(len(r.chunk_acc), 14)
        self.assertTrue(0.0 <= r.accuracy <= 1.0)
        self.assertTrue(0.0 <= r.post_drift_accuracy <= 1.0)
        self.assertTrue(0.0 <= r.recovery_chunks <= 5.0)

    def test_perfect_oracle_scores_one(self) -> None:
        s = make_stream("sudden", n_samples=1000, n_drifts=2, seed=1)

        class Oracle:
            def partial_fit(self, X, y):
                return self

            def predict(self, X):
                idx = np.array([np.flatnonzero((s.X == row).all(axis=1))[0] for row in X])
                return s.y[idx]

        r = prequential(Oracle(), s, "oracle")
        self.assertEqual(r.accuracy, 1.0)

    def test_invalid_args(self) -> None:
        s = make_stream("sudden", n_samples=500, n_drifts=1)
        with self.assertRaises(ValueError):
            prequential(OnlineMLP(3), s, "x", chunk_size=1)


if __name__ == "__main__":
    unittest.main()
