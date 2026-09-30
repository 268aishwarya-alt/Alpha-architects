import unittest

import numpy as np

from adaptive_evo.drift import PageHinkley
from adaptive_evo.streams import STREAM_KINDS, make_stream


class TestStreams(unittest.TestCase):
    def test_all_kinds_shapes_and_labels(self) -> None:
        for kind in STREAM_KINDS:
            s = make_stream(kind, n_samples=1000, n_features=5, n_classes=4, n_drifts=3, seed=1)
            self.assertEqual(s.X.shape, (1000, 5))
            self.assertEqual(s.y.shape, (1000,))
            self.assertTrue(s.y.min() >= 0 and s.y.max() < 4)
            self.assertEqual(len(s.drift_points), 3)
            self.assertTrue(np.all(np.isfinite(s.X)))

    def test_deterministic_given_seed(self) -> None:
        a, b = make_stream("gradual", seed=5), make_stream("gradual", seed=5)
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y, b.y)
        c = make_stream("gradual", seed=6)
        self.assertFalse(np.array_equal(a.y, c.y))

    def test_classes_not_degenerate(self) -> None:
        s = make_stream("sudden", seed=2)
        self.assertGreater(len(np.unique(s.y)), 1)

    def test_concept_actually_drifts(self) -> None:
        s = make_stream("sudden", n_samples=4000, n_drifts=1, label_noise=0.0, seed=3)
        # Same inputs, labels from concept 1 vs concept 0 should disagree often.
        dp = s.drift_points[0]
        from adaptive_evo.streams import _Teacher
        rng = np.random.default_rng(3)
        _ = rng.normal(size=s.X.shape)
        t0, t1 = _Teacher(rng, 8, 3), _Teacher(rng, 8, 3)
        agree = np.mean(t0.logits(s.X[dp:]).argmax(1) == t1.logits(s.X[dp:]).argmax(1))
        self.assertLess(agree, 0.7)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            make_stream("nope")
        with self.assertRaises(ValueError):
            make_stream("sudden", n_classes=1)
        with self.assertRaises(ValueError):
            make_stream("sudden", n_samples=10, n_drifts=4)
        with self.assertRaises(ValueError):
            make_stream("sudden", label_noise=1.0)


class TestPageHinkley(unittest.TestCase):
    def test_no_alarm_on_stationary_noise(self) -> None:
        rng = np.random.default_rng(0)
        ph = PageHinkley()
        alarms = [ph.update(float(np.clip(0.1 + 0.03 * rng.normal(), 0, 1))) for _ in range(300)]
        self.assertFalse(any(alarms))

    def test_alarm_on_step_increase(self) -> None:
        ph = PageHinkley()
        for _ in range(20):
            self.assertFalse(ph.update(0.1))
        fired = [ph.update(0.6) for _ in range(5)]
        self.assertTrue(any(fired))

    def test_resets_after_alarm(self) -> None:
        ph = PageHinkley(threshold=0.2)
        for _ in range(10):
            ph.update(0.1)
        while not ph.update(0.9):
            pass
        self.assertFalse(ph.update(0.9))

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            PageHinkley(threshold=0)


if __name__ == "__main__":
    unittest.main()
