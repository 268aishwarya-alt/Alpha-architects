import unittest

import numpy as np

from adaptive_evo.network import (
    cross_entropy, error_rate, loss_and_grads, predict_proba, random_genome, sgd_steps, softmax,
)


def _genome(seed: int = 0, hidden: int = 6):
    return random_genome(np.random.default_rng(seed), 4, 3, hidden, 2, 0.1, (0.05, 0.2))


class TestNetwork(unittest.TestCase):
    def test_softmax_rows_sum_to_one_and_stable(self) -> None:
        p = softmax(np.array([[1000.0, 1001.0, 999.0], [-5.0, 0.0, 5.0]]))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(12, 4))
        Y = np.eye(3)[rng.integers(0, 3, 12)]
        W1, b1 = rng.normal(size=(4, 5)) * 0.5, rng.normal(size=5) * 0.1
        W2, b2 = rng.normal(size=(5, 3)) * 0.5, rng.normal(size=3) * 0.1
        _, grads = loss_and_grads(W1, b1, W2, b2, X, Y, 1e-3)
        params = [W1, b1, W2, b2]
        eps = 1e-6
        for arr, gr in zip(params, grads):
            flat, gflat = arr.reshape(-1), gr.reshape(-1)
            for k in range(0, flat.size, max(1, flat.size // 4)):
                old = flat[k]
                flat[k] = old + eps
                lp = loss_and_grads(*params, X, Y, 1e-3)[0]
                flat[k] = old - eps
                lm = loss_and_grads(*params, X, Y, 1e-3)[0]
                flat[k] = old
                self.assertAlmostEqual((lp - lm) / (2 * eps), gflat[k], places=5)

    def test_sgd_reduces_loss(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
        g = _genome()
        before = cross_entropy(predict_proba(g, X), y)
        sgd_steps(g, X, y, 100, lr=0.2)
        self.assertLess(cross_entropy(predict_proba(g, X), y), before)

    def test_inactive_units_are_untouched_by_training(self) -> None:
        g = _genome()
        off = np.flatnonzero(~g.mask)
        w1_before, w2_before = g.W1[:, off].copy(), g.W2[off].copy()
        X = np.random.default_rng(0).normal(size=(30, 4))
        sgd_steps(g, X, np.zeros(30, dtype=np.int64), 10)
        np.testing.assert_array_equal(g.W1[:, off], w1_before)
        np.testing.assert_array_equal(g.W2[off], w2_before)

    def test_masking_changes_output(self) -> None:
        g = _genome()
        X = np.random.default_rng(0).normal(size=(10, 4))
        p1 = predict_proba(g, X)
        g.mask[np.flatnonzero(g.mask)[0]] = False
        self.assertFalse(np.allclose(p1, predict_proba(g, X)))

    def test_metrics(self) -> None:
        p = np.array([[0.9, 0.1], [0.2, 0.8]])
        y = np.array([0, 0])
        self.assertEqual(error_rate(p, y), 0.5)
        self.assertAlmostEqual(cross_entropy(p, y), -(np.log(0.9) + np.log(0.2)) / 2, places=6)

    def test_genome_copy_is_deep(self) -> None:
        g = _genome()
        c = g.copy()
        c.W1[0, 0] += 1.0
        c.mask[:] = False
        self.assertNotEqual(g.W1[0, 0], c.W1[0, 0])
        self.assertTrue(g.mask.any())


if __name__ == "__main__":
    unittest.main()
