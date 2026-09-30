"""Minimal NumPy MLP genome with an evolvable hidden-unit mask.

A genome stores weights for ``max_hidden`` units but only units flagged in
``mask`` participate in the forward pass, so *architecture* is evolvable
without changing array shapes. Gradient steps only touch active units
(Lamarckian local search: learned weights are written back to the genome).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass
class Genome:
    """One individual: weights, active-unit mask and self-adaptive strategy params."""

    W1: FloatArray
    b1: FloatArray
    W2: FloatArray
    b2: FloatArray
    mask: BoolArray
    lr: float
    sigma: float
    fitness: FloatArray = field(default_factory=lambda: np.zeros(0))
    rank: int = 0
    crowding: float = 0.0

    @property
    def n_active(self) -> int:
        return int(self.mask.sum())

    def copy(self) -> "Genome":
        return Genome(
            self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(),
            self.mask.copy(), self.lr, self.sigma, self.fitness.copy(), self.rank, self.crowding,
        )


def softmax(z: FloatArray) -> FloatArray:
    """Row-wise numerically stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def random_genome(
    rng: np.random.Generator,
    n_features: int,
    n_classes: int,
    max_hidden: int,
    min_hidden: int,
    sigma: float,
    lr_bounds: tuple[float, float],
) -> Genome:
    """Sample a random genome with a random number of active hidden units."""
    n_on = int(rng.integers(min_hidden, max_hidden + 1))
    mask = np.zeros(max_hidden, dtype=bool)
    mask[rng.permutation(max_hidden)[:n_on]] = True
    lr = float(np.exp(rng.uniform(np.log(lr_bounds[0]), np.log(lr_bounds[1]))))
    return Genome(
        W1=rng.normal(size=(n_features, max_hidden)) / np.sqrt(n_features),
        b1=np.zeros(max_hidden),
        W2=rng.normal(size=(max_hidden, n_classes)) / np.sqrt(max_hidden),
        b2=np.zeros(n_classes),
        mask=mask, lr=lr, sigma=sigma,
    )


def predict_proba(g: Genome, X: FloatArray) -> FloatArray:
    """Class probabilities using active units only."""
    idx = np.flatnonzero(g.mask)
    h = np.tanh(X @ g.W1[:, idx] + g.b1[idx])
    return softmax(h @ g.W2[idx] + g.b2)


def error_rate(probs: FloatArray, y: IntArray) -> float:
    return float(np.mean(probs.argmax(axis=1) != y))


def cross_entropy(probs: FloatArray, y: IntArray) -> float:
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def loss_and_grads(
    W1: FloatArray, b1: FloatArray, W2: FloatArray, b2: FloatArray,
    X: FloatArray, Y: FloatArray, l2: float,
) -> tuple[float, tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    """Mean cross-entropy + L2 loss and analytic gradients (``Y`` is one-hot)."""
    n = X.shape[0]
    a = np.tanh(X @ W1 + b1)
    p = softmax(a @ W2 + b2)
    loss = float(-np.mean(np.sum(Y * np.log(p + 1e-12), axis=1)))
    loss += 0.5 * l2 * float(np.sum(W1 * W1) + np.sum(W2 * W2))
    dz = (p - Y) / n
    dW2 = a.T @ dz + l2 * W2
    db2 = dz.sum(axis=0)
    dpre = (dz @ W2.T) * (1.0 - a * a)
    dW1 = X.T @ dpre + l2 * W1
    db1 = dpre.sum(axis=0)
    return loss, (dW1, db1, dW2, db2)


def sgd_steps(
    g: Genome, X: FloatArray, y: IntArray, steps: int,
    l2: float = 1e-4, lr: float | None = None, clip: float = 5.0,
) -> None:
    """Full-batch gradient descent on the active sub-network, in place."""
    if steps <= 0 or len(y) == 0:
        return
    idx = np.flatnonzero(g.mask)
    Y = np.eye(g.W2.shape[1])[y]
    step = g.lr if lr is None else lr
    W1, b1, W2, b2 = g.W1[:, idx], g.b1[idx], g.W2[idx], g.b2
    for _ in range(steps):
        _, grads = loss_and_grads(W1, b1, W2, b2, X, Y, l2)
        norm = float(np.sqrt(sum(float(np.sum(gr * gr)) for gr in grads)))
        scale = step * min(1.0, clip / (norm + 1e-12))
        W1 = W1 - scale * grads[0]
        b1 = b1 - scale * grads[1]
        W2 = W2 - scale * grads[2]
        b2 = b2 - scale * grads[3]
    g.W1[:, idx], g.b1[idx], g.W2[idx], g.b2 = W1, b1, W2, b2
