"""Synthetic non-stationary data streams with known ground-truth drift points.

Every concept is a random "teacher" network that labels Gaussian inputs, so the
decision boundary is non-linear and a linear model cannot solve it. Five drift
regimes are supported: ``sudden``, ``gradual``, ``incremental``, ``recurring``
(real concept drift) and ``covariate`` (input distribution shift only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

STREAM_KINDS: tuple[str, ...] = ("sudden", "gradual", "incremental", "recurring", "covariate")


@dataclass(frozen=True)
class Stream:
    """A finite labelled stream plus the indices where drift begins."""

    X: FloatArray
    y: IntArray
    drift_points: tuple[int, ...]
    kind: str
    n_classes: int


class _Teacher:
    """Random one-hidden-layer network used as a labelling concept."""

    def __init__(self, rng: np.random.Generator, d: int, k: int, hidden: int = 8) -> None:
        self.W1 = rng.normal(size=(d, hidden)) * 1.5 / np.sqrt(d)
        self.b1 = rng.normal(scale=0.3, size=hidden)
        self.W2 = rng.normal(size=(hidden, k))
        # Centre logits on a calibration sample so classes are roughly balanced.
        calib = rng.normal(size=(2000, d))
        self.offset = self._raw(calib).mean(axis=0)

    def _raw(self, X: FloatArray) -> FloatArray:
        return np.tanh(X @ self.W1 + self.b1) @ self.W2

    def logits(self, X: FloatArray) -> FloatArray:
        return self._raw(X) - self.offset


def make_stream(
    kind: str = "sudden",
    n_samples: int = 6000,
    n_features: int = 8,
    n_classes: int = 3,
    n_drifts: int = 4,
    label_noise: float = 0.03,
    seed: int = 0,
) -> Stream:
    """Generate a drifting stream of ``n_samples`` examples.

    Args:
        kind: one of :data:`STREAM_KINDS`.
        n_samples: total stream length.
        n_features: input dimensionality.
        n_classes: number of classes (>= 2).
        n_drifts: number of drift events; the stream has ``n_drifts + 1`` segments.
        label_noise: fraction of labels replaced by a random class.
        seed: RNG seed (streams are fully deterministic given the seed).
    """
    if kind not in STREAM_KINDS:
        raise ValueError(f"unknown stream kind {kind!r}; choose from {STREAM_KINDS}")
    if n_classes < 2 or n_features < 1 or n_drifts < 1:
        raise ValueError("need n_classes >= 2, n_features >= 1 and n_drifts >= 1")
    if not 0.0 <= label_noise < 1.0:
        raise ValueError("label_noise must be in [0, 1)")
    n_seg = n_drifts + 1
    seg_len = n_samples // n_seg
    if seg_len < 8:
        raise ValueError("n_samples too small for the requested number of drifts")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int64)

    if kind == "recurring":
        pool = [_Teacher(rng, n_features, n_classes) for _ in range(2)]
        teachers = [pool[s % 2] for s in range(n_seg)]
    elif kind == "covariate":
        base = _Teacher(rng, n_features, n_classes)
        teachers = [base] * n_seg
        for s in range(1, n_seg):
            lo, hi = s * seg_len, n_samples if s == n_seg - 1 else (s + 1) * seg_len
            X[lo:hi] += rng.normal(scale=1.5, size=n_features)
    else:
        teachers = [_Teacher(rng, n_features, n_classes) for _ in range(n_seg)]

    width = max(seg_len // 4, 1)
    for s in range(n_seg):
        lo = s * seg_len
        hi = n_samples if s == n_seg - 1 else (s + 1) * seg_len
        Xs = X[lo:hi]
        cur = teachers[s].logits(Xs)
        if s > 0 and kind in ("gradual", "incremental"):
            w = min(width, hi - lo)
            t = np.arange(w)[:, None] / w  # 0 -> 1 across the transition window
            prev = teachers[s - 1].logits(Xs[:w])
            if kind == "incremental":
                cur[:w] = (1.0 - t) * prev + t * cur[:w]
            else:
                pick_new = rng.random((w, 1)) < t
                cur[:w] = np.where(pick_new, cur[:w], prev)
        y[lo:hi] = cur.argmax(axis=1)

    flip = rng.random(n_samples) < label_noise
    y[flip] = rng.integers(0, n_classes, size=int(flip.sum()))
    drift_points = tuple(s * seg_len for s in range(1, n_seg))
    return Stream(X=X, y=y, drift_points=drift_points, kind=kind, n_classes=n_classes)
