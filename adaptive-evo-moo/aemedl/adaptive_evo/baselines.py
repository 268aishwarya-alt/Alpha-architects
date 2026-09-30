"""Reference learners the evolutionary model is compared against."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .network import Genome, predict_proba, random_genome, sgd_steps

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _MLPBaseline:
    """Fixed-architecture MLP; subclasses differ only in *how* they update."""

    def __init__(self, n_classes: int, hidden: int = 16, lr: float = 0.1, seed: int = 0) -> None:
        if n_classes < 2 or hidden < 1 or lr <= 0:
            raise ValueError("need n_classes >= 2, hidden >= 1, lr > 0")
        self.n_classes, self.hidden, self.lr = n_classes, hidden, lr
        self._rng = np.random.default_rng(seed)
        self._g: Genome | None = None

    def _ensure(self, d: int) -> Genome:
        if self._g is None:
            self._g = random_genome(self._rng, d, self.n_classes, self.hidden,
                                    self.hidden, 0.0, (self.lr, self.lr))
        return self._g

    def predict_proba(self, X: FloatArray) -> FloatArray:
        if self._g is None:
            raise RuntimeError("call partial_fit before predict")
        return predict_proba(self._g, np.asarray(X, dtype=np.float64))

    def predict(self, X: FloatArray) -> IntArray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


class StaticMLP(_MLPBaseline):
    """Trains on the first ``warmup_chunks`` chunks, then never updates (non-adaptive)."""

    def __init__(self, n_classes: int, warmup_chunks: int = 3, steps: int = 300, **kw: float) -> None:
        super().__init__(n_classes, **kw)  # type: ignore[arg-type]
        self.warmup_chunks, self.steps, self._seen = warmup_chunks, steps, 0
        self._X: list[FloatArray] = []
        self._y: list[IntArray] = []

    def partial_fit(self, X: FloatArray, y: IntArray) -> "StaticMLP":
        g = self._ensure(X.shape[1])
        if self._seen < self.warmup_chunks:
            self._X.append(X)
            self._y.append(y)
            sgd_steps(g, np.vstack(self._X), np.concatenate(self._y), self.steps)
            self._seen += 1
        return self


class OnlineMLP(_MLPBaseline):
    """Continual SGD on each incoming chunk (adapts, but forgets slowly and never restructures)."""

    def __init__(self, n_classes: int, steps: int = 20, **kw: float) -> None:
        super().__init__(n_classes, **kw)  # type: ignore[arg-type]
        self.steps = steps

    def partial_fit(self, X: FloatArray, y: IntArray) -> "OnlineMLP":
        sgd_steps(self._ensure(X.shape[1]), X, y, self.steps)
        return self


class WindowMLP(_MLPBaseline):
    """Warm-started SGD on a sliding window of the most recent samples."""

    def __init__(self, n_classes: int, window: int = 300, steps: int = 20, **kw: float) -> None:
        super().__init__(n_classes, **kw)  # type: ignore[arg-type]
        self.window, self.steps = window, steps
        self._X = np.zeros((0, 0))
        self._y = np.zeros(0, dtype=np.int64)

    def partial_fit(self, X: FloatArray, y: IntArray) -> "WindowMLP":
        g = self._ensure(X.shape[1])
        self._X = X if self._X.size == 0 else np.vstack([self._X, X])[-self.window:]
        self._y = y if self._y.size == 0 else np.concatenate([self._y, y])[-self.window:]
        sgd_steps(g, self._X, self._y, self.steps)
        return self
