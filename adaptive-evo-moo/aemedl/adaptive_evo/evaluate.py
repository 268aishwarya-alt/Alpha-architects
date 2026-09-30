"""Prequential (test-then-train) evaluation with drift-aware metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .streams import Stream

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class StreamClassifier(Protocol):
    def partial_fit(self, X: FloatArray, y: IntArray) -> object: ...
    def predict(self, X: FloatArray) -> IntArray: ...


@dataclass
class PrequentialResult:
    name: str
    chunk_acc: list[float] = field(default_factory=list)
    accuracy: float = 0.0
    post_drift_accuracy: float = 0.0
    recovery_chunks: float = 0.0
    seconds: float = 0.0


def prequential(
    model: StreamClassifier, stream: Stream, name: str, chunk_size: int = 100, horizon: int = 5
) -> PrequentialResult:
    """Predict each chunk *before* training on it. The first chunk is training-only.

    ``post_drift_accuracy`` is the mean accuracy over ``horizon`` chunks after each true drift;
    ``recovery_chunks`` counts chunks after a drift until accuracy is within 10% (relative) of the
    accuracy before it (capped at ``horizon``).
    """
    if chunk_size < 2 or horizon < 1:
        raise ValueError("chunk_size must be >= 2 and horizon >= 1")
    n = len(stream.y)
    accs: list[float] = []
    start = time.perf_counter()
    for lo in range(0, n - chunk_size + 1, chunk_size):
        Xc, yc = stream.X[lo:lo + chunk_size], stream.y[lo:lo + chunk_size]
        if lo > 0:
            accs.append(float(np.mean(model.predict(Xc) == yc)))
        model.partial_fit(Xc, yc)
    seconds = time.perf_counter() - start
    # accs[i] is the accuracy on chunk i+1
    post: list[float] = []
    rec: list[float] = []
    for dp in stream.drift_points:
        c = dp // chunk_size - 1  # index into accs of the first post-drift chunk
        if c < 3 or c >= len(accs):
            continue
        window = accs[c:c + horizon]
        post.append(float(np.mean(window)))
        before = float(np.mean(accs[c - 3:c]))
        need = 0.9 * before
        rec.append(float(next((i for i, a in enumerate(window) if a >= need), horizon)))
    return PrequentialResult(
        name=name, chunk_acc=accs, accuracy=float(np.mean(accs)),
        post_drift_accuracy=float(np.mean(post)) if post else float("nan"),
        recovery_chunks=float(np.mean(rec)) if rec else float("nan"), seconds=seconds,
    )
