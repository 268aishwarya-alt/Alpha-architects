"""Page-Hinkley change detector for a stream of per-chunk error rates."""
from __future__ import annotations


class PageHinkley:
    """Signals an *increase* in the mean of the monitored value.

    Args:
        delta: tolerated magnitude of change (noise allowance).
        threshold: alarm level for the cumulative deviation.
        min_samples: observations required before an alarm may fire.
    """

    def __init__(self, delta: float = 0.02, threshold: float = 0.5, min_samples: int = 3) -> None:
        if delta < 0 or threshold <= 0 or min_samples < 1:
            raise ValueError("need delta >= 0, threshold > 0, min_samples >= 1")
        self.delta = delta
        self.threshold = threshold
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._cum = 0.0
        self._min = 0.0

    def update(self, value: float) -> bool:
        """Feed one observation; returns True (and resets) when drift is flagged."""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._cum += value - self._mean - self.delta
        self._min = min(self._min, self._cum)
        if self._n >= self.min_samples and self._cum - self._min > self.threshold:
            self.reset()
            return True
        return False
