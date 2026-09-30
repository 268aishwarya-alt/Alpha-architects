"""Adaptive multi-objective evolutionary classifier for non-stationary streams.

Each ``partial_fit`` call receives one labelled chunk. The classifier
1. scores the deployed model on the chunk (prequential error) and feeds a
   Page-Hinkley detector,
2. on drift: clears stale memory, boosts mutation, injects random immigrants,
3. evolves a population with NSGA-II over four minimised objectives
   (recent error, stability on older data, noise robustness, complexity),
   using gradient refinement as Lamarckian local search,
4. deploys the Pareto-front member with the best drift-aware weighted score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drift import PageHinkley
from .network import (
    Genome, cross_entropy, error_rate, predict_proba, random_genome, sgd_steps,
)
from .nsga2 import binary_tournament, environmental_selection

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

OBJECTIVES: tuple[str, ...] = ("recent_error", "stability_error", "noise_error", "complexity")


@dataclass
class EvoConfig:
    """Hyper-parameters; validated on construction."""

    n_classes: int
    max_hidden: int = 24
    min_hidden: int = 2
    pop_size: int = 20
    generations: int = 2
    drift_generations: int = 6
    refine_steps: int = 5
    init_steps: int = 40
    recent_size: int = 200
    old_size: int = 300
    noise_std: float = 0.25
    mutation_sigma: float = 0.1
    weight_mut_prob: float = 0.3
    struct_mut_prob: float = 0.05
    crossover_prob: float = 0.8
    drift_boost: float = 3.0
    immigrant_frac: float = 0.3
    boost_chunks: int = 3
    keep_on_drift: int = 1
    l2: float = 1e-4
    lr_bounds: tuple[float, float] = (0.02, 0.6)
    ph_delta: float = 0.02
    ph_threshold: float = 0.5
    weights: tuple[float, float, float, float] = (0.55, 0.20, 0.20, 0.05)
    drift_weights: tuple[float, float, float, float] = (0.80, 0.02, 0.15, 0.03)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if not 1 <= self.min_hidden <= self.max_hidden:
            raise ValueError("need 1 <= min_hidden <= max_hidden")
        if self.pop_size < 4 or self.generations < 0 or self.drift_generations < 0:
            raise ValueError("pop_size must be >= 4 and generation counts >= 0")
        if self.recent_size < 10 or self.old_size < 0:
            raise ValueError("recent_size must be >= 10 and old_size >= 0")
        if not 0.0 <= self.immigrant_frac < 1.0:
            raise ValueError("immigrant_frac must be in [0, 1)")
        if not 0 < self.lr_bounds[0] <= self.lr_bounds[1]:
            raise ValueError("lr_bounds must satisfy 0 < low <= high")
        if len(self.weights) != 4 or len(self.drift_weights) != 4:
            raise ValueError("weights must have one entry per objective")


@dataclass
class _EvalSet:
    Xr: FloatArray
    yr: IntArray
    Xo: FloatArray
    yo: IntArray
    noise: FloatArray


class AdaptiveEvolutionaryClassifier:
    """Streaming classifier; use ``partial_fit`` per chunk and ``predict`` between chunks."""

    def __init__(self, config: EvoConfig) -> None:
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)
        self._detector = PageHinkley(config.ph_delta, config.ph_threshold)
        self._pop: list[Genome] = []
        self._best: Genome | None = None
        self._n_features = 0
        self._Xr = np.zeros((0, 0))
        self._yr = np.zeros(0, dtype=np.int64)
        self._Xo = np.zeros((0, 0))
        self._yo = np.zeros(0, dtype=np.int64)
        self._last_chunks: list[tuple[FloatArray, IntArray]] = []
        self._boost_left = 0
        self.history: list[dict[str, float]] = []
        self.drifts_detected: list[int] = []

    # ------------------------------------------------------------------ API
    def partial_fit(self, X: FloatArray, y: IntArray) -> "AdaptiveEvolutionaryClassifier":
        X, y = self._validate(X, y)
        first = not self._pop
        drift = False
        chunk_err = float("nan")
        if first:
            self._n_features = X.shape[1]
            self._Xr = np.zeros((0, self._n_features))
            self._Xo = np.zeros((0, self._n_features))
        else:
            assert self._best is not None
            chunk_err = error_rate(predict_proba(self._best, X), y)
            drift = self._detector.update(chunk_err)
        if drift:
            self._reset_memory()
            self.drifts_detected.append(len(self.history))
            self._boost_left = self.cfg.boost_chunks
        self._push(X, y)
        if first:
            self._pop = [self._new_genome() for _ in range(self.cfg.pop_size)]
            for g in self._pop:
                sgd_steps(g, self._Xr, self._yr, self.cfg.init_steps, self.cfg.l2, lr=0.1)
        elif drift:
            self._inject_immigrants()
        boosted = first or self._boost_left > 0
        self._evolve(self.cfg.drift_generations if boosted else self.cfg.generations, boosted)
        self._boost_left = max(0, self._boost_left - 1)
        self._select_best(boosted)
        assert self._best is not None
        self.history.append({
            "chunk_error": chunk_err, "drift": float(drift),
            "n_active": float(self._best.n_active), "front_size": float(len(self.pareto_front())),
        })
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        if self._best is None:
            raise RuntimeError("call partial_fit before predict")
        return predict_proba(self._best, np.asarray(X, dtype=np.float64))

    def predict(self, X: FloatArray) -> IntArray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

    def pareto_front(self) -> FloatArray:
        """Objective matrix (columns = :data:`OBJECTIVES`) of the current rank-0 set."""
        return np.array([g.fitness for g in self._pop if g.rank == 0])

    # ------------------------------------------------------------ internals
    def _validate(self, X: FloatArray, y: IntArray) -> tuple[FloatArray, IntArray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) < 2:
            raise ValueError("X must be 2-D, y 1-D, same length >= 2")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
        if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= self.cfg.n_classes:
            raise ValueError(f"y must be integers in [0, {self.cfg.n_classes})")
        if self._pop and X.shape[1] != self._n_features:
            raise ValueError("feature count changed between chunks")
        return X, y.astype(np.int64)

    def _new_genome(self) -> Genome:
        c = self.cfg
        return random_genome(self._rng, self._n_features, c.n_classes, c.max_hidden,
                             c.min_hidden, c.mutation_sigma, c.lr_bounds)

    def _push(self, X: FloatArray, y: IntArray) -> None:
        self._last_chunks = (self._last_chunks + [(X, y)])[-max(1, self.cfg.keep_on_drift):]
        self._Xr = np.vstack([self._Xr, X])
        self._yr = np.concatenate([self._yr, y])
        over = len(self._yr) - self.cfg.recent_size
        if over > 0:
            self._Xo = np.vstack([self._Xo, self._Xr[:over]])[-self.cfg.old_size:] \
                if self.cfg.old_size else self._Xo
            self._yo = np.concatenate([self._yo, self._yr[:over]])[-self.cfg.old_size:] \
                if self.cfg.old_size else self._yo
            self._Xr, self._yr = self._Xr[over:], self._yr[over:]

    def _reset_memory(self) -> None:
        """Drop obsolete data; keep only the most recent chunk(s) (likely post-drift)."""
        keep = self._last_chunks[-self.cfg.keep_on_drift:] if self.cfg.keep_on_drift else []
        self._Xr = np.vstack([k[0] for k in keep]) if keep else np.zeros((0, self._n_features))
        self._yr = np.concatenate([k[1] for k in keep]) if keep else np.zeros(0, dtype=np.int64)
        self._Xo = np.zeros((0, self._n_features))
        self._yo = np.zeros(0, dtype=np.int64)
        self._last_chunks = []

    def _inject_immigrants(self) -> None:
        n_new = int(round(self.cfg.immigrant_frac * len(self._pop)))
        order = sorted(range(len(self._pop)), key=lambda i: (self._pop[i].rank, -self._pop[i].crowding))
        for i in order[len(order) - n_new:]:
            g = self._new_genome()
            sgd_steps(g, self._Xr, self._yr, self.cfg.init_steps, self.cfg.l2, lr=0.1)
            self._pop[i] = g

    def _eval_set(self) -> _EvalSet:
        noise = self._rng.normal(0.0, self.cfg.noise_std, size=self._Xr.shape)
        return _EvalSet(self._Xr, self._yr, self._Xo, self._yo, noise)

    def _evaluate(self, g: Genome, ev: _EvalSet) -> FloatArray:
        p = predict_proba(g, ev.Xr)
        f_recent = error_rate(p, ev.yr) + 0.05 * min(cross_entropy(p, ev.yr), 3.0)
        if len(ev.yo):
            po = predict_proba(g, ev.Xo)
            f_stab = error_rate(po, ev.yo) + 0.05 * min(cross_entropy(po, ev.yo), 3.0)
        else:
            f_stab = f_recent
        f_noise = error_rate(predict_proba(g, ev.Xr + ev.noise), ev.yr)
        f_cx = g.n_active / self.cfg.max_hidden
        return np.array([f_recent, f_stab, f_noise, f_cx])

    def _refine_and_score(self, pop: list[Genome], ev: _EvalSet) -> None:
        for g in pop:
            sgd_steps(g, ev.Xr, ev.yr, self.cfg.refine_steps, self.cfg.l2)
            g.fitness = self._evaluate(g, ev)

    def _survive(self, merged: list[Genome]) -> list[Genome]:
        F = np.array([g.fitness for g in merged])
        idx, rank, crowd = environmental_selection(F, self.cfg.pop_size)
        out = []
        for i, r, c in zip(idx, rank, crowd):
            merged[int(i)].rank, merged[int(i)].crowding = int(r), float(c)
            out.append(merged[int(i)])
        return out

    def _evolve(self, generations: int, boosted: bool) -> None:
        ev = self._eval_set()
        self._refine_and_score(self._pop, ev)
        self._pop = self._survive(self._pop)
        for _ in range(generations):
            kids = self._offspring(boosted)
            self._refine_and_score(kids, ev)
            self._pop = self._survive(self._pop + kids)

    def _offspring(self, boosted: bool) -> list[Genome]:
        c = self.cfg
        rank = np.array([g.rank for g in self._pop])
        crowd = np.array([g.crowding for g in self._pop])
        n = len(self._pop)
        pa = binary_tournament(rank, crowd, self._rng, n)
        pb = binary_tournament(rank, crowd, self._rng, n)
        boost = c.drift_boost if boosted else 1.0
        kids = []
        for a, b in zip(pa, pb):
            child = self._pop[int(a)].copy()
            if self._rng.random() < c.crossover_prob:
                self._crossover(child, self._pop[int(b)])
            self._mutate(child, boost)
            kids.append(child)
        return kids

    def _crossover(self, child: Genome, other: Genome) -> None:
        take = self._rng.random(len(child.mask)) < 0.5
        child.W1[:, take] = other.W1[:, take]
        child.b1[take] = other.b1[take]
        child.W2[take] = other.W2[take]
        child.mask[take] = other.mask[take]
        child.b2 = 0.5 * (child.b2 + other.b2)
        child.lr = float(np.sqrt(child.lr * other.lr))
        child.sigma = 0.5 * (child.sigma + other.sigma)

    def _mutate(self, g: Genome, boost: float) -> None:
        c = self.cfg
        g.sigma = float(np.clip(g.sigma * np.exp(0.3 * self._rng.normal()), 1e-3, 0.5))
        s = g.sigma * boost
        for arr in (g.W1, g.b1, g.W2, g.b2):
            hit = self._rng.random(arr.shape) < c.weight_mut_prob
            arr += hit * self._rng.normal(0.0, s, size=arr.shape)
        flips = self._rng.random(len(g.mask)) < c.struct_mut_prob * min(boost, 2.0)
        g.mask ^= flips
        if g.n_active < c.min_hidden:
            off = np.flatnonzero(~g.mask)
            need = c.min_hidden - g.n_active
            g.mask[self._rng.choice(off, size=need, replace=False)] = True
        g.lr = float(np.clip(g.lr * np.exp(0.2 * self._rng.normal()), *c.lr_bounds))

    def _select_best(self, boosted: bool) -> None:
        front = [g for g in self._pop if g.rank == 0]
        w = np.array(self.cfg.drift_weights if boosted else self.cfg.weights)
        best = min(front, key=lambda g: float(w @ g.fitness))
        self._best = best.copy()
