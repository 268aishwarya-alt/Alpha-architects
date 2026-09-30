"""NSGA-II building blocks (all objectives are minimised)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def non_dominated_sort(F: FloatArray) -> list[list[int]]:
    """Split rows of ``F`` (n x m objective matrix) into Pareto fronts."""
    n = F.shape[0]
    if n == 0:
        return []
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt  # dom[i, j] is True when i dominates j
    counts = dom.sum(axis=0).astype(np.int64)
    fronts: list[list[int]] = []
    current = np.flatnonzero(counts == 0)
    while current.size:
        fronts.append(current.tolist())
        nxt: list[int] = []
        for i in current:
            for j in np.flatnonzero(dom[i]):
                counts[j] -= 1
                if counts[j] == 0:
                    nxt.append(int(j))
        current = np.array(nxt, dtype=np.int64)
    return fronts


def crowding_distance(F: FloatArray, front: list[int]) -> FloatArray:
    """Crowding distance of each member of ``front`` (boundary points get inf)."""
    k = len(front)
    dist = np.zeros(k)
    if k <= 2:
        dist[:] = np.inf
        return dist
    sub = F[front]
    for m in range(F.shape[1]):
        order = np.argsort(sub[:, m], kind="stable")
        span = sub[order[-1], m] - sub[order[0], m]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span <= 0:
            continue
        dist[order[1:-1]] += (sub[order[2:], m] - sub[order[:-2], m]) / span
    return dist


def environmental_selection(F: FloatArray, n: int) -> tuple[IntArray, IntArray, FloatArray]:
    """Pick ``n`` survivors. Returns (indices, rank per survivor, crowding per survivor)."""
    chosen: list[int] = []
    ranks: list[int] = []
    crowd: list[float] = []
    for r, front in enumerate(non_dominated_sort(F)):
        cd = crowding_distance(F, front)
        if len(chosen) + len(front) <= n:
            take = np.arange(len(front))
        else:
            take = np.argsort(-cd, kind="stable")[: n - len(chosen)]
        chosen += [front[i] for i in take]
        ranks += [r] * len(take)
        crowd += [float(cd[i]) for i in take]
        if len(chosen) >= n:
            break
    return np.array(chosen, dtype=np.int64), np.array(ranks, dtype=np.int64), np.array(crowd)


def binary_tournament(
    rank: IntArray, crowd: FloatArray, rng: np.random.Generator, n_picks: int
) -> IntArray:
    """Crowded-comparison binary tournament: lower rank wins, then larger crowding."""
    a = rng.integers(0, len(rank), size=n_picks)
    b = rng.integers(0, len(rank), size=n_picks)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
    return np.where(a_wins, a, b).astype(np.int64)
