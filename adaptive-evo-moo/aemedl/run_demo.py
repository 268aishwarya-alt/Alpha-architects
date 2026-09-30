"""Benchmark the adaptive evolutionary classifier against baselines on drifting streams.

Example:
    python run_demo.py --kinds sudden recurring --seeds 3 --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_evo import (
    STREAM_KINDS, AdaptiveEvolutionaryClassifier, EvoConfig, OnlineMLP, PrequentialResult,
    StaticMLP, WindowMLP, make_stream, prequential,
)

CHUNK = 100


def run_one(kind: str, seed: int, n_samples: int) -> tuple[dict[str, PrequentialResult], list[int], tuple[int, ...]]:
    stream = make_stream(kind, n_samples=n_samples, seed=seed)
    k = stream.n_classes
    evo = AdaptiveEvolutionaryClassifier(EvoConfig(n_classes=k, seed=seed))
    models = {
        "Static MLP": StaticMLP(k, seed=seed),
        "Online SGD MLP": OnlineMLP(k, seed=seed),
        "Sliding-window MLP": WindowMLP(k, seed=seed),
        "Adaptive Evo-MOO (ours)": evo,
    }
    results = {name: prequential(m, stream, name, CHUNK) for name, m in models.items()}
    return results, evo.drifts_detected, stream.drift_points


def plot(kind: str, results: dict[str, PrequentialResult], drifts: tuple[int, ...], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, r in results.items():
        ax.plot(np.arange(1, len(r.chunk_acc) + 1), r.chunk_acc, label=name,
                lw=2.4 if "ours" in name else 1.3)
    for d in drifts:
        ax.axvline(d / CHUNK, color="grey", ls="--", lw=0.8)
    ax.set(xlabel=f"chunk ({CHUNK} samples)", ylabel="prequential accuracy",
           title=f"{kind} drift (dashed = true drift)", ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", nargs="+", default=list(STREAM_KINDS), choices=STREAM_KINDS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for kind in args.kinds:
        agg: dict[str, list[PrequentialResult]] = {}
        for seed in range(args.seeds):
            res, detected, truth = run_one(kind, seed, args.samples)
            for name, r in res.items():
                agg.setdefault(name, []).append(r)
            if seed == 0:
                plot(kind, res, truth, args.out / f"{kind}.png")
                print(f"[{kind}] true drift chunks {[d // CHUNK for d in truth]} "
                      f"detected {detected}", flush=True)
        summary[kind] = {
            name: {
                "accuracy": float(np.mean([r.accuracy for r in rs])),
                "post_drift_accuracy": float(np.nanmean([r.post_drift_accuracy for r in rs])),
                "recovery_chunks": float(np.nanmean([r.recovery_chunks for r in rs])),
                "seconds": float(np.mean([r.seconds for r in rs])),
            }
            for name, rs in agg.items()
        }
        print(f"\n== {kind} (mean over {args.seeds} seeds)")
        print(f"{'model':26s} {'acc':>6s} {'post-drift':>11s} {'recovery':>9s} {'sec':>6s}")
        for name, m in summary[kind].items():
            print(f"{name:26s} {m['accuracy']:6.3f} {m['post_drift_accuracy']:11.3f} "
                  f"{m['recovery_chunks']:9.2f} {m['seconds']:6.1f}", flush=True)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
