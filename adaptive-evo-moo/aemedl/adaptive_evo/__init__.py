"""Adaptive multi-objective evolutionary deep learning for drifting data streams."""
from .baselines import OnlineMLP, StaticMLP, WindowMLP
from .drift import PageHinkley
from .evaluate import PrequentialResult, prequential
from .evolver import OBJECTIVES, AdaptiveEvolutionaryClassifier, EvoConfig
from .streams import STREAM_KINDS, Stream, make_stream

__all__ = [
    "AdaptiveEvolutionaryClassifier", "EvoConfig", "OBJECTIVES", "PageHinkley",
    "OnlineMLP", "StaticMLP", "WindowMLP", "PrequentialResult", "prequential",
    "STREAM_KINDS", "Stream", "make_stream",
]
__version__ = "1.0.0"
