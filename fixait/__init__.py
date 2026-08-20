"""Unified global and local fiXAIt package."""

from .config import FiXAItConfig
from .core import CalcFeatureWeight, SplitData
from .explainer import FiXAIt
from .preprocessing import PreprocessingSummary, TabularPreprocessor
from .results import (
    FaithfulnessResult,
    FidelityResult,
    GlobalExplanation,
    LocalExplanation,
    SelfConsistencyResult,
    StabilityResult,
)

__all__ = [
    "CalcFeatureWeight",
    "FaithfulnessResult",
    "FidelityResult",
    "FiXAIt",
    "FiXAItConfig",
    "GlobalExplanation",
    "LocalExplanation",
    "PreprocessingSummary",
    "SelfConsistencyResult",
    "StabilityResult",
    "TabularPreprocessor",
    "SplitData",
]

__version__ = "0.8.0"
