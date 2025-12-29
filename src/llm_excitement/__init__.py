"""
LLM Excitement - Identifying interpretable features in LLMs using SAEs and Gemma Scope 2
"""

__version__ = "0.1.0"

from .sae_extractor import SAEFeatureExtractor
from .reward_detector import RewardSignalDetector
from .data_loader import FeedbackDataLoader

__all__ = [
    "SAEFeatureExtractor",
    "RewardSignalDetector",
    "FeedbackDataLoader",
]
