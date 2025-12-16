"""
Beatloom Lyric Synchronization System

A modular pipeline for accurate lyric-to-audio synchronization using:
1. Vocal separation with Demucs
2. Forced alignment with SOFA
3. Confidence scoring
4. Output generation for Beatloom

Author: Manus AI
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Manus AI"

from .vocal_separator import VocalSeparator
from .forced_aligner import ForcedAligner
from .confidence_scorer import ConfidenceScorer
from .output_generator import OutputGenerator
from .pipeline import LyricSyncPipeline

__all__ = [
    "VocalSeparator",
    "ForcedAligner", 
    "ConfidenceScorer",
    "OutputGenerator",
    "LyricSyncPipeline"
]

