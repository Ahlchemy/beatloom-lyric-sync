"""
Confidence Scoring Component

This module evaluates the quality of lyric alignment results and provides
confidence scores to identify potentially problematic alignments.
"""

import logging
import numpy as np
import librosa
from typing import List, Tuple, Dict, Any, Optional
from .forced_aligner import AlignmentResult

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Confidence scoring for lyric alignment results.
    
    This class analyzes alignment results and audio features to determine
    the reliability of the synchronization, helping to identify cases
    that may need manual review.
    """
    
    def __init__(self, 
                 min_word_duration: float = 0.1,
                 max_word_duration: float = 5.0,
                 expected_words_per_second: float = 2.5):
        """
        Initialize the confidence scorer.
        
        Args:
            min_word_duration: Minimum expected word duration in seconds
            max_word_duration: Maximum expected word duration in seconds
            expected_words_per_second: Expected speaking/singing rate
        """
        self.min_word_duration = min_word_duration
        self.max_word_duration = max_word_duration
        self.expected_words_per_second = expected_words_per_second
    
    def score_alignment(self, 
                       alignment: AlignmentResult, 
                       vocal_audio_path: str,
                       original_lyrics: str) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence score for an alignment.
        
        Args:
            alignment: The alignment result to score
            vocal_audio_path: Path to the vocal audio file
            original_lyrics: Original lyrics text
            
        Returns:
            Dictionary containing confidence score and detailed metrics
        """
        logger.info("Calculating confidence score for alignment")
        
        try:
            # Load audio for analysis
            y, sr = librosa.load(vocal_audio_path)
            
            # Calculate individual confidence metrics
            duration_score = self._score_word_durations(alignment)
            coverage_score = self._score_text_coverage(alignment, original_lyrics)
            timing_score = self._score_timing_consistency(alignment)
            audio_score = self._score_audio_alignment(alignment, y, sr)
            
            # Combine scores with weights
            weights = {
                'duration': 0.25,
                'coverage': 0.30,
                'timing': 0.25,
                'audio': 0.20
            }
            
            overall_confidence = (
                weights['duration'] * duration_score +
                weights['coverage'] * coverage_score +
                weights['timing'] * timing_score +
                weights['audio'] * audio_score
            )
            
            # Include base confidence from aligner
            if hasattr(alignment, 'confidence') and alignment.confidence > 0:
                overall_confidence = (overall_confidence + alignment.confidence) / 2
            
            result = {
                'overall_confidence': float(overall_confidence),
                'scores': {
                    'duration': float(duration_score),
                    'coverage': float(coverage_score),
                    'timing': float(timing_score),
                    'audio': float(audio_score)
                },
                'metrics': {
                    'word_count': len(alignment.words),
                    'total_duration': alignment.duration,
                    'avg_word_duration': self._calculate_avg_word_duration(alignment),
                    'words_per_second': len(alignment.words) / max(alignment.duration, 1)
                },
                'quality_flags': self._identify_quality_issues(alignment, original_lyrics)
            }
            
            logger.info(f"Confidence score calculated: {overall_confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence score: {e}")
            return {
                'overall_confidence': 0.0,
                'scores': {'duration': 0.0, 'coverage': 0.0, 'timing': 0.0, 'audio': 0.0},
                'metrics': {},
                'quality_flags': ['scoring_failed'],
                'error': str(e)
            }
    
    def _score_word_durations(self, alignment: AlignmentResult) -> float:
        """Score based on word duration reasonableness."""
        if not alignment.words:
            return 0.0
        
        durations = [end - start for _, start, end in alignment.words]
        
        # Count words with reasonable durations
        reasonable_count = sum(1 for d in durations 
                             if self.min_word_duration <= d <= self.max_word_duration)
        
        # Penalize extremely short or long words
        duration_score = reasonable_count / len(durations)
        
        # Additional penalty for very inconsistent durations
        if len(durations) > 1:
            duration_std = np.std(durations)
            duration_mean = np.mean(durations)
            if duration_mean > 0:
                cv = duration_std / duration_mean  # Coefficient of variation
                consistency_penalty = max(0, min(1, 2 - cv))  # Penalty for high variation
                duration_score *= consistency_penalty
        
        return duration_score
    
    def _score_text_coverage(self, alignment: AlignmentResult, original_lyrics: str) -> float:
        """Score based on how well the alignment covers the original text."""
        if not alignment.words or not original_lyrics.strip():
            return 0.0
        
        original_words = original_lyrics.lower().split()
        aligned_words = [word.lower() for word, _, _ in alignment.words]
        
        # Calculate word coverage
        matched_words = sum(1 for word in aligned_words if word in original_words)
        coverage = matched_words / max(len(original_words), 1)
        
        # Penalize for significant length mismatch
        length_ratio = len(aligned_words) / max(len(original_words), 1)
        length_penalty = 1.0 - abs(1.0 - length_ratio) * 0.5
        length_penalty = max(0.0, length_penalty)
        
        return coverage * length_penalty
    
    def _score_timing_consistency(self, alignment: AlignmentResult) -> float:
        """Score based on timing consistency and monotonicity."""
        if len(alignment.words) < 2:
            return 1.0
        
        # Check for monotonic timing (words should be in order)
        times = [start for _, start, _ in alignment.words]
        monotonic_score = 1.0
        
        for i in range(1, len(times)):
            if times[i] < times[i-1]:
                monotonic_score -= 0.2  # Penalty for non-monotonic timing
        
        monotonic_score = max(0.0, monotonic_score)
        
        # Check for reasonable gaps between words
        gaps = [times[i] - times[i-1] for i in range(1, len(times))]
        reasonable_gaps = sum(1 for gap in gaps if 0.0 <= gap <= 2.0)
        gap_score = reasonable_gaps / max(len(gaps), 1)
        
        # Check speaking rate consistency
        total_duration = alignment.duration
        words_per_second = len(alignment.words) / max(total_duration, 1)
        rate_score = 1.0 - abs(words_per_second - self.expected_words_per_second) / 5.0
        rate_score = max(0.0, min(1.0, rate_score))
        
        return (monotonic_score + gap_score + rate_score) / 3
    
    def _score_audio_alignment(self, alignment: AlignmentResult, y: np.ndarray, sr: int) -> float:
        """Score based on correlation with audio features."""
        if not alignment.words:
            return 0.0
        
        try:
            # Calculate RMS energy
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Check if word timings correlate with energy
            energy_scores = []
            for word, start, end in alignment.words:
                # Find RMS values in word time range
                mask = (rms_times >= start) & (rms_times <= end)
                if np.any(mask):
                    word_energy = np.mean(rms[mask])
                    # Higher energy during word timing is good
                    energy_scores.append(min(1.0, word_energy * 10))  # Scale energy
                else:
                    energy_scores.append(0.0)
            
            if energy_scores:
                return np.mean(energy_scores)
            else:
                return 0.5  # Neutral score if no energy data
                
        except Exception as e:
            logger.warning(f"Audio alignment scoring failed: {e}")
            return 0.5  # Neutral score on failure
    
    def _calculate_avg_word_duration(self, alignment: AlignmentResult) -> float:
        """Calculate average word duration."""
        if not alignment.words:
            return 0.0
        
        durations = [end - start for _, start, end in alignment.words]
        return np.mean(durations)
    
    def _identify_quality_issues(self, alignment: AlignmentResult, original_lyrics: str) -> List[str]:
        """Identify specific quality issues with the alignment."""
        issues = []
        
        if not alignment.words:
            issues.append('no_words_aligned')
            return issues
        
        # Check for very short or long words
        durations = [end - start for _, start, end in alignment.words]
        if any(d < self.min_word_duration for d in durations):
            issues.append('very_short_words')
        if any(d > self.max_word_duration for d in durations):
            issues.append('very_long_words')
        
        # Check for non-monotonic timing
        times = [start for _, start, _ in alignment.words]
        if any(times[i] < times[i-1] for i in range(1, len(times))):
            issues.append('non_monotonic_timing')
        
        # Check for large gaps
        gaps = [times[i] - times[i-1] for i in range(1, len(times))]
        if any(gap > 3.0 for gap in gaps):
            issues.append('large_gaps')
        
        # Check for overlapping words
        if any(end > next_start for (_, _, end), (_, next_start, _) 
               in zip(alignment.words[:-1], alignment.words[1:])):
            issues.append('overlapping_words')
        
        # Check coverage
        original_words = original_lyrics.lower().split()
        aligned_words = [word.lower() for word, _, _ in alignment.words]
        coverage = len(set(aligned_words) & set(original_words)) / max(len(original_words), 1)
        if coverage < 0.7:
            issues.append('low_text_coverage')
        
        # Check speaking rate
        words_per_second = len(alignment.words) / max(alignment.duration, 1)
        if words_per_second > 5.0:
            issues.append('too_fast')
        elif words_per_second < 0.5:
            issues.append('too_slow')
        
        return issues
    
    def is_alignment_acceptable(self, confidence_result: Dict[str, Any], 
                              threshold: float = 0.6) -> bool:
        """
        Determine if an alignment is acceptable based on confidence score.
        
        Args:
            confidence_result: Result from score_alignment()
            threshold: Minimum confidence threshold
            
        Returns:
            True if alignment is acceptable, False otherwise
        """
        overall_confidence = confidence_result.get('overall_confidence', 0.0)
        quality_flags = confidence_result.get('quality_flags', [])
        
        # Check confidence threshold
        if overall_confidence < threshold:
            return False
        
        # Check for critical quality issues
        critical_issues = ['no_words_aligned', 'non_monotonic_timing', 'overlapping_words']
        if any(issue in quality_flags for issue in critical_issues):
            return False
        
        return True

