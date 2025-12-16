"""
Main Lyric Synchronization Pipeline

This module orchestrates the complete lyric synchronization process,
combining vocal separation, forced alignment, confidence scoring,
and output generation.
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

from .vocal_separator import VocalSeparator
from .forced_aligner import ForcedAligner, AlignmentResult
from .confidence_scorer import ConfidenceScorer
from .output_generator import OutputGenerator

logger = logging.getLogger(__name__)


class LyricSyncPipeline:
    """
    Complete lyric synchronization pipeline.
    
    This class orchestrates the entire process from audio input to
    synchronized lyrics output, managing all intermediate steps and
    providing comprehensive error handling and logging.
    """
    
    def __init__(self, 
                 use_gpu: bool = True,
                 temp_dir: Optional[str] = None,
                 confidence_threshold: float = 0.6):
        """
        Initialize the lyric synchronization pipeline.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
            temp_dir: Directory for temporary files
            confidence_threshold: Minimum confidence for acceptable results
        """
        self.use_gpu = use_gpu
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="beatloom_sync_")
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.vocal_separator = None
        self.forced_aligner = None
        self.confidence_scorer = None
        self.output_generator = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing lyric synchronization pipeline...")
            
            # Initialize vocal separator
            device = 'cuda' if self.use_gpu else 'cpu'
            self.vocal_separator = VocalSeparator(device=device)
            
            # Initialize forced aligner
            self.forced_aligner = ForcedAligner(use_sofa=True)
            
            # Initialize confidence scorer
            self.confidence_scorer = ConfidenceScorer()
            
            # Initialize output generator
            self.output_generator = OutputGenerator()
            
            logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def process_audio(self, 
                     audio_path: str,
                     lyrics_text: str,
                     output_dir: str,
                     base_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio file and lyrics to generate synchronized output.
        
        Args:
            audio_path: Path to input audio file
            lyrics_text: Lyrics text to synchronize
            output_dir: Directory for output files
            base_filename: Base name for output files (optional)
            
        Returns:
            Dictionary containing processing results and file paths
        """
        start_time = time.time()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not lyrics_text.strip():
            raise ValueError("Lyrics text cannot be empty")
        
        if base_filename is None:
            base_filename = Path(audio_path).stem
        
        logger.info(f"Starting lyric synchronization for: {audio_path}")
        
        try:
            # Create working directory
            work_dir = os.path.join(self.temp_dir, f"work_{int(time.time())}")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Vocal Separation
            logger.info("Step 1: Vocal separation")
            vocal_path, separation_info = self.vocal_separator.separate_vocals(
                audio_path, work_dir
            )
            
            # Step 2: Forced Alignment
            logger.info("Step 2: Forced alignment")
            alignment = self.forced_aligner.align_lyrics(
                vocal_path, lyrics_text, work_dir
            )
            
            # Step 3: Confidence Scoring
            logger.info("Step 3: Confidence scoring")
            confidence_result = self.confidence_scorer.score_alignment(
                alignment, vocal_path, lyrics_text
            )
            
            # Step 4: Output Generation
            logger.info("Step 4: Output generation")
            
            # Prepare metadata
            processing_time = time.time() - start_time
            metadata = {
                'input_audio': audio_path,
                'lyrics_text': lyrics_text,
                'processing_time': processing_time,
                'vocal_separation': separation_info,
                'aligner_info': self.forced_aligner.get_aligner_info(),
                'pipeline_version': '1.0.0'
            }
            
            # Generate all output formats
            output_files = self.output_generator.save_all_formats(
                alignment, confidence_result, output_dir, base_filename, metadata
            )
            
            # Generate summary report
            processing_info = {
                'vocal_separation': 'Demucs',
                'alignment_method': 'SOFA' if self.forced_aligner.sofa_available else 'Fallback',
                'processing_time': f"{processing_time:.2f} seconds"
            }
            
            summary_report = self.output_generator.create_summary_report(
                alignment, confidence_result, processing_info
            )
            
            # Save summary report
            report_path = os.path.join(output_dir, f"{base_filename}_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            output_files['report'] = report_path
            
            # Determine if result is acceptable
            is_acceptable = self.confidence_scorer.is_alignment_acceptable(
                confidence_result, self.confidence_threshold
            )
            
            # Prepare final result
            result = {
                'success': True,
                'processing_time': processing_time,
                'is_acceptable': is_acceptable,
                'confidence': confidence_result['overall_confidence'],
                'word_count': len(alignment.words),
                'duration': alignment.duration,
                'output_files': output_files,
                'quality_flags': confidence_result.get('quality_flags', []),
                'metadata': metadata,
                'summary_report': summary_report
            }
            
            logger.info(f"Lyric synchronization completed in {processing_time:.2f} seconds")
            logger.info(f"Confidence: {confidence_result['overall_confidence']:.3f}")
            logger.info(f"Acceptable: {is_acceptable}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Lyric synchronization failed after {processing_time:.2f} seconds: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'is_acceptable': False,
                'confidence': 0.0,
                'word_count': 0,
                'duration': 0.0,
                'output_files': {},
                'quality_flags': ['processing_failed'],
                'metadata': {'input_audio': audio_path, 'lyrics_text': lyrics_text}
            }
    
    def process_batch(self, 
                     audio_files: List[str],
                     lyrics_texts: List[str],
                     output_dir: str) -> List[Dict[str, Any]]:
        """
        Process multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            lyrics_texts: List of corresponding lyrics texts
            output_dir: Directory for output files
            
        Returns:
            List of processing results
        """
        if len(audio_files) != len(lyrics_texts):
            raise ValueError("Number of audio files must match number of lyrics texts")
        
        logger.info(f"Starting batch processing of {len(audio_files)} files")
        
        results = []
        for i, (audio_path, lyrics_text) in enumerate(zip(audio_files, lyrics_texts)):
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_path}")
            
            base_filename = f"batch_{i+1:03d}_{Path(audio_path).stem}"
            result = self.process_audio(audio_path, lyrics_text, output_dir, base_filename)
            results.append(result)
        
        # Generate batch summary
        successful = sum(1 for r in results if r['success'])
        acceptable = sum(1 for r in results if r.get('is_acceptable', False))
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
        
        batch_summary = {
            'total_files': len(audio_files),
            'successful': successful,
            'acceptable': acceptable,
            'average_confidence': avg_confidence,
            'results': results
        }
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed: {successful}/{len(audio_files)} successful")
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            'vocal_separator': self.vocal_separator.get_model_info() if self.vocal_separator else None,
            'forced_aligner': self.forced_aligner.get_aligner_info() if self.forced_aligner else None,
            'confidence_threshold': self.confidence_threshold,
            'temp_dir': self.temp_dir,
            'use_gpu': self.use_gpu
        }
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

