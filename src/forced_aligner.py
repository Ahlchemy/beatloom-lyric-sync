"""
Forced Alignment Component using SOFA (Singing-Oriented Forced Aligner)

This module handles the alignment of lyrics text with isolated vocal audio
using singing-specific forced alignment techniques.
"""

import os
import subprocess
import tempfile
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import textgrid
import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AlignmentResult:
    """Container for alignment results."""
    
    def __init__(self, words: List[Tuple[str, float, float]], confidence: float = 0.0):
        self.words = words  # List of (word, start_time, end_time)
        self.confidence = confidence
        self.duration = max([end for _, _, end in words]) if words else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'words': [{'text': word, 'start': start, 'end': end} 
                     for word, start, end in self.words],
            'confidence': self.confidence,
            'duration': self.duration,
            'word_count': len(self.words)
        }
    
    def to_lrc(self) -> str:
        """Convert to LRC format."""
        lrc_lines = []
        for word, start, _ in self.words:
            minutes = int(start // 60)
            seconds = start % 60
            lrc_lines.append(f"[{minutes:02d}:{seconds:05.2f}]{word}")
        return '\n'.join(lrc_lines)


class ForcedAligner:
    """
    Forced alignment using SOFA or fallback methods.
    
    This class provides singing-specific forced alignment capabilities,
    with SOFA as the primary method and fallback implementations for
    cases where SOFA is not available.
    """
    
    def __init__(self, use_sofa: bool = True, sofa_path: Optional[str] = None):
        """
        Initialize the forced aligner.

        Args:
            use_sofa: Whether to use SOFA (if available)
            sofa_path: Path to SOFA directory (optional)
        """
        self.use_sofa = use_sofa
        self.sofa_path = sofa_path or self._find_sofa()
        self.checkpoint_path = None
        self.sofa_available = self._check_sofa_availability()

        if self.use_sofa and not self.sofa_available:
            logger.warning("SOFA not available, will use fallback alignment method")
    
    def _find_sofa(self) -> Optional[str]:
        """Try to find SOFA Python project directory."""
        # Get the project root directory (parent of src/)
        current_file = Path(__file__)
        project_root = current_file.parent.parent

        # Check for SOFA directory in the project
        sofa_dir = project_root / "SOFA"
        if sofa_dir.exists() and (sofa_dir / "infer.py").exists():
            return str(sofa_dir)

        # Check for SOFA in common locations
        possible_paths = [
            Path.cwd() / "SOFA",
            Path.home() / "SOFA",
            Path("/opt/SOFA")
        ]

        for path in possible_paths:
            if path.exists() and (path / "infer.py").exists():
                return str(path)

        return None
    
    def _check_sofa_availability(self) -> bool:
        """Check if SOFA is available and working."""
        if not self.sofa_path:
            return False

        try:
            sofa_path = Path(self.sofa_path)

            # Check for required SOFA files
            required_files = [
                sofa_path / "infer.py",
                sofa_path / "train.py",
                sofa_path / "dictionary" / "opencpop-extension.txt"
            ]

            for file_path in required_files:
                if not file_path.exists():
                    logger.warning(f"SOFA file not found: {file_path}")
                    return False

            # Check for checkpoint file
            project_root = Path(__file__).parent.parent
            checkpoint_paths = [
                project_root / "models" / "tgm_en_v100.ckpt",
                sofa_path / "ckpt" / "tgm_en_v100.ckpt"
            ]

            checkpoint_found = False
            for ckpt_path in checkpoint_paths:
                if ckpt_path.exists():
                    checkpoint_found = True
                    self.checkpoint_path = str(ckpt_path)
                    break

            if not checkpoint_found:
                logger.warning("SOFA checkpoint file not found (tgm_en_v100.ckpt)")
                return False

            # Check for required Python packages
            try:
                import lightning
                import click
            except ImportError as e:
                logger.warning(f"SOFA dependency not available: {e}")
                return False

            logger.info(f"SOFA found at: {self.sofa_path}")
            logger.info(f"SOFA checkpoint: {self.checkpoint_path}")
            return True

        except Exception as e:
            logger.warning(f"Error checking SOFA availability: {e}")
            return False
    
    def align_lyrics(self, vocal_audio_path: str, lyrics_text: str, 
                    output_dir: Optional[str] = None) -> AlignmentResult:
        """
        Align lyrics with vocal audio.
        
        Args:
            vocal_audio_path: Path to isolated vocal audio
            lyrics_text: Text of the lyrics to align
            output_dir: Directory for temporary files
            
        Returns:
            AlignmentResult object with word-level timings
        """
        if not os.path.exists(vocal_audio_path):
            raise FileNotFoundError(f"Vocal audio file not found: {vocal_audio_path}")
        
        if not lyrics_text.strip():
            raise ValueError("Lyrics text cannot be empty")
        
        logger.info(f"Starting forced alignment for: {vocal_audio_path}")
        
        if self.sofa_available and self.use_sofa:
            return self._align_with_sofa(vocal_audio_path, lyrics_text, output_dir)
        else:
            return self._align_with_fallback(vocal_audio_path, lyrics_text)
    
    def _align_with_sofa(self, vocal_audio_path: str, lyrics_text: str,
                        output_dir: Optional[str] = None) -> AlignmentResult:
        """Perform alignment using SOFA."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sofa_align_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        try:
            import shutil
            import sys

            # Create SOFA-compatible directory structure
            # SOFA expects: segments/name/file.wav and segments/name/file.lab
            base_name = Path(vocal_audio_path).stem
            segments_dir = Path(output_dir) / "segments" / base_name
            segments_dir.mkdir(parents=True, exist_ok=True)

            # Copy audio file to SOFA directory
            audio_dest = segments_dir / f"{base_name}.wav"
            shutil.copy2(vocal_audio_path, audio_dest)

            # Create .lab file with lyrics text (SOFA expects .lab format)
            lab_file = segments_dir / f"{base_name}.lab"
            with open(lab_file, 'w', encoding='utf-8') as f:
                f.write(lyrics_text)

            logger.info(f"Created SOFA input files in: {segments_dir}")

            # Build SOFA command
            sofa_infer_script = Path(self.sofa_path) / "infer.py"
            dictionary_path = Path(self.sofa_path) / "dictionary" / "opencpop-extension.txt"
            segments_parent = Path(output_dir) / "segments"

            cmd = [
                sys.executable,  # Use current Python interpreter
                str(sofa_infer_script),
                "--ckpt", self.checkpoint_path,
                "--folder", str(segments_parent),
                "--g2p", "English",  # Use English G2P module
                "--out_formats", "textgrid"
            ]

            logger.info(f"Running SOFA command: {' '.join(cmd)}")

            # Run SOFA inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.sofa_path  # Run from SOFA directory
            )

            if result.returncode != 0:
                logger.error(f"SOFA failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"SOFA alignment failed: {result.stderr}")

            logger.info(f"SOFA completed successfully")
            logger.debug(f"SOFA output: {result.stdout}")

            # Find generated TextGrid file
            # SOFA creates: segments/name/textgrid/file.TextGrid
            textgrid_output = segments_dir / "textgrid" / f"{base_name}.TextGrid"

            if not textgrid_output.exists():
                # Try alternate location
                textgrid_output = segments_dir / f"{base_name}.TextGrid"

            if not textgrid_output.exists():
                logger.error(f"TextGrid file not found at expected locations:")
                logger.error(f"  - {segments_dir / 'textgrid' / f'{base_name}.TextGrid'}")
                logger.error(f"  - {segments_dir / f'{base_name}.TextGrid'}")
                raise RuntimeError("SOFA did not produce expected TextGrid output")

            logger.info(f"Found TextGrid output: {textgrid_output}")

            # Parse TextGrid output
            return self._parse_textgrid(str(textgrid_output))

        except subprocess.TimeoutExpired:
            logger.error("SOFA alignment timed out")
            raise RuntimeError("SOFA alignment timed out after 5 minutes")
        except Exception as e:
            logger.error(f"SOFA alignment failed: {e}")
            logger.info("Falling back to alternative alignment method")
            # Fall back to alternative method
            return self._align_with_fallback(vocal_audio_path, lyrics_text)
    
    def _parse_textgrid(self, textgrid_path: str) -> AlignmentResult:
        """Parse TextGrid file to extract word alignments."""
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # Find the words tier (usually named "words" or "word")
            word_tier = None
            for tier in tg.tiers:
                if tier.name.lower() in ['words', 'word', 'phones']:
                    word_tier = tier
                    break
            
            if word_tier is None:
                raise ValueError("No word tier found in TextGrid")
            
            # Extract word alignments
            words = []
            for interval in word_tier:
                if interval.mark.strip() and interval.mark.strip() != '<sil>':
                    words.append((
                        interval.mark.strip(),
                        float(interval.minTime),
                        float(interval.maxTime)
                    ))
            
            # Calculate basic confidence based on alignment coverage
            total_duration = max([end for _, _, end in words]) if words else 0
            confidence = min(1.0, len(words) / max(1, len(words) * 0.8))
            
            logger.info(f"Parsed {len(words)} word alignments from TextGrid")
            return AlignmentResult(words, confidence)
            
        except Exception as e:
            logger.error(f"Failed to parse TextGrid: {e}")
            raise RuntimeError(f"TextGrid parsing failed: {e}")
    
    def _align_with_fallback(self, vocal_audio_path: str, lyrics_text: str) -> AlignmentResult:
        """
        Fallback alignment method using audio analysis.
        
        This method uses onset detection and duration estimation to create
        a basic alignment when SOFA is not available.
        """
        logger.info("Using fallback alignment method")
        
        try:
            # Load audio
            y, sr = librosa.load(vocal_audio_path)
            duration = len(y) / sr
            
            # Split lyrics into words
            words = lyrics_text.split()
            if not words:
                return AlignmentResult([])
            
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            
            # If we have fewer onsets than words, create evenly spaced timing
            if len(onset_frames) < len(words):
                # Estimate vocal start (skip silence at beginning)
                vocal_start = self._estimate_vocal_start(y, sr)
                vocal_end = duration * 0.95  # Assume vocals end before track ends
                vocal_duration = vocal_end - vocal_start
                
                # Distribute words evenly across vocal duration
                word_duration = vocal_duration / len(words)
                alignments = []
                
                for i, word in enumerate(words):
                    start_time = vocal_start + i * word_duration
                    end_time = start_time + word_duration * 0.8  # Leave gaps between words
                    alignments.append((word, start_time, end_time))
                
            else:
                # Use onset detection to align words
                alignments = []
                for i, word in enumerate(words):
                    if i < len(onset_frames):
                        start_time = onset_frames[i]
                    else:
                        # Extrapolate for remaining words
                        avg_gap = np.mean(np.diff(onset_frames[:len(words)]))
                        start_time = onset_frames[-1] + (i - len(onset_frames) + 1) * avg_gap
                    
                    # Estimate end time
                    if i + 1 < len(onset_frames):
                        end_time = onset_frames[i + 1] * 0.9  # End before next onset
                    else:
                        end_time = start_time + 1.0  # Default 1 second duration
                    
                    alignments.append((word, start_time, end_time))
            
            # Calculate confidence (lower for fallback method)
            confidence = 0.3  # Fallback method has lower confidence
            
            logger.info(f"Fallback alignment created {len(alignments)} word alignments")
            return AlignmentResult(alignments, confidence)
            
        except Exception as e:
            logger.error(f"Fallback alignment failed: {e}")
            raise RuntimeError(f"All alignment methods failed: {e}")
    
    def _estimate_vocal_start(self, y: np.ndarray, sr: int) -> float:
        """Estimate when vocals start in the audio."""
        # Simple energy-based vocal start detection
        frame_length = 2048
        hop_length = 512
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find first significant energy increase
        threshold = np.mean(rms) + np.std(rms)
        vocal_frame = np.where(rms > threshold)[0]
        
        if len(vocal_frame) > 0:
            return librosa.frames_to_time(vocal_frame[0], sr=sr, hop_length=hop_length)
        else:
            return 5.0  # Default to 5 seconds if no clear start found
    
    def get_aligner_info(self) -> Dict[str, Any]:
        """Get information about the aligner configuration."""
        return {
            'sofa_available': self.sofa_available,
            'sofa_path': self.sofa_path,
            'use_sofa': self.use_sofa,
            'fallback_available': True
        }

