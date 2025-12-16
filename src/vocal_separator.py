"""
Vocal Separation Component using Demucs

This module handles the separation of vocals from instrumental accompaniment
using the state-of-the-art Demucs neural network model.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple
import torch
import torchaudio
import soundfile as sf
from demucs.separate import apply_model
from demucs.pretrained import get_model

logger = logging.getLogger(__name__)


class VocalSeparator:
    """
    Vocal separation using Demucs for isolating singing voice from music.
    
    This class provides a clean interface to the Demucs model for separating
    vocals from instrumental accompaniment, which is crucial for accurate
    lyric alignment.
    """
    
    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        """
        Initialize the vocal separator.
        
        Args:
            model_name: Demucs model to use ('htdemucs' is recommended)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Demucs model."""
        try:
            logger.info(f"Initializing Demucs model '{self.model_name}' on {self.device}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Demucs model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Demucs model: {e}")
            raise
    
    def separate_vocals(self, audio_path: str, output_dir: Optional[str] = None) -> Tuple[str, dict]:
        """
        Separate vocals from an audio file.
        
        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save separated tracks (optional)
            
        Returns:
            Tuple of (vocal_track_path, separation_info)
            
        Raises:
            FileNotFoundError: If input audio file doesn't exist
            RuntimeError: If separation fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Starting vocal separation for: {audio_path}")
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="demucs_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure stereo input (Demucs expects stereo)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]  # Take first two channels
            
            logger.info(f"Audio loaded: {waveform.shape}, sample_rate: {sample_rate}")
            
            # Perform separation using Demucs
            logger.info("Running Demucs separation...")
            
            # Use the apply_model function
            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    waveform[None],  # Add batch dimension
                    device=self.device,
                    progress=True
                )[0]  # Remove batch dimension
            
            # Extract vocals (index 3 in htdemucs: drums, bass, other, vocals)
            vocals = sources[3]  # vocals track
            
            # Save vocal track
            vocal_filename = f"vocals_{Path(audio_path).stem}.wav"
            vocal_path = os.path.join(output_dir, vocal_filename)
            
            # Convert to numpy and save
            vocals_np = vocals.cpu().numpy()
            sf.write(vocal_path, vocals_np.T, sample_rate)
            
            # Prepare separation info
            separation_info = {
                'input_file': audio_path,
                'output_dir': output_dir,
                'vocal_path': vocal_path,
                'sample_rate': sample_rate,
                'duration': vocals_np.shape[1] / sample_rate,
                'channels': vocals_np.shape[0],
                'model_used': self.model_name,
                'device_used': self.device
            }
            
            logger.info(f"Vocal separation completed. Vocal track saved to: {vocal_path}")
            return vocal_path, separation_info
            
        except Exception as e:
            logger.error(f"Vocal separation failed: {e}")
            raise RuntimeError(f"Failed to separate vocals: {e}")
    
    def separate_all_sources(self, audio_path: str, output_dir: Optional[str] = None) -> dict:
        """
        Separate all sources (drums, bass, other, vocals) from an audio file.
        
        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save separated tracks
            
        Returns:
            Dictionary with paths to all separated tracks
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="demucs_full_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure stereo input
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            
            # Perform separation
            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    waveform[None],  # Add batch dimension
                    device=self.device,
                    progress=True
                )[0]  # Remove batch dimension
            
            # Source names for htdemucs
            source_names = ['drums', 'bass', 'other', 'vocals']
            separated_paths = {}
            
            base_name = Path(audio_path).stem
            
            for i, source_name in enumerate(source_names):
                source_audio = sources[i].cpu().numpy()
                filename = f"{source_name}_{base_name}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, source_audio.T, sample_rate)
                separated_paths[source_name] = filepath
            
            logger.info(f"All sources separated and saved to: {output_dir}")
            return separated_paths
            
        except Exception as e:
            logger.error(f"Full source separation failed: {e}")
            raise RuntimeError(f"Failed to separate all sources: {e}")
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'available': self.model is not None
        }

