#!/usr/bin/env python3
"""
Setup script for Beatloom Lyric Synchronization System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True


def check_dependencies():
    """Check if required system dependencies are available."""
    print("Checking system dependencies...")
    
    # Check for ffmpeg (needed for audio processing)
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ FFmpeg not found - may be needed for some audio formats")
    
    return True


def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("✓ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")
    
    directories = ['audio', 'output', 'temp', 'tests']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def test_installation():
    """Test if the installation works."""
    print("Testing installation...")
    
    try:
        # Test imports
        sys.path.insert(0, 'src')
        from src.pipeline import LyricSyncPipeline
        from src.vocal_separator import VocalSeparator
        from src.forced_aligner import ForcedAligner
        from src.confidence_scorer import ConfidenceScorer
        from src.output_generator import OutputGenerator
        
        print("✓ All modules imported successfully")
        
        # Test pipeline initialization
        pipeline = LyricSyncPipeline(use_gpu=False)
        info = pipeline.get_pipeline_info()
        print("✓ Pipeline initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("Beatloom Lyric Synchronization System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check system dependencies
    if not check_dependencies():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\nSetup failed during dependency installation.")
        print("You may need to install dependencies manually:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Test installation
    if not test_installation():
        print("\nSetup completed but tests failed.")
        print("The system may still work, but some features might be limited.")
        return 1
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your audio files in the 'audio' directory")
    print("2. Create lyrics text files")
    print("3. Run: python beatloom_sync.py audio/song.wav audio/lyrics.txt --output output")
    print("4. Or try the example: python example.py")
    print("\nFor more information, see README.md")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

