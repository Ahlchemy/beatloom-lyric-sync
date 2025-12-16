#!/usr/bin/env python3
"""
Simple test script to verify the Beatloom Lyric Synchronization system works.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import LyricSyncPipeline


def create_simple_test_audio(duration=8, sample_rate=44100):
    """Create a simple test audio file."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create a simple vocal-like signal
    fundamental = 220  # A3 note
    signal = (
        0.6 * np.sin(2 * np.pi * fundamental * t) +
        0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t)
    )
    
    # Add some variation
    vibrato = 0.05 * np.sin(2 * np.pi * 5 * t)
    signal *= (1 + vibrato)
    
    # Add background
    bass = 0.2 * np.sin(2 * np.pi * 55 * t)
    mixed_signal = signal + bass
    
    # Normalize
    mixed_signal = mixed_signal / np.max(np.abs(mixed_signal)) * 0.8
    
    # Convert to stereo
    stereo_signal = np.column_stack([mixed_signal, mixed_signal])
    
    return stereo_signal, sample_rate


def test_system():
    """Test the complete system."""
    print("Testing Beatloom Lyric Synchronization System")
    print("=" * 50)
    
    # Create test data
    print("Creating test audio...")
    audio_data, sample_rate = create_simple_test_audio()
    lyrics_text = "Hello world this is a test song with lyrics to synchronize"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test audio
        audio_path = os.path.join(temp_dir, "test.wav")
        sf.write(audio_path, audio_data, sample_rate)
        print(f"Test audio saved: {audio_path}")
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test the pipeline
        print("\nInitializing pipeline...")
        try:
            with LyricSyncPipeline(use_gpu=False) as pipeline:
                print("Pipeline initialized successfully!")
                
                print("\nProcessing audio...")
                result = pipeline.process_audio(
                    audio_path=audio_path,
                    lyrics_text=lyrics_text,
                    output_dir=output_dir,
                    base_filename="test"
                )
                
                print("\n" + "=" * 30)
                print("RESULTS:")
                print("=" * 30)
                
                if result['success']:
                    print("✓ Processing completed successfully!")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Acceptable: {result['is_acceptable']}")
                    print(f"  Words aligned: {result['word_count']}")
                    print(f"  Duration: {result['duration']:.2f} seconds")
                    print(f"  Processing time: {result['processing_time']:.2f} seconds")
                    
                    if result['quality_flags']:
                        print(f"  Quality flags: {', '.join(result['quality_flags'])}")
                    
                    print(f"\n  Output files generated:")
                    for format_name, file_path in result['output_files'].items():
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            print(f"    {format_name}: {os.path.basename(file_path)} ({file_size} bytes)")
                    
                    # Show sample output
                    if 'beatloom' in result['output_files']:
                        print(f"\n  Sample Beatloom output:")
                        import json
                        with open(result['output_files']['beatloom'], 'r') as f:
                            data = json.load(f)
                        
                        print(f"    Version: {data.get('version')}")
                        print(f"    Confidence: {data.get('confidence'):.3f}")
                        print(f"    Word count: {data['lyrics']['word_count']}")
                        print(f"    First few words:")
                        for word in data['lyrics']['words'][:3]:
                            print(f"      {word['start_time']:.2f}s-{word['end_time']:.2f}s: '{word['text']}'")
                    
                    print(f"\n✓ System test PASSED!")
                    return True
                    
                else:
                    print(f"✗ Processing failed: {result['error']}")
                    print(f"✗ System test FAILED!")
                    return False
                    
        except Exception as e:
            print(f"✗ System test FAILED with exception: {e}")
            return False


if __name__ == '__main__':
    success = test_system()
    sys.exit(0 if success else 1)

