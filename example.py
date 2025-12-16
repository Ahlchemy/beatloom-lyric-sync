#!/usr/bin/env python3
"""
Beatloom Lyric Synchronization - Example Usage

This script demonstrates how to use the Beatloom lyric synchronization system
with both the Python API and command-line interface.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import LyricSyncPipeline


def create_test_audio(duration=10, sample_rate=44100):
    """
    Create a simple test audio file with synthetic vocals.
    
    This creates a basic audio signal that simulates speech/singing
    for testing purposes when real audio files are not available.
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create a simple melody with harmonics (simulating vocals)
    fundamental = 220  # A3 note
    
    # Generate vocal-like signal with multiple harmonics
    signal = (
        0.5 * np.sin(2 * np.pi * fundamental * t) +
        0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.2 * np.sin(2 * np.pi * fundamental * 3 * t) +
        0.1 * np.sin(2 * np.pi * fundamental * 4 * t)
    )
    
    # Add some variation to simulate singing
    vibrato = 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
    signal *= (1 + vibrato)
    
    # Add some background "instruments" (lower frequencies)
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)  # Bass line
    drums = 0.2 * np.random.normal(0, 0.1, len(t))  # Noise for drums
    
    # Combine all elements
    mixed_signal = signal + bass + drums
    
    # Normalize
    mixed_signal = mixed_signal / np.max(np.abs(mixed_signal)) * 0.8
    
    # Convert to stereo
    stereo_signal = np.column_stack([mixed_signal, mixed_signal])
    
    return stereo_signal, sample_rate


def example_basic_usage():
    """Demonstrate basic usage of the lyric synchronization system."""
    print("=== Basic Usage Example ===")
    
    # Create test audio and lyrics
    print("Creating test audio file...")
    audio_data, sample_rate = create_test_audio(duration=8)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "test_song.wav")
        lyrics_text = "Hello world this is a test song with some lyrics to synchronize"
        output_dir = os.path.join(temp_dir, "output")
        
        # Save test audio
        sf.write(audio_path, audio_data, sample_rate)
        print(f"Test audio saved to: {audio_path}")
        
        # Initialize and run pipeline
        print("Initializing lyric synchronization pipeline...")
        try:
            with LyricSyncPipeline(use_gpu=False) as pipeline:  # Use CPU for example
                print("Processing audio and lyrics...")
                result = pipeline.process_audio(
                    audio_path=audio_path,
                    lyrics_text=lyrics_text,
                    output_dir=output_dir,
                    base_filename="test_example"
                )
                
                # Display results
                print("\n--- Results ---")
                print(f"Success: {result['success']}")
                
                if result['success']:
                    print(f"Confidence: {result['confidence']:.3f}")
                    print(f"Acceptable Quality: {result['is_acceptable']}")
                    print(f"Words Aligned: {result['word_count']}")
                    print(f"Duration: {result['duration']:.2f} seconds")
                    print(f"Processing Time: {result['processing_time']:.2f} seconds")
                    
                    if result['quality_flags']:
                        print(f"Quality Flags: {', '.join(result['quality_flags'])}")
                    
                    print("\nOutput Files:")
                    for format_name, file_path in result['output_files'].items():
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            print(f"  {format_name}: {file_path} ({file_size} bytes)")
                    
                    # Show sample of Beatloom format
                    beatloom_file = result['output_files'].get('beatloom')
                    if beatloom_file and os.path.exists(beatloom_file):
                        print("\n--- Sample Beatloom Output ---")
                        import json
                        with open(beatloom_file, 'r') as f:
                            data = json.load(f)
                        
                        print(f"Version: {data.get('version')}")
                        print(f"Confidence: {data.get('confidence'):.3f}")
                        print(f"Word Count: {data['lyrics']['word_count']}")
                        print("First few words:")
                        for word in data['lyrics']['words'][:5]:
                            print(f"  {word['start_time']:.2f}s - {word['end_time']:.2f}s: '{word['text']}'")
                
                else:
                    print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Pipeline error: {e}")
            print("Note: This example uses synthetic audio and may not work perfectly.")
            print("For best results, use real audio files with clear vocals.")


def example_component_usage():
    """Demonstrate individual component usage."""
    print("\n=== Component Usage Example ===")
    
    # Create test audio
    print("Creating test audio...")
    audio_data, sample_rate = create_test_audio(duration=5)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "component_test.wav")
        sf.write(audio_path, audio_data, sample_rate)
        
        try:
            # Test Vocal Separator
            print("\n1. Testing Vocal Separator...")
            from src.vocal_separator import VocalSeparator
            
            separator = VocalSeparator(device='cpu')
            vocal_path, sep_info = separator.separate_vocals(audio_path, temp_dir)
            print(f"   Vocal track saved: {vocal_path}")
            print(f"   Separation info: {sep_info['model_used']} on {sep_info['device_used']}")
            
            # Test Forced Aligner
            print("\n2. Testing Forced Aligner...")
            from src.forced_aligner import ForcedAligner
            
            aligner = ForcedAligner(use_sofa=False)  # Use fallback for example
            lyrics_text = "Hello world test song lyrics"
            alignment = aligner.align_lyrics(vocal_path, lyrics_text, temp_dir)
            print(f"   Aligned {len(alignment.words)} words")
            print(f"   Duration: {alignment.duration:.2f} seconds")
            
            # Test Confidence Scorer
            print("\n3. Testing Confidence Scorer...")
            from src.confidence_scorer import ConfidenceScorer
            
            scorer = ConfidenceScorer()
            confidence_result = scorer.score_alignment(alignment, vocal_path, lyrics_text)
            print(f"   Overall confidence: {confidence_result['overall_confidence']:.3f}")
            print(f"   Quality flags: {confidence_result['quality_flags']}")
            
            # Test Output Generator
            print("\n4. Testing Output Generator...")
            from src.output_generator import OutputGenerator
            
            generator = OutputGenerator()
            output_files = generator.save_all_formats(
                alignment, confidence_result, temp_dir, "component_test"
            )
            print(f"   Generated {len(output_files)} output formats")
            for format_name in output_files:
                print(f"     - {format_name}")
                
        except Exception as e:
            print(f"Component test error: {e}")
            print("Note: Some components may not work with synthetic audio.")


def example_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test files
        audio_dir = os.path.join(temp_dir, "audio")
        lyrics_dir = os.path.join(temp_dir, "lyrics")
        output_dir = os.path.join(temp_dir, "output")
        
        os.makedirs(audio_dir)
        os.makedirs(lyrics_dir)
        
        # Create test files
        test_files = [
            ("song1", "Hello world this is the first test song"),
            ("song2", "This is the second test with different lyrics"),
            ("song3", "Third song has even more words to test the system")
        ]
        
        print(f"Creating {len(test_files)} test files...")
        audio_files = []
        lyrics_texts = []
        
        for i, (name, lyrics) in enumerate(test_files):
            # Create audio
            duration = 6 + i  # Varying durations
            audio_data, sample_rate = create_test_audio(duration)
            audio_path = os.path.join(audio_dir, f"{name}.wav")
            sf.write(audio_path, audio_data, sample_rate)
            audio_files.append(audio_path)
            
            # Create lyrics file
            lyrics_path = os.path.join(lyrics_dir, f"{name}.txt")
            with open(lyrics_path, 'w') as f:
                f.write(lyrics)
            lyrics_texts.append(lyrics)
            
            print(f"  Created: {name}.wav ({duration}s) with {len(lyrics.split())} words")
        
        # Process batch
        print("\nProcessing batch...")
        try:
            with LyricSyncPipeline(use_gpu=False) as pipeline:
                results = pipeline.process_batch(audio_files, lyrics_texts, output_dir)
                
                # Display batch results
                print("\n--- Batch Results ---")
                successful = sum(1 for r in results if r['success'])
                acceptable = sum(1 for r in results if r.get('is_acceptable', False))
                avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
                
                print(f"Total files: {len(results)}")
                print(f"Successful: {successful}")
                print(f"Acceptable quality: {acceptable}")
                print(f"Average confidence: {avg_confidence:.3f}")
                
                print("\nIndividual results:")
                for i, result in enumerate(results):
                    name = test_files[i][0]
                    if result['success']:
                        print(f"  {name}: ✓ Confidence: {result['confidence']:.3f}")
                    else:
                        print(f"  {name}: ✗ Error: {result.get('error', 'Unknown')}")
                        
        except Exception as e:
            print(f"Batch processing error: {e}")


def main():
    """Run all examples."""
    print("Beatloom Lyric Synchronization - Examples")
    print("=" * 50)
    
    print("Note: These examples use synthetic audio for demonstration.")
    print("For best results, use real audio files with clear vocals.\n")
    
    try:
        # Run examples
        example_basic_usage()
        example_component_usage()
        example_batch_processing()
        
        print("\n" + "=" * 50)
        print("Examples completed!")
        print("\nTo use with real audio files:")
        print("  python beatloom_sync.py your_song.wav your_lyrics.txt --output ./output")
        print("\nFor more information, see README.md")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nExample error: {e}")
        print("This may be due to missing dependencies or system limitations.")


if __name__ == '__main__':
    main()

