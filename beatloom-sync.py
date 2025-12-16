#!/usr/bin/env python3
"""
Beatloom Lyric Synchronization - Command Line Interface

A complete solution for synchronizing lyrics with audio using:
1. Vocal separation with Demucs
2. Forced alignment with SOFA
3. Confidence scoring
4. Multiple output formats

Usage:
    python beatloom_sync.py audio.wav lyrics.txt --output ./output
    python beatloom_sync.py --batch audio_dir lyrics_dir --output ./output
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import LyricSyncPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('beatloom_sync.log')
        ]
    )


def find_audio_files(directory: str) -> List[str]:
    """Find audio files in a directory."""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    audio_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            audio_files.append(str(file_path))
    
    return sorted(audio_files)


def find_lyrics_files(directory: str) -> List[str]:
    """Find lyrics files in a directory."""
    lyrics_extensions = {'.txt', '.lrc'}
    lyrics_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in lyrics_extensions:
            lyrics_files.append(str(file_path))
    
    return sorted(lyrics_files)


def match_audio_lyrics(audio_files: List[str], lyrics_files: List[str]) -> List[Tuple[str, str]]:
    """Match audio files with corresponding lyrics files."""
    matches = []
    
    for audio_file in audio_files:
        audio_stem = Path(audio_file).stem
        
        # Look for exact match first
        for lyrics_file in lyrics_files:
            lyrics_stem = Path(lyrics_file).stem
            if audio_stem == lyrics_stem:
                matches.append((audio_file, lyrics_file))
                break
        else:
            # Look for partial match
            for lyrics_file in lyrics_files:
                lyrics_stem = Path(lyrics_file).stem
                if audio_stem in lyrics_stem or lyrics_stem in audio_stem:
                    matches.append((audio_file, lyrics_file))
                    break
            else:
                print(f"Warning: No lyrics file found for {audio_file}")
    
    return matches


def process_single_file(args):
    """Process a single audio file."""
    # Read lyrics
    try:
        with open(args.lyrics, 'r', encoding='utf-8') as f:
            lyrics_text = f.read().strip()
    except Exception as e:
        print(f"Error reading lyrics file: {e}")
        return 1
    
    # Initialize pipeline
    try:
        with LyricSyncPipeline(
            use_gpu=not args.cpu_only,
            confidence_threshold=args.confidence_threshold
        ) as pipeline:
            
            # Process audio
            result = pipeline.process_audio(
                args.audio,
                lyrics_text,
                args.output,
                args.base_name
            )
            
            # Print results
            if result['success']:
                print(f"✓ Processing completed successfully!")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Acceptable: {result['is_acceptable']}")
                print(f"  Words aligned: {result['word_count']}")
                print(f"  Duration: {result['duration']:.2f} seconds")
                print(f"  Processing time: {result['processing_time']:.2f} seconds")
                
                if result['quality_flags']:
                    print(f"  Quality flags: {', '.join(result['quality_flags'])}")
                
                print(f"  Output files:")
                for format_name, file_path in result['output_files'].items():
                    print(f"    {format_name}: {file_path}")
                
                return 0 if result['is_acceptable'] else 2
            else:
                print(f"✗ Processing failed: {result['error']}")
                return 1
                
    except Exception as e:
        print(f"Error: {e}")
        return 1


def process_batch(args):
    """Process multiple files in batch."""
    # Find audio and lyrics files
    audio_files = find_audio_files(args.audio_dir)
    lyrics_files = find_lyrics_files(args.lyrics_dir)
    
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return 1
    
    if not lyrics_files:
        print(f"No lyrics files found in {args.lyrics_dir}")
        return 1
    
    # Match files
    matches = match_audio_lyrics(audio_files, lyrics_files)
    
    if not matches:
        print("No matching audio/lyrics pairs found")
        return 1
    
    print(f"Found {len(matches)} audio/lyrics pairs to process")
    
    # Read all lyrics
    audio_paths = []
    lyrics_texts = []
    
    for audio_file, lyrics_file in matches:
        try:
            with open(lyrics_file, 'r', encoding='utf-8') as f:
                lyrics_text = f.read().strip()
            
            audio_paths.append(audio_file)
            lyrics_texts.append(lyrics_text)
            
        except Exception as e:
            print(f"Error reading {lyrics_file}: {e}")
            continue
    
    if not audio_paths:
        print("No valid audio/lyrics pairs found")
        return 1
    
    # Process batch
    try:
        with LyricSyncPipeline(
            use_gpu=not args.cpu_only,
            confidence_threshold=args.confidence_threshold
        ) as pipeline:
            
            results = pipeline.process_batch(audio_paths, lyrics_texts, args.output)
            
            # Print summary
            successful = sum(1 for r in results if r['success'])
            acceptable = sum(1 for r in results if r.get('is_acceptable', False))
            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
            
            print(f"\nBatch processing completed:")
            print(f"  Total files: {len(results)}")
            print(f"  Successful: {successful}")
            print(f"  Acceptable quality: {acceptable}")
            print(f"  Average confidence: {avg_confidence:.3f}")
            
            return 0 if successful == len(results) else 1
            
    except Exception as e:
        print(f"Batch processing error: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Beatloom Lyric Synchronization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python beatloom_sync.py song.wav lyrics.txt --output ./output
  
  # Process batch with custom settings
  python beatloom_sync.py --batch ./audio ./lyrics --output ./output --confidence-threshold 0.7
  
  # Use CPU only (no GPU)
  python beatloom_sync.py song.wav lyrics.txt --output ./output --cpu-only
        """
    )
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('audio', nargs='?', help='Input audio file')
    group.add_argument('--batch', nargs=2, metavar=('AUDIO_DIR', 'LYRICS_DIR'),
                      help='Batch process: audio directory and lyrics directory')
    
    parser.add_argument('lyrics', nargs='?', help='Input lyrics file (for single file mode)')
    
    # Output arguments
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for synchronized lyrics')
    parser.add_argument('--base-name', help='Base name for output files (single file mode)')
    
    # Processing options
    parser.add_argument('--cpu-only', action='store_true',
                       help='Use CPU only (disable GPU acceleration)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence threshold for acceptable results (default: 0.6)')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-error output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch and not args.lyrics:
        parser.error("lyrics file is required for single file mode")
    
    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process files
    if args.batch:
        args.audio_dir, args.lyrics_dir = args.batch
        return process_batch(args)
    else:
        return process_single_file(args)


if __name__ == '__main__':
    sys.exit(main())

