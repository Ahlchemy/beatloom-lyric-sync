# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beatloom Lyric Synchronization is an AI-powered system that accurately aligns lyrics with audio using music-specific technologies. Unlike traditional speech-based approaches, this system uses vocal separation (Demucs) and singing-oriented forced alignment (SOFA) to handle the unique challenges of synchronizing lyrics to music.

## Core Architecture

The system uses a 4-stage pipeline architecture (`src/pipeline.py`):

1. **Vocal Separation** (`src/vocal_separator.py`) - Uses Demucs to isolate singing voice from instrumental accompaniment
2. **Forced Alignment** (`src/forced_aligner.py`) - Primary: SOFA (Singing-Oriented Forced Aligner); Fallback: audio analysis with onset detection
3. **Confidence Scoring** (`src/confidence_scorer.py`) - Multi-factor quality evaluation (duration, coverage, timing, audio correlation)
4. **Output Generation** (`src/output_generator.py`) - Generates multiple formats: Beatloom JSON, LRC, SRT, TextGrid, comprehensive JSON

All components are orchestrated by `LyricSyncPipeline` which manages resources, temporary files, and error handling. The pipeline is designed as a context manager for automatic cleanup.

## SOFA Integration

The `SOFA/` directory contains a complete copy of the SOFA (Singing-Oriented Forced Aligner) project. This is a sophisticated forced alignment system specifically designed for singing voice:

- SOFA handles vibrato, melisma, and pitch variations that speech-based aligners cannot
- The system uses SOFA's inference capabilities when available, falling back to built-in alignment otherwise
- SOFA requires checkpoint files (`.ckpt`) and dictionary files (in `SOFA/dictionary/`)
- The forced aligner checks for SOFA availability via `src/forced_aligner.py` and adapts accordingly

## Common Commands

### Running the System

```bash
# Single file processing
python beatloom_sync.py audio.wav lyrics.txt --output ./output

# Batch processing
python beatloom_sync.py --batch ./audio_dir ./lyrics_dir --output ./output

# CPU-only mode (no GPU)
python beatloom_sync.py audio.wav lyrics.txt --output ./output --cpu-only

# Custom confidence threshold
python beatloom_sync.py audio.wav lyrics.txt --output ./output --confidence-threshold 0.7
```

### Setup and Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script (creates directories, tests installation)
python setup.py

# Run test suite
python test_system.py

# Run examples (demonstrates API usage)
python example.py
```

### SOFA Training and Inference

```bash
# SOFA inference (if using SOFA directly)
cd SOFA
python infer.py --ckpt checkpoint_path --folder segments_path --dictionary dictionary/opencpop-extension.txt

# SOFA training
python binarize.py  # Process training data
python train.py -p path_to_pretrained_model
```

## Python API Usage

The system is designed for both CLI and programmatic use:

```python
from src.pipeline import LyricSyncPipeline

# Use as context manager for automatic cleanup
with LyricSyncPipeline(use_gpu=True, confidence_threshold=0.6) as pipeline:
    result = pipeline.process_audio(
        'song.wav',
        'lyrics text here',
        './output',
        base_filename='song'
    )

    if result['success']:
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Output files: {result['output_files']}")
```

Individual components can also be used separately (see `example.py` for component-level usage).

## Key Implementation Details

### Vocal Separation
- Uses Facebook's Demucs hybrid spectrogram/waveform model
- Default model: 'htdemucs' (configurable in `VocalSeparator`)
- Outputs isolated vocal track for alignment
- Supports GPU acceleration via CUDA

### Forced Alignment Strategy
- **Primary**: SOFA is checked first via `forced_aligner.py`
- **Fallback**: If SOFA unavailable, uses onset detection + duration estimation
- The fallback uses librosa for audio analysis and creates reasonable word boundaries
- SOFA provides phoneme-level accuracy when available

### Confidence Scoring
- Multi-factor scoring: duration score, coverage score, timing score, audio score
- Quality flags identify issues: very_short_words, very_long_words, non_monotonic_timing, large_gaps, overlapping_words, low_text_coverage, too_fast/too_slow
- Default threshold: 0.6 (configurable)
- Used to determine if output is acceptable for production use

### Output Formats
The system generates 5 output formats simultaneously:
1. **Beatloom JSON** - Custom format with frame numbers and confidence scores
2. **LRC** - Standard karaoke format with timestamps
3. **SRT** - Subtitle format with grouped words
4. **TextGrid** - Praat annotation format for linguistic analysis
5. **Comprehensive JSON** - Full metadata, processing info, quality metrics

## File Organization

- `beatloom_sync.py` - Main CLI entry point
- `src/` - Core pipeline modules (all components are in this directory)
- `SOFA/` - Complete SOFA project (singing-oriented forced aligner)
- `audio/` - Default directory for input audio files
- `output/` - Default directory for synchronized lyrics output
- `models/` - Model files and checkpoints
- `example.py` - Demonstrates API usage with synthetic audio
- `test_system.py` - Simple integration test
- `setup.py` - Setup and dependency installation script

## Dependencies

Key dependencies (from `requirements.txt`):
- `torch` + `torchaudio` - Neural network models (Demucs, SOFA)
- `demucs` - Vocal separation
- `librosa` - Audio analysis (fallback alignment)
- `soundfile` - Audio I/O
- `textgrid` - TextGrid format support
- `pydub` - Audio manipulation

Optional: CUDA-capable GPU for acceleration (automatically detected)

## Development Notes

- All pipeline components implement cleanup via context managers
- Temporary files are managed in `temp_dir` (default: system temp with prefix "beatloom_sync_")
- Logging is configured in `beatloom_sync.py` with dual output (console + file)
- The system handles missing SOFA gracefully by falling back to built-in alignment
- GPU/CPU switching is automatic based on CUDA availability and user preference
- Batch processing reuses pipeline components for efficiency

## Quality Thresholds

Default confidence threshold is 0.6, but this can be adjusted based on use case:
- **>0.8**: High quality, suitable for production
- **0.6-0.8**: Acceptable, may need manual review
- **<0.6**: Low quality, likely needs re-processing or better source audio

The confidence score combines:
- Word duration reasonableness (not too short/long)
- Text coverage (aligned text matches input)
- Timing consistency (monotonic, no large gaps)
- Audio feature correlation (alignment matches audio energy)
