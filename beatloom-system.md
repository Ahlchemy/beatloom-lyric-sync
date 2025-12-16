# Beatloom Lyric Synchronization System

A state-of-the-art lyric synchronization system that accurately aligns lyrics with audio using advanced AI techniques. This system addresses the fundamental limitations of traditional speech-based approaches by using music-specific technologies.

## Features

- **Vocal Separation**: Uses Demucs to isolate singing voice from instrumental accompaniment
- **Singing-Specific Alignment**: Uses SOFA (Singing-Oriented Forced Aligner) for accurate lyric timing
- **Confidence Scoring**: Evaluates alignment quality and identifies potential issues
- **Multiple Output Formats**: Supports Beatloom JSON, LRC, SRT, TextGrid, and comprehensive JSON
- **Batch Processing**: Process multiple files efficiently
- **GPU Acceleration**: Optional CUDA support for faster processing

## Architecture

The system uses a modular 4-stage pipeline:

1. **Vocal Separation** (Demucs) - Isolates vocals from instruments
2. **Forced Alignment** (SOFA/Fallback) - Aligns lyrics to vocal audio
3. **Confidence Scoring** - Evaluates alignment quality
4. **Output Generation** - Creates multiple output formats

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for acceleration)
- At least 4GB RAM (8GB+ recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install SOFA

For best results, install SOFA (Singing-Oriented Forced Aligner):

```bash
# Follow instructions at: https://github.com/qiuqiao/SOFA
# The system will fall back to a built-in alignment method if SOFA is not available
```

## Quick Start

### Single File Processing

```bash
python beatloom_sync.py audio.wav lyrics.txt --output ./output
```

### Batch Processing

```bash
python beatloom_sync.py --batch ./audio_dir ./lyrics_dir --output ./output
```

### Python API

```python
from src.pipeline import LyricSyncPipeline

# Initialize pipeline
with LyricSyncPipeline() as pipeline:
    # Process single file
    result = pipeline.process_audio(
        'song.wav',
        'Hello world, this is a test song...',
        './output'
    )
    
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Output files: {result['output_files']}")
```

## Command Line Options

```
usage: beatloom_sync.py [-h] [--batch AUDIO_DIR LYRICS_DIR] [--output OUTPUT]
                        [--base-name BASE_NAME] [--cpu-only]
                        [--confidence-threshold CONFIDENCE_THRESHOLD]
                        [--verbose] [--quiet]
                        [audio] [lyrics]

Options:
  audio                 Input audio file
  lyrics                Input lyrics file
  --batch AUDIO_DIR LYRICS_DIR
                        Batch process: audio directory and lyrics directory
  --output OUTPUT, -o OUTPUT
                        Output directory for synchronized lyrics
  --base-name BASE_NAME
                        Base name for output files
  --cpu-only            Use CPU only (disable GPU acceleration)
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Minimum confidence threshold (default: 0.6)
  --verbose, -v         Enable verbose logging
  --quiet, -q           Suppress non-error output
```

## Output Formats

The system generates multiple output formats:

### Beatloom JSON Format
```json
{
  "version": "1.0",
  "confidence": 0.85,
  "lyrics": {
    "words": [
      {
        "text": "Hello",
        "start_time": 1.23,
        "end_time": 1.67,
        "start_frame": 37,
        "end_frame": 50
      }
    ]
  }
}
```

### LRC Format
```
[00:01.23]Hello
[00:01.67]world
```

### SRT Format
```
1
00:00:01,230 --> 00:00:03,450
Hello world this is
```

## Configuration

### Pipeline Settings

```python
pipeline = LyricSyncPipeline(
    use_gpu=True,                    # Enable GPU acceleration
    confidence_threshold=0.6,        # Minimum acceptable confidence
    temp_dir='/tmp/beatloom_sync'    # Temporary files directory
)
```

### Vocal Separator Settings

```python
from src.vocal_separator import VocalSeparator

separator = VocalSeparator(
    model_name='htdemucs',  # Demucs model variant
    device='cuda'           # Processing device
)
```

## Quality Assessment

The system provides comprehensive quality metrics:

- **Overall Confidence**: Combined score (0.0 - 1.0)
- **Duration Score**: Word duration reasonableness
- **Coverage Score**: Text coverage accuracy
- **Timing Score**: Temporal consistency
- **Audio Score**: Correlation with audio features

### Quality Flags

The system identifies potential issues:

- `very_short_words` - Words with unreasonably short duration
- `very_long_words` - Words with unreasonably long duration
- `non_monotonic_timing` - Words not in chronological order
- `large_gaps` - Significant silence between words
- `overlapping_words` - Words that overlap in time
- `low_text_coverage` - Poor match between input and aligned text
- `too_fast` / `too_slow` - Unrealistic speaking rate

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Check audio quality (clear vocals, minimal noise)
   - Verify lyrics text matches the audio
   - Try different Demucs models

2. **SOFA Not Available**
   - System automatically falls back to built-in alignment
   - Install SOFA for better accuracy
   - Check SOFA installation and PATH

3. **GPU Memory Issues**
   - Use `--cpu-only` flag
   - Reduce batch size
   - Close other GPU applications

4. **Poor Vocal Separation**
   - Try different audio formats (WAV recommended)
   - Check for mono vs stereo issues
   - Verify audio sample rate (44.1kHz recommended)

### Performance Tips

- Use GPU acceleration when available
- Process WAV files for best quality
- Ensure lyrics text is clean and accurate
- Use batch processing for multiple files

## Technical Details

### Vocal Separation

Uses Facebook's Demucs neural network:
- Hybrid spectrogram and waveform model
- Pre-trained on large music dataset
- Separates drums, bass, other, and vocals

### Forced Alignment

Primary: SOFA (Singing-Oriented Forced Aligner)
- Purpose-built for singing voice
- Handles vibrato, melisma, pitch variations
- Phoneme-level accuracy

Fallback: Audio analysis
- Onset detection
- Duration estimation
- Energy-based timing

### Confidence Scoring

Multi-factor evaluation:
- Word duration analysis
- Text coverage assessment
- Timing consistency check
- Audio feature correlation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Demucs**: Facebook Research for vocal separation
- **SOFA**: qiuqiao for singing-oriented forced alignment
- **Beatloom**: Original visualization system
- **Research Community**: Academic papers that informed this approach

## References

1. Défossez, A., et al. "Hybrid Spectrogram and Waveform Source Separation"
2. Qiu, Q. "SOFA: Singing-Oriented Forced Aligner"
3. Huang, J., et al. "Improving Lyrics Alignment through Joint Pitch Detection"
4. Schulze-Forster, K., et al. "Phoneme Level Lyrics Alignment"

