# Beatloom Lyric Synchronization - Usage Guide

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or run the setup script
python setup.py
```

### 2. Basic Usage

```bash
# Process a single audio file
python beatloom_sync.py song.wav lyrics.txt --output ./output

# Process with custom settings
python beatloom_sync.py song.wav lyrics.txt --output ./output --confidence-threshold 0.7 --cpu-only
```

### 3. Test the System

```bash
# Run the system test
python test_system.py

# Run examples
python example.py
```

## Detailed Usage

### Command Line Interface

The main script `beatloom_sync.py` provides a complete command-line interface:

```bash
# Single file processing
python beatloom_sync.py <audio_file> <lyrics_file> --output <output_dir>

# Batch processing
python beatloom_sync.py --batch <audio_dir> <lyrics_dir> --output <output_dir>

# Options:
#   --cpu-only              Use CPU instead of GPU
#   --confidence-threshold  Minimum confidence (default: 0.6)
#   --verbose              Enable detailed logging
#   --quiet                Suppress output
#   --base-name            Custom output filename
```

### Python API

```python
from src.pipeline import LyricSyncPipeline

# Initialize pipeline
with LyricSyncPipeline(use_gpu=True, confidence_threshold=0.6) as pipeline:
    
    # Process single file
    result = pipeline.process_audio(
        audio_path="song.wav",
        lyrics_text="Hello world this is a test song",
        output_dir="./output",
        base_filename="my_song"
    )
    
    # Check results
    if result['success']:
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Words aligned: {result['word_count']}")
        print(f"Output files: {result['output_files']}")
    else:
        print(f"Error: {result['error']}")
```

### Individual Components

```python
# Vocal Separation
from src.vocal_separator import VocalSeparator

separator = VocalSeparator(device='cuda')
vocal_path, info = separator.separate_vocals("song.wav", "./temp")

# Forced Alignment
from src.forced_aligner import ForcedAligner

aligner = ForcedAligner(use_sofa=True)
alignment = aligner.align_lyrics(vocal_path, "lyrics text", "./temp")

# Confidence Scoring
from src.confidence_scorer import ConfidenceScorer

scorer = ConfidenceScorer()
confidence = scorer.score_alignment(alignment, vocal_path, "lyrics text")

# Output Generation
from src.output_generator import OutputGenerator

generator = OutputGenerator()
files = generator.save_all_formats(alignment, confidence, "./output", "song")
```

## Input Requirements

### Audio Files

**Supported Formats:**
- WAV (recommended)
- MP3
- M4A
- FLAC
- OGG
- AAC

**Recommendations:**
- Sample rate: 44.1kHz or 48kHz
- Bit depth: 16-bit or 24-bit
- Stereo or mono
- Clear vocals with minimal background noise
- Duration: 30 seconds to 10 minutes (longer files supported)

### Lyrics Files

**Format:** Plain text (.txt) or LRC (.lrc)

**Requirements:**
- UTF-8 encoding
- One line per phrase or sentence
- Clean text without timestamps
- Match the actual sung lyrics

**Example:**
```
Hello world this is a test song
With lyrics that we want to synchronize
Every word should have a timing
That matches when it's sung
```

## Output Formats

The system generates multiple output formats:

### 1. Beatloom JSON Format
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
    ],
    "duration": 180.5,
    "word_count": 45
  },
  "timing": {
    "fps": 30,
    "frame_count": 5415
  }
}
```

### 2. LRC Format
```
[ar:Beatloom Sync]
[ti:Generated Lyrics]
[length:180.50]

[00:01.23]Hello
[00:01.67]world
[00:02.15]this
```

### 3. SRT Subtitle Format
```
1
00:00:01,230 --> 00:00:03,450
Hello world this is

2
00:00:03,450 --> 00:00:05,670
a test song with
```

### 4. TextGrid Format (Praat)
```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 180.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        intervals: size = 45
        intervals [1]:
            xmin = 1.23
            xmax = 1.67
            text = "Hello"
```

### 5. Comprehensive JSON
Complete data including confidence scores, quality metrics, and metadata.

### 6. Summary Report
Human-readable text report with processing details and quality assessment.

## Quality Assessment

### Confidence Scores

- **Overall Confidence**: 0.0 - 1.0 (higher is better)
- **Acceptable Threshold**: Default 0.6 (configurable)

**Score Components:**
- Duration Score: Word length reasonableness
- Coverage Score: Text matching accuracy
- Timing Score: Temporal consistency
- Audio Score: Correlation with audio features

### Quality Flags

The system identifies potential issues:

- `very_short_words`: Words shorter than 0.1 seconds
- `very_long_words`: Words longer than 5 seconds
- `non_monotonic_timing`: Words not in chronological order
- `large_gaps`: Gaps longer than 3 seconds between words
- `overlapping_words`: Words that overlap in time
- `low_text_coverage`: Poor match between input and output text
- `too_fast`: Speaking rate > 5 words/second
- `too_slow`: Speaking rate < 0.5 words/second

### Improving Results

**For Low Confidence:**
1. Check audio quality (clear vocals, minimal noise)
2. Verify lyrics match the audio exactly
3. Try different audio formats (WAV recommended)
4. Ensure proper stereo/mono format
5. Consider manual correction for critical applications

**For Quality Flags:**
- `very_short_words`: May indicate over-segmentation
- `large_gaps`: May indicate instrumental sections
- `low_text_coverage`: Check lyrics accuracy
- `too_fast/slow`: May indicate genre-specific singing style

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU (default if available)
pipeline = LyricSyncPipeline(use_gpu=True)

# Force CPU usage
pipeline = LyricSyncPipeline(use_gpu=False)
```

### Batch Processing

```python
# Process multiple files efficiently
audio_files = ["song1.wav", "song2.wav", "song3.wav"]
lyrics_texts = ["lyrics 1", "lyrics 2", "lyrics 3"]

results = pipeline.process_batch(audio_files, lyrics_texts, "./output")
```

### Memory Management

- Close pipeline when done: `pipeline.cleanup()`
- Use context manager: `with LyricSyncPipeline() as pipeline:`
- Process files individually for large batches
- Monitor GPU memory usage

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Check Python version (3.8+ required)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. GPU Memory Issues**
```bash
# Use CPU instead
python beatloom_sync.py song.wav lyrics.txt --output ./output --cpu-only
```

**3. Poor Results**
- Check audio quality and format
- Verify lyrics accuracy
- Try different confidence thresholds
- Consider manual correction

**4. SOFA Not Available**
- System automatically uses fallback alignment
- Install SOFA for better accuracy (optional)
- Fallback method still produces usable results

### Performance Issues

**Slow Processing:**
- Use GPU acceleration
- Check available system memory
- Process shorter audio segments
- Close other applications

**High Memory Usage:**
- Use CPU-only mode
- Process files individually
- Reduce batch size
- Monitor system resources

## Advanced Usage

### Custom Configuration

```python
# Custom vocal separator
separator = VocalSeparator(
    model_name='htdemucs',  # or 'mdx_extra'
    device='cuda'
)

# Custom confidence scorer
scorer = ConfidenceScorer(
    min_word_duration=0.05,
    max_word_duration=3.0,
    expected_words_per_second=2.0
)

# Custom pipeline
pipeline = LyricSyncPipeline(
    use_gpu=True,
    confidence_threshold=0.7,
    temp_dir='/tmp/custom'
)
```

### Integration with Beatloom

The Beatloom JSON format is designed for direct integration:

```python
# Load Beatloom output
import json
with open('output/song_beatloom.json', 'r') as f:
    sync_data = json.load(f)

# Use in Beatloom visualization
for word in sync_data['lyrics']['words']:
    start_frame = word['start_frame']
    end_frame = word['end_frame']
    text = word['text']
    # Apply to visualization at frames start_frame to end_frame
```

### Extending the System

The modular architecture allows easy extension:

```python
# Custom output format
class CustomOutputGenerator(OutputGenerator):
    def generate_custom_format(self, alignment):
        # Your custom format logic
        pass

# Custom confidence metrics
class CustomConfidenceScorer(ConfidenceScorer):
    def _score_custom_metric(self, alignment):
        # Your custom scoring logic
        pass
```

## Support and Contributing

### Getting Help

1. Check this usage guide
2. Review the README.md
3. Run the test system: `python test_system.py`
4. Check the example: `python example.py`

### Reporting Issues

When reporting issues, include:
- Python version
- Operating system
- Audio file format and duration
- Complete error message
- Steps to reproduce

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

