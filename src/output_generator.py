"""
Output Generation Component

This module converts alignment results into various output formats
suitable for Beatloom and other applications.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .forced_aligner import AlignmentResult

logger = logging.getLogger(__name__)


class OutputGenerator:
    """
    Generate various output formats from alignment results.
    
    This class converts alignment results into formats suitable for
    Beatloom visualization and other applications.
    """
    
    def __init__(self):
        """Initialize the output generator."""
        pass
    
    def generate_beatloom_format(self, 
                                alignment: AlignmentResult,
                                confidence_result: Dict[str, Any],
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Beatloom-compatible format.
        
        Args:
            alignment: Alignment result
            confidence_result: Confidence scoring result
            metadata: Additional metadata
            
        Returns:
            Dictionary in Beatloom format
        """
        beatloom_data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'confidence': confidence_result.get('overall_confidence', 0.0),
            'quality_flags': confidence_result.get('quality_flags', []),
            'lyrics': {
                'words': [],
                'duration': alignment.duration,
                'word_count': len(alignment.words)
            },
            'timing': {
                'fps': 30,  # Default FPS for Beatloom
                'frame_count': int(alignment.duration * 30)
            }
        }
        
        # Convert words to Beatloom format
        for word, start, end in alignment.words:
            word_data = {
                'text': word,
                'start_time': float(start),
                'end_time': float(end),
                'duration': float(end - start),
                'start_frame': int(start * 30),
                'end_frame': int(end * 30)
            }
            beatloom_data['lyrics']['words'].append(word_data)
        
        return beatloom_data
    
    def generate_lrc_format(self, alignment: AlignmentResult) -> str:
        """
        Generate LRC (Lyric) format.
        
        Args:
            alignment: Alignment result
            
        Returns:
            LRC format string
        """
        lrc_lines = []
        
        # Add header
        lrc_lines.append("[ar:Beatloom Sync]")
        lrc_lines.append("[ti:Generated Lyrics]")
        lrc_lines.append(f"[length:{alignment.duration:.2f}]")
        lrc_lines.append("")
        
        # Add timed lyrics
        for word, start, _ in alignment.words:
            minutes = int(start // 60)
            seconds = start % 60
            lrc_lines.append(f"[{minutes:02d}:{seconds:05.2f}]{word}")
        
        return '\n'.join(lrc_lines)
    
    def generate_srt_format(self, alignment: AlignmentResult, words_per_subtitle: int = 5) -> str:
        """
        Generate SRT (SubRip) subtitle format.
        
        Args:
            alignment: Alignment result
            words_per_subtitle: Number of words per subtitle line
            
        Returns:
            SRT format string
        """
        srt_lines = []
        subtitle_index = 1
        
        # Group words into subtitles
        for i in range(0, len(alignment.words), words_per_subtitle):
            word_group = alignment.words[i:i + words_per_subtitle]
            
            if not word_group:
                continue
            
            # Get timing for this group
            start_time = word_group[0][1]  # Start of first word
            end_time = word_group[-1][2]   # End of last word
            
            # Format times for SRT
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            # Create subtitle text
            subtitle_text = ' '.join(word[0] for word in word_group)
            
            # Add to SRT
            srt_lines.append(str(subtitle_index))
            srt_lines.append(f"{start_srt} --> {end_srt}")
            srt_lines.append(subtitle_text)
            srt_lines.append("")
            
            subtitle_index += 1
        
        return '\n'.join(srt_lines)
    
    def generate_json_format(self, 
                           alignment: AlignmentResult,
                           confidence_result: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive JSON format.
        
        Args:
            alignment: Alignment result
            confidence_result: Confidence scoring result
            metadata: Additional metadata
            
        Returns:
            Comprehensive JSON data
        """
        json_data = {
            'format': 'beatloom_sync_v1',
            'generated_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'alignment': {
                'words': [
                    {
                        'text': word,
                        'start_time': float(start),
                        'end_time': float(end),
                        'duration': float(end - start),
                        'index': i
                    }
                    for i, (word, start, end) in enumerate(alignment.words)
                ],
                'total_duration': float(alignment.duration),
                'word_count': len(alignment.words)
            },
            'confidence': confidence_result,
            'statistics': {
                'avg_word_duration': sum(end - start for _, start, end in alignment.words) / max(len(alignment.words), 1),
                'words_per_second': len(alignment.words) / max(alignment.duration, 1),
                'total_words': len(alignment.words),
                'total_duration': float(alignment.duration)
            }
        }
        
        return json_data
    
    def generate_textgrid_format(self, alignment: AlignmentResult) -> str:
        """
        Generate Praat TextGrid format.
        
        Args:
            alignment: Alignment result
            
        Returns:
            TextGrid format string
        """
        textgrid_lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            '',
            'xmin = 0',
            f'xmax = {alignment.duration}',
            'tiers? <exists>',
            'size = 1',
            'item []:',
            '    item [1]:',
            '        class = "IntervalTier"',
            '        name = "words"',
            '        xmin = 0',
            f'        xmax = {alignment.duration}',
            f'        intervals: size = {len(alignment.words)}'
        ]
        
        for i, (word, start, end) in enumerate(alignment.words, 1):
            textgrid_lines.extend([
                f'        intervals [{i}]:',
                f'            xmin = {start}',
                f'            xmax = {end}',
                f'            text = "{word}"'
            ])
        
        return '\n'.join(textgrid_lines)
    
    def save_all_formats(self, 
                        alignment: AlignmentResult,
                        confidence_result: Dict[str, Any],
                        output_dir: str,
                        base_filename: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Save alignment results in all supported formats.
        
        Args:
            alignment: Alignment result
            confidence_result: Confidence scoring result
            output_dir: Output directory
            base_filename: Base filename (without extension)
            metadata: Additional metadata
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        try:
            # Beatloom JSON format
            beatloom_data = self.generate_beatloom_format(alignment, confidence_result, metadata)
            beatloom_path = os.path.join(output_dir, f"{base_filename}_beatloom.json")
            with open(beatloom_path, 'w', encoding='utf-8') as f:
                json.dump(beatloom_data, f, indent=2, ensure_ascii=False)
            saved_files['beatloom'] = beatloom_path
            
            # LRC format
            lrc_content = self.generate_lrc_format(alignment)
            lrc_path = os.path.join(output_dir, f"{base_filename}.lrc")
            with open(lrc_path, 'w', encoding='utf-8') as f:
                f.write(lrc_content)
            saved_files['lrc'] = lrc_path
            
            # SRT format
            srt_content = self.generate_srt_format(alignment)
            srt_path = os.path.join(output_dir, f"{base_filename}.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            saved_files['srt'] = srt_path
            
            # Comprehensive JSON
            json_data = self.generate_json_format(alignment, confidence_result, metadata)
            json_path = os.path.join(output_dir, f"{base_filename}_full.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            saved_files['json'] = json_path
            
            # TextGrid format
            textgrid_content = self.generate_textgrid_format(alignment)
            textgrid_path = os.path.join(output_dir, f"{base_filename}.TextGrid")
            with open(textgrid_path, 'w', encoding='utf-8') as f:
                f.write(textgrid_content)
            saved_files['textgrid'] = textgrid_path
            
            logger.info(f"Saved alignment results in {len(saved_files)} formats to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save output formats: {e}")
            raise RuntimeError(f"Output generation failed: {e}")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def create_summary_report(self, 
                            alignment: AlignmentResult,
                            confidence_result: Dict[str, Any],
                            processing_info: Dict[str, Any]) -> str:
        """
        Create a human-readable summary report.
        
        Args:
            alignment: Alignment result
            confidence_result: Confidence scoring result
            processing_info: Information about the processing pipeline
            
        Returns:
            Summary report as string
        """
        report_lines = [
            "Beatloom Lyric Synchronization Report",
            "=" * 40,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "ALIGNMENT SUMMARY:",
            f"  Total words aligned: {len(alignment.words)}",
            f"  Total duration: {alignment.duration:.2f} seconds",
            f"  Average word duration: {sum(end - start for _, start, end in alignment.words) / max(len(alignment.words), 1):.2f} seconds",
            f"  Words per second: {len(alignment.words) / max(alignment.duration, 1):.2f}",
            "",
            "CONFIDENCE SCORES:",
            f"  Overall confidence: {confidence_result.get('overall_confidence', 0):.3f}",
            f"  Duration score: {confidence_result.get('scores', {}).get('duration', 0):.3f}",
            f"  Coverage score: {confidence_result.get('scores', {}).get('coverage', 0):.3f}",
            f"  Timing score: {confidence_result.get('scores', {}).get('timing', 0):.3f}",
            f"  Audio score: {confidence_result.get('scores', {}).get('audio', 0):.3f}",
            "",
            "QUALITY FLAGS:",
        ]
        
        quality_flags = confidence_result.get('quality_flags', [])
        if quality_flags:
            for flag in quality_flags:
                report_lines.append(f"  - {flag.replace('_', ' ').title()}")
        else:
            report_lines.append("  No quality issues detected")
        
        report_lines.extend([
            "",
            "PROCESSING INFO:",
            f"  Vocal separation: {processing_info.get('vocal_separation', 'Unknown')}",
            f"  Alignment method: {processing_info.get('alignment_method', 'Unknown')}",
            f"  Processing time: {processing_info.get('processing_time', 'Unknown')}",
            "",
            "WORD TIMINGS:",
        ])
        
        # Add first few word timings as examples
        for i, (word, start, end) in enumerate(alignment.words[:10]):
            report_lines.append(f"  {start:6.2f}s - {end:6.2f}s: {word}")
        
        if len(alignment.words) > 10:
            report_lines.append(f"  ... and {len(alignment.words) - 10} more words")
        
        return '\n'.join(report_lines)

