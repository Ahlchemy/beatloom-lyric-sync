"""
Beatloom Video Generator with Mandala Visualizer and Synchronized Lyrics
Combines audio-reactive mandala visualization with synchronized lyric overlay.
"""

import os
import sys
import json
import shutil
import subprocess
import pathlib
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb
from matplotlib.text import Text

try:
    import librosa
    import soundfile as sf  # noqa: F401 (librosa backend)
except ImportError as e:
    raise SystemExit("Missing dependency: please `python -m pip install librosa soundfile` in your venv") from e

# ---------- config ----------
# Portrait 9:16 for phones
W, H = 1080, 1920
FPS = 30
OUT_DIR = "video_output"

# Visual style
BG_COLOR    = "#0b1c1f"
RING_COLOR  = "#66c3b0"
SPOKE_COLOR = "#3c7a6f"
PARTICLE_COLOR = "#9be7d8"
LYRIC_COLOR = "#ffffff"
LYRIC_HIGHLIGHT_COLOR = "#ffff66"

# Color cycling based on spectral centroid
COLOR_CYCLE_ENABLE = True
COLOR_HUES = [0.33, 0.36, 0.47, 0.5, 0.53, 0.6, 0.63, 0.67]
HUE_BASE = random.choice(COLOR_HUES)
HUE_RANGE = 0.12
SAT = 0.65
VAL = 0.95

# Mandala parameters
RINGS = 6
SPOKES = random.randint(8, 16)
POINTS_PER_RING = 256
LINEWIDTH = 2.0

# Sensitivity / mapping
RMS_GAIN = 0.8
MEL_GAIN = 0.9
BEAT_JOLT = 0.25
ANG_MOD_N = 8
ANG_MOD_GAIN = 0.08
ROTATION_FACTOR = 0.35

# Particles config
PARTICLES_ENABLE = True
MAX_PARTICLES = 2000
EMIT_BASE = 2
EMIT_RMS = 20
EMIT_BEAT = 60
EMIT_BEAT_STRONG = 160
PARTICLE_SPEED_MIN = 120
PARTICLE_SPEED_MAX = 520
PARTICLE_LIFETIME = 1.6
PARTICLE_SIZE = 18

# Center particles config (replacing spokes)
CENTER_PARTICLES_ENABLE = True
MAX_CENTER_PARTICLES = 800
CENTER_EMIT_BASE = 3
CENTER_EMIT_RMS = 15
CENTER_EMIT_BEAT = 40
CENTER_PARTICLE_SPEED_MIN = 60
CENTER_PARTICLE_SPEED_MAX = 180
CENTER_PARTICLE_LIFETIME = 2.5
CENTER_PARTICLE_SIZE = 12
CENTER_FLOW_DISTANCE = 0.33  # Flow 1/3 of the way across screen

# Spotlight config (for vocals) - DISABLED
SPOTLIGHT_ENABLE = False
SPOTLIGHT_COUNT = 2  # Number of spotlights (1-2 large ones as requested)
SPOTLIGHT_WIDTH = 25.0  # Much thicker than regular lines
SPOTLIGHT_VOCAL_GAIN = 1.5  # Higher sensitivity to vocals
SPOTLIGHT_BEAT_RESPONSE = 0.6
SPOTLIGHT_RANDOM_CHANGE = 0.3  # How often spotlights change direction (seconds)
SPOTLIGHT_ALPHA = 0.8  # Transparency for dramatic effect

# Smoke config
SMOKE_ENABLE = True
SMOKE_RES = (270, 480)
SMOKE_SPEED = 0.25
SMOKE_ALPHA_BASE = 0.10
SMOKE_ALPHA_GAIN = 0.20

# Oscilloscope (outer ring waveform)
OSC_ENABLE = True
OSC_POINTS = 512
OSC_GAIN = 0.25
OSC_WIDTH = 2.0

# Lyric display config
LYRIC_FONT_SIZE = 48
LYRIC_FONT_SIZE_CURRENT = 56
LYRIC_Y_POSITION = H * 0.15  # Bottom 15% of screen
LYRIC_LINE_SPACING = 80
LYRIC_MAX_LINES = 3
LYRIC_FADE_DURATION = 0.3  # seconds for fade in/out
LYRIC_HIGHLIGHT_DURATION = 0.1  # seconds for highlight effect

# Center lyric display config (for key refrain words)
CENTER_LYRIC_ENABLE = True
CENTER_LYRIC_FONT_SIZE = 80  # Larger for center display
CENTER_LYRIC_MAX_SIZE = 140  # Maximum size during effects
CENTER_LYRIC_COLOR = "#ffffff"
CENTER_LYRIC_HIGHLIGHT_COLOR = "#ffff00"
CENTER_LYRIC_EFFECT_DURATION = 2.0  # seconds for full effect cycle
CENTER_LYRIC_FADE_IN = 0.4  # seconds to fade in
CENTER_LYRIC_FADE_OUT = 0.6  # seconds to fade out
CENTER_LYRIC_WARP_INTENSITY = 0.4  # How much to warp with beat
CENTER_LYRIC_PULSE_SPEED = 6  # Speed of pulsing effect

class LyricRenderer:
    """Handles synchronized lyric display with timing and effects."""
    
    def __init__(self, sync_data: Dict, ax):
        self.sync_data = sync_data
        self.ax = ax
        self.current_word_idx = 0
        self.lyric_texts = []
        self.setup_lyrics()
    
    def setup_lyrics(self):
        """Initialize lyric text objects."""
        for i in range(LYRIC_MAX_LINES):
            text = Text(W/2, LYRIC_Y_POSITION + i * LYRIC_LINE_SPACING, 
                       "", fontsize=LYRIC_FONT_SIZE, color=LYRIC_COLOR,
                       ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            self.ax.add_artist(text)
            self.lyric_texts.append(text)
    
    def get_current_lyrics(self, time_sec: float) -> Tuple[List[str], int]:
        """Get current lyrics to display and highlight index."""
        if not self.sync_data.get('words'):
            return [], -1
        
        words = self.sync_data['words']
        current_word = -1
        
        # Find current word
        for i, word_data in enumerate(words):
            if word_data['start'] <= time_sec <= word_data['end']:
                current_word = i
                break
        
        # Get context words (previous and next)
        context_range = 8  # words before and after
        start_idx = max(0, current_word - context_range)
        end_idx = min(len(words), current_word + context_range + 1)
        
        # Build lines of text
        lines = []
        current_line = []
        current_highlight = -1
        word_count = 0
        
        for i in range(start_idx, end_idx):
            word_text = words[i]['word']
            current_line.append(word_text)
            
            if i == current_word:
                current_highlight = word_count
            
            word_count += 1
            
            # Break into lines (roughly 6-8 words per line)
            if len(current_line) >= 7 or i == end_idx - 1:
                lines.append(' '.join(current_line))
                if i != current_word:
                    word_count = 0  # Reset if current word not in this line
                current_line = []
        
        return lines[:LYRIC_MAX_LINES], current_highlight
    
    def update(self, time_sec: float, beat_intensity: float):
        """Update lyric display for current time."""
        lines, highlight_idx = self.get_current_lyrics(time_sec)
        
        # Update text objects
        for i, text_obj in enumerate(self.lyric_texts):
            if i < len(lines):
                text_obj.set_text(lines[i])
                
                # Highlight current line with beat intensity
                if highlight_idx >= 0 and i == 0:  # Assume current word is in first line
                    alpha = 0.8 + 0.2 * beat_intensity
                    color = LYRIC_HIGHLIGHT_COLOR if beat_intensity > 0.5 else LYRIC_COLOR
                    size = LYRIC_FONT_SIZE_CURRENT + int(10 * beat_intensity)
                else:
                    alpha = 0.7
                    color = LYRIC_COLOR
                    size = LYRIC_FONT_SIZE
                
                text_obj.set_color(color)
                text_obj.set_fontsize(size)
                text_obj.set_alpha(alpha)
                text_obj.set_visible(True)
            else:
                text_obj.set_visible(False)


class CenterLyricRenderer:
    """Handles center lyric display for key refrain words with dynamic effects."""
    
    def __init__(self, sync_data: Dict, ax, cx: float, cy: float):
        self.sync_data = sync_data
        self.ax = ax
        self.cx = cx
        self.cy = cy
        self.center_text = None
        self.current_word = ""
        self.effect_start_time = 0
        self.word_start_time = 0
        self.setup_center_text()
        
        # Define key refrain words to highlight in center
        self.key_words = {
            "anyway", "despite", "warning", "together", "harmony", 
            "did", "it", "music", "lyrics", "synchronized", "test",
            "timing", "working", "sample", "song"
        }
    
    def setup_center_text(self):
        """Initialize center text object."""
        if CENTER_LYRIC_ENABLE:
            self.center_text = Text(self.cx, self.cy, "", 
                                  fontsize=CENTER_LYRIC_FONT_SIZE, 
                                  color=CENTER_LYRIC_COLOR,
                                  ha='center', va='center', 
                                  weight='bold', zorder=15,
                                  bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor='black', alpha=0.7,
                                          edgecolor='white', linewidth=2))
            self.ax.add_artist(self.center_text)
    
    def get_current_center_word(self, time_sec: float) -> str:
        """Get current word to display in center if it's a key word."""
        if not self.sync_data.get('words'):
            return ""
        
        words = self.sync_data['words']
        
        # Find current word
        for word_data in words:
            if word_data['start'] <= time_sec <= word_data['end']:
                word_text = word_data['word'].lower().strip('.,!?')
                # Check if it's a key refrain word
                if word_text in self.key_words:
                    return word_text.upper()
        
        return ""
    
    def update(self, time_sec: float, beat_intensity: float, vocal_intensity: float):
        """Update center lyric display with dynamic effects."""
        if not CENTER_LYRIC_ENABLE or not self.center_text:
            return
        
        current_word = self.get_current_center_word(time_sec)
        
        # Check if we have a new word
        if current_word != self.current_word:
            self.current_word = current_word
            if current_word:
                self.effect_start_time = time_sec
                self.word_start_time = time_sec
        
        if self.current_word:
            # Calculate effect progress
            effect_time = time_sec - self.effect_start_time
            
            if effect_time <= CENTER_LYRIC_EFFECT_DURATION:
                # Word is active, apply effects
                self.center_text.set_text(self.current_word)
                
                # Fade in/out effect
                if effect_time <= CENTER_LYRIC_FADE_IN:
                    alpha = effect_time / CENTER_LYRIC_FADE_IN
                elif effect_time >= CENTER_LYRIC_EFFECT_DURATION - CENTER_LYRIC_FADE_OUT:
                    fade_progress = (CENTER_LYRIC_EFFECT_DURATION - effect_time) / CENTER_LYRIC_FADE_OUT
                    alpha = max(0, fade_progress)
                else:
                    alpha = 1.0
                
                # Dynamic size effect with beat and vocal intensity
                base_size = CENTER_LYRIC_FONT_SIZE
                beat_boost = CENTER_LYRIC_WARP_INTENSITY * beat_intensity * 40
                vocal_boost = vocal_intensity * 25
                pulse = 15 * np.sin(effect_time * CENTER_LYRIC_PULSE_SPEED)  # Pulsing effect
                
                size = min(CENTER_LYRIC_MAX_SIZE, base_size + beat_boost + vocal_boost + pulse)
                
                # Color effect based on intensity
                if beat_intensity > 0.6 or vocal_intensity > 0.7:
                    color = CENTER_LYRIC_HIGHLIGHT_COLOR
                else:
                    # Interpolate between white and highlight color
                    intensity = max(beat_intensity, vocal_intensity)
                    color = CENTER_LYRIC_COLOR if intensity < 0.3 else CENTER_LYRIC_HIGHLIGHT_COLOR
                
                # Apply effects
                self.center_text.set_fontsize(size)
                self.center_text.set_color(color)
                self.center_text.set_alpha(alpha * (0.8 + 0.2 * vocal_intensity))
                self.center_text.set_visible(True)
                
                # Warping effect - position offset with beat and vocal intensity
                warp_multiplier = CENTER_LYRIC_WARP_INTENSITY * (beat_intensity + vocal_intensity * 0.5)
                warp_x = self.cx + warp_multiplier * 25 * np.sin(time_sec * 4 + effect_time * 2)
                warp_y = self.cy + warp_multiplier * 20 * np.cos(time_sec * 3 + effect_time * 1.5)
                self.center_text.set_position((warp_x, warp_y))
                
                # Update bbox style based on intensity
                if beat_intensity > 0.7:
                    bbox_props = dict(boxstyle="round,pad=0.4", 
                                    facecolor='yellow', alpha=0.3,
                                    edgecolor='white', linewidth=3)
                else:
                    bbox_props = dict(boxstyle="round,pad=0.3", 
                                    facecolor='black', alpha=0.7,
                                    edgecolor='white', linewidth=2)
                self.center_text.set_bbox(bbox_props)
            else:
                # Effect finished, hide text
                self.center_text.set_visible(False)
                self.current_word = ""
        else:
            # No key word, hide text
            self.center_text.set_visible(False)


class BeatloomVideoGenerator:
    """Main video generator class combining mandala visuals with synchronized lyrics."""
    
    def __init__(self, audio_path: str, sync_json_path: str, output_dir: str = "video_output"):
        self.audio_path = audio_path
        self.sync_json_path = sync_json_path
        self.output_dir = output_dir
        
        # Load synchronization data
        with open(sync_json_path, 'r') as f:
            self.sync_data = json.load(f)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check ffmpeg
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg not found – install it: brew install ffmpeg")
    
    def analyze_audio(self, sr_target: int = 22050, hop_length: int = 512):
        """Analyze audio for visualization features."""
        y, sr = librosa.load(self.audio_path, sr=sr_target, mono=True)
        duration = len(y) / sr
        
        # Frame-based features
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=RINGS,
                                             hop_length=hop_length, fmax=sr/2)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                                    hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        frame_times = np.arange(mel_db.shape[1]) * (hop_length / sr)
        rms_times = np.arange(rms.shape[0]) * (hop_length / sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Separate vocal and instrumental frequency ranges
        # Vocals typically 80Hz-1100Hz, instruments have broader range
        vocal_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=8,
                                                  hop_length=hop_length, 
                                                  fmin=80, fmax=1100)
        vocal_db = librosa.power_to_db(vocal_mel, ref=np.max)
        vocal_rms = np.mean(vocal_db, axis=0)  # Average across vocal frequency bands
        
        # Instrumental focus on mid-low frequencies (excludes vocal range)
        inst_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=RINGS,
                                                 hop_length=hop_length,
                                                 fmin=1100, fmax=sr/2)
        inst_db = librosa.power_to_db(inst_mel, ref=np.max)

        return {
            "y": y, "sr": sr, "duration": duration, "tempo": tempo,
            "mel_db": mel_db, "frame_times": frame_times,
            "rms": rms, "rms_times": rms_times, "beat_times": beat_times,
            "centroid": centroid,
            "vocal_db": vocal_db, "vocal_rms": vocal_rms,
            "inst_db": inst_db,
        }
    
    def interp_features_to_frames(self, features, times_src, total_frames):
        """Interpolate audio features to video frame rate."""
        t_video = np.arange(total_frames) / FPS
        if features.ndim == 1:
            return np.interp(t_video, times_src, features)
        else:
            bands = features.shape[0]
            out = np.zeros((bands, total_frames), dtype=float)
            for b in range(bands):
                out[b] = np.interp(t_video, times_src, features[b])
            return out
    
    def gaussian_smooth(self, x, sigma_frames=1.5):
        """Apply gaussian smoothing to features."""
        radius = int(3 * sigma_frames)
        if radius <= 0:
            return x
        idx = np.arange(-radius, radius + 1)
        kernel = np.exp(-(idx**2) / (2 * sigma_frames**2))
        kernel /= kernel.sum()
        if x.ndim == 1:
            return np.convolve(x, kernel, mode='same')
        else:
            return np.vstack([np.convolve(row, kernel, mode='same') for row in x])
    
    def _smoke_field(self, X_sm, Y_sm, t: float):
        """Generate animated smoke field."""
        s1 = np.sin(1.2*X_sm + 0.9*t) * np.cos(0.7*Y_sm - 0.6*t)
        s2 = np.sin(0.6*X_sm - 0.8*t + 0.5*Y_sm)
        s3 = np.cos(0.9*(X_sm*np.cos(0.3*t) + Y_sm*np.sin(0.2*t)) + 0.4*t)
        f = (s1 + 0.7*s2 + 0.5*s3)
        f = (f - f.min()) / (np.ptp(f) + 1e-8)
        return f
    
    def generate_video(self, preview_seconds: Optional[float] = None):
        """Generate the complete video with mandala visuals and synchronized lyrics."""
        print("Beatloom Video Generator – Analyzing audio...")
        
        # Analyze audio
        audio = self.analyze_audio()
        duration = audio["duration"]
        if preview_seconds is not None:
            duration = min(duration, float(preview_seconds))
        
        total_frames = int(np.ceil(duration * FPS))
        
        # Interpolate features to video frames
        mel_v = self.interp_features_to_frames(audio["mel_db"], audio["frame_times"], total_frames)
        rms_v = self.interp_features_to_frames(audio["rms"], audio["rms_times"], total_frames)
        centroid_v = self.interp_features_to_frames(audio["centroid"], audio["frame_times"], total_frames)
        
        # Interpolate vocal and instrumental features
        vocal_v = self.interp_features_to_frames(audio["vocal_rms"], audio["frame_times"], total_frames)
        inst_v = self.interp_features_to_frames(audio["inst_db"], audio["frame_times"], total_frames)
        
        # Normalize and smooth
        mel_v = (mel_v - np.min(mel_v, axis=1, keepdims=True)) / (
            np.ptp(mel_v, axis=1, keepdims=True) + 1e-8)
        mel_v = self.gaussian_smooth(mel_v, sigma_frames=2)
        
        rms_v = (rms_v - np.min(rms_v)) / (np.ptp(rms_v) + 1e-8)
        rms_v = self.gaussian_smooth(rms_v, sigma_frames=2)
        
        centroid_v = (centroid_v - np.min(centroid_v)) / (np.ptp(centroid_v) + 1e-8)
        centroid_v = self.gaussian_smooth(centroid_v, sigma_frames=2)
        
        # Normalize vocal and instrumental features
        vocal_v = (vocal_v - np.min(vocal_v)) / (np.ptp(vocal_v) + 1e-8)
        vocal_v = self.gaussian_smooth(vocal_v, sigma_frames=2)
        
        if inst_v.ndim == 1:
            inst_v = (inst_v - np.min(inst_v)) / (np.ptp(inst_v) + 1e-8)
        else:
            inst_v = (inst_v - np.min(inst_v, axis=1, keepdims=True)) / (
                np.ptp(inst_v, axis=1, keepdims=True) + 1e-8)
        inst_v = self.gaussian_smooth(inst_v, sigma_frames=2)
        
        # Beat envelopes
        beat_env = np.zeros(total_frames, dtype=float)
        for bt in audio["beat_times"]:
            idx = int(round(bt * FPS))
            if 0 <= idx < total_frames:
                beat_env[max(0, idx-1):min(total_frames, idx+2)] = 1.0
        beat_env = self.gaussian_smooth(beat_env, sigma_frames=1)
        
        print(f"Setting up visualization ({total_frames} frames, ~{duration:.2f}s)...")
        
        # Setup matplotlib figure
        fig = plt.figure(figsize=(W/100, H/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal', adjustable='box')
        ax.axis("off")
        fig.patch.set_facecolor(BG_COLOR)
        
        cx, cy = W/2, H/2
        max_r = min(W, H) * 0.46
        min_r = min(W, H) * 0.10
        ring_base = np.linspace(min_r, max_r, RINGS)
        theta = np.linspace(0, 2*np.pi, POINTS_PER_RING, endpoint=False)
        
        # Initialize lyric renderers
        lyric_renderer = LyricRenderer(self.sync_data, ax)
        center_lyric_renderer = CenterLyricRenderer(self.sync_data, ax, cx, cy)
        
        # Smoke background
        smoke_img = None
        X_sm, Y_sm = None, None
        if SMOKE_ENABLE:
            sh, sw = SMOKE_RES
            x_sm = np.linspace(-np.pi, np.pi, sw)
            y_sm = np.linspace(-np.pi, np.pi, sh)
            X_sm, Y_sm = np.meshgrid(x_sm, y_sm)
            smoke_img = ax.imshow(np.zeros((sh, sw)), extent=[0, W, 0, H], origin='lower',
                                  cmap='gray', interpolation='bilinear', alpha=SMOKE_ALPHA_BASE,
                                  zorder=0)
        
        # Rings
        ring_artists = []
        for _ in range(RINGS):
            line = Line2D([], [], color=RING_COLOR, linewidth=LINEWIDTH, antialiased=True, zorder=2)
            ax.add_line(line)
            ring_artists.append(line)
        
        # Center particles (replacing spokes)
        center_particle_scatter = None
        cp_x, cp_y, cp_vx, cp_vy, cp_life, cp_life_max = [], [], [], [], [], []
        if CENTER_PARTICLES_ENABLE:
            center_particle_scatter = ax.scatter([], [], s=[], c=[], marker='o', linewidths=0, zorder=2)
        
        # Spotlights (for vocals) - large stage lighting beams
        spotlight_artists = []
        spotlight_angles = []
        spotlight_last_change = []
        if SPOTLIGHT_ENABLE:
            # Initialize with random directions
            for i in range(SPOTLIGHT_COUNT):
                # Random initial angle
                ang = np.random.uniform(0, 2*np.pi)
                spotlight_angles.append(ang)
                spotlight_last_change.append(0.0)
                
                # Create thick line for spotlight beam
                line = Line2D([], [], color=SPOKE_COLOR, linewidth=SPOTLIGHT_WIDTH, 
                             alpha=SPOTLIGHT_ALPHA, zorder=3, solid_capstyle='round')
                ax.add_line(line)
                spotlight_artists.append(line)
        
        # Oscilloscope
        osc_line = None
        theta_osc = None
        if OSC_ENABLE:
            theta_osc = np.linspace(0, 2*np.pi, OSC_POINTS, endpoint=False)
            osc_line = Line2D([], [], color=PARTICLE_COLOR, linewidth=OSC_WIDTH, zorder=4)
            ax.add_line(osc_line)
        
        # Particles
        particle_scatter = None
        p_x, p_y, p_vx, p_vy, p_life, p_life_max = [], [], [], [], [], []
        if PARTICLES_ENABLE:
            particle_scatter = ax.scatter([], [], s=[], c=[], marker='o', linewidths=0, zorder=3)
        
        def update(frame):
            """Update function for animation."""
            time_sec = frame / FPS
            rms = rms_v[frame]
            beat = beat_env[frame]
            tempo = audio["tempo"]
            rot = ROTATION_FACTOR * 2*np.pi * (tempo/60.0) * time_sec
            
            artists = []
            
            # Update lyrics
            lyric_renderer.update(time_sec, beat)
            
            # Update center lyrics with vocal intensity
            vocal_intensity = vocal_v[frame] if frame < len(vocal_v) else 0
            center_lyric_renderer.update(time_sec, beat, vocal_intensity)
            
            # Smoke
            if SMOKE_ENABLE and smoke_img is not None:
                f = self._smoke_field(X_sm, Y_sm, time_sec * SMOKE_SPEED)
                smoke_img.set_data(f)
                energy = 0.5*float(np.mean(mel_v[:, frame])) + 0.5*rms
                smoke_img.set_alpha(SMOKE_ALPHA_BASE + SMOKE_ALPHA_GAIN * energy)
                artists.append(smoke_img)
            
            # Color cycling
            hue = (HUE_BASE + HUE_RANGE * (2*centroid_v[frame]-1)) % 1.0 if COLOR_CYCLE_ENABLE else HUE_BASE
            rgb = hsv_to_rgb([hue, SAT, VAL])
            ring_color = tuple(rgb)
            
            # Complementary colors for center particles
            center_hue = (hue + 0.5) % 1.0  # Opposite on color wheel
            center_color = hsv_to_rgb([center_hue, SAT * 0.8, VAL * 0.9])
            
            # Update ring colors (for instruments)
            for line in ring_artists:
                line.set_color(ring_color)
            
            # Rings (respond to instruments)
            for i in range(RINGS):
                base = ring_base[i]
                # Use instrumental data for rings
                if inst_v.ndim > 1 and i < inst_v.shape[0]:
                    band = inst_v[i, frame]
                else:
                    band = mel_v[i, frame] if i < mel_v.shape[0] else 0
                radius = base * (1 + MEL_GAIN*band + RMS_GAIN*rms + BEAT_JOLT*beat)
                ang_mod = 1 + ANG_MOD_GAIN * np.sin(ANG_MOD_N*theta + rot + i*0.6)
                rr = radius * ang_mod
                xs = cx + rr * np.cos(theta)
                ys = cy + rr * np.sin(theta)
                ring_artists[i].set_data(xs, ys)
            
            # Center particles (replacing spokes)
            if CENTER_PARTICLES_ENABLE and center_particle_scatter is not None:
                # Emit new center particles
                center_emits = int(CENTER_EMIT_BASE + CENTER_EMIT_RMS * rms + CENTER_EMIT_BEAT * beat)
                if center_emits > 0:
                    center_emits = min(center_emits, MAX_CENTER_PARTICLES - len(cp_x))
                    if center_emits > 0:
                        angles = np.random.uniform(0, 2*np.pi, size=center_emits)
                        speeds = np.random.uniform(CENTER_PARTICLE_SPEED_MIN, CENTER_PARTICLE_SPEED_MAX, size=center_emits) / FPS
                        # Add pulsation based on beat
                        speed_multiplier = 1.0 + 0.5 * beat
                        for a, spd in zip(angles, speeds):
                            cp_x.append(cx)
                            cp_y.append(cy)
                            cp_vx.append(spd * speed_multiplier * np.cos(a))
                            cp_vy.append(spd * speed_multiplier * np.sin(a))
                            life = CENTER_PARTICLE_LIFETIME * (0.8 + 0.4*np.random.rand())
                            cp_life.append(life)
                            cp_life_max.append(life)
                
                # Update existing center particles
                max_distance = min(W, H) * CENTER_FLOW_DISTANCE
                i = 0
                while i < len(cp_x):
                    cp_x[i] += cp_vx[i]
                    cp_y[i] += cp_vy[i]
                    cp_life[i] -= 1.0 / FPS
                    
                    # Remove particles that are too far from center or expired
                    distance_from_center = np.sqrt((cp_x[i] - cx)**2 + (cp_y[i] - cy)**2)
                    if cp_life[i] <= 0 or distance_from_center > max_distance:
                        for arr in (cp_x, cp_y, cp_vx, cp_vy, cp_life, cp_life_max):
                            arr.pop(i)
                        continue
                    i += 1
                
                # Render center particles
                if len(cp_x) > 0:
                    offs = np.column_stack([cp_x, cp_y])
                    life_frac = np.array(cp_life) / np.array(cp_life_max)
                    # Pulsating size based on beat and life
                    pulse = 1.0 + 0.3 * beat * np.sin(time_sec * 8)  # Fast pulsation
                    sizes = CENTER_PARTICLE_SIZE * pulse * (0.3 + 1.7*np.sin(np.clip(1-life_frac, 0, 1)*np.pi))
                    alphas = np.clip(life_frac, 0, 1) * (0.7 + 0.3*rms)
                    
                    # Use complementary color
                    colors = np.column_stack([
                        np.full_like(alphas, center_color[0]),
                        np.full_like(alphas, center_color[1]),
                        np.full_like(alphas, center_color[2]),
                        alphas,
                    ])
                    center_particle_scatter.set_offsets(offs)
                    center_particle_scatter.set_sizes(sizes)
                    center_particle_scatter.set_facecolors(colors)
                else:
                    center_particle_scatter.set_offsets(np.empty((0, 2)))
                    center_particle_scatter.set_sizes([])
                    center_particle_scatter.set_facecolors([])
                artists.append(center_particle_scatter)
            
            # Spotlights (respond to vocals) - large stage lighting beams
            if SPOTLIGHT_ENABLE and spotlight_artists:
                vocal_intensity = vocal_v[frame] if frame < len(vocal_v) else 0
                spotlight_len = (min_r*0.2) + (max_r*0.9) * (SPOTLIGHT_VOCAL_GAIN*vocal_intensity + SPOTLIGHT_BEAT_RESPONSE*beat)
                
                # Update spotlight directions randomly based on time
                for s in range(len(spotlight_artists)):
                    # Check if it's time to change direction
                    if time_sec - spotlight_last_change[s] > SPOTLIGHT_RANDOM_CHANGE + s*0.1:
                        # Random new direction with some influence from beat
                        if beat > 0.5:  # On strong beats, bigger direction changes
                            spotlight_angles[s] = np.random.uniform(0, 2*np.pi)
                        else:  # Subtle changes otherwise
                            spotlight_angles[s] += np.random.uniform(-0.5, 0.5)
                        spotlight_last_change[s] = time_sec
                    
                    # Add slight continuous rotation and beat response
                    ang_cur = spotlight_angles[s] + rot*0.1 + beat*0.2*np.sin(time_sec*2 + s)
                    
                    # Spotlight length varies with vocals and beat
                    beam_length = spotlight_len * (0.7 + 0.3*np.sin(time_sec*1.5 + s*1.2))
                    
                    # Create spotlight beam from center outward
                    x0, y0 = float(cx), float(cy)
                    x1 = float(cx) + float(beam_length) * float(np.cos(ang_cur))
                    y1 = float(cy) + float(beam_length) * float(np.sin(ang_cur))
                    spotlight_artists[s].set_data([x0, x1], [y0, y1])
                    
                    # Vary spotlight intensity with vocals and beat
                    intensity = SPOTLIGHT_ALPHA * (0.5 + 0.5*vocal_intensity + 0.3*beat)
                    spotlight_artists[s].set_alpha(intensity)
            
            artists += ring_artists
            
            # Oscilloscope
            if OSC_ENABLE and osc_line is not None:
                y = audio["y"]
                sr = audio["sr"]
                n = int(time_sec * sr)
                seg = y[n:n+2048]
                if seg.size < 2048:
                    seg = np.pad(seg, (0, 2048 - seg.size))
                xsamp = np.linspace(0, seg.size-1, OSC_POINTS)
                wav = np.interp(xsamp, np.arange(seg.size), seg)
                if np.max(np.abs(wav)) > 1e-8:
                    wav = wav / np.max(np.abs(wav))
                outer_base = max_r * 0.96
                rr = outer_base * (1 + OSC_GAIN * 0.5 * (wav * (0.6 + 0.8*rms)))
                xo = cx + rr * np.cos(theta_osc)
                yo = cy + rr * np.sin(theta_osc)
                osc_line.set_data(xo, yo)
                artists.append(osc_line)
            
            # Particles
            if PARTICLES_ENABLE and particle_scatter is not None:
                # Emit new particles
                emits = int(EMIT_BASE + EMIT_RMS * rms + EMIT_BEAT * beat)
                if emits > 0:
                    emits = min(emits, MAX_PARTICLES - len(p_x))
                    if emits > 0:
                        angles = np.random.uniform(0, 2*np.pi, size=emits)
                        speeds = np.random.uniform(PARTICLE_SPEED_MIN, PARTICLE_SPEED_MAX, size=emits) / FPS
                        for a, spd in zip(angles, speeds):
                            p_x.append(cx)
                            p_y.append(cy)
                            p_vx.append(spd * np.cos(a))
                            p_vy.append(spd * np.sin(a))
                            life = PARTICLE_LIFETIME * (0.7 + 0.6*np.random.rand())
                            p_life.append(life)
                            p_life_max.append(life)
                
                # Update existing particles
                i = 0
                while i < len(p_x):
                    p_x[i] += p_vx[i]
                    p_y[i] += p_vy[i]
                    p_life[i] -= 1.0 / FPS
                    if p_life[i] <= 0 or not (-W*0.2 <= p_x[i] <= W*1.2 and -H*0.2 <= p_y[i] <= H*1.2):
                        for arr in (p_x, p_y, p_vx, p_vy, p_life, p_life_max):
                            arr.pop(i)
                        continue
                    i += 1
                
                # Render particles
                if len(p_x) > 0:
                    offs = np.column_stack([p_x, p_y])
                    life_frac = np.array(p_life) / np.array(p_life_max)
                    sizes = PARTICLE_SIZE * (0.5 + 1.5*np.sin(np.clip(1-life_frac, 0, 1)*np.pi))
                    alphas = np.clip(life_frac, 0, 1) * (0.6 + 0.4*rms)
                    colors = np.column_stack([
                        np.full_like(alphas, 0.6),
                        np.full_like(alphas, 0.9),
                        np.full_like(alphas, 0.85),
                        alphas,
                    ])
                    particle_scatter.set_offsets(offs)
                    particle_scatter.set_sizes(sizes)
                    particle_scatter.set_facecolors(colors)
                else:
                    particle_scatter.set_offsets(np.empty((0, 2)))
                    particle_scatter.set_sizes([])
                    particle_scatter.set_facecolors([])
                artists.append(particle_scatter)
            
            return artists
        
        # Generate video
        base_name = pathlib.Path(self.audio_path).stem
        suffix = "_preview" if preview_seconds else ""
        out_file = os.path.join(self.output_dir, f"{base_name}_beatloom{suffix}.mp4")
        out_file_muxed = os.path.join(self.output_dir, f"{base_name}_beatloom{suffix}_with_audio.mp4")
        
        print(f"Rendering video → {out_file}")
        anim = manim.FuncAnimation(fig, update, frames=total_frames, blit=False)
        
        writer = manim.FFMpegWriter(fps=FPS, codec="h264", bitrate=12000,
                                    extra_args=["-pix_fmt", "yuv420p"])
        anim.save(out_file, writer=writer)
        
        # Mux original audio
        try:
            cmd = [
                "ffmpeg", "-y", "-i", out_file, "-i", self.audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_file_muxed
            ]
            print("Muxing original audio into video...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Video complete → {pathlib.Path(out_file_muxed).absolute()}")
            return out_file_muxed
        except Exception as e:
            print(f"Audio mux failed: {e}")
            print(f"Manual command: ffmpeg -y -i '{out_file}' -i '{self.audio_path}' -c:v copy -c:a aac -b:a 192k -shortest '{out_file_muxed}'")
            return out_file


def main():
    """Command line interface for video generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Beatloom lyric video with mandala visualization")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("sync_json", help="Path to Beatloom synchronization JSON file")
    parser.add_argument("--output-dir", default="video_output", help="Output directory")
    parser.add_argument("--preview", type=float, help="Generate preview of N seconds")
    
    args = parser.parse_args()
    
    generator = BeatloomVideoGenerator(args.audio_file, args.sync_json, args.output_dir)
    output_file = generator.generate_video(preview_seconds=args.preview)
    
    print(f"\n🎬 Video generation complete!")
    print(f"📁 Output: {output_file}")


if __name__ == "__main__":
    main()

