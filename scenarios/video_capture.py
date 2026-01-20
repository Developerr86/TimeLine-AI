"""
Video Capture Handler for the TimeLine Content Capture Pipeline.
Handles video lecture recording with frame capture and audio recording.
"""

import os
import threading
import time
import wave
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import cv2
import numpy as np

# Monkeypatch numpy.fromstring to fix compability with soundcard < 0.4.2 and numpy >= 1.20
# soundcard uses numpy.fromstring for binary data, which was removed.
try:
    if not hasattr(np, '_is_patched_for_soundcard'):
        _original_fromstring = np.fromstring
        
        def _patched_fromstring(string, dtype=float, count=-1, sep=''):
            if sep == '':
                # Binary mode - use frombuffer
                return np.frombuffer(string, dtype=dtype, count=count)
            return _original_fromstring(string, dtype=dtype, count=count, sep=sep)
            
        np.fromstring = _patched_fromstring
        np._is_patched_for_soundcard = True
        print("🔧 Applied monkeypatch for numpy.fromstring (soundcard compatibility)")
except Exception as e:
    print(f"⚠️ Failed to patch numpy: {e}")
from PIL import ImageGrab

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Whisper transcription (load on demand to save RAM)
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# Audio recording via soundcard (system loopback)
try:
    import soundcard as sc
    HAS_SOUNDCARD = True
    # Suppress non-critical data discontinuity warnings (common during recording)
    import warnings
    warnings.filterwarnings('ignore', message='data discontinuity in recording')
except ImportError:
    HAS_SOUNDCARD = False

# Fallback: pyaudiowpatch for Windows loopback (check independently)
try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# GPU availability detection
def _detect_gpu() -> bool:
    """Check if a compatible GPU is available for inference."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        # Alternative: check if CUDA is available via ctypes
        import ctypes
        ctypes.CDLL("nvcuda.dll")
        return True
    except (OSError, FileNotFoundError):
        return False

HAS_GPU = _detect_gpu()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    get_db, CaptureSession, CapturedText, CapturedMedia,
    SessionType, MediaType, MEDIA_VIDEO_DIR
)


class StreamingTranscriber:
    """
    Real-time audio recording and transcription using soundcard + faster-whisper.
    Records system audio (loopback) and transcribes in chunks.
    """
    
    def __init__(self, 
                 chunk_duration: float = 5.0,
                 sample_rate: int = 16000,
                 model_size: str = "tiny"):
        """
        Initialize streaming transcriber.
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds (default 5)
            sample_rate: Audio sample rate (16000 for Whisper)
            model_size: Whisper model size (tiny, base, small, etc.)
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.model_size = model_size
        
        self._is_recording = False
        self._stop_event = threading.Event()
        self._record_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None
        
        # Audio buffer
        self._audio_chunks: List[np.ndarray] = []
        self._audio_lock = threading.Lock()
        
        # Transcript buffer
        self._transcript_parts: List[Dict[str, Any]] = []  # Now stores {time, text}
        self._transcript_lock = threading.Lock()
        self._full_transcript = ""
        
        # Stats
        self._chunks_processed = 0
        self._total_audio_duration = 0.0
        self._whisper_model = None
        
        # Model loading state (for UI)
        self._model_loading = False
        self._model_loaded = False
        self._device_type = "cpu"  # "cpu" or "cuda"
        
        # Language detection (auto-detect from first chunk)
        self._detected_language: Optional[str] = None
        self._language_detection_done = False
        
        # Output files
        self._audio_file_path: Optional[Path] = None
        self._transcript_file_path: Optional[Path] = None
        self._all_audio_data: List[np.ndarray] = []
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    @property
    def current_transcript(self) -> str:
        with self._transcript_lock:
            return self._full_transcript
    
    def start(self, output_dir: Path) -> bool:
        """
        Start recording and real-time transcription.
        
        Args:
            output_dir: Directory to save audio file
            
        Returns:
            True if started successfully
        """
        if self._is_recording:
            print("⚠️ Already recording")
            return False
        
        # Diagnostic: Print library availability
        print("=" * 60)
        print("🔍 AUDIO DIAGNOSTIC INFO")
        print("=" * 60)
        print(f"  soundcard library: {'✅ Available' if HAS_SOUNDCARD else '❌ Not installed'}")
        print(f"  pyaudiowpatch library: {'✅ Available' if HAS_PYAUDIO else '❌ Not installed'}")
        print(f"  faster-whisper: {'✅ Available' if HAS_WHISPER else '❌ Not installed'}")
        print(f"  GPU (CUDA): {'✅ Available' if HAS_GPU else '❌ Not available (using CPU)'}")
        
        if not HAS_SOUNDCARD and not HAS_PYAUDIO:
            print("❌ No audio recording library available")
            print("   Install soundcard: pip install soundcard")
            print("   Or pyaudiowpatch: pip install pyaudiowpatch")
            print("=" * 60)
            return False
        
        # Diagnostic: List available audio devices
        if HAS_SOUNDCARD:
            try:
                print("\n📢 Available Speakers (for loopback):")
                speakers = sc.all_speakers()
                for i, speaker in enumerate(speakers):
                    print(f"  [{i}] {speaker.name}")
                
                default_speaker = sc.default_speaker()
                print(f"\n🔊 Default Speaker: {default_speaker.name}")
                print(f"   Will record loopback from this device")
                
                print("\n🎤 Available Microphones:")
                mics = sc.all_microphones(include_loopback=True)
                for i, mic in enumerate(mics):
                    loopback_tag = " (LOOPBACK)" if "loopback" in mic.name.lower() else ""
                    print(f"  [{i}] {mic.name}{loopback_tag}")
            except Exception as e:
                print(f"⚠️ Error listing audio devices: {e}")
        
        print("=" * 60)
        
        self._stop_event.clear()
        self._audio_chunks.clear()
        self._transcript_parts.clear()
        self._full_transcript = ""
        self._chunks_processed = 0
        self._total_audio_duration = 0.0
        self._all_audio_data.clear()
        
        # Reset model and language detection states
        self._model_loading = False
        self._model_loaded = False
        self._detected_language = None
        self._language_detection_done = False
        self._device_type = "cuda" if HAS_GPU else "cpu"
        
        # Output files
        self._audio_file_path = output_dir / "audio.wav"
        self._transcript_file_path = output_dir / "transcript.txt"
        print(f"📁 Audio will be saved to: {self._audio_file_path}")
        print(f"📁 Transcript will be saved to: {self._transcript_file_path}")
        
        # Start recording thread
        print("🚀 Starting audio recording thread...")
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        
        # Start transcription thread
        if HAS_WHISPER:
            print("🚀 Starting transcription thread...")
            self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
            self._transcribe_thread.start()
        else:
            print("⚠️ Whisper not available - audio will be recorded but not transcribed in real-time")
        
        self._is_recording = True
        print(f"🎙️ Started streaming transcription (chunk: {self.chunk_duration}s)")
        return True
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop recording and return final transcript.
        
        Returns:
            Dict with transcript and stats
        """
        if not self._is_recording:
            return {"transcript": "", "duration": 0}
        
        self._stop_event.set()
        self._is_recording = False
        
        # Wait for threads to finish
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=10.0)
        
        # Process any remaining chunks
        self._process_remaining_chunks()
        
        # Save full audio file
        if self._all_audio_data and self._audio_file_path:
            self._save_audio_file()
        
        # Unload Whisper model to free RAM
        if self._whisper_model:
            del self._whisper_model
            self._whisper_model = None
        
        print(f"⏹️ Stopped streaming transcription")
        print(f"   Chunks processed: {self._chunks_processed}")
        print(f"   Total duration: {self._total_audio_duration:.1f}s")
        print(f"   Transcript length: {len(self._full_transcript)} chars")
        
        return {
            "transcript": self._full_transcript,
            "duration": self._total_audio_duration,
            "chunks_processed": self._chunks_processed
        }
    
    def _record_loop(self):
        """Record audio in chunks using configured library."""
        print("🎧 Record loop started")
        
        # Read audio_library setting from config.json
        audio_library = "pyaudiowpatch"  # default
        try:
            import json
            config_path = Path(__file__).parent.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    audio_library = config.get("audio_library", "pyaudiowpatch")
                    print(f"📋 Audio library from config: {audio_library}")
        except Exception as e:
            print(f"⚠️ Could not read config.json, using default: {e}")
        
        try:
            if audio_library == "pyaudiowpatch" and HAS_PYAUDIO:
                print("   Using pyaudiowpatch library for recording (from config)")
                self._record_with_pyaudio()
            elif audio_library == "soundcard" and HAS_SOUNDCARD:
                print("   Using soundcard library for recording (from config)")
                self._record_with_soundcard()
            elif HAS_PYAUDIO:
                print("   Fallback: Using pyaudiowpatch library for recording")
                self._record_with_pyaudio()
            elif HAS_SOUNDCARD:
                print("   Fallback: Using soundcard library for recording")
                self._record_with_soundcard()
            else:
                print("❌ No recording library available in record loop!")
        except Exception as e:
            import traceback
            print(f"❌ Recording error: {e}")
            traceback.print_exc()
        print("🎧 Record loop ended")
    
    def _record_with_soundcard(self):
        """Record using soundcard library (preferred)."""
        try:
            # Search for 'Stereo Mix' device first
            preferred_mic = None
            print("🔍 Searching for 'Stereo Mix' audio device...")
            
            try:
                # Get all microphones including loopback
                all_mics = sc.all_microphones(include_loopback=True)
                for mic in all_mics:
                    if "stereo mix" in mic.name.lower():
                        preferred_mic = mic
                        print(f"✅ Found preferred device: {mic.name}")
                        break
            except Exception as e:
                print(f"⚠️ Error searching for mics: {e}")

            # Fallback to default speaker loopback
            if preferred_mic:
                recorder_context = preferred_mic.recorder(samplerate=self.sample_rate)
                source_name = preferred_mic.name
            else:
                default_speaker = sc.default_speaker()
                print(f"⚠️ 'Stereo Mix' not found. Falling back to default loopback: {default_speaker.name}")
                recorder_context = sc.get_microphone(
                    id=str(default_speaker.name),
                    include_loopback=True
                ).recorder(samplerate=self.sample_rate)
                source_name = f"Loopback: {default_speaker.name}"

            # Calculate samples per chunk
            samples_per_chunk = int(self.sample_rate * self.chunk_duration)
            print(f"   Sample rate: {self.sample_rate} Hz")
            print(f"   Chunk duration: {self.chunk_duration}s")
            print(f"   Samples per chunk: {samples_per_chunk}")
            
            print(f"🎤 Opening audio source: {source_name}...")
            with recorder_context as mic:
                print(f"✅ Audio source opened successfully: {source_name}")
                print("🎙️ Starting to record audio chunks...")
                
                chunk_count = 0
                while not self._stop_event.is_set():
                    # Record a chunk
                    data = mic.record(numframes=samples_per_chunk)
                    chunk_count += 1
                    
                    # Convert stereo to mono if needed
                    if len(data.shape) > 1 and data.shape[1] > 1:
                        data = np.mean(data, axis=1)
                    
                    # Normalize to float32 in range [-1, 1]
                    data = data.astype(np.float32)
                    
                    # Check audio levels
                    max_level = np.max(np.abs(data))
                    rms_level = np.sqrt(np.mean(data**2))
                    
                    # Add to buffers
                    with self._audio_lock:
                        self._audio_chunks.append(data)
                    self._all_audio_data.append(data)
                    
                    self._total_audio_duration += self.chunk_duration
                    
                    # Detailed audio level info
                    if max_level < 0.001:
                        audio_status = "🔇 SILENT"
                    elif max_level < 0.1:
                        audio_status = "🔈 Low"
                    elif max_level < 0.5:
                        audio_status = "🔉 Medium"
                    else:
                        audio_status = "🔊 Loud"
                    
                    print(f"  🎤 Chunk {chunk_count}: {self.chunk_duration}s | {audio_status} (max: {max_level:.4f}, rms: {rms_level:.4f}) | Total: {self._total_audio_duration:.1f}s")
                
                print(f"🛑 Recording stopped. Total chunks: {chunk_count}")
                    
        except Exception as e:
            import traceback
            print(f"❌ Soundcard recording error: {e}")
            traceback.print_exc()
    
    def _record_with_pyaudio(self):
        """Record using pyaudiowpatch (fallback for Windows)."""
        try:
            p = pyaudio.PyAudio()
            
            # Find WASAPI loopback device
            wasapi_info = None
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device.get("isLoopbackDevice"):
                    wasapi_info = device
                    break
            
            if not wasapi_info:
                print("❌ No loopback device found")
                p.terminate()
                return
            
            print(f"🔊 Recording from: {wasapi_info['name']}")
            
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=int(wasapi_info["maxInputChannels"]),
                rate=int(wasapi_info["defaultSampleRate"]),
                input=True,
                input_device_index=wasapi_info["index"],
                frames_per_buffer=int(wasapi_info["defaultSampleRate"] * self.chunk_duration)
            )
            
            while not self._stop_event.is_set():
                data = stream.read(int(wasapi_info["defaultSampleRate"] * self.chunk_duration))
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Resample to target sample rate if needed
                if int(wasapi_info["defaultSampleRate"]) != self.sample_rate:
                    # Simple decimation (could use scipy.signal.resample for better quality)
                    ratio = int(wasapi_info["defaultSampleRate"]) / self.sample_rate
                    audio_data = audio_data[::int(ratio)]
                
                # Convert stereo to mono
                channels = int(wasapi_info["maxInputChannels"])
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels).mean(axis=1)
                
                with self._audio_lock:
                    self._audio_chunks.append(audio_data)
                self._all_audio_data.append(audio_data)
                
                self._total_audio_duration += self.chunk_duration
                print(f"  🎤 Recorded chunk ({self.chunk_duration}s)")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"❌ PyAudio recording error: {e}")
    
    def _transcribe_loop(self):
        """Process audio chunks and transcribe."""
        if not HAS_WHISPER:
            print("⚠️ Whisper not available - skipping transcription")
            return
        
        # Track model loading state
        self._model_loading = True
        self._device_type = "cuda" if HAS_GPU else "cpu"
        
        # Load Whisper model
        device = "cuda" if HAS_GPU else "cpu"
        compute_type = "float16" if HAS_GPU else "int8"
        
        print(f"📦 Loading Whisper '{self.model_size}' model (device: {device})...")
        print(f"   This may take a moment...")
        
        try:
            self._whisper_model = WhisperModel(
                self.model_size, 
                device=device, 
                compute_type=compute_type
            )
            self._model_loading = False
            self._model_loaded = True
            print("✅ Whisper model loaded successfully!")
            print("🎙️ Ready to transcribe audio...")
        except Exception as e:
            self._model_loading = False
            print(f"❌ Failed to load Whisper model: {e}")
            return
        
        while not self._stop_event.is_set():
            # Check for audio chunks to transcribe
            chunk = None
            with self._audio_lock:
                if self._audio_chunks:
                    chunk = self._audio_chunks.pop(0)
            
            if chunk is not None:
                self._transcribe_chunk(chunk)
            else:
                time.sleep(0.5)
    
    def _transcribe_chunk(self, audio_data: np.ndarray):
        """Transcribe a single audio chunk and save to .txt file."""
        try:
            # Auto-detect language from first chunk, then use detected language
            if not self._language_detection_done:
                # First chunk: auto-detect language
                segments, info = self._whisper_model.transcribe(
                    audio_data,
                    beam_size=1,
                    vad_filter=True,
                    language=None  # Auto-detect
                )
                # Convert generator to list immediately so we can both detect language AND get text
                segments_list = list(segments)
                self._detected_language = info.language
                self._language_detection_done = True
                print(f"  🌐 Detected language: {self._detected_language}")
            else:
                # Subsequent chunks: use detected language
                segments, _ = self._whisper_model.transcribe(
                    audio_data,
                    beam_size=1,
                    vad_filter=True,
                    language=self._detected_language
                )
                segments_list = list(segments)
            
            text_parts = []
            for segment in segments_list:
                text_parts.append(segment.text.strip())
            
            chunk_text = " ".join(text_parts)
            
            if chunk_text:
                # Calculate timestamp for this chunk
                chunk_time = self._chunks_processed * self.chunk_duration
                minutes = int(chunk_time // 60)
                seconds = int(chunk_time % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                
                with self._transcript_lock:
                    # Store as dict with time and text
                    self._transcript_parts.append({
                        "time": chunk_time,
                        "timestamp": timestamp,
                        "text": chunk_text
                    })
                    # Build full transcript (just text)
                    self._full_transcript = " ".join([p["text"] for p in self._transcript_parts])
                
                self._chunks_processed += 1
                
                # Append to .txt file in real-time
                if self._transcript_file_path:
                    try:
                        with open(self._transcript_file_path, 'a', encoding='utf-8') as f:
                            f.write(f"{timestamp} {chunk_text}\n")
                    except Exception as e:
                        print(f"⚠️ Failed to write transcript chunk: {e}")
                
                print(f"  📝 Chunk {self._chunks_processed} {timestamp}: \"{chunk_text[:50]}...\"")
            
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
    
    def _process_remaining_chunks(self):
        """Process any remaining audio chunks after stop."""
        if not HAS_WHISPER or not self._whisper_model:
            return
        
        with self._audio_lock:
            remaining = list(self._audio_chunks)
            self._audio_chunks.clear()
        
        for chunk in remaining:
            self._transcribe_chunk(chunk)
    
    def _save_audio_file(self):
        """Save all recorded audio to a WAV file."""
        try:
            if not self._all_audio_data:
                return
            
            # Concatenate all audio data
            full_audio = np.concatenate(self._all_audio_data)
            
            # Convert to 16-bit PCM
            audio_int16 = (full_audio * 32767).astype(np.int16)
            
            # Save as WAV
            with wave.open(str(self._audio_file_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            print(f"💾 Audio saved: {self._audio_file_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save audio: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current transcriber status including model loading state."""
        with self._transcript_lock:
            transcript_chunks = list(self._transcript_parts)
        
        return {
            "streaming_enabled": True,
            "is_recording": self._is_recording,
            "model_loading": self._model_loading,
            "model_loaded": self._model_loaded,
            "device_type": self._device_type,
            "current_transcript": self.current_transcript,
            "transcript_chunks": transcript_chunks,  # List of {time, timestamp, text}
            "audio_duration_seconds": self._total_audio_duration,
            "chunks_processed": self._chunks_processed,
            "chunk_duration_seconds": self.chunk_duration,
            "has_soundcard": HAS_SOUNDCARD,
            "has_pyaudio": HAS_PYAUDIO,
            "has_whisper": HAS_WHISPER,
            "has_gpu": HAS_GPU
        }


class VideoCaptureHandler:
    """
    Handles video lecture capture:
    - Captures screenshots every N seconds
    - SSIM-based frame deduplication
    - Integrates with StreamingTranscriber for real-time audio transcription
    - Stores frames and audio in database
    """
    
    def __init__(self, 
                 capture_interval: float = 2.0,
                 ssim_threshold: float = 0.5,
                 media_listener = None,
                 chunk_duration: float = 5.0):
        """
        Initialize the video capture handler.
        
        Args:
            capture_interval: Seconds between frame captures (default 2)
            ssim_threshold: Similarity threshold for deduplication (default 0.5 = 50%)
                           Lower values = more frames saved (images must be 50%+ different)
            media_listener: Optional MediaListener instance for audio (legacy)
            chunk_duration: Duration of audio chunks for transcription (default 5s)
        """
        self.capture_interval = capture_interval
        self.ssim_threshold = ssim_threshold
        self.media_listener = media_listener
        self.chunk_duration = chunk_duration
        
        self._is_recording = False
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._saved_frame_count = 0
        
        # Streaming transcriber for real-time audio
        self._transcriber: Optional[StreamingTranscriber] = None
        
        # Callbacks
        self._on_frame_saved: Optional[Callable] = None
        self._on_status_change: Optional[Callable] = None
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    def start(self, session_id: str) -> Dict[str, Any]:
        """
        Start video capture session.
        
        Args:
            session_id: The capture session ID
            
        Returns:
            Dict with status and any errors
        """
        result = {
            "success": False,
            "session_id": session_id,
            "errors": []
        }
        
        if self._is_recording:
            result["errors"].append("Already recording")
            return result
        
        try:
            self._session_id = session_id
            self._session_dir = MEDIA_VIDEO_DIR / session_id
            self._frames_dir = self._session_dir / "frames"
            
            # Create directories
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._frames_dir.mkdir(exist_ok=True)
            
            # Reset counters
            self._last_frame = None
            self._frame_count = 0
            self._saved_frame_count = 0
            
            # Start streaming transcriber (new - replaces media_listener)
            self._transcriber = StreamingTranscriber(
                chunk_duration=self.chunk_duration,
                sample_rate=16000,
                model_size="tiny"
            )
            if not self._transcriber.start(self._session_dir):
                result["errors"].append("Failed to start audio recording")
                # Continue anyway - frame capture can still work
            
            # Legacy: Start audio recording if media listener available
            if self.media_listener:
                audio_path = self._session_dir / "audio.wav"
                self.media_listener.start_recording(str(audio_path))
            
            # Note: Frame capture is now handled by the browser extension
            # via /api/ingest/frame endpoint, not by system screenshot loop.
            # The extension captures actual video frames from the <video> element.
            self._stop_event.clear()
            self._is_recording = True
            
            result["success"] = True
            result["frames_dir"] = str(self._frames_dir)
            result["streaming_transcription"] = self._transcriber.is_recording if self._transcriber else False
            print(f"▶️ Video capture started: {session_id}")
            print(f"   📷 Waiting for frames from browser extension...")
            
        except Exception as e:
            result["errors"].append(f"Failed to start: {str(e)}")
            self._is_recording = False
        
        return result
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop video capture session.
        
        Returns:
            Dict with session stats and any errors
        """
        result = {
            "success": False,
            "session_id": self._session_id,
            "total_frames": self._frame_count,
            "saved_frames": self._saved_frame_count,
            "audio_recorded": False,
            "transcript": None,
            "errors": []
        }
        
        if not self._is_recording:
            result["errors"].append("Not recording")
            return result
        
        try:
            # Signal stop
            self._stop_event.set()
            self._is_recording = False
            
            # Wait for capture thread
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=5.0)
            
            # Stop streaming transcriber (new)
            transcriber_result = None
            if self._transcriber:
                transcriber_result = self._transcriber.stop()
                result["audio_recorded"] = True
                result["audio_duration"] = transcriber_result.get("duration", 0)
                result["transcript"] = transcriber_result.get("transcript", "")
                result["chunks_processed"] = transcriber_result.get("chunks_processed", 0)
            
            # Legacy: Stop audio recording
            audio_path = None
            if self.media_listener:
                audio_path = self.media_listener.stop_recording()
                if audio_path:
                    result["audio_recorded"] = True
                    result["audio_path"] = audio_path
            
            # Store in database
            db = get_db()
            try:
                # Update session
                session = db.query(CaptureSession).filter_by(id=self._session_id).first()
                if session:
                    session.end_time = datetime.utcnow()
                
                # Store audio file reference if recorded
                if self._session_dir:
                    audio_file = self._session_dir / "audio.wav"
                    if audio_file.exists():
                        audio_record = CapturedMedia(
                            session_id=self._session_id,
                            file_path=str(audio_file),
                            media_type=MediaType.AUDIO
                        )
                        db.add(audio_record)
                
                # Store transcript if available
                if transcriber_result and transcriber_result.get("transcript"):
                    text_record = CapturedText(
                        session_id=self._session_id,
                        content=transcriber_result["transcript"],
                        source_url_or_path=str(self._session_dir / "audio.wav"),
                        content_type="transcript"
                    )
                    db.add(text_record)
                    print(f"💾 Transcript saved: {len(transcriber_result['transcript'])} chars")
                
                db.commit()
                
            except Exception as e:
                db.rollback()
                result["errors"].append(f"Database error: {str(e)}")
            finally:
                db.close()
            
            result["success"] = True
            print(f"⏹️ Video capture stopped: {self._saved_frame_count}/{self._frame_count} frames saved")
            
        except Exception as e:
            result["errors"].append(f"Failed to stop: {str(e)}")
        
        return result
    
    def _capture_loop(self):
        """Main capture loop running in background thread."""
        print(f"🎬 Capture loop started (interval: {self.capture_interval}s)")
        
        while not self._stop_event.is_set():
            try:
                # Check media state if listener available
                should_capture = True
                if self.media_listener:
                    is_playing = self.media_listener.is_media_playing()
                    should_capture = is_playing
                    
                    # Also control audio recording based on playback state
                    if is_playing:
                        self.media_listener.resume_recording()
                    else:
                        self.media_listener.pause_recording()
                
                if should_capture:
                    self._capture_frame()
                
                # Wait for next interval
                self._stop_event.wait(timeout=self.capture_interval)
                
            except Exception as e:
                print(f"⚠️ Capture loop error: {e}")
        
        print("🎬 Capture loop ended")
    
    def _capture_frame(self):
        """Capture a single frame and save if sufficiently different."""
        self._frame_count += 1
        
        try:
            # Capture screenshot
            screenshot = ImageGrab.grab()
            frame = np.array(screenshot)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Check similarity with last frame
            should_save = True
            if self._last_frame is not None and HAS_SKIMAGE:
                # Resize if needed for comparison
                if frame_gray.shape != self._last_frame.shape:
                    self._last_frame = cv2.resize(
                        self._last_frame, 
                        (frame_gray.shape[1], frame_gray.shape[0])
                    )
                
                similarity, _ = ssim(frame_gray, self._last_frame, full=True)
                
                # Only save if sufficiently different
                # Note: threshold 0.5 means we save if images are LESS than 50% similar
                # (i.e., they must be at least 50% different)
                should_save = similarity < self.ssim_threshold
                
                if not should_save:
                    print(f"  📷 Frame {self._frame_count}: skipped (similarity: {similarity:.2%})")
            
            if should_save:
                self._save_frame(frame)
                self._last_frame = frame_gray
                
        except Exception as e:
            print(f"⚠️ Frame capture error: {e}")
    
    def _save_frame(self, frame: np.ndarray):
        """Save a frame to disk and database."""
        self._saved_frame_count += 1
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{self._saved_frame_count:05d}_{timestamp}.jpg"
        filepath = self._frames_dir / filename
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Store in database
        db = get_db()
        try:
            media_record = CapturedMedia(
                session_id=self._session_id,
                file_path=str(filepath),
                media_type=MediaType.FRAME
            )
            db.add(media_record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"⚠️ Failed to save frame record: {e}")
        finally:
            db.close()
        
        print(f"  📷 Frame {self._saved_frame_count} saved: {filename}")
        
        if self._on_frame_saved:
            self._on_frame_saved(filepath)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current capture status."""
        return {
            "is_recording": self._is_recording,
            "session_id": self._session_id,
            "total_frames": self._frame_count,
            "saved_frames": self._saved_frame_count,
            "capture_interval": self.capture_interval,
            "ssim_threshold": self.ssim_threshold,
            "ingested_frames": getattr(self, '_ingested_frame_count', 0),
        }
    
    # =========================================================================
    # Extension-based Frame Ingestion (NEW)
    # =========================================================================
    
    def ingest_frame(self, frame_data: str, frame_number: int = None, 
                     video_time: float = None, timestamp: int = None) -> Dict[str, Any]:
        """
        Ingest a frame from the browser extension.
        
        Args:
            frame_data: Base64 encoded JPEG image (data URL format)
            frame_number: Frame number from extension
            video_time: Video playback time
            timestamp: Unix timestamp when frame was captured
            
        Returns:
            Dict with status: whether frame was saved and similarity score
        """
        result = {
            "saved": False,
            "similarity": None,
            "frame_number": frame_number,
            "errors": []
        }
        
        if not self._is_recording:
            result["errors"].append("Not recording")
            return result
        
        try:
            # Initialize ingested frame counter
            if not hasattr(self, '_ingested_frame_count'):
                self._ingested_frame_count = 0
            self._ingested_frame_count += 1
            
            # Decode base64 image
            # Remove data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            import base64
            image_bytes = base64.b64decode(frame_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                result["errors"].append("Failed to decode image")
                return result
            
            # Convert to grayscale for SSIM comparison
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use dedicated variable for ingested frames (separate from system screenshot loop)
            if not hasattr(self, '_last_ingested_frame'):
                self._last_ingested_frame = None
            
            # Check similarity with last ingested frame
            # Threshold: 0.6 means we ONLY save if similarity is LESS than 60%
            # (i.e., frames must be at least 40% different)
            extension_ssim_threshold = 0.6
            should_save = True
            
            if self._last_ingested_frame is not None and HAS_SKIMAGE:
                # Resize for comparison if dimensions differ
                compare_frame = self._last_ingested_frame
                if frame_gray.shape != compare_frame.shape:
                    compare_frame = cv2.resize(
                        compare_frame, 
                        (frame_gray.shape[1], frame_gray.shape[0])
                    )
                
                similarity, _ = ssim(frame_gray, compare_frame, full=True)
                result["similarity"] = float(similarity)
                
                # Save only if similarity is LESS than threshold (frames are different enough)
                should_save = similarity < extension_ssim_threshold
                
                if should_save:
                    print(f"  📷 Frame {self._ingested_frame_count}: SAVED (similarity: {similarity:.1%} < {extension_ssim_threshold:.0%})")
                else:
                    print(f"  📷 Frame {self._ingested_frame_count}: skipped (similarity: {similarity:.1%} >= {extension_ssim_threshold:.0%})")
            else:
                print(f"  📷 Frame {self._ingested_frame_count}: SAVED (first frame or no SSIM)")
            
            if should_save:
                self._save_ingested_frame(frame, video_time)
                self._last_ingested_frame = frame_gray.copy()  # Store copy for next comparison
                result["saved"] = True
            
        except Exception as e:
            result["errors"].append(f"Frame ingestion error: {str(e)}")
            print(f"⚠️ Frame ingestion error: {e}")
        
        return result
    
    def _save_ingested_frame(self, frame: np.ndarray, video_time: float = None):
        """Save an ingested frame to disk and database."""
        self._saved_frame_count += 1
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        video_time_str = f"_t{int(video_time)}" if video_time else ""
        filename = f"frame_{self._saved_frame_count:05d}{video_time_str}_{timestamp_str}.jpg"
        filepath = self._frames_dir / filename
        
        # Frame is already in BGR format from cv2.imdecode
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Store in database
        db = get_db()
        try:
            media_record = CapturedMedia(
                session_id=self._session_id,
                file_path=str(filepath),
                media_type=MediaType.FRAME
            )
            db.add(media_record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"⚠️ Failed to save ingested frame record: {e}")
        finally:
            db.close()
        
        print(f"  💾 Ingested frame {self._saved_frame_count} saved: {filename}")
        
        if self._on_frame_saved:
            self._on_frame_saved(filepath)
    
    def get_transcriber_status(self) -> Dict[str, Any]:
        """Get streaming transcriber status."""
        if self._transcriber:
            return self._transcriber.get_status()
        return {
            "streaming_enabled": HAS_SOUNDCARD or HAS_PYAUDIO,
            "is_recording": False,
            "current_transcript": "",
            "audio_duration_seconds": 0,
            "chunks_processed": 0,
            "has_soundcard": HAS_SOUNDCARD,
            "has_pyaudio": HAS_PYAUDIO,
            "has_whisper": HAS_WHISPER,
            "has_gpu": HAS_GPU
        }
    
    def transcribe_session(self, session_id: str) -> Dict[str, Any]:
        """
        Transcribe recorded audio for a video session using Whisper.
        Loads Whisper model on demand to save RAM when not in use.
        
        Args:
            session_id: The capture session ID
            
        Returns:
            Dict with transcription status and content
        """
        result = {
            "success": False,
            "session_id": session_id,
            "transcript": None,
            "duration_seconds": None,
            "errors": []
        }
        
        if not HAS_WHISPER:
            result["errors"].append("faster-whisper not installed. Run: pip install faster-whisper")
            return result
        
        # Find audio file in session directory
        session_dir = MEDIA_VIDEO_DIR / session_id
        audio_path = session_dir / "audio.wav"
        
        if not audio_path.exists():
            result["errors"].append(f"Audio file not found: {audio_path}")
            return result
        
        try:
            print(f"🎙️ Transcribing audio for session: {session_id}")
            
            # Load Whisper model on demand with GPU auto-detection
            device = "cuda" if HAS_GPU else "cpu"
            compute_type = "float16" if HAS_GPU else "int8"
            
            print(f"  📦 Loading Whisper 'tiny' model (device: {device})...")
            model = WhisperModel("tiny", device=device, compute_type=compute_type)
            
            # Transcribe audio
            print(f"  🔄 Processing audio...")
            segments, info = model.transcribe(str(audio_path), beam_size=5)
            
            # Collect transcript text
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text)
            
            transcript = " ".join(transcript_parts).strip()
            result["transcript"] = transcript
            result["duration_seconds"] = info.duration
            
            print(f"  ✅ Transcription complete: {len(transcript)} chars, {info.duration:.1f}s audio")
            
            # Save transcript to database
            db = get_db()
            try:
                text_record = CapturedText(
                    session_id=session_id,
                    content=transcript,
                    source_url_or_path=str(audio_path),
                    content_type="transcript"
                )
                db.add(text_record)
                db.commit()
                print(f"  💾 Transcript saved to database")
                result["success"] = True
                
                # Index transcript for RAG search
                try:
                    from rag_engine import get_rag_engine
                    rag = get_rag_engine()
                    
                    # Get session title
                    session = db.query(CaptureSession).filter_by(id=session_id).first()
                    title = session.title if session else "Video Session"
                    
                    index_result = rag.index_session(
                        session_id=session_id,
                        text=transcript,
                        source="video",
                        title=title
                    )
                    if index_result.get("success"):
                        print(f"  🔍 Indexed {index_result.get('chunks_indexed', 0)} chunks for RAG search")
                    else:
                        print(f"  ⚠️ RAG indexing failed: {index_result.get('errors', [])}")
                except Exception as rag_error:
                    print(f"  ⚠️ RAG indexing error (non-fatal): {rag_error}")
                
            except Exception as e:
                db.rollback()
                result["errors"].append(f"Database error: {str(e)}")
            finally:
                db.close()
            
            # Clean up model to free RAM
            del model
            
        except Exception as e:
            result["errors"].append(f"Transcription failed: {str(e)}")
            print(f"  ❌ Transcription error: {e}")
        
        return result
    
    @staticmethod
    def get_capabilities() -> Dict[str, Any]:
        """Return available capabilities for video capture."""
        return {
            "ssim_deduplication": HAS_SKIMAGE,
            "transcription": HAS_WHISPER,
            "gpu_available": HAS_GPU,
            "audio_recording_soundcard": HAS_SOUNDCARD,
            "audio_recording_pyaudio": HAS_PYAUDIO,
            "streaming_transcription": (HAS_SOUNDCARD or HAS_PYAUDIO) and HAS_WHISPER,
        }


if __name__ == "__main__":
    # Quick test
    handler = VideoCaptureHandler()
    print(f"Capabilities: {VideoCaptureHandler.get_capabilities()}")
    print(f"Status: {handler.get_status()}")

