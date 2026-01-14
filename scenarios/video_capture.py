"""
Video Capture Handler for the TimeLine Content Capture Pipeline.
Handles video lecture recording with frame capture and audio recording.
"""

import os
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import cv2
import numpy as np
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


class VideoCaptureHandler:
    """
    Handles video lecture capture:
    - Captures screenshots every N seconds
    - SSIM-based frame deduplication
    - Integrates with media_listener for audio recording
    - Stores frames and audio in database
    """
    
    def __init__(self, 
                 capture_interval: float = 2.0,
                 ssim_threshold: float = 0.5,
                 media_listener = None):
        """
        Initialize the video capture handler.
        
        Args:
            capture_interval: Seconds between frame captures (default 2)
            ssim_threshold: Similarity threshold for deduplication (default 0.5 = 50%)
                           Lower values = more frames saved (images must be 50%+ different)
            media_listener: Optional MediaListener instance for audio
        """
        self.capture_interval = capture_interval
        self.ssim_threshold = ssim_threshold
        self.media_listener = media_listener
        
        self._is_recording = False
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._saved_frame_count = 0
        
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
            
            # Start audio recording if media listener available
            if self.media_listener:
                audio_path = self._session_dir / "audio.wav"
                self.media_listener.start_recording(str(audio_path))
            
            # Start capture thread
            self._stop_event.clear()
            self._is_recording = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            result["success"] = True
            result["frames_dir"] = str(self._frames_dir)
            print(f"▶️ Video capture started: {session_id}")
            
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
            
            # Stop audio recording
            audio_path = None
            if self.media_listener:
                audio_path = self.media_listener.stop_recording()
                result["audio_recorded"] = audio_path is not None
                result["audio_path"] = audio_path
            
            # Store in database
            db = get_db()
            try:
                # Update session
                session = db.query(CaptureSession).filter_by(id=self._session_id).first()
                if session:
                    session.end_time = datetime.utcnow()
                
                # Store audio file reference if recorded
                if audio_path:
                    audio_record = CapturedMedia(
                        session_id=self._session_id,
                        file_path=str(audio_path),
                        media_type=MediaType.AUDIO
                    )
                    db.add(audio_record)
                
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
        }


if __name__ == "__main__":
    # Quick test
    handler = VideoCaptureHandler()
    print(f"Capabilities: {VideoCaptureHandler.get_capabilities()}")
    print(f"Status: {handler.get_status()}")

