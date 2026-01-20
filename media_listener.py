"""
Media Listener for the TimeLine Content Capture Pipeline.
Handles Windows Media Transport Controls and System Audio Recording.

This module is Windows-specific and requires:
- winsdk for media control detection
- soundcard for audio loopback recording
"""

import asyncio
import threading
import wave
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import time

# Windows-specific imports
try:
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus
    )
    HAS_WINSDK = True
except ImportError:
    HAS_WINSDK = False

try:
    import soundcard as sc
    HAS_SOUNDCARD = True
except ImportError:
    HAS_SOUNDCARD = False

try:
    import numpy as np
    HAS_NUMPY = True
    
    # Monkeypatch numpy.fromstring to fix compability with soundcard < 0.4.2 and numpy >= 1.20
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
            print("🔧 Applied monkeypatch for numpy.fromstring (soundcard compatibility in media_listener)")
    except Exception as e:
        print(f"⚠️ Failed to patch numpy in media_listener: {e}")

except ImportError:
    HAS_NUMPY = False


class MediaListener:
    """
    Windows Media Transport Controls listener and audio recorder.
    
    Features:
    - Detects system-wide media playback state (Playing/Paused)
    - Records system audio (loopback) to WAV files
    - Automatically pauses/resumes recording based on playback state
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        """
        Initialize the media listener.
        
        Args:
            sample_rate: Audio sample rate (default 44100)
            channels: Number of audio channels (default 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Media playback state
        self._media_playing = False
        self._media_session_active = False
        self._media_title: Optional[str] = None
        self._media_artist: Optional[str] = None
        
        # Audio recording state
        self._is_recording = False
        self._is_paused = False
        self._recording_thread: Optional[threading.Thread] = None
        self._stop_recording_event = threading.Event()
        self._output_path: Optional[str] = None
        self._audio_frames = []
        
        # Asyncio event loop for winsdk
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_playback_change: Optional[Callable] = None
        
        # Start media listener if available
        if HAS_WINSDK:
            self._start_media_listener()
    
    def _start_media_listener(self):
        """Start the asyncio event loop for media control detection."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._media_listener_loop())
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        print("✅ Media listener started")
    
    async def _media_listener_loop(self):
        """Main async loop for checking media state."""
        try:
            manager = await MediaManager.request_async()
            
            while True:
                try:
                    session = manager.get_current_session()
                    
                    if session:
                        self._media_session_active = True
                        
                        # Get playback info
                        playback_info = session.get_playback_info()
                        status = playback_info.playback_status
                        
                        was_playing = self._media_playing
                        self._media_playing = (status == PlaybackStatus.PLAYING)
                        
                        # Get media info
                        try:
                            media_properties = await session.try_get_media_properties_async()
                            if media_properties:
                                self._media_title = media_properties.title
                                self._media_artist = media_properties.artist
                        except Exception:
                            pass
                        
                        # Trigger callback on state change
                        if was_playing != self._media_playing and self._on_playback_change:
                            self._on_playback_change(self._media_playing)
                    else:
                        self._media_session_active = False
                        self._media_playing = False
                        self._media_title = None
                        self._media_artist = None
                    
                    await asyncio.sleep(0.5)  # Check every 500ms
                    
                except Exception as e:
                    print(f"⚠️ Media listener error: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"❌ Failed to start media manager: {e}")
    
    def is_media_playing(self) -> bool:
        """Check if media is currently playing."""
        return self._media_playing
    
    def is_session_active(self) -> bool:
        """Check if there's an active media session."""
        return self._media_session_active
    
    def get_media_info(self) -> dict:
        """Get current media information."""
        return {
            "is_playing": self._media_playing,
            "session_active": self._media_session_active,
            "title": self._media_title,
            "artist": self._media_artist,
        }
    
    def set_playback_callback(self, callback: Callable):
        """Set callback for playback state changes."""
        self._on_playback_change = callback
    
    # Audio Recording Methods
    
    def start_recording(self, output_path: str) -> bool:
        """
        Start recording system audio.
        
        Args:
            output_path: Path to save the WAV file
            
        Returns:
            True if recording started successfully
        """
        if not HAS_SOUNDCARD:
            print("⚠️ soundcard not installed, cannot record audio")
            return False
        
        if self._is_recording:
            print("⚠️ Already recording")
            return False
        
        self._output_path = output_path
        self._audio_frames = []
        self._stop_recording_event.clear()
        self._is_recording = True
        self._is_paused = False
        
        self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._recording_thread.start()
        
        print(f"🎙️ Audio recording started: {output_path}")
        return True
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save audio file.
        
        Returns:
            Path to the saved audio file, or None if failed
        """
        if not self._is_recording:
            return None
        
        self._stop_recording_event.set()
        self._is_recording = False
        
        # Wait for recording thread
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=5.0)
        
        # Save audio to file
        if self._audio_frames and self._output_path:
            try:
                return self._save_audio()
            except Exception as e:
                print(f"❌ Failed to save audio: {e}")
                return None
        
        return None
    
    def pause_recording(self):
        """Pause audio recording."""
        if self._is_recording and not self._is_paused:
            self._is_paused = True
            print("⏸️ Audio recording paused")
    
    def resume_recording(self):
        """Resume audio recording."""
        if self._is_recording and self._is_paused:
            self._is_paused = False
            print("▶️ Audio recording resumed")
    
    def _recording_loop(self):
        """Main audio recording loop."""
        try:
            # Search for 'Stereo Mix' device first
            preferred_mic = None
            print("🔍 Searching for 'Stereo Mix' audio device (media listener)...")
            
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
            
            if preferred_mic:
                recorder_context = preferred_mic.recorder(samplerate=self.sample_rate, channels=self.channels)
                source_name = preferred_mic.name
            else:
                # Fallback to default loopback
                default_speaker = sc.default_speaker()
                print(f"⚠️ 'Stereo Mix' not found. Falling back to default loopback: {default_speaker.name}")
                recorder_context = sc.get_microphone(
                    id=str(default_speaker.name),
                    include_loopback=True
                ).recorder(samplerate=self.sample_rate, channels=self.channels)
                source_name = f"Loopback: {default_speaker.name}"
            
            if recorder_context is None:
                print("❌ Could not open audio device")
                return
            
            print(f"🎤 Recording from: {source_name}")
            
            with recorder_context as recorder:
                while not self._stop_recording_event.is_set():
                    # Record small chunks
                    data = recorder.record(numframes=int(self.sample_rate * 0.1))  # 100ms chunks
                    
                    if not self._is_paused:
                        self._audio_frames.append(data)
                    
                    time.sleep(0.01)  # Small sleep to prevent CPU overuse
                    
        except Exception as e:
            print(f"❌ Recording error: {e}")
    
    def _save_audio(self) -> str:
        """Save recorded audio to WAV file."""
        if not self._audio_frames:
            return None
        
        import numpy as np
        
        # Concatenate all frames
        audio_data = np.concatenate(self._audio_frames, axis=0)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save to WAV
        output_path = Path(self._output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(str(output_path), 'w') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        duration = len(audio_data) / self.sample_rate
        print(f"💾 Audio saved: {output_path} ({duration:.1f}s)")
        
        self._audio_frames = []
        return str(output_path)
    
    def get_recording_status(self) -> dict:
        """Get current recording status."""
        return {
            "is_recording": self._is_recording,
            "is_paused": self._is_paused,
            "output_path": self._output_path,
            "frames_recorded": len(self._audio_frames),
        }
    
    @staticmethod
    def get_capabilities() -> dict:
        """Return available capabilities."""
        return {
            "media_controls": HAS_WINSDK,
            "audio_recording": HAS_SOUNDCARD,
            "platform": "windows" if HAS_WINSDK else "unknown",
        }


# Singleton instance
_media_listener: Optional[MediaListener] = None


def get_media_listener() -> MediaListener:
    """Get or create the global media listener instance."""
    global _media_listener
    if _media_listener is None:
        _media_listener = MediaListener()
    return _media_listener


if __name__ == "__main__":
    # Quick test
    print(f"Capabilities: {MediaListener.get_capabilities()}")
    
    if HAS_WINSDK and HAS_SOUNDCARD:
        listener = get_media_listener()
        print("Checking media state for 5 seconds...")
        
        for i in range(10):
            time.sleep(0.5)
            print(f"  Media info: {listener.get_media_info()}")
