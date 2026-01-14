"""
Orchestrator for the TimeLine Content Capture Pipeline.
Implements the state-machine-based monitoring loop using FastVLM for activity detection.
"""

import threading
import time
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from PIL import ImageGrab
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from state_manager import (
    get_state_manager, StateManager, AppState, 
    ScenarioType, PendingScenario
)
from models import (
    init_db, ensure_media_dirs, get_db,
    CaptureSession, SessionType, MEDIA_BASE_DIR
)


class Orchestrator:
    """
    Main orchestrator for the Content Capture Pipeline.
    
    Responsibilities:
    - Run the monitoring loop (screenshot + VLM analysis every N seconds)
    - Detect user activity scenarios (Web/Doc/Video)
    - Coordinate state transitions
    - Manage capture sessions
    """
    
    # VLM endpoint configuration
    DEFAULT_VLM_URL = "http://localhost:5001/predict"
    
    # Scenario detection prompt for FastVLM
    SCENARIO_DETECTION_PROMPT = """Analyze this screen. Is the user:
1) Reading a website article or blog post?
2) Viewing a PDF, document, or reading material?
3) Watching a Video (YouTube, video player, lecture)?
4) Other activity?

Respond with ONLY valid JSON in this exact format:
{"scenario": "WEB", "confidence": 0.85}

Where scenario is one of: WEB, DOC, VIDEO, OTHER
And confidence is a float from 0.0 to 1.0"""
    
    def __init__(self, 
                 monitoring_interval: float = 20.0,
                 confidence_threshold: float = 0.7,
                 vlm_url: str = None,
                 similarity_threshold: float = 0.95):
        """
        Initialize the orchestrator.
        
        Args:
            monitoring_interval: Seconds between VLM checks (default 20)
            confidence_threshold: Min confidence to trigger scenario (default 0.7)
            vlm_url: FastVLM endpoint URL
            similarity_threshold: SSIM threshold for skipping similar screenshots
        """
        self.monitoring_interval = monitoring_interval
        self.confidence_threshold = confidence_threshold
        self.vlm_url = vlm_url or self.DEFAULT_VLM_URL
        self.similarity_threshold = similarity_threshold
        
        self._state_manager = get_state_manager()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Screenshot tracking
        self._screenshots_dir = Path("screenshots")
        self._screenshots_dir.mkdir(exist_ok=True)
        self._last_screenshot_path: Optional[Path] = None
        
        # Stats
        self._loop_count = 0
        self._scenarios_detected = 0
        self._last_analysis_time: Optional[datetime] = None
        self._last_analysis_result: Optional[Dict] = None
    
    def start(self) -> bool:
        """
        Start the monitoring loop.
        
        Returns:
            True if started successfully
        """
        if self._state_manager.monitoring_enabled:
            print("⚠️ Monitoring already running")
            return False
        
        # Initialize database and directories
        init_db()
        ensure_media_dirs()
        
        self._state_manager.set_monitoring_enabled(True)
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"🚀 Orchestrator started (interval: {self.monitoring_interval}s)")
        return True
    
    def stop(self) -> bool:
        """
        Stop the monitoring loop.
        
        Returns:
            True if stopped successfully
        """
        if not self._state_manager.monitoring_enabled:
            return False
        
        self._state_manager.set_monitoring_enabled(False)
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        print("⏹️ Orchestrator stopped")
        return True
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        print("⏳ Waiting 5 seconds before starting monitoring...")
        time.sleep(5)
        print("✅ Monitoring loop active")
        
        while not self._stop_event.is_set() and self._state_manager.monitoring_enabled:
            try:
                # Only run if in MONITORING state
                if not self._state_manager.is_monitoring:
                    print(f"  ⏸️ Paused (state: {self._state_manager.current_state.value})")
                    self._stop_event.wait(timeout=2.0)
                    continue
                
                self._loop_count += 1
                print(f"\n🔄 Monitoring loop #{self._loop_count}")
                
                # Take screenshot
                screenshot_path = self._take_screenshot()
                if not screenshot_path:
                    continue
                
                # Check similarity with last screenshot
                if self._should_skip_similar(screenshot_path):
                    print("  ⏭️ Skipping similar screenshot")
                    screenshot_path.unlink()  # Delete duplicate
                    self._stop_event.wait(timeout=self.monitoring_interval)
                    continue
                
                self._last_screenshot_path = screenshot_path
                
                # Analyze with VLM
                analysis = self._analyze_screenshot(screenshot_path)
                self._last_analysis_time = datetime.utcnow()
                self._last_analysis_result = analysis
                
                if analysis and analysis.get("scenario"):
                    scenario = analysis["scenario"]
                    confidence = analysis.get("confidence", 0.0)
                    
                    print(f"  📊 Detected: {scenario} (confidence: {confidence:.2%})")
                    
                    # Check if we should trigger scenario
                    if (scenario != "OTHER" and 
                        confidence >= self.confidence_threshold):
                        
                        self._trigger_scenario(scenario, confidence, screenshot_path)
                
                # Wait for next interval
                self._stop_event.wait(timeout=self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Monitoring loop error: {e}")
                self._stop_event.wait(timeout=5.0)
        
        print("🔄 Monitoring loop ended")
    
    def _take_screenshot(self) -> Optional[Path]:
        """Capture and save a screenshot."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self._screenshots_dir / f"screenshot_{timestamp}.png"
            
            screenshot = ImageGrab.grab()
            screenshot.save(filename)
            
            print(f"  📸 Screenshot: {filename.name}")
            return filename
            
        except Exception as e:
            print(f"  ❌ Screenshot failed: {e}")
            return None
    
    def _should_skip_similar(self, current_path: Path) -> bool:
        """Check if current screenshot is too similar to the last one."""
        if not self._last_screenshot_path or not self._last_screenshot_path.exists():
            return False
        
        if not HAS_SKIMAGE:
            return False
        
        try:
            img1 = cv2.imread(str(self._last_screenshot_path))
            img2 = cv2.imread(str(current_path))
            
            if img1 is None or img2 is None:
                return False
            
            # Resize if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            similarity, _ = ssim(gray1, gray2, full=True)
            
            return similarity >= self.similarity_threshold
            
        except Exception as e:
            print(f"  ⚠️ SSIM check failed: {e}")
            return False
    
    def _analyze_screenshot(self, image_path: Path) -> Optional[Dict]:
        """
        Send screenshot to FastVLM for scenario analysis.
        
        Returns:
            Dict with 'scenario' and 'confidence', or None if failed
        """
        try:
            print(f"  🧠 Analyzing with VLM...")
            
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'prompt': self.SCENARIO_DETECTION_PROMPT}
                
                response = requests.post(
                    self.vlm_url, 
                    files=files, 
                    data=data,
                    timeout=30
                )
            
            if response.status_code != 200:
                print(f"  ⚠️ VLM request failed: {response.status_code}")
                return None
            
            response_json = response.json()
            
            # FastVLM returns {"response": "..."}
            response_text = response_json.get('response', '')
            
            # Parse the JSON response from VLM
            return self._parse_vlm_response(response_text)
            
        except requests.RequestException as e:
            print(f"  ⚠️ VLM connection failed: {e}")
            return None
        except Exception as e:
            print(f"  ⚠️ VLM analysis failed: {e}")
            return None
    
    def _parse_vlm_response(self, response_text: str) -> Optional[Dict]:
        """Parse the VLM response to extract scenario and confidence."""
        try:
            # Clean up response
            text = response_text.strip()
            
            # Handle markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Find JSON object in response
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                
                scenario = data.get('scenario', 'OTHER').upper()
                confidence = float(data.get('confidence', 0.0))
                
                # Validate scenario
                if scenario not in ['WEB', 'DOC', 'VIDEO', 'OTHER']:
                    scenario = 'OTHER'
                
                return {
                    'scenario': scenario,
                    'confidence': min(max(confidence, 0.0), 1.0)
                }
            
            return None
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  ⚠️ Failed to parse VLM response: {e}")
            return None
    
    def _trigger_scenario(self, scenario: str, confidence: float, screenshot_path: Path):
        """Trigger a pending scenario for user confirmation."""
        scenario_type = ScenarioType(scenario)
        
        pending = PendingScenario(
            scenario_type=scenario_type,
            confidence=confidence,
            screenshot_path=str(screenshot_path)
        )
        
        if self._state_manager.set_pending_scenario(pending):
            self._scenarios_detected += 1
            print(f"  🔔 Scenario triggered: {scenario} ({confidence:.2%})")
    
    def force_analyze(self) -> Optional[Dict]:
        """
        Force an immediate screenshot and analysis.
        Useful for testing or manual triggers.
        
        Returns:
            Analysis result dict
        """
        screenshot_path = self._take_screenshot()
        if screenshot_path:
            return self._analyze_screenshot(screenshot_path)
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "monitoring_enabled": self._state_manager.monitoring_enabled,
            "current_state": self._state_manager.current_state.value,
            "monitoring_interval": self.monitoring_interval,
            "confidence_threshold": self.confidence_threshold,
            "vlm_url": self.vlm_url,
            "loop_count": self._loop_count,
            "scenarios_detected": self._scenarios_detected,
            "last_analysis_time": self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            "last_analysis_result": self._last_analysis_result,
        }


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def configure_orchestrator(**kwargs) -> Orchestrator:
    """
    Configure and get the orchestrator instance.
    
    Args:
        monitoring_interval: Seconds between checks
        confidence_threshold: Min confidence for scenarios
        vlm_url: FastVLM endpoint URL
        similarity_threshold: SSIM threshold
    """
    global _orchestrator
    _orchestrator = Orchestrator(**kwargs)
    return _orchestrator


if __name__ == "__main__":
    # Quick test
    print("Testing orchestrator...")
    orch = get_orchestrator()
    print(f"Status: {orch.get_status()}")
