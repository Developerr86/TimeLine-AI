"""
Orchestrator for the TimeLine Content Capture Pipeline.
Event-driven orchestrator that receives heartbeats from the browser extension.
"""

import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

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
    - Receive heartbeats from browser extension
    - Detect scenario changes (Web/Doc/Video)
    - Coordinate state transitions
    - Manage capture sessions
    
    This is now event-driven (passive) rather than polling-based (active).
    """
    
    def __init__(self, 
                 auto_trigger_video: bool = False,
                 scenario_change_threshold: int = 1):
        """
        Initialize the orchestrator.
        
        Args:
            auto_trigger_video: If True, auto-start video capture when detected
            scenario_change_threshold: Number of consistent heartbeats before triggering scenario
        """
        self.auto_trigger_video = auto_trigger_video
        self.scenario_change_threshold = scenario_change_threshold
        
        self._state_manager = get_state_manager()
        self._lock = threading.Lock()
        
        # Browser context from extension
        self._browser_context: Dict[str, Any] = {
            "url": None,
            "title": None,
            "scenario": None,
            "timestamp": None,
            "idle_state": None
        }
        
        # Scenario tracking for change detection
        self._scenario_streak = 0
        self._last_scenario = None
        
        # Track processed URLs to prevent duplicate notifications
        # Key: (url, scenario) tuple, Value: {"title": str, "processed": bool, "triggered_at": datetime}
        self._captured_scenarios: Dict[tuple, Dict[str, Any]] = {}
        
        # Stats
        self._heartbeat_count = 0
        self._scenarios_triggered = 0
        self._last_heartbeat_time: Optional[datetime] = None
    
    def start(self) -> bool:
        """
        Enable the orchestrator to receive heartbeats.
        
        Returns:
            True if started successfully
        """
        if self._state_manager.monitoring_enabled:
            print("⚠️ Orchestrator already enabled")
            return False
        
        # Initialize database and directories
        init_db()
        ensure_media_dirs()
        
        self._state_manager.set_monitoring_enabled(True)
        print("🚀 Orchestrator enabled (waiting for browser extension heartbeats)")
        return True
    
    def stop(self) -> bool:
        """
        Disable the orchestrator.
        
        Returns:
            True if stopped successfully
        """
        if not self._state_manager.monitoring_enabled:
            return False
        
        self._state_manager.set_monitoring_enabled(False)
        
        # Clear browser context but preserve captured scenarios
        with self._lock:
            self._browser_context = {
                "url": None,
                "title": None,
                "scenario": None,
                "timestamp": None,
                "idle_state": None
            }
            self._scenario_streak = 0
            self._last_scenario = None
        
        self._state_manager.reset()
        
        print("⏹️ Orchestrator disabled")
        return True
    
    def handle_heartbeat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a heartbeat from the browser extension.
        
        Args:
            data: Heartbeat data with url, title, scenario, timestamp
            
        Returns:
            Response dict with status and optional command
        """
        self._heartbeat_count += 1
        self._last_heartbeat_time = datetime.utcnow()
        
        url = data.get("url")
        title = data.get("title")
        scenario = data.get("scenario", "OTHER")
        timestamp = data.get("timestamp")
        idle_state = data.get("idle_state")
        
        # Update browser context
        with self._lock:
            self._browser_context = {
                "url": url,
                "title": title,
                "scenario": scenario,
                "timestamp": timestamp,
                "idle_state": idle_state
            }
        
        # Build response
        response = {
            "status": "ok",
            "command": None,
            "current_state": self._state_manager.current_state.value
        }
        
        # Skip processing if not monitoring or idle
        if not self._state_manager.monitoring_enabled:
            return response
        
        if idle_state in ("locked", "idle"):
            # User is idle, reset streak
            self._scenario_streak = 0
            return response
        
        # Track scenario consistency
        if scenario == self._last_scenario:
            self._scenario_streak += 1
        else:
            self._scenario_streak = 1
            self._last_scenario = scenario
        
        # Check if we should trigger a scenario change
        if self._should_trigger_scenario(scenario, url):
            self._trigger_scenario(scenario, url, title)
            response["triggered_scenario"] = scenario
        
        return response
    
    def _should_trigger_scenario(self, scenario: str, url: str) -> bool:
        """Check if we should trigger a scenario based on current state."""
        # Only trigger from MONITORING state
        if not self._state_manager.is_monitoring:
            return False
        
        # Ignore OTHER scenarios
        if scenario == "OTHER":
            return False
        
        # Require consistent heartbeats
        if self._scenario_streak < self.scenario_change_threshold:
            return False
        
        # Check if this URL+scenario combination was already triggered
        scenario_key = (url, scenario)
        if scenario_key in self._captured_scenarios:
            return False
        
        return True
    
    def _trigger_scenario(self, scenario: str, url: str, title: str):
        """Trigger a pending scenario for user confirmation."""
        try:
            scenario_type = ScenarioType(scenario)
        except ValueError:
            print(f"⚠️ Unknown scenario type: {scenario}")
            return
        
        # Record this scenario as captured (pending processing)
        scenario_key = (url, scenario)
        self._captured_scenarios[scenario_key] = {
            "title": title,
            "processed": False,
            "triggered_at": datetime.utcnow()
        }
        
        pending = PendingScenario(
            scenario_type=scenario_type,
            confidence=1.0,  # Extension detection is binary, not probabilistic
            metadata={
                "url": url,
                "title": title,
                "source": "browser_extension"
            }
        )
        
        if self._state_manager.set_pending_scenario(pending):
            self._scenarios_triggered += 1
            print(f"🔔 Scenario triggered: {scenario} - {title or url}")
    
    @property
    def browser_context(self) -> Dict[str, Any]:
        """Get current browser context (thread-safe)."""
        with self._lock:
            return self._browser_context.copy()
    
    def get_captured_scenarios(self) -> list:
        """Get list of captured scenarios for frontend display."""
        scenarios = []
        for (url, scenario_type), data in self._captured_scenarios.items():
            scenarios.append({
                "url": url,
                "scenario": scenario_type,
                "title": data.get("title"),
                "processed": data.get("processed", False),
                "triggered_at": data.get("triggered_at").isoformat() if data.get("triggered_at") else None
            })
        # Sort by triggered_at descending (newest first)
        scenarios.sort(key=lambda x: x.get("triggered_at") or "", reverse=True)
        return scenarios
    
    def mark_scenario_processed(self, url: str, scenario: str) -> bool:
        """Mark a scenario as fully processed (no re-prompting)."""
        scenario_key = (url, scenario)
        if scenario_key in self._captured_scenarios:
            self._captured_scenarios[scenario_key]["processed"] = True
            return True
        return False
    
    def clear_captured_scenarios(self):
        """Clear all captured scenarios (allows re-detection)."""
        self._captured_scenarios.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "monitoring_enabled": self._state_manager.monitoring_enabled,
            "is_monitoring": self._state_manager.monitoring_enabled,
            "current_state": self._state_manager.current_state.value,
            "browser_context": self.browser_context,
            "heartbeat_count": self._heartbeat_count,
            "scenarios_triggered": self._scenarios_triggered,
            "last_heartbeat_time": self._last_heartbeat_time.isoformat() if self._last_heartbeat_time else None,
            "scenario_streak": self._scenario_streak,
            "captured_scenarios": self.get_captured_scenarios(),
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
        auto_trigger_video: Auto-start video capture
        scenario_change_threshold: Consistent heartbeats before triggering
    """
    global _orchestrator
    _orchestrator = Orchestrator(**kwargs)
    return _orchestrator


if __name__ == "__main__":
    # Quick test
    print("Testing orchestrator...")
    orch = get_orchestrator()
    print(f"Status: {orch.get_status()}")
    
    # Simulate heartbeats
    orch.start()
    
    result1 = orch.handle_heartbeat({
        "url": "https://youtube.com/watch?v=test",
        "title": "Test Video",
        "scenario": "VIDEO",
        "timestamp": 12345
    })
    print(f"Heartbeat 1: {result1}")
    
    result2 = orch.handle_heartbeat({
        "url": "https://youtube.com/watch?v=test",
        "title": "Test Video",
        "scenario": "VIDEO",
        "timestamp": 12346
    })
    print(f"Heartbeat 2: {result2}")
    
    print(f"Final status: {orch.get_status()}")
