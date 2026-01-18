"""
State Manager for the TimeLine Content Capture Pipeline.
Implements a thread-safe singleton to track application state.
"""

import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


class AppState(str, Enum):
    """Application states for the orchestrator."""
    MONITORING = "MONITORING"                   # Default loop, VLM checking screen
    AWAITING_USER_INPUT = "AWAITING_USER_INPUT" # Scenario detected, waiting for confirmation
    CAPTURING_WEB = "CAPTURING_WEB"             # Actively scraping a URL
    CAPTURING_DOC = "CAPTURING_DOC"             # Actively processing a document
    CAPTURING_VIDEO = "CAPTURING_VIDEO"         # Actively recording video/audio


class ScenarioType(str, Enum):
    """Types of detected scenarios from VLM."""
    WEB = "WEB"
    DOC = "DOC"
    VIDEO = "VIDEO"
    OTHER = "OTHER"


@dataclass
class PendingScenario:
    """Data class for a scenario awaiting user confirmation."""
    scenario_type: ScenarioType
    confidence: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    screenshot_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type.value,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
            "screenshot_path": self.screenshot_path,
        }


@dataclass
class ActiveCapture:
    """Data class for an active capture session."""
    session_id: str
    capture_type: ScenarioType
    started_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "capture_type": self.capture_type.value,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
        }


class StateManager:
    """
    Thread-safe singleton state manager for the application.
    Tracks current state, pending scenarios, and active captures.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._state_lock = threading.RLock()
        self._current_state: AppState = AppState.MONITORING
        self._pending_scenario: Optional[PendingScenario] = None
        self._active_capture: Optional[ActiveCapture] = None
        self._last_state_change: datetime = datetime.utcnow()
        self._monitoring_enabled: bool = False
        self._initialized = True
        
        print("✅ StateManager initialized")
    
    @property
    def current_state(self) -> AppState:
        """Get the current application state."""
        with self._state_lock:
            return self._current_state
    
    @property
    def is_monitoring(self) -> bool:
        """Check if currently in monitoring state."""
        with self._state_lock:
            return self._current_state == AppState.MONITORING
    
    @property
    def is_capturing(self) -> bool:
        """Check if currently in any capturing state."""
        with self._state_lock:
            return self._current_state in (
                AppState.CAPTURING_WEB,
                AppState.CAPTURING_DOC,
                AppState.CAPTURING_VIDEO
            )
    
    @property
    def is_awaiting_input(self) -> bool:
        """Check if awaiting user input."""
        with self._state_lock:
            return self._current_state == AppState.AWAITING_USER_INPUT
    
    @property
    def pending_scenario(self) -> Optional[PendingScenario]:
        """Get the pending scenario if any."""
        with self._state_lock:
            return self._pending_scenario
    
    @property
    def active_capture(self) -> Optional[ActiveCapture]:
        """Get the active capture if any."""
        with self._state_lock:
            return self._active_capture
    
    @property
    def monitoring_enabled(self) -> bool:
        """Check if monitoring loop is enabled."""
        with self._state_lock:
            return self._monitoring_enabled
    
    def set_monitoring_enabled(self, enabled: bool):
        """Enable or disable the monitoring loop."""
        with self._state_lock:
            self._monitoring_enabled = enabled
            print(f"🔄 Monitoring {'enabled' if enabled else 'disabled'}")
    
    def transition_to(self, new_state: AppState) -> bool:
        """
        Attempt to transition to a new state.
        Returns True if transition was successful, False otherwise.
        """
        with self._state_lock:
            old_state = self._current_state
            
            # Validate transitions
            valid_transitions = {
                AppState.MONITORING: [AppState.AWAITING_USER_INPUT],
                AppState.AWAITING_USER_INPUT: [
                    AppState.MONITORING,
                    AppState.CAPTURING_WEB,
                    AppState.CAPTURING_DOC,
                    AppState.CAPTURING_VIDEO
                ],
                AppState.CAPTURING_WEB: [AppState.MONITORING],
                AppState.CAPTURING_DOC: [AppState.MONITORING],
                AppState.CAPTURING_VIDEO: [AppState.MONITORING],
            }
            
            if new_state not in valid_transitions.get(old_state, []):
                print(f"⚠️ Invalid state transition: {old_state} -> {new_state}")
                return False
            
            self._current_state = new_state
            self._last_state_change = datetime.utcnow()
            print(f"🔄 State transition: {old_state.value} -> {new_state.value}")
            return True
    
    def set_pending_scenario(self, scenario: PendingScenario) -> bool:
        """
        Set a pending scenario and transition to AWAITING_USER_INPUT.
        Only works if currently in MONITORING state.
        """
        with self._state_lock:
            if self._current_state != AppState.MONITORING:
                print(f"⚠️ Cannot set pending scenario while in {self._current_state}")
                return False
            
            self._pending_scenario = scenario
            self._current_state = AppState.AWAITING_USER_INPUT
            self._last_state_change = datetime.utcnow()
            print(f"📋 Pending scenario set: {scenario.scenario_type.value} ({scenario.confidence:.2f})")
            return True
    
    def confirm_scenario(self, session_id: str) -> Optional[PendingScenario]:
        """
        Confirm the pending scenario and start capture.
        Returns the confirmed scenario for processing.
        """
        with self._state_lock:
            if self._current_state != AppState.AWAITING_USER_INPUT:
                print(f"⚠️ No pending scenario to confirm")
                return None
            
            if self._pending_scenario is None:
                return None
            
            scenario = self._pending_scenario
            
            # Map scenario type to capture state
            state_map = {
                ScenarioType.WEB: AppState.CAPTURING_WEB,
                ScenarioType.DOC: AppState.CAPTURING_DOC,
                ScenarioType.VIDEO: AppState.CAPTURING_VIDEO,
            }
            
            new_state = state_map.get(scenario.scenario_type)
            if new_state is None:
                print(f"⚠️ Cannot capture scenario type: {scenario.scenario_type}")
                self.dismiss_scenario()
                return None
            
            self._current_state = new_state
            self._active_capture = ActiveCapture(
                session_id=session_id,
                capture_type=scenario.scenario_type,
                metadata=scenario.metadata
            )
            self._pending_scenario = None
            self._last_state_change = datetime.utcnow()
            
            print(f"✅ Scenario confirmed, starting {new_state.value}")
            return scenario
    
    def dismiss_scenario(self):
        """Dismiss the pending scenario and return to monitoring after a delay."""
        with self._state_lock:
            if self._pending_scenario:
                print(f"❌ Dismissed pending scenario: {self._pending_scenario.scenario_type.value}")
            
            self._pending_scenario = None
            self._current_state = AppState.MONITORING
            self._last_state_change = datetime.utcnow()
        
        # Delay before resuming monitoring to prevent immediate re-triggering
        print("⏳ Waiting 3 seconds before resuming monitoring...")
        import time
        time.sleep(3)
    
    def start_capture(self, session_id: str, capture_type: ScenarioType, metadata: Dict[str, Any] = None):
        """
        Start a capture session directly (bypassing scenario detection).
        Used when user manually triggers capture.
        """
        with self._state_lock:
            if self.is_capturing:
                print(f"⚠️ Already capturing, cannot start new session")
                return False
            
            state_map = {
                ScenarioType.WEB: AppState.CAPTURING_WEB,
                ScenarioType.DOC: AppState.CAPTURING_DOC,
                ScenarioType.VIDEO: AppState.CAPTURING_VIDEO,
            }
            
            new_state = state_map.get(capture_type)
            if new_state is None:
                return False
            
            self._current_state = new_state
            self._active_capture = ActiveCapture(
                session_id=session_id,
                capture_type=capture_type,
                metadata=metadata or {}
            )
            self._pending_scenario = None
            self._last_state_change = datetime.utcnow()
            
            print(f"▶️ Started capture: {capture_type.value} (session: {session_id})")
            return True
    
    def end_capture(self) -> Optional[ActiveCapture]:
        """
        End the current capture session and return to monitoring.
        Returns the ended capture for finalization.
        """
        with self._state_lock:
            if not self.is_capturing:
                return None
            
            capture = self._active_capture
            self._active_capture = None
            self._current_state = AppState.MONITORING
            self._last_state_change = datetime.utcnow()
            
            if capture:
                print(f"⏹️ Ended capture: {capture.capture_type.value}")
        
        # Delay before resuming monitoring to prevent immediate re-triggering
        print("⏳ Waiting 3 seconds before resuming monitoring...")
        import time
        time.sleep(3)
        
        return capture
    
    def get_status(self) -> Dict[str, Any]:
        """Get the full status for API responses."""
        with self._state_lock:
            return {
                "state": self._current_state.value,
                "monitoring_enabled": self._monitoring_enabled,
                "last_state_change": self._last_state_change.isoformat(),
                "pending_scenario": self._pending_scenario.to_dict() if self._pending_scenario else None,
                "active_capture": self._active_capture.to_dict() if self._active_capture else None,
            }
    
    def reset(self):
        """Reset state manager to initial state."""
        with self._state_lock:
            self._current_state = AppState.MONITORING
            self._pending_scenario = None
            self._active_capture = None
            self._last_state_change = datetime.utcnow()
            self._monitoring_enabled = False
            print("🔄 StateManager reset")


# Global singleton instance
state_manager = StateManager()


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    return state_manager


if __name__ == "__main__":
    # Quick test
    sm = get_state_manager()
    print(f"Initial state: {sm.current_state}")
    print(f"Status: {sm.get_status()}")
