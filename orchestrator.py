"""
Orchestrator for the TimeLine Content Capture Pipeline.
Event-driven orchestrator that receives heartbeats from the browser extension.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List

from state_manager import (
    get_state_manager, StateManager, AppState, 
    ScenarioType, PendingScenario
)
from models import (
    init_db, ensure_media_dirs, get_db,
    CaptureSession, SessionType, MEDIA_BASE_DIR
)


@dataclass
class PendingActivity:
    """Represents a queued activity waiting for user processing."""
    id: str
    scenario: str  # VIDEO, DOC, WEB
    url: str
    title: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False
    processing: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)  # Store snapshot_path, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scenario": self.scenario,
            "url": self.url,
            "title": self.title,
            "detected_at": self.detected_at.isoformat(),
            "processed": self.processed,
            "processing": self.processing,
            "metadata": self.metadata
        }


class Orchestrator:
    """
    Main orchestrator for the Content Capture Pipeline.
    
    Responsibilities:
    - Receive heartbeats from browser extension
    - Detect scenario changes (Web/Doc/Video)
    - Queue activities for user processing (instead of auto-processing)
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
        
        # NEW: Pending activities queue (activities waiting for user to process)
        self._pending_activities: List[PendingActivity] = []
        self._current_activity: Optional[PendingActivity] = None
        
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
        """Queue a new activity for user processing (instead of auto-triggering)."""
        try:
            scenario_type = ScenarioType(scenario)
        except ValueError:
            print(f"⚠️ Unknown scenario type: {scenario}")
            return
        
        # Check if this URL is already in pending activities
        with self._lock:
            for activity in self._pending_activities:
                if activity.url == url and activity.scenario == scenario:
                    print(f"⏭️ Activity already queued: {scenario} - {title or url}")
                    return
        
        # Record this scenario as captured (pending processing)
        scenario_key = (url, scenario)
        self._captured_scenarios[scenario_key] = {
            "title": title,
            "processed": False,
            "triggered_at": datetime.utcnow()
        }
        
        # Create and queue the new activity
        activity = PendingActivity(
            id=str(uuid.uuid4()),
            scenario=scenario,
            url=url,
            title=title or url,
            detected_at=datetime.utcnow()
        )
        
        with self._lock:
            self._pending_activities.append(activity)
        
        self._scenarios_triggered += 1
        print(f"📥 Activity queued: {scenario} - {title or url} (ID: {activity.id[:8]}...)")
    
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
    
    # =========================================================================
    # Activity Queue Management (NEW)
    # =========================================================================
    
    def get_pending_activities(self) -> List[Dict[str, Any]]:
        """Get list of pending activities for frontend display."""
        with self._lock:
            return [activity.to_dict() for activity in self._pending_activities]
    
    def get_activity_by_id(self, activity_id: str) -> Optional[PendingActivity]:
        """Get a specific activity by ID."""
        with self._lock:
            for activity in self._pending_activities:
                if activity.id == activity_id:
                    return activity
        return None
    
    def get_activity_by_url(self, url: str, scenario: str = None) -> Optional[PendingActivity]:
        """Get a specific activity by URL (and optionally scenario)."""
        with self._lock:
            for activity in self._pending_activities:
                if activity.url == url:
                    if scenario is None or activity.scenario == scenario:
                        return activity
        return None
    
    def update_activity_metadata(self, activity_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing activity.
        
        Args:
            activity_id: The activity ID to update
            metadata: Dict of metadata to merge
            
        Returns:
            True if activity was found and updated
        """
        with self._lock:
            for activity in self._pending_activities:
                if activity.id == activity_id:
                    activity.metadata.update(metadata)
                    print(f"📎 Updated activity metadata: {activity_id[:8]}... -> {list(metadata.keys())}")
                    return True
        return False
    
    def start_activity_processing(self, activity_id: str) -> Dict[str, Any]:
        """
        Start processing a specific activity.
        Transitions state from MONITORING to PROCESSING, then to CAPTURING_VIDEO.
        
        Args:
            activity_id: The ID of the activity to process
            
        Returns:
            Dict with status and any errors
        """
        result = {
            "success": False,
            "activity_id": activity_id,
            "errors": []
        }
        
        # Find the activity
        activity = self.get_activity_by_id(activity_id)
        if not activity:
            result["errors"].append("Activity not found")
            return result
        
        if activity.processing:
            result["errors"].append("Activity is already being processed")
            return result
        
        if activity.processed:
            result["errors"].append("Activity has already been processed")
            return result
        
        # Transition state to PROCESSING
        if not self._state_manager.transition_to(AppState.PROCESSING):
            result["errors"].append(f"Cannot transition to PROCESSING from {self._state_manager.current_state.value}")
            return result
        
        # Mark activity as processing
        with self._lock:
            activity.processing = True
            self._current_activity = activity
        
        result["success"] = True
        result["activity"] = activity.to_dict()
        print(f"▶️ Started processing activity: {activity.scenario} - {activity.title}")
        
        return result
    
    def stop_activity_processing(self) -> Dict[str, Any]:
        """
        Stop processing the current activity and return to MONITORING.
        
        Returns:
            Dict with status and activity info
        """
        result = {
            "success": False,
            "activity": None,
            "errors": []
        }
        
        with self._lock:
            if self._current_activity is None:
                result["errors"].append("No activity is currently being processed")
                return result
            
            activity = self._current_activity
            activity.processing = False
            activity.processed = True
            
            # Remove from pending list
            self._pending_activities = [a for a in self._pending_activities if a.id != activity.id]
            
            result["activity"] = activity.to_dict()
            self._current_activity = None
        
        # Transition back to MONITORING
        self._state_manager.reset()  # Force reset to MONITORING
        
        result["success"] = True
        print(f"⏹️ Stopped processing activity: {activity.scenario} - {activity.title}")
        
        return result
    
    def dismiss_activity(self, activity_id: str) -> bool:
        """
        Dismiss an activity from the pending list without processing.
        
        Args:
            activity_id: The ID of the activity to dismiss
            
        Returns:
            True if activity was found and dismissed
        """
        with self._lock:
            for activity in self._pending_activities:
                if activity.id == activity_id:
                    self._pending_activities.remove(activity)
                    # Also remove from captured scenarios to allow re-detection
                    scenario_key = (activity.url, activity.scenario)
                    if scenario_key in self._captured_scenarios:
                        del self._captured_scenarios[scenario_key]
                    print(f"🗑️ Dismissed activity: {activity.scenario} - {activity.title}")
                    return True
        return False
    
    def clear_pending_activities(self):
        """Clear all pending activities."""
        with self._lock:
            self._pending_activities.clear()
            self._current_activity = None
        self._captured_scenarios.clear()
        print("🗑️ Cleared all pending activities")
    
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
            # NEW: Activity queue status
            "pending_activities": self.get_pending_activities(),
            "pending_activities_count": len(self._pending_activities),
            "current_activity": self._current_activity.to_dict() if self._current_activity else None,
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
