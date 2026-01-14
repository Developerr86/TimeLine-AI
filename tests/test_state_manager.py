"""
Tests for state manager.
Run with: python -m pytest tests/test_state_manager.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from state_manager import (
    StateManager, AppState, ScenarioType, 
    PendingScenario, ActiveCapture
)


@pytest.fixture
def state_manager():
    """Create a fresh state manager for each test."""
    # Create new instance (bypass singleton for testing)
    sm = object.__new__(StateManager)
    sm._initialized = False
    sm.__init__()
    return sm


class TestStateManager:
    """Tests for StateManager."""
    
    def test_initial_state(self, state_manager):
        """Test initial state is MONITORING."""
        assert state_manager.current_state == AppState.MONITORING
        assert not state_manager.is_capturing
        assert not state_manager.is_awaiting_input
        assert state_manager.is_monitoring
    
    def test_set_monitoring_enabled(self, state_manager):
        """Test enabling/disabling monitoring."""
        assert not state_manager.monitoring_enabled
        
        state_manager.set_monitoring_enabled(True)
        assert state_manager.monitoring_enabled
        
        state_manager.set_monitoring_enabled(False)
        assert not state_manager.monitoring_enabled
    
    def test_set_pending_scenario(self, state_manager):
        """Test setting a pending scenario."""
        scenario = PendingScenario(
            scenario_type=ScenarioType.WEB,
            confidence=0.85
        )
        
        result = state_manager.set_pending_scenario(scenario)
        
        assert result is True
        assert state_manager.is_awaiting_input
        assert state_manager.pending_scenario is not None
        assert state_manager.pending_scenario.scenario_type == ScenarioType.WEB
    
    def test_cannot_set_pending_while_capturing(self, state_manager):
        """Test that pending scenario can't be set while capturing."""
        # Start a capture
        state_manager.start_capture("test-session", ScenarioType.VIDEO)
        
        # Try to set pending scenario
        scenario = PendingScenario(scenario_type=ScenarioType.WEB, confidence=0.9)
        result = state_manager.set_pending_scenario(scenario)
        
        assert result is False
        assert state_manager.pending_scenario is None
    
    def test_confirm_scenario(self, state_manager):
        """Test confirming a pending scenario."""
        scenario = PendingScenario(
            scenario_type=ScenarioType.DOC,
            confidence=0.75
        )
        state_manager.set_pending_scenario(scenario)
        
        confirmed = state_manager.confirm_scenario("session-123")
        
        assert confirmed is not None
        assert confirmed.scenario_type == ScenarioType.DOC
        assert state_manager.current_state == AppState.CAPTURING_DOC
        assert state_manager.active_capture is not None
        assert state_manager.active_capture.session_id == "session-123"
    
    def test_dismiss_scenario(self, state_manager):
        """Test dismissing a pending scenario."""
        scenario = PendingScenario(
            scenario_type=ScenarioType.VIDEO,
            confidence=0.8
        )
        state_manager.set_pending_scenario(scenario)
        assert state_manager.is_awaiting_input
        
        state_manager.dismiss_scenario()
        
        assert state_manager.is_monitoring
        assert state_manager.pending_scenario is None
    
    def test_start_capture_directly(self, state_manager):
        """Test starting a capture directly (without scenario detection)."""
        result = state_manager.start_capture(
            "session-456", 
            ScenarioType.WEB,
            {"url": "https://example.com"}
        )
        
        assert result is True
        assert state_manager.current_state == AppState.CAPTURING_WEB
        assert state_manager.active_capture.session_id == "session-456"
        assert state_manager.active_capture.metadata.get("url") == "https://example.com"
    
    def test_cannot_start_capture_while_capturing(self, state_manager):
        """Test that capture can't start while already capturing."""
        state_manager.start_capture("session-1", ScenarioType.VIDEO)
        
        result = state_manager.start_capture("session-2", ScenarioType.WEB)
        
        assert result is False
        assert state_manager.active_capture.session_id == "session-1"
    
    def test_end_capture(self, state_manager):
        """Test ending a capture session."""
        state_manager.start_capture("session-789", ScenarioType.VIDEO)
        assert state_manager.is_capturing
        
        ended = state_manager.end_capture()
        
        assert ended is not None
        assert ended.session_id == "session-789"
        assert state_manager.is_monitoring
        assert state_manager.active_capture is None
    
    def test_reset(self, state_manager):
        """Test resetting state manager."""
        state_manager.set_monitoring_enabled(True)
        state_manager.start_capture("session-test", ScenarioType.DOC)
        
        state_manager.reset()
        
        assert state_manager.is_monitoring
        assert not state_manager.monitoring_enabled
        assert state_manager.pending_scenario is None
        assert state_manager.active_capture is None
    
    def test_get_status(self, state_manager):
        """Test status dictionary."""
        status = state_manager.get_status()
        
        assert "state" in status
        assert "monitoring_enabled" in status
        assert "last_state_change" in status
        assert "pending_scenario" in status
        assert "active_capture" in status


class TestStateTransitions:
    """Test valid and invalid state transitions."""
    
    def test_valid_transitions(self, state_manager):
        """Test valid state transition sequence."""
        # MONITORING -> AWAITING_USER_INPUT
        scenario = PendingScenario(scenario_type=ScenarioType.WEB, confidence=0.9)
        assert state_manager.set_pending_scenario(scenario)
        assert state_manager.current_state == AppState.AWAITING_USER_INPUT
        
        # AWAITING_USER_INPUT -> CAPTURING_WEB
        state_manager.confirm_scenario("session-test")
        assert state_manager.current_state == AppState.CAPTURING_WEB
        
        # CAPTURING_WEB -> MONITORING
        state_manager.end_capture()
        assert state_manager.current_state == AppState.MONITORING
    
    def test_invalid_transition(self, state_manager):
        """Test invalid state transition."""
        # Try to transition from MONITORING directly to CAPTURING (without AWAITING)
        # This should fail via transition_to
        result = state_manager.transition_to(AppState.CAPTURING_VIDEO)
        assert result is False
        assert state_manager.current_state == AppState.MONITORING


class TestPendingScenario:
    """Tests for PendingScenario dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        scenario = PendingScenario(
            scenario_type=ScenarioType.DOC,
            confidence=0.82,
            metadata={"pages": 5}
        )
        
        data = scenario.to_dict()
        
        assert data["scenario_type"] == "DOC"
        assert data["confidence"] == 0.82
        assert data["metadata"]["pages"] == 5
        assert "detected_at" in data


class TestActiveCapture:
    """Tests for ActiveCapture dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        capture = ActiveCapture(
            session_id="abc-123",
            capture_type=ScenarioType.VIDEO,
            metadata={"frames": 42}
        )
        
        data = capture.to_dict()
        
        assert data["session_id"] == "abc-123"
        assert data["capture_type"] == "VIDEO"
        assert data["metadata"]["frames"] == 42
        assert "started_at" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
