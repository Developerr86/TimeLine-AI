"""
Tests for database models.
Run with: python -m pytest tests/test_models.py -v
"""

import pytest
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    init_db, get_db, Base, engine,
    CaptureSession, CapturedText, CapturedMedia,
    SessionType, MediaType
)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    db = get_db()
    yield db
    
    # Cleanup
    db.close()
    # Drop all tables after test
    Base.metadata.drop_all(bind=engine)


class TestCaptureSession:
    """Tests for CaptureSession model."""
    
    def test_create_web_session(self, db_session):
        """Test creating a web capture session."""
        session = CaptureSession(
            type=SessionType.WEB,
            title="Test Article",
            source_url="https://example.com/article"
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.id is not None
        assert session.type == SessionType.WEB
        assert session.title == "Test Article"
        assert session.source_url == "https://example.com/article"
        assert session.start_time is not None
        assert session.end_time is None
    
    def test_create_doc_session(self, db_session):
        """Test creating a document capture session."""
        session = CaptureSession(
            type=SessionType.DOC,
            title="Test PDF",
            source_path="/path/to/document.pdf"
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.type == SessionType.DOC
        assert session.source_path == "/path/to/document.pdf"
    
    def test_create_video_session(self, db_session):
        """Test creating a video capture session."""
        session = CaptureSession(type=SessionType.VIDEO)
        db_session.add(session)
        db_session.commit()
        
        assert session.type == SessionType.VIDEO
    
    def test_session_to_dict(self, db_session):
        """Test session serialization."""
        session = CaptureSession(
            type=SessionType.WEB,
            title="Test"
        )
        db_session.add(session)
        db_session.commit()
        
        data = session.to_dict()
        assert "id" in data
        assert data["type"] == "WEB"
        assert data["title"] == "Test"


class TestCapturedText:
    """Tests for CapturedText model."""
    
    def test_create_text(self, db_session):
        """Test creating captured text record."""
        session = CaptureSession(type=SessionType.WEB)
        db_session.add(session)
        db_session.commit()
        
        text = CapturedText(
            session_id=session.id,
            content="This is the captured article content.",
            source_url_or_path="https://example.com",
            content_type="article"
        )
        db_session.add(text)
        db_session.commit()
        
        assert text.id is not None
        assert text.session_id == session.id
        assert "article content" in text.content
    
    def test_text_relationship(self, db_session):
        """Test text-session relationship."""
        session = CaptureSession(type=SessionType.DOC)
        db_session.add(session)
        db_session.commit()
        
        text1 = CapturedText(session_id=session.id, content="Page 1 content")
        text2 = CapturedText(session_id=session.id, content="Page 2 content")
        db_session.add_all([text1, text2])
        db_session.commit()
        
        # Refresh session to load relationships
        db_session.refresh(session)
        assert len(session.texts) == 2


class TestCapturedMedia:
    """Tests for CapturedMedia model."""
    
    def test_create_image_media(self, db_session):
        """Test creating image media record."""
        session = CaptureSession(type=SessionType.WEB)
        db_session.add(session)
        db_session.commit()
        
        media = CapturedMedia(
            session_id=session.id,
            file_path="/media/web/session1/image.jpg",
            media_type=MediaType.IMAGE,
            original_url="https://example.com/image.jpg"
        )
        db_session.add(media)
        db_session.commit()
        
        assert media.id is not None
        assert media.media_type == MediaType.IMAGE
    
    def test_create_audio_media(self, db_session):
        """Test creating audio media record."""
        session = CaptureSession(type=SessionType.VIDEO)
        db_session.add(session)
        db_session.commit()
        
        media = CapturedMedia(
            session_id=session.id,
            file_path="/media/video/session1/audio.wav",
            media_type=MediaType.AUDIO,
            duration_seconds=120.5
        )
        db_session.add(media)
        db_session.commit()
        
        assert media.media_type == MediaType.AUDIO
        assert media.duration_seconds == 120.5
    
    def test_create_frame_media(self, db_session):
        """Test creating video frame media record."""
        session = CaptureSession(type=SessionType.VIDEO)
        db_session.add(session)
        db_session.commit()
        
        media = CapturedMedia(
            session_id=session.id,
            file_path="/media/video/session1/frames/frame001.jpg",
            media_type=MediaType.FRAME
        )
        db_session.add(media)
        db_session.commit()
        
        assert media.media_type == MediaType.FRAME


class TestCascadeDelete:
    """Test cascade delete behavior."""
    
    def test_delete_session_cascades(self, db_session):
        """Test that deleting a session deletes associated records."""
        session = CaptureSession(type=SessionType.WEB)
        db_session.add(session)
        db_session.commit()
        session_id = session.id
        
        text = CapturedText(session_id=session_id, content="Test content")
        media = CapturedMedia(
            session_id=session_id, 
            file_path="/test.jpg",
            media_type=MediaType.IMAGE
        )
        db_session.add_all([text, media])
        db_session.commit()
        
        # Delete session
        db_session.delete(session)
        db_session.commit()
        
        # Verify cascade
        assert db_session.query(CaptureSession).filter_by(id=session_id).first() is None
        assert db_session.query(CapturedText).filter_by(session_id=session_id).first() is None
        assert db_session.query(CapturedMedia).filter_by(session_id=session_id).first() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
