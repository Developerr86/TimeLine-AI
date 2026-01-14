"""
Database models for the TimeLine Content Capture Pipeline.
Uses SQLAlchemy with SQLite backend.
"""

import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pathlib import Path

# Database setup
DATABASE_PATH = Path(__file__).parent / "timeline.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SessionType(str, Enum):
    """Types of capture sessions."""
    WEB = "WEB"
    DOC = "DOC"
    VIDEO = "VIDEO"


class MediaType(str, Enum):
    """Types of captured media."""
    AUDIO = "AUDIO"
    IMAGE = "IMAGE"
    FRAME = "FRAME"  # Video frame screenshot


class CaptureSession(Base):
    """
    Represents a single capture session.
    A session groups related captured content (text, images, audio) together.
    """
    __tablename__ = "capture_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type = Column(SQLEnum(SessionType), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    
    # Metadata
    title = Column(String(500), nullable=True)
    source_url = Column(Text, nullable=True)  # For WEB captures
    source_path = Column(Text, nullable=True)  # For DOC captures
    
    # Relationships
    texts = relationship("CapturedText", back_populates="session", cascade="all, delete-orphan")
    media = relationship("CapturedMedia", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CaptureSession(id={self.id}, type={self.type}, start_time={self.start_time})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value if self.type else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "title": self.title,
            "source_url": self.source_url,
            "source_path": self.source_path,
        }


class CapturedText(Base):
    """
    Represents captured text content from a session.
    Can be article text, OCR output, or transcribed audio.
    """
    __tablename__ = "captured_texts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("capture_sessions.id"), nullable=False)
    content = Column(Text, nullable=False)
    source_url_or_path = Column(Text, nullable=True)
    captured_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Metadata for different source types
    content_type = Column(String(50), nullable=True)  # e.g., "article", "ocr", "transcript"
    confidence = Column(Float, nullable=True)  # For OCR confidence scores
    
    # Relationship
    session = relationship("CaptureSession", back_populates="texts")

    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<CapturedText(id={self.id}, content={content_preview})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "source_url_or_path": self.source_url_or_path,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "content_type": self.content_type,
            "confidence": self.confidence,
        }


class CapturedMedia(Base):
    """
    Represents captured media files (images, audio, video frames).
    """
    __tablename__ = "captured_media"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("capture_sessions.id"), nullable=False)
    file_path = Column(Text, nullable=False)
    media_type = Column(SQLEnum(MediaType), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Metadata
    original_url = Column(Text, nullable=True)  # For downloaded images
    file_size_bytes = Column(String(20), nullable=True)
    duration_seconds = Column(Float, nullable=True)  # For audio files
    width = Column(String(10), nullable=True)  # For images
    height = Column(String(10), nullable=True)  # For images
    
    # Relationship
    session = relationship("CaptureSession", back_populates="media")

    def __repr__(self):
        return f"<CapturedMedia(id={self.id}, type={self.media_type}, path={self.file_path})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "file_path": self.file_path,
            "media_type": self.media_type.value if self.media_type else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "original_url": self.original_url,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
        }


def init_db():
    """
    Initialize the database, creating all tables if they don't exist.
    Call this on application startup.
    """
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized at {DATABASE_PATH}")


def get_db():
    """
    Get a database session.
    Use as a context manager or ensure close() is called.
    """
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def get_db_context():
    """
    Context manager for database sessions.
    Usage: with get_db_context() as db: ...
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Media directories setup
MEDIA_BASE_DIR = Path(__file__).parent / "media"
MEDIA_WEB_DIR = MEDIA_BASE_DIR / "web"
MEDIA_DOCS_DIR = MEDIA_BASE_DIR / "docs"
MEDIA_VIDEO_DIR = MEDIA_BASE_DIR / "video"


def ensure_media_dirs():
    """Create media directories if they don't exist."""
    MEDIA_BASE_DIR.mkdir(exist_ok=True)
    MEDIA_WEB_DIR.mkdir(exist_ok=True)
    MEDIA_DOCS_DIR.mkdir(exist_ok=True)
    MEDIA_VIDEO_DIR.mkdir(exist_ok=True)
    print(f"✅ Media directories initialized at {MEDIA_BASE_DIR}")


if __name__ == "__main__":
    # Quick test of database initialization
    init_db()
    ensure_media_dirs()
    print("Database and media directories ready!")
