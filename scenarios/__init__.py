"""
Scenarios package for the TimeLine Content Capture Pipeline.
Contains modules for web, document, and video capture scenarios.
"""

from .web_capture import WebCaptureHandler
from .doc_capture import DocCaptureHandler
from .video_capture import VideoCaptureHandler

__all__ = ['WebCaptureHandler', 'DocCaptureHandler', 'VideoCaptureHandler']
