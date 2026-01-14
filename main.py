"""
TimeLine Content Capture Pipeline - Flask API Server

This is the main entry point for the Python backend.
It provides REST API endpoints for the Electron frontend.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import uuid
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

# Local modules
from models import (
    init_db, ensure_media_dirs, get_db, 
    CaptureSession, CapturedText, CapturedMedia,
    SessionType, MediaType, MEDIA_BASE_DIR,
    MEDIA_WEB_DIR, MEDIA_DOCS_DIR, MEDIA_VIDEO_DIR
)
from state_manager import get_state_manager, AppState, ScenarioType
from orchestrator import get_orchestrator, configure_orchestrator
from scenarios.web_capture import WebCaptureHandler
from scenarios.doc_capture import DocCaptureHandler
from scenarios.video_capture import VideoCaptureHandler
from media_listener import get_media_listener, MediaListener

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Allowed file extensions for document upload
ALLOWED_DOC_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_DOC_EXTENSIONS


# ============================================================================
# Status & State Endpoints
# ============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get full application status including orchestrator and state manager status.
    The frontend should poll this endpoint to detect pending scenarios.
    """
    state_mgr = get_state_manager()
    orch = get_orchestrator()
    
    # Get session counts from database
    db = get_db()
    try:
        session_count = db.query(CaptureSession).count()
        text_count = db.query(CapturedText).count()
        media_count = db.query(CapturedMedia).count()
    finally:
        db.close()
    
    return jsonify({
        "state": state_mgr.get_status(),
        "orchestrator": orch.get_status(),
        "database": {
            "sessions": session_count,
            "texts": text_count,
            "media": media_count
        }
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get just the state manager status (lightweight polling endpoint)."""
    return jsonify(get_state_manager().get_status())


# ============================================================================
# Monitoring Control Endpoints
# ============================================================================

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start the monitoring loop."""
    try:
        data = request.get_json(silent=True) or {}
        
        # Optional configuration
        interval = data.get('interval', 20)
        confidence = data.get('confidence_threshold', 0.7)
        vlm_url = data.get('vlm_url')
        
        orch = configure_orchestrator(
            monitoring_interval=interval,
            confidence_threshold=confidence,
            vlm_url=vlm_url
        ) if any([interval != 20, confidence != 0.7, vlm_url]) else get_orchestrator()
        
        if orch.start():
            return jsonify({'status': 'started', 'config': orch.get_status()})
        else:
            # Already running is not an error - return success with current status
            return jsonify({'status': 'already_running', 'config': orch.get_status()})
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Error in /api/start: {error_msg}")
        return jsonify({'status': 'error', 'message': str(e), 'details': error_msg}), 500


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop the monitoring loop."""
    orch = get_orchestrator()
    
    if orch.stop():
        return jsonify({'status': 'stopped'})
    else:
        return jsonify({'status': 'not_running'}), 400


@app.route('/api/analyze', methods=['POST'])
def force_analyze():
    """Force an immediate screenshot and VLM analysis."""
    orch = get_orchestrator()
    result = orch.force_analyze()
    
    if result:
        return jsonify({'status': 'success', 'analysis': result})
    else:
        return jsonify({'status': 'error', 'message': 'Analysis failed'}), 500


# ============================================================================
# Scenario Management Endpoints
# ============================================================================

@app.route('/api/scenario/confirm', methods=['POST'])
def confirm_scenario():
    """
    Confirm the pending scenario and start capture.
    The frontend calls this when user confirms the detected scenario.
    """
    state_mgr = get_state_manager()
    
    if not state_mgr.is_awaiting_input:
        return jsonify({
            'status': 'error', 
            'message': 'No pending scenario to confirm'
        }), 400
    
    pending = state_mgr.pending_scenario
    if not pending:
        return jsonify({'status': 'error', 'message': 'No pending scenario'}), 400
    
    # Create database session
    db = get_db()
    try:
        session_type = SessionType(pending.scenario_type.value)
        session = CaptureSession(
            id=str(uuid.uuid4()),
            type=session_type
        )
        db.add(session)
        db.commit()
        session_id = session.id
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()
    
    # Confirm in state manager
    confirmed = state_mgr.confirm_scenario(session_id)
    
    return jsonify({
        'status': 'confirmed',
        'session_id': session_id,
        'scenario_type': pending.scenario_type.value
    })


@app.route('/api/scenario/dismiss', methods=['POST'])
def dismiss_scenario():
    """Dismiss the pending scenario and return to monitoring."""
    state_mgr = get_state_manager()
    state_mgr.dismiss_scenario()
    
    return jsonify({'status': 'dismissed'})


# ============================================================================
# Web Capture Endpoints
# ============================================================================

@app.route('/api/capture/web', methods=['POST'])
def capture_web():
    """
    Capture content from a web URL.
    Request body: { "url": "https://example.com/article" }
    """
    data = request.get_json()
    if not data or not data.get('url'):
        return jsonify({'status': 'error', 'message': 'URL required'}), 400
    
    url = data['url']
    state_mgr = get_state_manager()
    
    # Create database session
    db = get_db()
    try:
        session = CaptureSession(
            id=str(uuid.uuid4()),
            type=SessionType.WEB,
            source_url=url
        )
        db.add(session)
        db.commit()
        session_id = session.id
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()
    
    # Start capture
    state_mgr.start_capture(session_id, ScenarioType.WEB, {'url': url})
    
    try:
        handler = WebCaptureHandler()
        result = handler.capture(url, session_id)
        
        # Update session end time
        db = get_db()
        try:
            session = db.query(CaptureSession).filter_by(id=session_id).first()
            if session:
                session.end_time = datetime.utcnow()
                db.commit()
        finally:
            db.close()
        
        return jsonify(result)
        
    finally:
        # Always return to monitoring state
        state_mgr.end_capture()


# ============================================================================
# Document Capture Endpoints
# ============================================================================

@app.route('/api/capture/doc', methods=['POST'])
def capture_doc():
    """
    Capture content from an uploaded document.
    Expects multipart/form-data with 'file' field.
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error', 
            'message': f'Unsupported file type. Allowed: {", ".join(ALLOWED_DOC_EXTENSIONS)}'
        }), 400
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = UPLOAD_FOLDER / filename
    file.save(temp_path)
    
    state_mgr = get_state_manager()
    
    # Create database session
    db = get_db()
    try:
        session = CaptureSession(
            id=str(uuid.uuid4()),
            type=SessionType.DOC,
            source_path=str(temp_path)
        )
        db.add(session)
        db.commit()
        session_id = session.id
    except Exception as e:
        db.rollback()
        temp_path.unlink(missing_ok=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()
    
    # Start capture
    state_mgr.start_capture(session_id, ScenarioType.DOC, {'filename': filename})
    
    try:
        handler = DocCaptureHandler()
        result = handler.capture(str(temp_path), session_id, filename)
        
        # Update session end time
        db = get_db()
        try:
            session = db.query(CaptureSession).filter_by(id=session_id).first()
            if session:
                session.end_time = datetime.utcnow()
                db.commit()
        finally:
            db.close()
        
        # Clean up temp file (original is copied to media dir)
        temp_path.unlink(missing_ok=True)
        
        return jsonify(result)
        
    finally:
        state_mgr.end_capture()


@app.route('/api/capture/doc/capabilities', methods=['GET'])
def doc_capabilities():
    """Get document capture capabilities (what's installed)."""
    return jsonify(DocCaptureHandler.get_capabilities())


# ============================================================================
# Video Capture Endpoints
# ============================================================================

# Global video capture handler (needs to persist between start/stop calls)
_video_handler: VideoCaptureHandler = None


def get_video_handler() -> VideoCaptureHandler:
    global _video_handler
    if _video_handler is None:
        media_listener = get_media_listener()
        _video_handler = VideoCaptureHandler(media_listener=media_listener)
    return _video_handler


@app.route('/api/capture/video/start', methods=['POST'])
def start_video_capture():
    """Start video/lecture capture (screenshots + audio recording)."""
    state_mgr = get_state_manager()
    
    if state_mgr.is_capturing:
        return jsonify({
            'status': 'error',
            'message': f'Already capturing: {state_mgr.current_state.value}'
        }), 400
    
    # Create database session
    db = get_db()
    try:
        session = CaptureSession(
            id=str(uuid.uuid4()),
            type=SessionType.VIDEO
        )
        db.add(session)
        db.commit()
        session_id = session.id
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()
    
    # Start capture
    state_mgr.start_capture(session_id, ScenarioType.VIDEO)
    
    handler = get_video_handler()
    result = handler.start(session_id)
    
    if not result.get('success'):
        state_mgr.end_capture()
        return jsonify(result), 500
    
    return jsonify(result)


@app.route('/api/capture/video/stop', methods=['POST'])
def stop_video_capture():
    """Stop video/lecture capture."""
    state_mgr = get_state_manager()
    
    if state_mgr.current_state != AppState.CAPTURING_VIDEO:
        return jsonify({
            'status': 'error',
            'message': 'Not currently recording video'
        }), 400
    
    handler = get_video_handler()
    result = handler.stop()
    
    # Update session end time
    if result.get('session_id'):
        db = get_db()
        try:
            session = db.query(CaptureSession).filter_by(id=result['session_id']).first()
            if session:
                session.end_time = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    state_mgr.end_capture()
    
    return jsonify(result)


@app.route('/api/capture/video/status', methods=['GET'])
def video_capture_status():
    """Get current video capture status."""
    handler = get_video_handler()
    media_listener = get_media_listener()
    
    return jsonify({
        'capture': handler.get_status(),
        'media': media_listener.get_media_info(),
        'recording': media_listener.get_recording_status(),
        'capabilities': MediaListener.get_capabilities()
    })


@app.route('/api/capture/video/transcribe', methods=['POST'])
def transcribe_video():
    """
    Trigger transcription for a video session.
    Request body: { "session_id": "uuid-here" }
    
    This should be called after stopping video capture to transcribe
    the recorded audio. Transcription runs synchronously and may take
    a while depending on audio length.
    """
    data = request.get_json()
    if not data or not data.get('session_id'):
        return jsonify({'status': 'error', 'message': 'session_id required'}), 400
    
    session_id = data['session_id']
    
    # Verify session exists
    db = get_db()
    try:
        session = db.query(CaptureSession).filter_by(id=session_id).first()
        if not session:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404
        if session.type != SessionType.VIDEO:
            return jsonify({'status': 'error', 'message': 'Not a video session'}), 400
    finally:
        db.close()
    
    # Run transcription
    handler = get_video_handler()
    result = handler.transcribe_session(session_id)
    
    if result.get('success'):
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'transcript_length': len(result.get('transcript', '')),
            'duration_seconds': result.get('duration_seconds'),
            'message': 'Transcription complete'
        })
    else:
        return jsonify({
            'status': 'error',
            'session_id': session_id,
            'errors': result.get('errors', [])
        }), 500


@app.route('/api/capture/video/capabilities', methods=['GET'])
def video_capture_capabilities():
    """Get video capture capabilities (what's installed)."""
    from scenarios.video_capture import VideoCaptureHandler
    return jsonify(VideoCaptureHandler.get_capabilities())


# ============================================================================
# Session & Data Endpoints
# ============================================================================

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all capture sessions."""
    db = get_db()
    try:
        sessions = db.query(CaptureSession).order_by(
            CaptureSession.start_time.desc()
        ).limit(100).all()
        
        return jsonify([s.to_dict() for s in sessions])
    finally:
        db.close()


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get details for a specific session including captured content."""
    db = get_db()
    try:
        session = db.query(CaptureSession).filter_by(id=session_id).first()
        if not session:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404
        
        texts = db.query(CapturedText).filter_by(session_id=session_id).all()
        media = db.query(CapturedMedia).filter_by(session_id=session_id).all()
        
        return jsonify({
            'session': session.to_dict(),
            'texts': [t.to_dict() for t in texts],
            'media': [m.to_dict() for m in media]
        })
    finally:
        db.close()


@app.route('/api/responses', methods=['GET'])
def api_responses():
    """
    Legacy endpoint for backward compatibility with old frontend.
    Returns sessions in the old responses.json format.
    """
    db = get_db()
    try:
        sessions = db.query(CaptureSession).order_by(
            CaptureSession.start_time.desc()
        ).limit(100).all()
        
        # Convert to old format
        responses = []
        for session in sessions:
            # Get first text content if available
            text = db.query(CapturedText).filter_by(session_id=session.id).first()
            
            # Get first media/screenshot if available
            media = db.query(CapturedMedia).filter_by(
                session_id=session.id,
                media_type=MediaType.IMAGE
            ).first()
            
            response = {
                'timestamp': session.start_time.isoformat() if session.start_time else None,
                'title': session.title or 'Captured Content',
                'summary': text.content[:200] + '...' if text and len(text.content) > 200 else (text.content if text else 'No content'),
                'model': 'database',
                'model_name': f'{session.type.value} capture',
                'image_path': media.file_path if media else None,
                'session_id': session.id
            }
            responses.append(response)
        
        return jsonify(responses)
    finally:
        db.close()


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """Delete a capture session and all associated data."""
    db = get_db()
    try:
        session = db.query(CaptureSession).filter_by(id=session_id).first()
        if not session:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404
        
        # Delete associated media files
        media_items = db.query(CapturedMedia).filter_by(session_id=session_id).all()
        for item in media_items:
            try:
                Path(item.file_path).unlink(missing_ok=True)
            except Exception:
                pass
        
        # Delete session (cascades to texts and media records)
        db.delete(session)
        db.commit()
        
        return jsonify({'status': 'deleted', 'session_id': session_id})
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()


# ============================================================================
# Media Serving Endpoints
# ============================================================================

@app.route('/media/<path:filepath>')
def serve_media(filepath):
    """Serve media files (images, audio, etc.)."""
    return send_from_directory(MEDIA_BASE_DIR, filepath)


@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    """Serve screenshot files."""
    return send_from_directory(SCREENSHOTS_DIR, filename)


# ============================================================================
# Legacy Endpoints (Backward Compatibility)
# ============================================================================

# Legacy config file path
import json
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "interval": 20,
    "model_type": "remote",
    "ollama_model": "qwen2.5vl:3b",
    "gemini_model": "gemini-2.5-pro",
    "remote_url": "http://localhost:5001/predict",
    "similarity_threshold": 0.95,
    "notes_history_limit": 5,
    "notes_model_provider": "gemini",
    "notes_ollama_model": "llama3",
    "enabled": False
}


def load_config():
    """Load config from file or return defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save config to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Legacy config endpoint for old frontend."""
    if request.method == 'POST':
        data = request.json
        config = load_config()
        
        config['interval'] = int(data.get('interval', 20))
        config['model_type'] = data.get('model_type', 'remote')
        config['ollama_model'] = data.get('ollama_model', config['ollama_model'])
        config['gemini_model'] = data.get('gemini_model', config['gemini_model'])
        config['remote_url'] = data.get('remote_url', config.get('remote_url', 'http://localhost:5001/predict'))
        config['similarity_threshold'] = float(data.get('similarity_threshold', 0.95))
        config['notes_history_limit'] = int(data.get('notes_history_limit', 5))
        config['notes_model_provider'] = data.get('notes_model_provider', 'gemini')
        config['notes_ollama_model'] = data.get('notes_ollama_model', 'llama3')
        
        save_config(config)
        return jsonify({'status': 'success', 'config': config})
    else:
        return jsonify(load_config())

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """
    Legacy endpoint for image upload from old frontend.
    Redirects to the new document capture flow.
    """
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    # Save to screenshots directory (legacy behavior)
    from werkzeug.utils import secure_filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"uploaded_{timestamp}.png"
    filepath = SCREENSHOTS_DIR / filename
    
    file.save(filepath)
    print(f"📤 Image uploaded: {filename}")
    
    # Create a quick analysis session (simplified version)
    db = get_db()
    try:
        session = CaptureSession(
            id=str(uuid.uuid4()),
            type=SessionType.DOC,
            title=f"Uploaded: {file.filename}",
            source_path=str(filepath)
        )
        db.add(session)
        
        # Save image reference
        media = CapturedMedia(
            session_id=session.id,
            file_path=str(filepath),
            media_type=MediaType.IMAGE
        )
        db.add(media)
        
        db.commit()
        session_id = session.id
        
        return jsonify({
            'status': 'success',
            'title': f'Uploaded: {file.filename}',
            'summary': 'Image uploaded successfully. Analyze it using the new capture flow.',
            'image_path': str(filepath),
            'session_id': session_id
        })
        
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.route('/api/reset', methods=['POST'])
def reset_state():
    """Reset the state manager (for debugging/recovery)."""
    state_mgr = get_state_manager()
    orch = get_orchestrator()
    
    # Stop monitoring if running
    orch.stop()
    
    # Reset state
    state_mgr.reset()
    
    return jsonify({'status': 'reset'})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    # Initialize database and directories
    init_db()
    ensure_media_dirs()
    
    print("=" * 60)
    print("TimeLine Content Capture Pipeline")
    print("=" * 60)
    print(f"API Server: http://localhost:5000")
    print(f"VLM Expected: http://localhost:5001/predict")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
