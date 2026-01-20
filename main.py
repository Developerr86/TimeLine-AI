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
from rag_engine import get_rag_engine, RAGEngine

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
    """Start the orchestrator to receive browser extension heartbeats."""
    try:
        data = request.get_json(silent=True) or {}
        
        # Optional configuration
        auto_trigger_video = data.get('auto_trigger_video', False)
        scenario_threshold = data.get('scenario_change_threshold', 2)
        
        if auto_trigger_video or scenario_threshold != 2:
            orch = configure_orchestrator(
                auto_trigger_video=auto_trigger_video,
                scenario_change_threshold=scenario_threshold
            )
        else:
            orch = get_orchestrator()
        
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
    """Stop the orchestrator."""
    orch = get_orchestrator()
    
    if orch.stop():
        return jsonify({'status': 'stopped'})
    else:
        return jsonify({'status': 'not_running'}), 400


@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """
    Receive heartbeat from browser extension.
    Request body: {
        "url": "https://...",
        "title": "Page Title",
        "scenario": "VIDEO" | "DOC" | "WEB" | "OTHER",
        "timestamp": 1234567890
    }
    
    Response includes command for extension:
    - START_CAPTURE: Start capturing video frames
    - STOP_CAPTURE: Stop capturing video frames
    - None: No action needed
    """
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    orch = get_orchestrator()
    result = orch.handle_heartbeat(data)
    
    # Check if we should send capture command to extension
    handler = get_video_handler()
    if handler.is_recording:
        result['command'] = 'START_CAPTURE'
    else:
        # If we were capturing but now stopped, send stop command
        result['command'] = None
    
    return jsonify(result)


@app.route('/api/ingest/frame', methods=['POST'])
def ingest_frame():
    """
    Receive a captured video frame from the browser extension.
    Request body: {
        "frame_data": "data:image/jpeg;base64,...",
        "frame_number": 1,
        "video_time": 123.45,
        "timestamp": 1234567890
    }
    
    The frame is processed by VideoCaptureHandler which:
    - Computes SSIM with previous frame for deduplication
    - Saves unique frames to media/video/<session_id>/frames/
    """
    data = request.get_json()
    if not data or not data.get('frame_data'):
        return jsonify({'status': 'error', 'message': 'No frame data provided'}), 400
    
    handler = get_video_handler()
    
    if not handler.is_recording:
        return jsonify({
            'status': 'not_recording',
            'message': 'Video capture not active'
        }), 400
    
    # Extract base64 data (remove data URL prefix if present)
    frame_data = data.get('frame_data', '')
    if frame_data.startswith('data:'):
        # Remove "data:image/jpeg;base64," prefix
        frame_data = frame_data.split(',', 1)[-1]
    
    # Ingest the frame
    result = handler.ingest_frame(
        frame_data=frame_data,
        video_time=data.get('video_time', 0),
        frame_number=data.get('frame_number', 0)
    )
    
    return jsonify(result)


@app.route('/api/ingest/web', methods=['POST'])
def ingest_web():
    """
    Receive HTML snapshot from browser extension for web articles.
    Request body: {
        "url": "https://...",
        "title": "Article Title",
        "html_content": "<html>...</html>"
    }
    
    Saves the HTML to a temp file and attaches the path to the pending activity.
    """
    data = request.get_json()
    if not data or not data.get('html_content'):
        return jsonify({'status': 'error', 'message': 'No HTML content provided'}), 400
    
    url = data.get('url', '')
    title = data.get('title', 'Untitled')
    html_content = data.get('html_content', '')
    
    # Create temp directory if needed
    temp_dir = Path('media/temp')
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from URL hash
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    timestamp = int(datetime.now().timestamp())
    filename = f"web_{url_hash}_{timestamp}.html"
    snapshot_path = temp_dir / filename
    
    try:
        # Save HTML content
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📸 Saved WEB snapshot: {snapshot_path} ({len(html_content)} bytes)")
        
        # Find or wait for the pending activity to attach metadata
        orch = get_orchestrator()
        activity = orch.get_activity_by_url(url, 'WEB')
        
        if activity:
            # Update activity with snapshot path
            orch.update_activity_metadata(activity.id, {
                'snapshot_path': str(snapshot_path),
                'snapshot_size': len(html_content)
            })
        else:
            # Activity might not exist yet (heartbeat arrives after snapshot)
            # Store in a temp mapping that can be checked when activity is created
            print(f"  ⏳ No pending activity found for {url[:50]}... (will attach later)")
        
        return jsonify({
            'status': 'success',
            'snapshot_path': str(snapshot_path),
            'size': len(html_content),
            'activity_found': activity is not None
        })
        
    except Exception as e:
        print(f"❌ Failed to save WEB snapshot: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# Activity Queue Endpoints (NEW - "Process Later" Workflow)
# ============================================================================

@app.route('/api/activities', methods=['GET'])
def get_activities():
    """
    Get list of pending activities waiting for user processing.
    Returns activities that have been detected but not yet processed.
    """
    orch = get_orchestrator()
    activities = orch.get_pending_activities()
    
    return jsonify({
        "activities": activities,
        "count": len(activities),
        "current_activity": orch._current_activity.to_dict() if orch._current_activity else None
    })


@app.route('/api/activity/process', methods=['POST'])
def process_activity():
    """
    Start processing a specific activity.
    Request body: { "activity_id": "uuid-here" }
    
    This transitions the system from MONITORING to PROCESSING state,
    and initiates the capture flow for the activity.
    """
    data = request.get_json()
    if not data or not data.get('activity_id'):
        return jsonify({'status': 'error', 'message': 'activity_id required'}), 400
    
    activity_id = data['activity_id']
    orch = get_orchestrator()
    
    result = orch.start_activity_processing(activity_id)
    
    if not result.get('success'):
        return jsonify({
            'status': 'error',
            'errors': result.get('errors', [])
        }), 400
    
    # Get the activity details to determine the capture type
    activity = orch.get_activity_by_id(activity_id)
    
    # For VIDEO activities, start the video capture handler
    if activity and activity.scenario == 'VIDEO':
        import uuid
        from models import CaptureSession, SessionType
        
        # Create a new capture session in the database
        db = get_db()
        try:
            session_id = str(uuid.uuid4())
            session = CaptureSession(
                id=session_id,
                type=SessionType.VIDEO,
                source_url=activity.url,
                title=activity.title
            )
            db.add(session)
            db.commit()
            print(f"📹 Created video capture session: {session_id}")
            
            # Start the video capture handler (this will start audio recording + transcription)
            handler = get_video_handler()
            capture_result = handler.start(session_id)
            
            if capture_result.get('success'):
                print(f"✅ Video capture started successfully")
                result['capture_session_id'] = session_id
                result['streaming_transcription'] = capture_result.get('streaming_transcription', False)
            else:
                print(f"⚠️ Video capture start had errors: {capture_result.get('errors')}")
                result['capture_errors'] = capture_result.get('errors', [])
                
        except Exception as e:
            db.rollback()
            print(f"❌ Failed to create capture session: {e}")
            result['capture_errors'] = [str(e)]
        finally:
            db.close()
    
    # For DOC activities, start the document capture handler
    elif activity and activity.scenario == 'DOC':
        import uuid
        import threading
        from models import CaptureSession, SessionType
        
        # Create a new capture session in the database
        db = get_db()
        try:
            session_id = str(uuid.uuid4())
            session = CaptureSession(
                id=session_id,
                type=SessionType.DOC,
                source_url=activity.url,
                title=activity.title
            )
            db.add(session)
            db.commit()
            print(f"📄 Created document capture session: {session_id}")
            
            # Run document processing in background thread (OCR can be slow)
            def process_doc_async():
                try:
                    handler = DocCaptureHandler()
                    doc_result = handler.process_activity(activity.url, session_id)
                    if doc_result.get('success'):
                        print(f"✅ Document processed: {doc_result.get('text_length', 0)} chars extracted")
                        # Update session end time
                        db = get_db()
                        try:
                            session = db.query(CaptureSession).filter_by(id=session_id).first()
                            if session:
                                session.end_time = datetime.utcnow()
                                db.commit()
                        finally:
                            db.close()
                    else:
                        print(f"⚠️ Document processing had errors: {doc_result.get('errors')}")
                except Exception as e:
                    print(f"❌ Document processing failed: {e}")
            
            thread = threading.Thread(target=process_doc_async, daemon=True)
            thread.start()
            
            result['capture_session_id'] = session_id
            print(f"🔄 Document processing started in background")
            
        except Exception as e:
            db.rollback()
            print(f"❌ Failed to create document session: {e}")
            result['capture_errors'] = [str(e)]
        finally:
            db.close()
    
    # For WEB activities, start the web capture handler
    elif activity and activity.scenario == 'WEB':
        import uuid
        import threading
        from models import CaptureSession, SessionType
        
        # Create a new capture session in the database
        db = get_db()
        try:
            session_id = str(uuid.uuid4())
            session = CaptureSession(
                id=session_id,
                type=SessionType.WEB,
                source_url=activity.url,
                title=activity.title
            )
            db.add(session)
            db.commit()
            print(f"🌐 Created web capture session: {session_id}")
            
            # Get snapshot path from activity metadata (if available)
            snapshot_path = activity.metadata.get('snapshot_path') if activity.metadata else None
            
            # Run web processing in background thread
            def process_web_async():
                try:
                    handler = WebCaptureHandler()
                    
                    if snapshot_path and Path(snapshot_path).exists():
                        # Process from saved snapshot (offline)
                        print(f"📸 Processing from snapshot: {snapshot_path}")
                        web_result = handler.process_snapshot(snapshot_path, session_id, activity.url)
                    else:
                        # Fallback: fetch from URL (online)
                        print(f"🌐 No snapshot found, fetching from URL...")
                        web_result = handler.capture(activity.url, session_id)
                    
                    if web_result.get('success'):
                        print(f"✅ Web article processed: {web_result.get('text_length', 0)} chars extracted")
                        # Update session end time
                        db = get_db()
                        try:
                            session = db.query(CaptureSession).filter_by(id=session_id).first()
                            if session:
                                session.end_time = datetime.utcnow()
                                db.commit()
                        finally:
                            db.close()
                    else:
                        print(f"⚠️ Web processing had errors: {web_result.get('errors')}")
                except Exception as e:
                    print(f"❌ Web processing failed: {e}")
            
            thread = threading.Thread(target=process_web_async, daemon=True)
            thread.start()
            
            result['capture_session_id'] = session_id
            print(f"🔄 Web processing started in background")
            
        except Exception as e:
            db.rollback()
            print(f"❌ Failed to create web session: {e}")
            result['capture_errors'] = [str(e)]
        finally:
            db.close()
    
    return jsonify({
        'status': 'processing',
        'activity_id': activity_id,
        'activity': result.get('activity'),
        'capture_session_id': result.get('capture_session_id'),
        'streaming_transcription': result.get('streaming_transcription', False),
        'message': f'Started processing {activity.scenario} activity'
    })


@app.route('/api/activity/stop', methods=['POST'])
def stop_activity():
    """
    Stop processing the current activity and return to monitoring.
    This also stops any active video capture and saves the transcript.
    """
    # First, stop the video capture handler (if recording)
    handler = get_video_handler()
    capture_result = None
    if handler.is_recording:
        print("⏹️ Stopping video capture...")
        capture_result = handler.stop()
        if capture_result.get('success'):
            print(f"✅ Video capture stopped successfully")
            print(f"   Frames saved: {capture_result.get('saved_frames', 0)}")
            print(f"   Audio duration: {capture_result.get('audio_duration', 0):.1f}s")
            if capture_result.get('transcript'):
                print(f"   Transcript length: {len(capture_result.get('transcript', ''))} chars")
        else:
            print(f"⚠️ Video capture stop had errors: {capture_result.get('errors')}")
    
    # Then stop the activity processing in orchestrator
    orch = get_orchestrator()
    result = orch.stop_activity_processing()
    
    if not result.get('success'):
        return jsonify({
            'status': 'error',
            'errors': result.get('errors', [])
        }), 400
    
    response = {
        'status': 'stopped',
        'activity': result.get('activity')
    }
    
    # Include capture results if available
    if capture_result:
        response['capture_result'] = {
            'success': capture_result.get('success'),
            'saved_frames': capture_result.get('saved_frames', 0),
            'audio_duration': capture_result.get('audio_duration', 0),
            'transcript_length': len(capture_result.get('transcript', '')),
            'chunks_processed': capture_result.get('chunks_processed', 0)
        }
    
    return jsonify(response)


@app.route('/api/activity/dismiss', methods=['POST'])
def dismiss_activity():
    """
    Dismiss a pending activity without processing.
    Request body: { "activity_id": "uuid-here" }
    """
    data = request.get_json()
    if not data or not data.get('activity_id'):
        return jsonify({'status': 'error', 'message': 'activity_id required'}), 400
    
    activity_id = data['activity_id']
    orch = get_orchestrator()
    
    if orch.dismiss_activity(activity_id):
        return jsonify({'status': 'dismissed', 'activity_id': activity_id})
    else:
        return jsonify({'status': 'error', 'message': 'Activity not found'}), 404


# ============================================================================
# Session Details Endpoints (for Scenario Details Page)
# ============================================================================

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_details(session_id):
    """
    Get session details including capture status and transcriber state.
    Used by the Scenario Details page.
    """
    from models import CaptureSession, MEDIA_VIDEO_DIR
    
    db = get_db()
    try:
        session = db.query(CaptureSession).filter_by(id=session_id).first()
        if not session:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 404
        
        # Get video handler status
        handler = get_video_handler()
        capture_status = handler.get_status()
        transcriber_status = handler.get_transcriber_status()
        
        # Check if this session is currently active
        is_active = handler._session_id == session_id and handler.is_recording
        
        return jsonify({
            'session': {
                'id': session.id,
                'type': session.type.value if session.type else None,
                'source_url': session.source_url,
                'title': session.title,
                'start_time': session.start_time.isoformat() if session.start_time else None,
                'end_time': session.end_time.isoformat() if session.end_time else None,
            },
            'is_active': is_active,
            'capture_status': capture_status if is_active else None,
            'transcriber_status': transcriber_status if is_active else None
        })
    finally:
        db.close()


@app.route('/api/session/<session_id>/frames', methods=['GET'])
def get_session_frames(session_id):
    """
    Get list of captured frames with timestamps for a session.
    """
    from models import MEDIA_VIDEO_DIR
    import os
    import re
    
    frames_dir = MEDIA_VIDEO_DIR / session_id / "frames"
    
    if not frames_dir.exists():
        return jsonify({'frames': [], 'count': 0})
    
    frames = []
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = frames_dir / filename
            stat = filepath.stat()
            
            # Try to extract video time from filename (format: frame_00001_t123_timestamp.jpg)
            video_time = None
            time_match = re.search(r'_t(\d+)_', filename)
            if time_match:
                video_time = int(time_match.group(1))
            
            frames.append({
                'filename': filename,
                'path': f'/api/media/video/{session_id}/frames/{filename}',
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'video_time': video_time
            })
    
    return jsonify({
        'frames': frames,
        'count': len(frames)
    })


@app.route('/api/session/<session_id>/transcript', methods=['GET'])
def get_session_transcript(session_id):
    """
    Read the transcript.txt file for a session.
    Supports VIDEO (media/video), DOC (media/docs), and WEB (media/web) sessions.
    """
    from models import MEDIA_VIDEO_DIR, MEDIA_DOCS_DIR, MEDIA_WEB_DIR, CaptureSession, SessionType
    
    # First, determine session type from database
    db = get_db()
    try:
        session = db.query(CaptureSession).filter_by(id=session_id).first()
        if session and session.type == SessionType.DOC:
            transcript_path = MEDIA_DOCS_DIR / session_id / "transcript.txt"
        elif session and session.type == SessionType.WEB:
            transcript_path = MEDIA_WEB_DIR / session_id / "transcript.txt"
        else:
            transcript_path = MEDIA_VIDEO_DIR / session_id / "transcript.txt"
    finally:
        db.close()
    
    if not transcript_path.exists():
        return jsonify({
            'content': '',
            'exists': False,
            'chunks': []
        })
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse chunks from format: [MM:SS] text
        import re
        chunks = []
        for line in content.strip().split('\n'):
            if line.strip():
                match = re.match(r'\[(\d+):(\d+)\]\s*(.+)', line)
                if match:
                    minutes, seconds, text = match.groups()
                    time_seconds = int(minutes) * 60 + int(seconds)
                    chunks.append({
                        'time': time_seconds,
                        'timestamp': f"[{minutes}:{seconds}]",
                        'text': text.strip()
                    })
                else:
                    chunks.append({'text': line.strip()})
        
        return jsonify({
            'content': content,
            'exists': True,
            'chunks': chunks
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/session/<session_id>/transcript', methods=['PUT'])
def update_session_transcript(session_id):
    """
    Save edited transcript content to transcript.txt file.
    """
    from models import MEDIA_VIDEO_DIR
    
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({'status': 'error', 'message': 'content required'}), 400
    
    session_dir = MEDIA_VIDEO_DIR / session_id
    if not session_dir.exists():
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    transcript_path = session_dir / "transcript.txt"
    
    try:
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(data['content'])
        
        return jsonify({
            'status': 'success',
            'message': 'Transcript saved',
            'path': str(transcript_path)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/media/video/<session_id>/frames/<filename>')
def serve_video_frame(session_id, filename):
    """
    Serve captured frame images for the scenario details page.
    """
    frames_dir = MEDIA_VIDEO_DIR / session_id / "frames"
    
    if not frames_dir.exists():
        return jsonify({'status': 'error', 'message': 'Frames directory not found'}), 404
    
    # Secure the filename to prevent path traversal
    safe_filename = secure_filename(filename)
    
    return send_from_directory(frames_dir, safe_filename)


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



@app.route('/api/current_session', methods=['GET'])
def get_current_session():
    """
    Get the current capture session status including live transcript.
    Used by frontend to display real-time transcription progress.
    """
    handler = get_video_handler()
    state_mgr = get_state_manager()
    orch = get_orchestrator()
    
    # Get transcription status if available
    transcriber_status = handler.get_transcriber_status() if hasattr(handler, 'get_transcriber_status') else {}
    
    return jsonify({
        'is_recording': handler.is_recording,
        'session_id': handler._session_id,
        'current_state': state_mgr.current_state.value,
        'current_activity': orch._current_activity.to_dict() if orch._current_activity else None,
        'capture_status': handler.get_status(),
        'transcriber': transcriber_status
    })


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


@app.route('/api/sessions/processed', methods=['GET'])
def list_processed_sessions():
    """
    List all sessions that have been processed and indexed.
    These are sessions with text content that can be used for notes generation.
    """
    db = get_db()
    try:
        # Get sessions that have associated text content
        sessions = db.query(CaptureSession).join(
            CapturedText, CaptureSession.id == CapturedText.session_id
        ).order_by(
            CaptureSession.start_time.desc()
        ).distinct().limit(100).all()
        
        result = []
        for session in sessions:
            session_dict = session.to_dict()
            # Check if session is indexed in RAG
            rag = get_rag_engine()
            stats = rag.get_stats()
            indexed_sessions = {s["session_id"] for s in stats.get("sessions", [])}
            session_dict["indexed"] = session.id in indexed_sessions
            result.append(session_dict)
        
        return jsonify({
            "sessions": result,
            "count": len(result)
        })
    finally:
        db.close()


@app.route('/api/generate_notes', methods=['POST'])
def generate_notes():
    """
    Generate notes from selected processed sessions.
    
    Request body: {
        "session_ids": ["uuid-1", "uuid-2", ...]
    }
    
    Response: {
        "status": "success",
        "notes": "...",
        "session_count": 2,
        "chunks_used": 10
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'No data provided'
        }), 400
    
    session_ids = data.get('session_ids', [])
    
    if not session_ids:
        return jsonify({
            'status': 'error',
            'message': 'No session_ids provided'
        }), 400
    
    # Get notes model from config
    config = load_config()
    notes_model = config.get('notes_ollama_model', 'llama3')
    
    try:
        rag = get_rag_engine(notes_model)
        result = rag.generate_notes(session_ids, notes_model)
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'notes': result.get('notes', ''),
                'session_count': result.get('session_count', 0),
                'chunks_used': result.get('chunks_used', 0)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('errors', ['Notes generation failed'])[0] if result.get('errors') else 'Notes generation failed',
                'errors': result.get('errors', [])
            }), 500
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Notes generation error: {error_msg}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


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
    "notes_ollama_model": "phi3:3.8b",
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
        config['notes_ollama_model'] = data.get('notes_ollama_model', 'phi3:3.8b')
        
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
# RAG Chat Endpoints
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with your notes using RAG (Retrieval-Augmented Generation).
    
    Request body: {
        "query": "What did I learn about machine learning?"
    }
    
    Response: {
        "response": "Based on your notes...",
        "sources": [{"session_id": "...", "title": "...", "source": "video"}],
        "success": true
    }
    """
    data = request.get_json()
    if not data or not data.get('query'):
        return jsonify({
            'status': 'error',
            'message': 'No query provided'
        }), 400
    
    query = data.get('query', '').strip()
    
    try:
        rag = get_rag_engine()
        result = rag.query(query)
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'response': result.get('response', ''),
                'sources': result.get('sources', []),
                'chunks_used': result.get('chunks_used', 0)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('errors', ['Unknown error'])[0] if result.get('errors') else 'Query failed',
                'errors': result.get('errors', [])
            }), 500
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Chat error: {error_msg}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
    """Get RAG engine statistics (indexed sessions, chunk counts, etc.)."""
    try:
        rag = get_rag_engine()
        stats = rag.get_stats()
        return jsonify({
            'status': 'success',
            **stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/rag/capabilities', methods=['GET'])
def rag_capabilities():
    """Get RAG engine capabilities (which dependencies are installed)."""
    return jsonify({
        'status': 'success',
        'capabilities': RAGEngine.get_capabilities()
    })


@app.route('/api/rag/index', methods=['POST'])
def rag_index_session():
    """
    Manually index a session's content.
    
    Request body: {
        "session_id": "uuid-here"
    }
    """
    data = request.get_json()
    if not data or not data.get('session_id'):
        return jsonify({
            'status': 'error',
            'message': 'No session_id provided'
        }), 400
    
    session_id = data.get('session_id')
    
    db = get_db()
    try:
        # Get session info
        session = db.query(CaptureSession).filter_by(id=session_id).first()
        if not session:
            return jsonify({
                'status': 'error',
                'message': f'Session not found: {session_id}'
            }), 404
        
        # Get text content
        text_record = db.query(CapturedText).filter_by(session_id=session_id).first()
        if not text_record or not text_record.content:
            return jsonify({
                'status': 'error',
                'message': 'No text content found for session'
            }), 404
        
        # Index the content
        rag = get_rag_engine()
        result = rag.index_session(
            session_id=session_id,
            text=text_record.content,
            source=session.type.value.lower() if session.type else "unknown",
            title=session.title
        )
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'chunks_indexed': result.get('chunks_indexed', 0),
                'session_id': session_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('errors', ['Indexing failed'])[0]
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
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


@app.route('/api/clear_context', methods=['POST'])
def clear_context():
    """
    Clear all captured data - sessions, texts, media, and screenshot files.
    This is a destructive operation used by the Settings "Clear All Data" button.
    """
    import shutil
    
    state_mgr = get_state_manager()
    orch = get_orchestrator()
    
    # Stop monitoring if running
    orch.stop()
    
    # Reset state
    state_mgr.reset()
    
    deleted_sessions = 0
    deleted_files = 0
    errors = []
    
    db = get_db()
    try:
        # Delete all media files first
        media_items = db.query(CapturedMedia).all()
        for item in media_items:
            try:
                path = Path(item.file_path)
                if path.exists():
                    path.unlink()
                    deleted_files += 1
            except Exception as e:
                errors.append(f"Failed to delete {item.file_path}: {str(e)}")
        
        # Delete all sessions (cascades to texts and media records)
        deleted_sessions = db.query(CaptureSession).delete()
        db.commit()
        
    except Exception as e:
        db.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Database error: {str(e)}'
        }), 500
    finally:
        db.close()
    
    # Clear screenshot directory
    try:
        if SCREENSHOTS_DIR.exists():
            for file in SCREENSHOTS_DIR.iterdir():
                if file.is_file():
                    file.unlink()
                    deleted_files += 1
    except Exception as e:
        errors.append(f"Failed to clear screenshots: {str(e)}")
    
    # Clear media directories
    for media_dir in [MEDIA_WEB_DIR, MEDIA_DOCS_DIR, MEDIA_VIDEO_DIR]:
        try:
            if media_dir.exists():
                for item in media_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                        deleted_files += 1
                    else:
                        item.unlink()
                        deleted_files += 1
        except Exception as e:
            errors.append(f"Failed to clear {media_dir}: {str(e)}")
    
    return jsonify({
        'status': 'success',
        'message': f'Cleared {deleted_sessions} sessions and {deleted_files} files',
        'deleted_sessions': deleted_sessions,
        'deleted_files': deleted_files,
        'errors': errors if errors else None
    })


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
