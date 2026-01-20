# Timeline-AI: Project Context & Architecture

## 1. Project Manifesto
**Timeline-AI** is an automated, offline-first, privacy-focused note-taking application designed for students. It runs locally on the user's machine, observing their study sessions to generate detailed, structured notes without sending data to the cloud.

* **Core Philosophy:** "Capture Now, Process Later." We prioritize low-latency data capture (screenshots/audio/html) during study sessions and defer heavy AI processing (OCR, Transcription, Summarization) to when the user explicitly triggers it via the Activity Queue.
* **Privacy:** All AI models (`faster-whisper-large-v3`, `EasyOCR`, `FastVLM`, `Llama`) run on `localhost`. No user data leaves the device.

---

## 2. System Architecture
The project follows a **"Sidecar" Microservices Architecture** wrapped in a Desktop Application.
### **A. The Frontend (The Shell)**
* **Tech:** Electron + React (Vite) + TailwindCSS.
* **Role:** The visual interface.
    * **Timeline View:** Displays processed notes.
    * **Activity List:** (NEW) Displays pending detected scenarios (Video, Doc, Web) waiting for user approval.
    * **Scenario Details:** Real-time view of transcription and capture progress.
* **Communication:** Talks to the Python Backend via HTTP REST API (`http://localhost:5000`).

### **B. The Backend (The Orchestrator)**
* **Tech:** Python (Flask), SQLAlchemy (SQLite).
* **Role:** The state manager.
    * **Input:** Receives "Heartbeats" and "Ingest Data" (Frames/HTML) from the Browser Extension.
    * **Queue Management:** Maintains a list of `PendingActivity` items instead of auto-triggering processing.
    * **Execution:** Manages threaded processing for Audio Recording, Transcription, and OCR.

### **C. The Watchdog (The Trigger)**
* **Tech:** Chrome/Edge Browser Extension (Manifest V3).
* **Role:** detection and raw data extraction.
    * **Heartbeat:** reports URL/Title to Backend every 3s.
    * **Video Capture:** Injects `content.js` to extract frames from `<video>` elements (1fps) and sends to Backend.
    * **Web Capture:** Captures full DOM HTML snapshots immediately upon detection.

### **D. The AI Engine (The Worker)**
* **Tech:** On-demand Python processes (using `threading`).
* **Models:**
    * **Audio:** `faster-whisper` (**large-v3**, float16, CUDA) for real-time multilingual transcription.
    * **Vision:** `scikit-image` (SSIM) for frame deduplication.
    * **OCR:** `EasyOCR` / `PyMuPDF` for document text extraction.
    * **Web:** `Trafilatura` for HTML parsing.

---

## 3. Data Flow & Scenarios

### **Scenario 1: The Video Lecture (YouTube/Zoom)**
1.  **Detection:** Extension detects video URL -> Sends Heartbeat.
2.  **Queueing:** Orchestrator adds to `ActivityList` (Status: Pending).
3.  **Process Trigger:** User clicks "Process" in Frontend.
4.  **Capture (Real-time):**
    * **Visual:** Extension extracts frames (1fps) -> Sends to `POST /api/ingest/frame`.
    * **Backend:** Calculates SSIM. If similarity < 60% (unique), save frame.
    * **Audio:** Backend records System Audio (Loopback) -> buffers chunks.
5.  **Processing:**
    * **Transcription:** `faster-whisper` (large-v3) processes audio chunks in real-time.
    * **UI:** Live transcript updates in `ScenarioDetailsView`.

### **Scenario 2: The Document (PDF/Slides)**
1.  **Detection:** Extension detects `.pdf` or `file://` URL.
2.  **Queueing:** Orchestrator adds to `ActivityList`.
3.  **Process Trigger:** User clicks "Process".
4.  **Processing:**
    * **Retrieval:** Backend downloads PDF (web) or accesses path (local).
    * **Extraction:** Checks for selectable text via `PyMuPDF`.
    * **Fallback:** If scan detected, uses `EasyOCR` (GPU) to extract text from pages.
    * **Output:** Searchable text summary.

### **Scenario 3: The Web Article**
1.  **Detection:** Extension detects generic article URL.
2.  **Capture (Immediate):** Extension captures full HTML snapshot -> Sends to `POST /api/ingest/web`.
3.  **Queueing:** Orchestrator adds to `ActivityList`, attaching the snapshot path.
4.  **Process Trigger:** User clicks "Process".
5.  **Processing:**
    * Backend reads saved HTML snapshot.
    * **Trafilatura** extracts main content, title, and metadata offline.
    * **Output:** Clean markdown note.

---

## 4. Implementation Status (Changelog)

### **Phase 1-5: Foundation & Basics (Completed)**
* [x] Basic Electron/React/Python setup.
* [x] Database Schema & State Management.
* [x] Browser Extension Heartbeat logic.

### **Phase 6: "Process Later" Workflow (Completed)**
* [x] **Activity Queue:** Implemented `pending_activities` list in Orchestrator.
* [x] **Frontend UI:** Added `ActivityList.tsx` sidebar component with "Process" buttons.
* [x] **Video Pipeline:**
    * Implemented `content.js` for canvas-based frame extraction.
    * Implemented `ingest_frame` endpoint with SSIM deduplication.
    * Configured `faster-whisper` with `large-v3` model for Hindi/English support.
* [x] **Doc Pipeline:**
    * Implemented `DocCaptureHandler.process_activity` for downloading/OCR.
* [x] **Web Pipeline:**
    * Implemented HTML snapshot ingestion (`POST /api/ingest/web`).
    * Offline processing via `trafilatura`.

### **Phase 7: Audio Recording Improvements (Completed)**
* [x] **Numpy Compatibility Fix:** 
    * Added monkeypatch for `numpy.fromstring` → `numpy.frombuffer` to fix `soundcard` library crash on numpy 1.20+.
* [x] **Stereo Mix Support:**
    * Audio recording now searches for "Stereo Mix" device first for cleaner audio.
    * Falls back to default speaker loopback if unavailable.
* [x] **Configurable Audio Library:**
    * Added `audio_library` setting in `config.json` (options: `soundcard`, `pyaudiowpatch`).
    * Added UI toggle in Settings page under "🎤 Audio Recording Library".
    * Default changed to `pyaudiowpatch` for better Windows audio capture reliability.
* [x] **First Chunk Transcription Fix:**
    * Fixed bug where first audio chunk (used for language detection) was not transcribed.

---

## 5. API Endpoints

### **Activity Queue (NEW)**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/activities` | List pending detected activities |
| POST | `/api/activity/process` | Start processing an activity (Video/Doc/Web) |
| POST | `/api/activity/stop` | Stop current processing (save & exit) |
| POST | `/api/activity/dismiss` | Remove activity from queue |

### **Ingestion (NEW)**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/ingest/frame` | Receive base64 video frame from Extension |
| POST | `/api/ingest/web` | Receive HTML snapshot from Extension |

### **Session & Live Status**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/current_session` | Get live status (recording state, live transcript) |
| GET | `/api/session/<id>` | Get static session details |

---

## 6. Configuration (`config.json`)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `interval` | int | 5 | Screenshot interval (seconds) |
| `model_type` | string | `"remote"` | AI model provider |
| `audio_library` | string | `"pyaudiowpatch"` | **Audio recording library** (`soundcard` or `pyaudiowpatch`) |
| `similarity_threshold` | float | 0.8 | Frame deduplication threshold |

---

## 7. Directory Structure
```text
Timeline-AI/
├── electron/
├── src/
│   ├── components/
│   │   ├── ActivityList.tsx       # NEW: Queue UI
│   │   ├── ScenarioDetailsView.tsx # Live transcript/frames
│   │   ├── SettingsView.tsx       # Settings with audio library toggle
│   │   └── ...
│   └── App.tsx
├── extension/
│   ├── background.js       # Coordinator
│   ├── content.js          # NEW: Video Frame Extractor
│   └── manifest.json
├── backend/
│   ├── main.py             # API with new /ingest endpoints
│   ├── orchestrator.py     # Manages PendingActivity queue
│   ├── config.json         # App configuration (includes audio_library)
│   ├── scenarios/
│   │   ├── video_capture.py # Whisper + Audio Recording (soundcard/pyaudiowpatch)
│   │   ├── doc_capture.py   # PyMuPDF + EasyOCR
│   │   └── web_capture.py   # Trafilatura + Snapshot handling
│   └── media/              # Storage