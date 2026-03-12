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
    * Added UI toggle in Settings page under "Audio Recording Library".
    * Default changed to `pyaudiowpatch` for better Windows audio capture reliability.
* [x] **First Chunk Transcription Fix:**
    * Fixed bug where first audio chunk (used for language detection) was not transcribed.

### **Phase 8: UI & Feature Enhancements (Completed)**
* [x] **Sidebar UI Updates:**
    * Removed analytics, info, and calendar icons from sidebar
    * Added Notes icon and Notes page
* [x] **Image Upload Processing:**
    * Implemented `upload_image` endpoint to process uploaded images with vision model
    * Added progress bar display during image processing
    * Shows real-time status updates to user
* [x] **Custom Delete Confirmation:**
    * Added custom Electron-style confirmation dialog (not native browser popup)
    * Used in ActivityCard for delete functionality
    * Reusable `ConfirmDialog` component
* [x] **IMG vs DOC Type Separation:**
    * Uploaded images now get IMG type with user-provided description
    * Documents (PDFs) get DOC type
    * Different processing logic for each type
* [x] **Transcript for IMG Sessions:**
    * Shows description in transcript section for IMG sessions
    * Displays as "Image Description:" in the transcript view

### **Phase 9: Video Processing Pipeline (Completed)**
* [x] **Contact Sheet Generation:**
    * Created `scenarios/contact_sheet.py` utility
    * Generates 4x4 grid contact sheets (16 frames each)
    * Thumbnails are 300x300px, resulting in 1200x1200px contact sheets
    * Saves to `media/video/<session_id>/contact_sheets/`
* [x] **AI Frame Analysis & Selection:**
    * Vision model analyzes contact sheets to identify relevant frames
    * Includes "educational" indicator in vision model JSON output
    * Disabled thinking mode (`think: false`) for Ollama vision model calls
    * Shows batch number during contact sheet analysis
* [x] **Manual Frame Selection Fallback:**
    * If AI selects no frames, displays contact sheets to user for manual selection
    * 4x4 grid UI with batch navigation
    * User can select/deselect individual frames
* [x] **Processing Enhancements:**
    * Added collapsible logs panel during note generation
    * Includes previous frame context in video frame analysis
    * Better error handling and status updates

### **Phase 10: Notes Persistence (Completed)**
* [x] **Local Note Storage:**
    * Generated notes saved as markdown files in `media/notes/` directory
    * Database table `generated_notes` stores note metadata
    * Each note linked to session_id and includes title, content, created_at
* [x] **Notes Page:**
    * New `NotesView.tsx` component displays all saved notes
    * Shows note title, session info, and creation date
    * View, save (download), and delete functionality
* [x] **Notes API:**
    * `GET /api/notes` - List all saved notes
    * `GET /api/notes/<id>` - Get specific note
    * `DELETE /api/notes/<id>` - Delete note
    * `GET /api/notes/<id>/download` - Download note as markdown

### **Phase 11: UI Streaming & Rendering Polish (Completed)**
* [x] **Real-time AI Streaming (Server-Sent Events):**
    * Refactored `/api/session/<id>/analyze_contact_sheets` and `process_frames` APIs to yield JSON streams (SSE) when `stream=true` is requested.
    * Streams live status updates, the URL of the image currently being processed, and raw text generation tokens from the local Ollama model.
* [x] **Frontend SSE Consumers:**
    * Added `analyzeContactSheetsStream` and `processVideoFramesStream` using native Fetch `ReadableStream` line-by-line parsing in `api.ts`.
* [x] **NotesModal UI Transparency:**
    * UI natively previews the current image (contact sheet or video frame) being analyzed.
    * New `token_stream` log type progressively appends LLM text chunks for a typewriter effect.
* [x] **Markdown Notes Formatting:**
    * Replaced manual regex parsing with `react-markdown` and `remark-gfm` in `NotesView.tsx`.
    * Added `@tailwindcss/typography` plugin to apply clean, dark-mode-compatible styles (`.prose`, `.prose-invert`) to generated notes, fixing unformatted tables and text walls.
* [x] **Quality of Life & Bug Fixes:**
    * Changed default notes Ollama model to `qwen3.5:2b`.
    * Fixed runtime crash (`Cannot set properties of null (setting 'innerHTML')`) in `ActivityCard.tsx` by replacing direct DOM manipulation with standard React state.
    * Removed unused variables (`isDocSession`, `isWebSession`) in `ScenarioDetailsView`.
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

### **Video Processing (NEW)**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/session/<id>/analyze_contact_sheets` | Analyze contact sheets for frame selection |
| POST | `/api/session/<id>/process_frames` | Process selected video frames |
| GET | `/api/session/<id>/frame_data` | Get frame analysis data |
| GET | `/api/media/video/<id>/contact_sheets/<filename>` | Serve contact sheet images |

### **Notes (NEW)**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/notes` | List all saved notes |
| GET | `/api/notes/<id>` | Get specific note |
| DELETE | `/api/notes/<id>` | Delete note |
| GET | `/api/notes/<id>/download` | Download note as markdown |

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
│   │   ├── ActivityList.tsx       # Queue UI
│   │   ├── ScenarioDetailsView.tsx # Live transcript/frames
│   │   ├── SettingsView.tsx       # Settings with audio library toggle
│   │   ├── NotesView.tsx          # Saved notes page
│   │   ├── NotesModal.tsx         # Note generation with frame selection
│   │   ├── ConfirmDialog.tsx      # Custom confirmation dialog
│   │   ├── TimelineView.tsx       # Home page with activities
│   │   ├── ControlPanel.tsx       # Upload and progress UI
│   │   ├── ActivityCard.tsx       # Activity display with delete
│   │   └── ...
│   ├── services/
│   │   └── api.ts                 # API service with new endpoints
│   └── App.tsx
├── extension/
│   ├── background.js       # Coordinator
│   ├── content.js          # Video Frame Extractor
│   └── manifest.json
├── backend/
│   ├── main.py             # API with new /ingest endpoints
│   ├── orchestrator.py     # Manages PendingActivity queue
│   ├── models.py           # Database models (includes GeneratedNote)
│   ├── rag_engine.py       # Note generation with frame data
│   ├── config.json         # App configuration
│   ├── scenarios/
│   │   ├── video_capture.py # Whisper + Audio Recording
│   │   ├── doc_capture.py   # PyMuPDF + EasyOCR
│   │   ├── web_capture.py   # Trafilatura + Snapshot handling
│   │   └── contact_sheet.py # Contact sheet generation utility
│   └── media/              # Storage (notes/, video/, frames/)
```
