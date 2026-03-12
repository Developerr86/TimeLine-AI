const API_BASE = 'http://localhost:5000';

export interface Activity {
    id?: number;
    timestamp: string;
    model: string;
    model_name: string;
    image_path: string;
    title: string;
    summary: string;
    raw_response?: string;
    token_usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
    };
    error?: string;
    uploaded?: boolean;
    // For scenario-based activities
    session_id?: string;
    scenario?: 'VIDEO' | 'DOC' | 'WEB';
}

export interface Config {
    interval: number;
    model_type: 'ollama' | 'gemini' | 'remote';
    ollama_model: string;
    gemini_model: string;
    remote_url?: string;
    similarity_threshold: number;
    notes_history_limit: number;
    notes_model_provider: 'gemini' | 'ollama';
    notes_ollama_model: string;
    enabled: boolean;
    audio_library: 'soundcard' | 'pyaudiowpatch';
}

export interface PendingScenario {
    scenario_type: 'WEB' | 'DOC' | 'VIDEO';
    confidence: number;
    detected_at: string;
    metadata: Record<string, unknown>;
    screenshot_path: string | null;
}

export interface StateStatus {
    state: 'MONITORING' | 'AWAITING_USER_INPUT' | 'CAPTURING_WEB' | 'CAPTURING_DOC' | 'CAPTURING_VIDEO';
    monitoring_enabled: boolean;
    last_state_change: string;
    pending_scenario: PendingScenario | null;
    active_capture: {
        session_id: string;
        capture_type: string;
        started_at: string;
    } | null;
}

export interface OrchestratorStatus {
    is_monitoring: boolean;
    monitoring_enabled: boolean;
    current_state: string;
    browser_context: {
        url: string | null;
        title: string | null;
        scenario: string | null;
        timestamp: number | null;
        idle_state: string | null;
    } | null;
    heartbeat_count: number;
    scenarios_triggered: number;
    last_heartbeat_time: string | null;
    scenario_streak: number;
    // Activity queue fields (NEW)
    pending_activities: PendingActivity[];
    pending_activities_count: number;
    current_activity: PendingActivity | null;
    // Legacy fields for backward compatibility
    current_screenshot?: string | null;
    last_analysis_time?: string | null;
    analysis_count?: number;
    captured_scenarios?: unknown[];
}

export interface StatusResponse {
    state: StateStatus;
    orchestrator: OrchestratorStatus;
    database: {
        sessions: number;
        texts: number;
        media: number;
    };
    // Legacy fields for backward compatibility
    enabled?: boolean;
    total_responses?: number;
    config?: Config;
}

export interface NotesResponse {
    status: 'success' | 'error';
    notes?: string;
    message?: string;
    session_count?: number;
    chunks_used?: number;
}

export interface ProcessedSession {
    id: string;
    type: 'VIDEO' | 'DOC' | 'WEB' | null;
    title: string | null;
    source_url: string | null;
    source_path: string | null;
    start_time: string | null;
    end_time: string | null;
    indexed: boolean;
}

export interface ProcessedSessionsResponse {
    sessions: ProcessedSession[];
    count: number;
}

export interface PendingActivity {
    id: string;
    scenario: 'VIDEO' | 'DOC' | 'WEB';
    url: string;
    title: string;
    detected_at: string;
    processed: boolean;
    processing: boolean;
}

export interface PendingActivityResponse {
    activities: PendingActivity[];
    count: number;
    current_activity: PendingActivity | null;
}

export interface CurrentSessionResponse {
    session_id: string | null;
    is_recording: boolean;
    transcript: string;
    frames_captured: number;
    audio_duration_seconds: number;
}

export interface TranscriberStatus {
    streaming_enabled: boolean;
    is_recording: boolean;
    model_loading: boolean;
    model_loaded: boolean;
    device_type: 'cpu' | 'cuda';
    current_transcript: string;
    transcript_chunks: TranscriptChunk[];
    audio_duration_seconds: number;
    chunks_processed: number;
    chunk_duration_seconds: number;
    has_soundcard: boolean;
    has_pyaudio: boolean;
    has_whisper: boolean;
    has_gpu: boolean;
}

export interface CaptureStatus {
    is_recording: boolean;
    session_id: string | null;
    total_frames: number;
    saved_frames: number;
    capture_interval: number;
    ssim_threshold: number;
    ingested_frames: number;
}

export interface SessionDetails {
    id: string;
    type: string | null;
    source_url: string | null;
    title: string | null;
    start_time: string | null;
    end_time: string | null;
}

export interface SessionDetailsResponse {
    session: SessionDetails;
    is_active: boolean;
    capture_status: CaptureStatus | null;
    transcriber_status: TranscriberStatus | null;
}

export interface SessionFrame {
    filename: string;
    path: string;
    size: number;
    modified: number;
    video_time: number | null;
}

export interface SessionFramesResponse {
    frames: SessionFrame[];
    count: number;
}

export interface TranscriptChunk {
    time?: number;
    timestamp?: string;
    text: string;
}

export interface SessionTranscriptResponse {
    content: string;
    exists: boolean;
    chunks: TranscriptChunk[];
}

class ApiService {
    private baseUrl: string;

    constructor() {
        this.baseUrl = API_BASE;
    }

    private async request<T>(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<T> {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return response.json();
    }

    async getStatus(): Promise<StatusResponse> {
        return this.request<StatusResponse>('/api/status');
    }

    async getResponses(): Promise<Activity[]> {
        return this.request<Activity[]>('/api/responses');
    }

    async getConfig(): Promise<Config> {
        return this.request<Config>('/api/config');
    }

    async updateConfig(config: Partial<Config>): Promise<{ status: string; config: Config }> {
        return this.request('/api/config', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    }

    async startCapture(): Promise<{ status: string }> {
        return this.request('/api/start', { method: 'POST' });
    }

    async stopCapture(): Promise<{ status: string }> {
        return this.request('/api/stop', { method: 'POST' });
    }

    async generateNotes(sessionIds: string[]): Promise<NotesResponse> {
        return this.request('/api/generate_notes', {
            method: 'POST',
            body: JSON.stringify({ session_ids: sessionIds }),
        });
    }

    async getProcessedSessions(): Promise<ProcessedSessionsResponse> {
        return this.request<ProcessedSessionsResponse>('/api/sessions/processed');
    }


    async uploadImage(file: File): Promise<{
        status: string;
        title?: string;
        description?: string;
        summary?: string;
        educational?: string;
        image_path?: string;
        session_id?: string;
        type?: string;
        message?: string;
    }> {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${this.baseUrl}/api/upload_image`, {
            method: 'POST',
            body: formData,
        });

        return response.json();
    }

    async deleteSession(sessionId: string): Promise<{ status: string; message?: string }> {
        return this.request(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    }

    async clearContext(): Promise<{ status: string; message?: string }> {
        return this.request('/api/clear_context', { method: 'POST' });
    }

    getScreenshotUrl(imagePath: string | null | undefined): string {
        if (!imagePath) {
            return '';
        }
        // Extract filename from path
        const filename = imagePath.split(/[/\\]/).pop() || imagePath;
        return `${this.baseUrl}/screenshots/${filename}`;
    }

    // =========================================================================
    // Activity Queue Methods (NEW - "Process Later" Workflow)
    // =========================================================================

    async getActivities(): Promise<PendingActivityResponse> {
        return this.request<PendingActivityResponse>('/api/activities');
    }

    async processActivity(activityId: string): Promise<{ status: string; activity?: PendingActivity; message?: string }> {
        return this.request('/api/activity/process', {
            method: 'POST',
            body: JSON.stringify({ activity_id: activityId }),
        });
    }

    async stopActivity(): Promise<{ status: string; activity?: PendingActivity }> {
        return this.request('/api/activity/stop', { method: 'POST' });
    }

    async dismissActivity(activityId: string): Promise<{ status: string; activity_id?: string }> {
        return this.request('/api/activity/dismiss', {
            method: 'POST',
            body: JSON.stringify({ activity_id: activityId }),
        });
    }

    async getCurrentSession(): Promise<CurrentSessionResponse> {
        return this.request<CurrentSessionResponse>('/api/current_session');
    }

    // =========================================================================
    // Session Details Methods (for Scenario Details Page)
    // =========================================================================

    async getSessionDetails(sessionId: string): Promise<SessionDetailsResponse> {
        return this.request<SessionDetailsResponse>(`/api/session/${sessionId}`);
    }

    async getSessionFrames(sessionId: string): Promise<SessionFramesResponse> {
        return this.request<SessionFramesResponse>(`/api/session/${sessionId}/frames`);
    }

    async getSessionTranscript(sessionId: string): Promise<SessionTranscriptResponse> {
        return this.request<SessionTranscriptResponse>(`/api/session/${sessionId}/transcript`);
    }

    async updateSessionTranscript(sessionId: string, content: string): Promise<{ status: string; message?: string }> {
        return this.request(`/api/session/${sessionId}/transcript`, {
            method: 'PUT',
            body: JSON.stringify({ content }),
        });
    }

    getFrameUrl(sessionId: string, filename: string): string {
        return `${this.baseUrl}/api/media/video/${sessionId}/frames/${filename}`;
    }

    // =========================================================================
    // RAG Chat Methods
    // =========================================================================

    async chat(query: string): Promise<ChatResponse> {
        return this.request<ChatResponse>('/api/chat', {
            method: 'POST',
            body: JSON.stringify({ query }),
        });
    }

    async getRagStats(): Promise<RagStatsResponse> {
        return this.request<RagStatsResponse>('/api/rag/stats');
    }

    async getRagCapabilities(): Promise<RagCapabilitiesResponse> {
        return this.request<RagCapabilitiesResponse>('/api/rag/capabilities');
    }

    async indexSession(sessionId: string): Promise<{ status: string; chunks_indexed?: number; message?: string }> {
        return this.request('/api/rag/index', {
            method: 'POST',
            body: JSON.stringify({ session_id: sessionId }),
        });
    }
}

export interface ChatSource {
    session_id: string;
    source: string;
    title: string;
    relevance?: number;
}

export interface ChatResponse {
    status: string;
    response?: string;
    sources?: ChatSource[];
    chunks_used?: number;
    message?: string;
    errors?: string[];
}

export interface RagSession {
    session_id: string;
    source: string;
    title: string;
    chunk_count: number;
}

export interface RagStatsResponse {
    status: string;
    initialized?: boolean;
    total_chunks?: number;
    sessions?: RagSession[];
    embedding_model?: string;
    llm_model?: string;
    error?: string;
}

export interface RagCapabilitiesResponse {
    status: string;
    capabilities: {
        chromadb: boolean;
        sentence_transformers: boolean;
        ollama: boolean;
        fully_available: boolean;
    };
}

export const api = new ApiService();
