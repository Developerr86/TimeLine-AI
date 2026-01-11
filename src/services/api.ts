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
}

export interface StatusResponse {
    enabled: boolean;
    total_responses: number;
    config: Config;
}

export interface NotesResponse {
    status: 'success' | 'error';
    notes?: string;
    message?: string;
    activity_count?: number;
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

    async generateNotes(): Promise<NotesResponse> {
        return this.request('/api/generate_notes', { method: 'POST' });
    }

    async uploadImage(file: File): Promise<{
        status: string;
        title?: string;
        summary?: string;
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

    async clearContext(): Promise<{ status: string; message?: string }> {
        return this.request('/api/clear_context', { method: 'POST' });
    }

    getScreenshotUrl(imagePath: string): string {
        // Extract filename from path
        const filename = imagePath.split(/[/\\]/).pop() || imagePath;
        return `${this.baseUrl}/screenshots/${filename}`;
    }
}

export const api = new ApiService();
