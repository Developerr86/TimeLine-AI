/// <reference types="vite/client" />

export interface PendingScenario {
    scenario_type: 'WEB' | 'DOC' | 'VIDEO';
    confidence: number;
    timestamp: string;
    context?: Record<string, unknown>;
}

declare global {
    interface Window {
        electronAPI?: {
            // Window controls
            minimize: () => Promise<void>;
            maximize: () => Promise<void>;
            close: () => Promise<void>;

            // Backend URL
            getBackendUrl: () => Promise<string>;

            // Scenario handling
            confirmScenario: () => Promise<{ status: string; session_id?: string }>;
            dismissScenario: () => Promise<{ status: string }>;
            getPendingScenario: () => Promise<PendingScenario | null>;
            onShowScenarioDialog: (callback: (scenario: PendingScenario) => void) => void;

            // Platform info
            platform: string;
        };
    }
}

export { };
