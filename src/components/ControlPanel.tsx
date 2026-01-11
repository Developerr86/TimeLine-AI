import { Play, Square, Upload, Activity } from 'lucide-react';
import { useRef } from 'react';
import { StatusResponse, api } from '../services/api';

interface ControlPanelProps {
    status: StatusResponse | null;
    onStart: () => void;
    onStop: () => void;
}

export default function ControlPanel({ status, onStart, onStop }: ControlPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            await api.uploadImage(file);
            window.location.reload();
        } catch (error) {
            console.error('Upload failed:', error);
        }

        // Reset input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const isActive = status?.enabled ?? false;

    return (
        <div className="glass rounded-2xl p-4 mb-6 flex items-center justify-between">
            {/* Left: Filter tabs (placeholder for future) */}
            <div className="flex items-center gap-2">
                <button className="px-4 py-2 rounded-lg bg-white/10 text-sm font-medium text-white">
                    📋 All tasks
                </button>
                <button className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:bg-white/5 transition-colors">
                    💼 Core tasks
                </button>
                <button className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:bg-white/5 transition-colors">
                    ⭐ Personal
                </button>
            </div>

            {/* Right: Status and controls */}
            <div className="flex items-center gap-4">
                {/* Status Badge */}
                <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${isActive ? 'bg-emerald-500/20 text-emerald-400' : 'bg-white/10 text-gray-400'
                    }`}>
                    <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-400 animate-pulse-glow' : 'bg-gray-500'
                        }`} />
                    <span className="text-sm font-medium">
                        {isActive ? 'Recording' : 'Idle'}
                    </span>
                </div>

                {/* Stats */}
                {status && (
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Activity className="w-4 h-4" />
                        <span>{status.total_responses} activities</span>
                    </div>
                )}

                {/* Upload Button */}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                />
                <button
                    onClick={handleUploadClick}
                    className="btn-secondary no-drag"
                >
                    <Upload className="w-4 h-4" />
                    Upload
                </button>

                {/* Start/Stop Buttons */}
                <button
                    onClick={onStart}
                    disabled={isActive}
                    className="btn-success no-drag"
                >
                    <Play className="w-4 h-4" />
                    Start
                </button>
                <button
                    onClick={onStop}
                    disabled={!isActive}
                    className="btn-danger no-drag"
                >
                    <Square className="w-4 h-4" />
                    Stop
                </button>
            </div>
        </div>
    );
}
