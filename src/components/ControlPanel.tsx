import { Play, Square, Upload, Activity, Image, Loader2, ClipboardList, BookOpen } from 'lucide-react';
import { useRef, useState } from 'react';
import { StatusResponse, api } from '../services/api';

interface ControlPanelProps {
    status: StatusResponse | null;
    onStart: () => void;
    onStop: () => void;
}

export default function ControlPanel({ status, onStart, onStop }: ControlPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [showScreenshot, setShowScreenshot] = useState(true);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadedImage, setUploadedImage] = useState<{ path: string; title: string; summary: string; type: string; educational: string } | null>(null);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        setUploadProgress(10);
        setUploadedImage(null);

        try {
            setUploadProgress(30);
            const result = await api.uploadImage(file);
            setUploadProgress(90);
            
            if (result.status === 'success') {
                setUploadedImage({
                    path: result.image_path || '',
                    title: result.title || 'Uploaded File',
                    summary: result.description || result.summary || '',
                    type: result.type || 'UNKNOWN',
                    educational: result.educational || 'no'
                });
            }
            setUploadProgress(100);
            
            // Refresh after a short delay
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        } catch (error) {
            console.error('Upload failed:', error);
            setUploading(false);
            setUploadProgress(0);
        }

        // Reset input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Determine if monitoring is active from the new API structure
    const isMonitoring = status?.state?.monitoring_enabled || status?.orchestrator?.is_monitoring || false;
    const currentState = status?.state?.state || 'MONITORING';

    // Get current screenshot path (only from pending scenario now)
    const screenshotPath = status?.state?.pending_scenario?.screenshot_path || null;

    // Get browser context from extension
    const browserContext = status?.orchestrator?.browser_context;

    // Total activities
    const totalActivities = status?.database?.sessions || status?.total_responses || 0;

    return (
        <div className="space-y-4 mb-6">
            {/* Control Bar */}
            <div className="glass rounded-2xl p-4 flex items-center justify-between">
                {/* Left: Filter tabs (placeholder for future) */}
                <div className="flex items-center gap-2">
                    <button className="px-4 py-2 rounded-lg bg-white/10 text-sm font-medium text-white">
                        <ClipboardList className="w-4 h-4 inline mr-1" /> All tasks
                    </button>
                </div>

                {/* Right: Status and controls */}
                <div className="flex items-center gap-4">
                    {/* Status Badge */}
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${isMonitoring ? 'bg-emerald-500/20 text-emerald-400' : 'bg-white/10 text-gray-400'
                        }`}>
                        <span className={`w-2 h-2 rounded-full ${isMonitoring ? 'bg-emerald-400' : 'bg-gray-500'
                            }`} />
                        <span className="text-sm font-medium">
                            {isMonitoring ? (currentState === 'AWAITING_USER_INPUT' ? 'Paused' : 'Recording') : 'Idle'}
                        </span>
                    </div>

                    {/* Stats */}
                    {status && (
                        <div className="flex items-center gap-2 text-sm text-gray-400">
                            <Activity className="w-4 h-4" />
                            <span>{totalActivities} activities</span>
                        </div>
                    )}

                    {/* Toggle Screenshot Preview */}
                    <button
                        onClick={() => setShowScreenshot(!showScreenshot)}
                        className={`p-2 rounded-lg transition-colors ${showScreenshot ? 'bg-indigo-500/20 text-indigo-400' : 'bg-white/10 text-gray-400 hover:bg-white/15'
                            }`}
                        title={showScreenshot ? 'Hide context preview' : 'Show context preview'}
                    >
                        <Image className="w-4 h-4" />
                    </button>

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
                        disabled={isMonitoring}
                        className={`btn-success no-drag ${isMonitoring ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                        <Play className="w-4 h-4" />
                        Start
                    </button>
                    <button
                        onClick={onStop}
                        disabled={!isMonitoring}
                        className={`btn-danger no-drag ${!isMonitoring ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                        <Square className="w-4 h-4" />
                        Stop
                    </button>
                </div>
            </div>

            {/* Browser Context Preview (from extension) */}
            {showScreenshot && browserContext?.url && (
                <div className="glass rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-medium text-gray-300">Browser Context</h3>
                        <span className={`text-xs px-2 py-1 rounded ${browserContext.scenario === 'VIDEO' ? 'bg-red-500/20 text-red-400' :
                                browserContext.scenario === 'DOC' ? 'bg-blue-500/20 text-blue-400' :
                                    browserContext.scenario === 'WEB' ? 'bg-green-500/20 text-green-400' :
                                        'bg-gray-500/20 text-gray-400'
                            }`}>
                            {browserContext.scenario || 'UNKNOWN'}
                        </span>
                    </div>
                    <div className="text-sm text-white truncate mb-1">
                        {browserContext.title || 'Untitled'}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                        {browserContext.url}
                    </div>
                </div>
            )}

            {/* Screenshot Preview (only when pending scenario has screenshot) */}
            {showScreenshot && screenshotPath && (
                <div className="glass rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-medium text-gray-300">Pending Screenshot</h3>
                        <span className="text-xs text-gray-500">
                            {currentState === 'AWAITING_USER_INPUT' ? 'Awaiting response...' : 'Processing...'}
                        </span>
                    </div>
                    <div className="relative rounded-lg overflow-hidden bg-black/20">
                        <img
                            src={api.getScreenshotUrl(screenshotPath)}
                            alt="Screenshot from pending scenario"
                            className="w-full h-auto max-h-64 object-contain"
                            onError={(e) => {
                                // Hide image on error
                                (e.target as HTMLImageElement).style.display = 'none';
                            }}
                        />
                    </div>
                </div>
            )}

            {/* Upload Progress / Uploaded Image Preview */}
            {uploading && (
                <div className="glass rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-medium text-gray-300">Processing Upload</h3>
                        <span className="text-xs text-gray-500 flex items-center gap-2">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Analyzing image...
                        </span>
                    </div>
                    {/* Progress bar */}
                    <div className="w-full bg-gray-700 rounded-full h-2 mb-3">
                        <div 
                            className="bg-coral-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${uploadProgress}%` }}
                        />
                    </div>
                    <p className="text-xs text-gray-400">
                        {uploadProgress < 30 && "Uploading image..."}
                        {uploadProgress >= 30 && uploadProgress < 90 && "Analyzing with AI vision model..."}
                        {uploadProgress >= 90 && "Finalizing..."}
                    </p>
                </div>
            )}

            {/* Uploaded Image Preview (after processing) */}
            {uploadedImage && !uploading && uploadedImage.path && (
                <div className="glass rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-medium text-gray-300">Uploaded File Processed</h3>
                        <span className="text-xs px-2 py-1 rounded bg-green-500/20 text-green-400">
                            ✓ Complete
                        </span>
                    </div>
                    <div className="relative rounded-lg overflow-hidden bg-black/20 mb-3">
                        <img
                            src={api.getScreenshotUrl(uploadedImage.path)}
                            alt="Uploaded and processed"
                            className="w-full h-auto max-h-64 object-contain"
                            onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                            }}
                        />
                    </div>
                    {uploadedImage.title && (
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <span className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">
                                {uploadedImage.type}
                            </span>
                            {uploadedImage.educational === 'yes' && (
                                <span className="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">
                                    <BookOpen className="w-3 h-3 inline mr-1" /> Educational
                                </span>
                            )}
                            <div className="text-sm text-white font-medium">
                                {uploadedImage.title}
                            </div>
                        </div>
                    )}
                    {uploadedImage.summary && (
                        <div className="text-xs text-gray-400 line-clamp-3">
                            {uploadedImage.summary}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

