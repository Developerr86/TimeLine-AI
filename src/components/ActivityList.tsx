import { useState, useEffect, useCallback } from 'react';
import { Play, Square, X, Video, FileText, Globe, Clock, ExternalLink, Loader2 } from 'lucide-react';
import { api } from '../services/api';

export interface PendingActivity {
    id: string;
    scenario: 'VIDEO' | 'DOC' | 'WEB';
    url: string;
    title: string;
    detected_at: string;
    processed: boolean;
    processing: boolean;
}

interface ActivityListProps {
    onProcess?: (activity: PendingActivity) => void;
    onDismiss?: (activity: PendingActivity) => void;
    showNotification?: (message: string, type: 'info' | 'success' | 'error') => void;
}

const scenarioIcons = {
    VIDEO: Video,
    DOC: FileText,
    WEB: Globe,
};

const scenarioColors = {
    VIDEO: 'from-red-500 to-pink-500',
    DOC: 'from-blue-500 to-cyan-500',
    WEB: 'from-green-500 to-emerald-500',
};

export default function ActivityList({ onProcess, onDismiss, showNotification }: ActivityListProps) {
    const [activities, setActivities] = useState<PendingActivity[]>([]);
    const [loading, setLoading] = useState(false);
    const [processingId, setProcessingId] = useState<string | null>(null);
    const [stoppingId, setStoppingId] = useState<string | null>(null);

    const fetchActivities = useCallback(async () => {
        try {
            const data = await api.getActivities();
            setActivities(data.activities || []);
        } catch (error) {
            console.error('Failed to fetch activities:', error);
        }
    }, []);

    useEffect(() => {
        fetchActivities();
        const interval = setInterval(fetchActivities, 3000);
        return () => clearInterval(interval);
    }, [fetchActivities]);

    const handleProcess = async (activity: PendingActivity) => {
        setProcessingId(activity.id);
        setLoading(true);
        try {
            const result = await api.processActivity(activity.id);
            if (result.status === 'processing') {
                showNotification?.(`▶️ Started processing: ${activity.title}`, 'success');
                onProcess?.(activity);
            } else {
                showNotification?.(`❌ Failed to start processing`, 'error');
            }
            fetchActivities();
        } catch (error) {
            console.error('Failed to process activity:', error);
            showNotification?.(`❌ Error: ${error}`, 'error');
        } finally {
            setLoading(false);
            setProcessingId(null);
        }
    };

    const handleDismiss = async (activity: PendingActivity) => {
        try {
            const result = await api.dismissActivity(activity.id);
            if (result.status === 'dismissed') {
                showNotification?.(`🗑️ Dismissed: ${activity.title}`, 'info');
                onDismiss?.(activity);
            }
            fetchActivities();
        } catch (error) {
            console.error('Failed to dismiss activity:', error);
        }
    };

    const handleStop = async (activity: PendingActivity) => {
        setStoppingId(activity.id);
        try {
            const result = await api.stopActivity();
            if (result.status === 'stopped') {
                showNotification?.(`⏹️ Stopped processing: ${activity.title}`, 'info');
            } else {
                showNotification?.(`❌ Failed to stop processing`, 'error');
            }
            fetchActivities();
        } catch (error) {
            console.error('Failed to stop activity:', error);
            showNotification?.(`❌ Error: ${error}`, 'error');
        } finally {
            setStoppingId(null);
        }
    };

    const formatTimeAgo = (dateString: string) => {
        const date = new Date(dateString);
        const now = new Date();
        const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

        if (seconds < 60) return 'just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    };

    if (activities.length === 0) {
        return null;
    }

    return (
        <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    Pending Activities
                    <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded-full text-xs">
                        {activities.length}
                    </span>
                </h3>
            </div>

            <div className="space-y-2">
                {activities.map((activity) => {
                    const Icon = scenarioIcons[activity.scenario] || Globe;
                    const gradientColor = scenarioColors[activity.scenario] || 'from-gray-500 to-gray-600';
                    const isProcessing = processingId === activity.id;

                    return (
                        <div
                            key={activity.id}
                            className={`glass rounded-xl p-4 transition-all duration-200 ${activity.processing ? 'ring-2 ring-purple-500/50' : ''
                                }`}
                        >
                            <div className="flex items-start gap-3">
                                {/* Icon */}
                                <div className={`p-2 rounded-lg bg-gradient-to-br ${gradientColor} flex-shrink-0`}>
                                    <Icon className="w-4 h-4 text-white" />
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-medium text-gray-400 uppercase">
                                            {activity.scenario}
                                        </span>
                                        <span className="text-xs text-gray-500">
                                            {formatTimeAgo(activity.detected_at)}
                                        </span>
                                        {activity.processing && (
                                            <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full text-xs flex items-center gap-1">
                                                <Loader2 className="w-3 h-3 animate-spin" />
                                                Processing
                                            </span>
                                        )}
                                    </div>

                                    <h4 className="text-sm font-medium text-gray-200 truncate mt-1">
                                        {activity.title}
                                    </h4>

                                    <a
                                        href={activity.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-xs text-gray-500 hover:text-gray-400 truncate flex items-center gap-1 mt-1"
                                    >
                                        <ExternalLink className="w-3 h-3 flex-shrink-0" />
                                        <span className="truncate">{activity.url}</span>
                                    </a>
                                </div>

                                {/* Actions */}
                                <div className="flex items-center gap-2 flex-shrink-0">
                                    {activity.processing ? (
                                        /* Show Stop button when processing */
                                        <button
                                            onClick={() => handleStop(activity)}
                                            disabled={stoppingId === activity.id}
                                            className="p-2 rounded-lg bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600 text-white transition-all duration-200 disabled:opacity-50"
                                            title="Stop processing"
                                        >
                                            {stoppingId === activity.id ? (
                                                <Loader2 className="w-4 h-4 animate-spin" />
                                            ) : (
                                                <Square className="w-4 h-4" />
                                            )}
                                        </button>
                                    ) : (
                                        /* Show Play/Dismiss buttons when not processing */
                                        <>
                                            <button
                                                onClick={() => handleProcess(activity)}
                                                disabled={loading}
                                                className="p-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white transition-all duration-200 disabled:opacity-50"
                                                title="Process this activity"
                                            >
                                                {isProcessing ? (
                                                    <Loader2 className="w-4 h-4 animate-spin" />
                                                ) : (
                                                    <Play className="w-4 h-4" />
                                                )}
                                            </button>
                                            <button
                                                onClick={() => handleDismiss(activity)}
                                                className="p-2 rounded-lg hover:bg-gray-700/50 text-gray-400 hover:text-gray-300 transition-all duration-200"
                                                title="Dismiss"
                                            >
                                                <X className="w-4 h-4" />
                                            </button>
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
