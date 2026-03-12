import { useNavigate } from 'react-router-dom';
import { api, Activity } from '../services/api';
import {
    Youtube,
    Globe,
    Github,
    Code,
    Smartphone,
    AlertTriangle,
    Trash2
} from 'lucide-react';

interface ActivityCardProps {
    activity: Activity;
    onDelete?: (activity: Activity) => void;
}

function getActivityIcon(title: string) {
    const lowerTitle = title.toLowerCase();

    if (lowerTitle.includes('code') || lowerTitle.includes('vs code') || lowerTitle.includes('vscode')) {
        return { icon: Code, color: 'from-blue-500 to-cyan-500' };
    }
    if (lowerTitle.includes('youtube') || lowerTitle.includes('video')) {
        return { icon: Youtube, color: 'from-red-500 to-pink-500' };
    }
    if (lowerTitle.includes('github')) {
        return { icon: Github, color: 'from-gray-600 to-gray-800' };
    }
    if (lowerTitle.includes('browse') || lowerTitle.includes('web') || lowerTitle.includes('chrome')) {
        return { icon: Globe, color: 'from-emerald-500 to-teal-500' };
    }

    return { icon: Smartphone, color: 'from-orange-400 to-pink-500' };
}

function formatTime(timestamp: string): string {
    try {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    } catch {
        return timestamp.slice(11, 16);
    }
}

export default function ActivityCard({ activity, onDelete }: ActivityCardProps) {
    const navigate = useNavigate();
    const { icon: IconComponent, color } = getActivityIcon(activity.title);
    const screenshotUrl = activity.image_path ? api.getScreenshotUrl(activity.image_path) : null;
    const hasSession = !!activity.session_id;

    const handleClick = () => {
        if (hasSession) {
            navigate(`/scenario/${activity.session_id}`);
        }
    };

    const handleDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        
        if (!activity.session_id) {
            return;
        }

        if (onDelete) {
            onDelete(activity);
        }
    };

    return (
        <div
            className={`glass rounded-2xl overflow-hidden card-hover animate-slide-up ${hasSession ? 'cursor-pointer hover:ring-2 hover:ring-purple-500/50' : ''
                }`}
            onClick={handleClick}
        >
            {/* Header */}
            <div className="glass-light p-4 flex items-center gap-3">
                <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg`}>
                    <IconComponent className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-white truncate">{activity.title}</h3>
                    <p className="text-sm text-gray-400">{formatTime(activity.timestamp)}</p>
                </div>
                {hasSession && (
                    <button
                        onClick={handleDelete}
                        className="p-2 rounded-lg hover:bg-red-500/20 text-gray-400 hover:text-red-400 transition-colors"
                        title="Delete activity"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                )}
            </div>

            {/* Screenshot Preview */}
            {activity.image_path && (
                <div className="h-48 bg-black/30 overflow-hidden">
                    <img
                        src={screenshotUrl || undefined}
                        alt="Screenshot"
                        className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                        onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.parentElement!.innerHTML = `
                <div class="w-full h-full flex items-center justify-center text-gray-500">
                  <span>📸 Screenshot unavailable</span>
                </div>
              `;
                        }}
                    />
                </div>
            )}

            {/* Body */}
            <div className="p-4">
                {/* Summary */}
                <div className="mb-4">
                    <div className="text-xs uppercase tracking-wider text-gray-500 mb-1">Summary</div>
                    <p className="text-sm text-gray-300 leading-relaxed">{activity.summary}</p>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="glass-light rounded-lg p-2.5">
                        <div className="flex items-center justify-between">
                            <span className="text-xs uppercase tracking-wider text-gray-500">Type</span>
                            <span className="text-xs font-medium text-coral-400 truncate max-w-[80px]">
                                {activity.model_name?.slice(0, 12) || activity.model}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Error Badge */}
                {activity.error && (
                    <div className="mt-3 flex items-center gap-2 bg-red-500/10 text-red-400 px-3 py-2 rounded-lg text-sm">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                        <span className="truncate">{activity.error}</span>
                    </div>
                )}
            </div>
        </div>
    );
}
