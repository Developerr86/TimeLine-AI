import { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Sparkles, RefreshCw } from 'lucide-react';
import ActivityCard from './ActivityCard';
import { api, Activity } from '../services/api';

interface TimelineViewProps {
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
    onRefresh: () => void;
}

export default function TimelineView({ showNotification, onRefresh }: TimelineViewProps) {
    const [activities, setActivities] = useState<Activity[]>([]);
    const [loading, setLoading] = useState(true);
    const [notes, setNotes] = useState<string | null>(null);
    const [generatingNotes, setGeneratingNotes] = useState(false);

    const fetchActivities = useCallback(async () => {
        try {
            const data = await api.getResponses();
            setActivities(data);
        } catch (error) {
            console.error('Failed to fetch activities:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchActivities();
        const interval = setInterval(fetchActivities, 5000);
        return () => clearInterval(interval);
    }, [fetchActivities]);

    const handleGenerateNotes = async () => {
        setGeneratingNotes(true);
        try {
            const response = await api.generateNotes();
            if (response.status === 'success' && response.notes) {
                setNotes(response.notes);
                showNotification(`✅ Notes generated from ${response.activity_count} activities`, 'success');
            } else {
                showNotification(`❌ ${response.message || 'Failed to generate notes'}`, 'error');
            }
        } catch (error) {
            showNotification('❌ Failed to generate notes', 'error');
        } finally {
            setGeneratingNotes(false);
        }
    };

    const handleRefresh = () => {
        setLoading(true);
        fetchActivities();
        onRefresh();
    };

    const today = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

    return (
        <div className="animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-3xl font-light text-white">Timeline</h1>

                <div className="flex items-center gap-4">
                    {/* Date Navigation */}
                    <div className="glass rounded-xl px-4 py-2 flex items-center gap-3">
                        <button className="text-gray-400 hover:text-white transition-colors">
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        <span className="text-sm text-gray-300">Today, {today}</span>
                        <button className="text-gray-400 hover:text-white transition-colors">
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Refresh */}
                    <button
                        onClick={handleRefresh}
                        className="btn-secondary"
                        disabled={loading}
                    >
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            {/* Stats Summary */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="glass rounded-xl p-4">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Total Activities</div>
                    <div className="text-2xl font-semibold text-coral-500">{activities.length}</div>
                </div>
                <div className="glass rounded-xl p-4">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Today's Sessions</div>
                    <div className="text-2xl font-semibold text-coral-500">
                        {activities.filter(a => a.timestamp.startsWith(new Date().toISOString().split('T')[0])).length}
                    </div>
                </div>
                <div className="glass rounded-xl p-4">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Screenshot Interval</div>
                    <div className="text-2xl font-semibold text-coral-500">5s</div>
                </div>
            </div>

            {/* AI Notes Section */}
            <div className="glass rounded-xl p-5 mb-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 gradient-purple rounded-lg flex items-center justify-center">
                            <Sparkles className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h3 className="font-medium text-white">AI Activity Notes</h3>
                            <p className="text-xs text-gray-500">Based on recent history</p>
                        </div>
                    </div>
                    <button
                        onClick={handleGenerateNotes}
                        disabled={generatingNotes}
                        className="btn-primary text-sm"
                    >
                        {generatingNotes ? (
                            <>
                                <span className="spinner w-4 h-4" />
                                Generating...
                            </>
                        ) : (
                            <>
                                <Sparkles className="w-4 h-4" />
                                Generate Notes
                            </>
                        )}
                    </button>
                </div>

                <div className="text-sm text-gray-400 leading-relaxed">
                    {notes ? (
                        <div
                            className="prose prose-invert prose-sm max-w-none"
                            dangerouslySetInnerHTML={{
                                __html: notes
                                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                    .replace(/- (.*?)(?=\n|$)/g, '<li>$1</li>')
                                    .replace(/\n/g, '<br>')
                            }}
                        />
                    ) : (
                        <p className="italic">Click "Generate Notes" to analyze your recent activities using AI...</p>
                    )}
                </div>
            </div>

            {/* Activity Grid */}
            {loading && activities.length === 0 ? (
                <div className="flex items-center justify-center py-20">
                    <div className="spinner w-8 h-8" />
                </div>
            ) : activities.length === 0 ? (
                <div className="glass rounded-2xl p-12 text-center">
                    <div className="text-6xl mb-4">📸</div>
                    <h2 className="text-2xl font-medium text-white mb-2">No activities captured yet</h2>
                    <p className="text-gray-400 mb-6">Click "Start" to begin capturing and analyzing your screen activity.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
                    {activities.map((activity, index) => (
                        <ActivityCard key={activity.id || index} activity={activity} />
                    ))}
                </div>
            )}
        </div>
    );
}
