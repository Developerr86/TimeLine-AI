import { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import ActivityCard from './ActivityCard';
import ConfirmDialog from './ConfirmDialog';
import { api, Activity } from '../services/api';

interface TimelineViewProps {
    onRefresh: () => void;
}

export default function TimelineView({ onRefresh }: TimelineViewProps) {
    const [activities, setActivities] = useState<Activity[]>([]);
    const [loading, setLoading] = useState(true);
    const [deleteDialog, setDeleteDialog] = useState<{
        isOpen: boolean;
        activity: Activity | null;
    }>({ isOpen: false, activity: null });

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

    const handleRefresh = () => {
        setLoading(true);
        fetchActivities();
        onRefresh();
    };

    const handleDeleteClick = (activity: Activity) => {
        setDeleteDialog({ isOpen: true, activity });
    };

    const handleConfirmDelete = async () => {
        if (!deleteDialog.activity?.session_id) {
            setDeleteDialog({ isOpen: false, activity: null });
            return;
        }

        try {
            await api.deleteSession(deleteDialog.activity.session_id);
            setDeleteDialog({ isOpen: false, activity: null });
            // Refresh activities after deletion
            fetchActivities();
            onRefresh();
        } catch (error) {
            console.error('Failed to delete activity:', error);
            setDeleteDialog({ isOpen: false, activity: null });
        }
    };

    const handleCancelDelete = () => {
        setDeleteDialog({ isOpen: false, activity: null });
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
                        <ActivityCard 
                            key={activity.id || index} 
                            activity={activity} 
                            onDelete={handleDeleteClick}
                        />
                    ))}
                </div>
            )}

            {/* Delete Confirmation Dialog */}
            <ConfirmDialog
                isOpen={deleteDialog.isOpen}
                title="Delete Activity"
                message={`Are you sure you want to delete "${deleteDialog.activity?.title}"? This will remove all associated data including screenshots, transcripts, and media files.`}
                confirmLabel="Delete"
                cancelLabel="Cancel"
                onConfirm={handleConfirmDelete}
                onCancel={handleCancelDelete}
                destructive
            />
        </div>
    );
}
