import { useState, useEffect } from 'react';
import { X, Sparkles, FileText, Video, Globe, Check } from 'lucide-react';
import { api, ProcessedSession } from '../services/api';

interface NotesModalProps {
    isOpen: boolean;
    onClose: () => void;
    onNotesGenerated: (notes: string, sessionCount: number) => void;
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
}

export default function NotesModal({ isOpen, onClose, onNotesGenerated, showNotification }: NotesModalProps) {
    const [sessions, setSessions] = useState<ProcessedSession[]>([]);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);

    useEffect(() => {
        if (isOpen) {
            fetchSessions();
        }
    }, [isOpen]);

    const fetchSessions = async () => {
        setLoading(true);
        try {
            const response = await api.getProcessedSessions();
            setSessions(response.sessions);
            // Pre-select indexed sessions
            const indexedIds = new Set(
                response.sessions.filter(s => s.indexed).map(s => s.id)
            );
            setSelectedIds(indexedIds);
        } catch (error) {
            showNotification('Failed to load sessions', 'error');
        } finally {
            setLoading(false);
        }
    };

    const toggleSession = (sessionId: string) => {
        const newSelected = new Set(selectedIds);
        if (newSelected.has(sessionId)) {
            newSelected.delete(sessionId);
        } else {
            newSelected.add(sessionId);
        }
        setSelectedIds(newSelected);
    };

    const selectAll = () => {
        setSelectedIds(new Set(sessions.map(s => s.id)));
    };

    const selectNone = () => {
        setSelectedIds(new Set());
    };

    const handleGenerate = async () => {
        if (selectedIds.size === 0) {
            showNotification('Please select at least one session', 'error');
            return;
        }

        setGenerating(true);
        try {
            const response = await api.generateNotes(Array.from(selectedIds));
            if (response.status === 'success' && response.notes) {
                onNotesGenerated(response.notes, response.session_count || selectedIds.size);
                onClose();
            } else {
                showNotification(`❌ ${response.message || 'Failed to generate notes'}`, 'error');
            }
        } catch (error) {
            showNotification('❌ Failed to generate notes', 'error');
        } finally {
            setGenerating(false);
        }
    };

    const getScenarioIcon = (type: string | null) => {
        switch (type) {
            case 'VIDEO':
                return <Video className="w-4 h-4 text-red-400" />;
            case 'DOC':
                return <FileText className="w-4 h-4 text-blue-400" />;
            case 'WEB':
                return <Globe className="w-4 h-4 text-green-400" />;
            default:
                return <FileText className="w-4 h-4 text-gray-400" />;
        }
    };

    const formatDate = (dateString: string | null) => {
        if (!dateString) return 'Unknown date';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
            <div className="glass rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col mx-4">
                {/* Header */}
                <div className="flex items-center justify-between p-5 border-b border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 gradient-purple rounded-xl flex items-center justify-center">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-medium text-white">Notes Generation</h2>
                            <p className="text-xs text-gray-400">Select activities to generate notes from</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-white transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-5">
                    {loading ? (
                        <div className="flex items-center justify-center py-12">
                            <div className="spinner w-8 h-8" />
                        </div>
                    ) : sessions.length === 0 ? (
                        <div className="text-center py-12">
                            <div className="text-4xl mb-4">📭</div>
                            <h3 className="text-lg font-medium text-white mb-2">No processed activities</h3>
                            <p className="text-gray-400 text-sm">
                                Process some Video, Document, or Web activities first to generate notes.
                            </p>
                        </div>
                    ) : (
                        <>
                            {/* Selection controls */}
                            <div className="flex items-center justify-between mb-4">
                                <span className="text-sm text-gray-400">
                                    {selectedIds.size} of {sessions.length} selected
                                </span>
                                <div className="flex gap-2">
                                    <button
                                        onClick={selectAll}
                                        className="text-xs text-coral-400 hover:text-coral-300 transition-colors"
                                    >
                                        Select All
                                    </button>
                                    <span className="text-gray-600">|</span>
                                    <button
                                        onClick={selectNone}
                                        className="text-xs text-gray-400 hover:text-gray-300 transition-colors"
                                    >
                                        Clear
                                    </button>
                                </div>
                            </div>

                            {/* Session list */}
                            <div className="space-y-2">
                                {sessions.map((session) => (
                                    <button
                                        key={session.id}
                                        onClick={() => toggleSession(session.id)}
                                        className={`w-full flex items-center gap-3 p-3 rounded-xl border transition-all ${selectedIds.has(session.id)
                                                ? 'border-coral-500/50 bg-coral-500/10'
                                                : 'border-white/10 bg-white/5 hover:bg-white/10'
                                            }`}
                                    >
                                        {/* Checkbox */}
                                        <div className={`w-5 h-5 rounded flex items-center justify-center border ${selectedIds.has(session.id)
                                                ? 'bg-coral-500 border-coral-500'
                                                : 'border-gray-500'
                                            }`}>
                                            {selectedIds.has(session.id) && (
                                                <Check className="w-3 h-3 text-white" />
                                            )}
                                        </div>

                                        {/* Icon */}
                                        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                                            {getScenarioIcon(session.type)}
                                        </div>

                                        {/* Info */}
                                        <div className="flex-1 text-left">
                                            <div className="text-sm text-white truncate">
                                                {session.title || 'Untitled Session'}
                                            </div>
                                            <div className="text-xs text-gray-500">
                                                {session.type} • {formatDate(session.start_time)}
                                            </div>
                                        </div>

                                        {/* Indexed badge */}
                                        {session.indexed && (
                                            <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400">
                                                Indexed
                                            </span>
                                        )}
                                    </button>
                                ))}
                            </div>
                        </>
                    )}
                </div>

                {/* Footer */}
                <div className="p-5 border-t border-white/10 flex items-center justify-between">
                    <p className="text-xs text-gray-500">
                        {sessions.length > 0 && !loading && (
                            <>Only indexed sessions will be used for note generation</>
                        )}
                    </p>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="btn-secondary"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleGenerate}
                            disabled={generating || selectedIds.size === 0}
                            className="btn-primary"
                        >
                            {generating ? (
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
                </div>
            </div>
        </div>
    );
}
