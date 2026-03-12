import { useState, useEffect, useCallback } from 'react';
import { Sparkles, RefreshCw, FileText, Video, Globe, Trash2, Copy, Check } from 'lucide-react';
import NotesModal from './NotesModal';
import ConfirmDialog from './ConfirmDialog';
import { api, ProcessedSession } from '../services/api';

interface NotesViewProps {
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
}

export default function NotesView({ showNotification }: NotesViewProps) {
    const [sessions, setSessions] = useState<ProcessedSession[]>([]);
    const [loading, setLoading] = useState(true);
    const [notes, setNotes] = useState<string | null>(null);
    const [showNotesModal, setShowNotesModal] = useState(false);
    const [copied, setCopied] = useState(false);
    const [deleteDialog, setDeleteDialog] = useState<{
        isOpen: boolean;
        session: ProcessedSession | null;
    }>({ isOpen: false, session: null });

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const sessionsData = await api.getProcessedSessions();
            setSessions(sessionsData.sessions);
        } catch (error) {
            console.error('Failed to fetch data:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, [fetchData]);

    const handleOpenNotesModal = () => {
        setShowNotesModal(true);
    };

    const handleNotesGenerated = (generatedNotes: string, sessionCount: number) => {
        setNotes(generatedNotes);
        showNotification(`✅ Notes generated from ${sessionCount} session(s)`, 'success');
    };

    const handleCopyNotes = async () => {
        if (notes) {
            await navigator.clipboard.writeText(notes);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
            showNotification('📋 Notes copied to clipboard', 'success');
        }
    };

    const handleClearNotes = () => {
        setNotes(null);
        showNotification('🗑️ Notes cleared', 'info');
    };

    const handleDeleteClick = (session: ProcessedSession) => {
        setDeleteDialog({ isOpen: true, session });
    };

    const handleConfirmDelete = async () => {
        if (!deleteDialog.session?.id) {
            setDeleteDialog({ isOpen: false, session: null });
            return;
        }

        try {
            await api.deleteSession(deleteDialog.session.id);
            setDeleteDialog({ isOpen: false, session: null });
            showNotification('🗑️ Session deleted successfully', 'success');
            fetchData();
        } catch (error) {
            console.error('Failed to delete session:', error);
            showNotification('❌ Failed to delete session', 'error');
            setDeleteDialog({ isOpen: false, session: null });
        }
    };

    const handleCancelDelete = () => {
        setDeleteDialog({ isOpen: false, session: null });
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

    const renderNotes = (notesContent: string) => {
        return notesContent
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/- (.*?)(?=\n|$)/g, '<li>$1</li>')
            .replace(/\n/g, '<br>');
    };

    const indexedSessions = sessions.filter(s => s.indexed);

    return (
        <div className="animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-3xl font-light text-white">Notes</h1>
                <button
                    onClick={() => fetchData()}
                    className="btn-secondary"
                    disabled={loading}
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            {/* Stats Summary */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="glass rounded-xl p-4">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Total Sessions</div>
                    <div className="text-2xl font-semibold text-coral-500">{sessions.length}</div>
                </div>
                <div className="glass rounded-xl p-4">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Indexed Sessions</div>
                    <div className="text-2xl font-semibold text-coral-500">{indexedSessions.length}</div>
                </div>
            </div>

            {/* Notes Section */}
            <div className="glass rounded-xl p-5 mb-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 gradient-purple rounded-lg flex items-center justify-center">
                            <Sparkles className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h3 className="font-medium text-white">Generated Notes</h3>
                            <p className="text-xs text-gray-500">AI-generated study notes from your sessions</p>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        {notes && (
                            <>
                                <button
                                    onClick={handleCopyNotes}
                                    className="btn-secondary text-sm"
                                    title="Copy to clipboard"
                                >
                                    {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                                </button>
                                <button
                                    onClick={handleClearNotes}
                                    className="btn-secondary text-sm"
                                    title="Clear notes"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </>
                        )}
                        <button
                            onClick={handleOpenNotesModal}
                            className="btn-primary text-sm"
                        >
                            <Sparkles className="w-4 h-4" />
                            Generate Notes
                        </button>
                    </div>
                </div>

                <div className="text-sm text-gray-400 leading-relaxed min-h-[120px]">
                    {notes ? (
                        <div
                            className="prose prose-invert prose-sm max-w-none bg-white/5 rounded-xl p-4 border border-white/10"
                            dangerouslySetInnerHTML={{
                                __html: renderNotes(notes)
                            }}
                        />
                    ) : (
                        <div className="flex flex-col items-center justify-center py-8 text-gray-500">
                            <Sparkles className="w-8 h-8 mb-2 opacity-50" />
                            <p className="italic">Click "Generate Notes" to select sessions and create study notes...</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Processed Sessions */}
            <div className="mb-6">
                <h2 className="text-xl font-light text-white mb-4">Processed Sessions</h2>
                {loading && sessions.length === 0 ? (
                    <div className="flex items-center justify-center py-12">
                        <div className="spinner w-8 h-8" />
                    </div>
                ) : sessions.length === 0 ? (
                    <div className="glass rounded-2xl p-8 text-center">
                        <div className="text-4xl mb-4">📭</div>
                        <h3 className="text-lg font-medium text-white mb-2">No processed sessions</h3>
                        <p className="text-gray-400 text-sm">
                            Process some Video, Document, or Web activities first to generate notes.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {sessions.map((session) => (
                            <div
                                key={session.id}
                                className={`flex items-center gap-3 p-3 rounded-xl border ${
                                    session.indexed
                                        ? 'border-green-500/30 bg-green-500/5'
                                        : 'border-white/10 bg-white/5'
                                }`}
                            >
                                {/* Icon */}
                                <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                                    {getScenarioIcon(session.type)}
                                </div>

                                {/* Info */}
                                <div className="flex-1 min-w-0">
                                    <div className="text-sm text-white truncate">
                                        {session.title || 'Untitled Session'}
                                    </div>
                                    <div className="text-xs text-gray-500">
                                        {session.type} • {formatDate(session.start_time)}
                                    </div>
                                </div>

                                {/* Indexed badge */}
                                {session.indexed ? (
                                    <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400">
                                        Indexed
                                    </span>
                                ) : (
                                    <span className="text-xs px-2 py-0.5 rounded-full bg-gray-500/20 text-gray-400">
                                        Not Indexed
                                    </span>
                                )}

                                {/* Delete button */}
                                <button
                                    onClick={() => handleDeleteClick(session)}
                                    className="p-2 rounded-lg hover:bg-red-500/20 text-gray-400 hover:text-red-400 transition-colors"
                                    title="Delete session"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Delete Confirmation Dialog */}
            <ConfirmDialog
                isOpen={deleteDialog.isOpen}
                title="Delete Session"
                message={`Are you sure you want to delete "${deleteDialog.session?.title}"? This will remove all associated data including screenshots, transcripts, and media files.`}
                confirmLabel="Delete"
                cancelLabel="Cancel"
                onConfirm={handleConfirmDelete}
                onCancel={handleCancelDelete}
                destructive
            />

            {/* Notes Modal */}
            <NotesModal
                isOpen={showNotesModal}
                onClose={() => setShowNotesModal(false)}
                onNotesGenerated={handleNotesGenerated}
                showNotification={showNotification}
            />
        </div>
    );
}
