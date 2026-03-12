import { useState, useEffect, useCallback } from 'react';
import { Sparkles, RefreshCw, FileText, Video, Globe, Trash2, Copy, Check, Download, BookOpen } from 'lucide-react';
import NotesModal from './NotesModal';
import ConfirmDialog from './ConfirmDialog';
import { api, ProcessedSession } from '../services/api';

interface SavedNote {
    id: string;
    title: string;
    content: string;
    session_ids: string;
    created_at: string;
    updated_at: string;
}

interface NotesViewProps {
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
}

export default function NotesView({ showNotification }: NotesViewProps) {
    const [sessions, setSessions] = useState<ProcessedSession[]>([]);
    const [savedNotes, setSavedNotes] = useState<SavedNote[]>([]);
    const [loading, setLoading] = useState(true);
    const [activeNote, setActiveNote] = useState<SavedNote | null>(null);
    const [showNotesModal, setShowNotesModal] = useState(false);
    const [copied, setCopied] = useState(false);
    const [viewMode, setViewMode] = useState<'generated' | 'saved'>('saved');
    const [deleteDialog, setDeleteDialog] = useState<{
        isOpen: boolean;
        type: 'session' | 'note';
        id: string;
        title: string;
    }>({ isOpen: false, type: 'session', id: '', title: '' });

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const [sessionsData, notesData] = await Promise.all([
                api.getProcessedSessions(),
                api.getNotes()
            ]);
            setSessions(sessionsData.sessions);
            setSavedNotes(notesData.notes);
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

    const handleNotesGenerated = (_generatedNotes: string, sessionCount: number) => {
        showNotification(`✅ Notes generated from ${sessionCount} session(s)`, 'success');
        fetchData();
        setViewMode('saved');
    };

    const handleCopyNotes = async () => {
        if (activeNote) {
            await navigator.clipboard.writeText(activeNote.content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
            showNotification('📋 Notes copied to clipboard', 'success');
        }
    };

    const handleDownloadNotes = async () => {
        if (activeNote) {
            try {
                const result = await api.downloadNote(activeNote.id);
                if (result.status === 'success') {
                    const blob = new Blob([result.content], { type: 'text/markdown' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = result.filename;
                    a.click();
                    URL.revokeObjectURL(url);
                    showNotification('📥 Notes downloaded', 'success');
                }
            } catch (error) {
                showNotification('❌ Failed to download notes', 'error');
            }
        }
    };

    const handleDeleteClick = (type: 'session' | 'note', id: string, title: string) => {
        setDeleteDialog({ isOpen: true, type, id, title });
    };

    const handleConfirmDelete = async () => {
        try {
            if (deleteDialog.type === 'note') {
                await api.deleteNote(deleteDialog.id);
                showNotification('🗑️ Note deleted successfully', 'success');
                if (activeNote?.id === deleteDialog.id) {
                    setActiveNote(null);
                }
            } else {
                await api.deleteSession(deleteDialog.id);
                showNotification('🗑️ Session deleted successfully', 'success');
            }
            setDeleteDialog({ isOpen: false, type: 'session', id: '', title: '' });
            fetchData();
        } catch (error) {
            console.error('Failed to delete:', error);
            showNotification('❌ Failed to delete', 'error');
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

            {/* View Toggle */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setViewMode('saved')}
                    className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                        viewMode === 'saved' 
                            ? 'bg-purple-500/20 text-purple-400' 
                            : 'glass text-gray-400 hover:text-white'
                    }`}
                >
                    <BookOpen className="w-4 h-4" />
                    Saved Notes
                    {savedNotes.length > 0 && (
                        <span className="px-2 py-0.5 bg-purple-500/30 rounded-full text-xs">
                            {savedNotes.length}
                        </span>
                    )}
                </button>
                <button
                    onClick={() => setViewMode('generated')}
                    className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                        viewMode === 'generated' 
                            ? 'bg-purple-500/20 text-purple-400' 
                            : 'glass text-gray-400 hover:text-white'
                    }`}
                >
                    <Sparkles className="w-4 h-4" />
                    Generate New
                </button>
            </div>

            {/* Saved Notes View */}
            {viewMode === 'saved' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Notes List */}
                    <div className="lg:col-span-1 space-y-3">
                        <div className="flex items-center justify-between mb-2">
                            <h2 className="text-lg font-light text-white">Your Notes</h2>
                            <span className="text-xs text-gray-500">{savedNotes.length} notes</span>
                        </div>
                        
                        {loading && savedNotes.length === 0 ? (
                            <div className="glass rounded-xl p-8 text-center">
                                <div className="spinner w-8 h-8 mx-auto" />
                            </div>
                        ) : savedNotes.length === 0 ? (
                            <div className="glass rounded-xl p-8 text-center">
                                <div className="text-4xl mb-4">📝</div>
                                <h3 className="text-lg font-medium text-white mb-2">No saved notes</h3>
                                <p className="text-gray-400 text-sm mb-4">
                                    Generate notes from your sessions to save them here.
                                </p>
                                <button
                                    onClick={() => setViewMode('generated')}
                                    className="btn-primary"
                                >
                                    <Sparkles className="w-4 h-4" />
                                    Generate Notes
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {savedNotes.map((note) => (
                                    <button
                                        key={note.id}
                                        onClick={() => setActiveNote(note)}
                                        className={`w-full text-left p-3 rounded-xl border transition-all ${
                                            activeNote?.id === note.id
                                                ? 'border-purple-500/50 bg-purple-500/10'
                                                : 'border-white/10 bg-white/5 hover:bg-white/10'
                                        }`}
                                    >
                                        <div className="text-sm text-white truncate mb-1">
                                            {note.title}
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <div className="text-xs text-gray-500">
                                                {formatDate(note.created_at)}
                                            </div>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeleteClick('note', note.id, note.title);
                                                }}
                                                className="p-1 rounded hover:bg-red-500/20 text-gray-500 hover:text-red-400"
                                            >
                                                <Trash2 className="w-3 h-3" />
                                            </button>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Note Content */}
                    <div className="lg:col-span-2">
                        {activeNote ? (
                            <div className="glass rounded-xl p-5 h-full">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-lg font-medium text-white truncate">
                                        {activeNote.title}
                                    </h3>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={handleCopyNotes}
                                            className="btn-secondary text-sm"
                                            title="Copy to clipboard"
                                        >
                                            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                                        </button>
                                        <button
                                            onClick={handleDownloadNotes}
                                            className="btn-secondary text-sm"
                                            title="Download as markdown"
                                        >
                                            <Download className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                                <div className="text-xs text-gray-500 mb-4">
                                    Created: {formatDate(activeNote.created_at)}
                                </div>
                                <div 
                                    className="prose prose-invert prose-sm max-w-none bg-white/5 rounded-xl p-4 border border-white/10 overflow-y-auto max-h-[60vh]"
                                    dangerouslySetInnerHTML={{
                                        __html: renderNotes(activeNote.content)
                                    }}
                                />
                            </div>
                        ) : (
                            <div className="glass rounded-xl p-8 text-center h-full flex flex-col items-center justify-center">
                                <BookOpen className="w-12 h-12 text-gray-600 mb-4" />
                                <p className="text-gray-500">
                                    Select a note from the list to view its content
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Generate New View */}
            {viewMode === 'generated' && (
                <>
                    {/* Stats Summary */}
                    <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="glass rounded-xl p-4">
                            <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Total Sessions</div>
                            <div className="text-2xl font-semibold text-coral-500">{sessions.length}</div>
                        </div>
                        <div className="glass rounded-xl p-4">
                            <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Indexed Sessions</div>
                            <div className="text-2xl font-semibold text-coral-500">{indexedSessions.length}</div>
                        </div>
                        <div className="glass rounded-xl p-4">
                            <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Saved Notes</div>
                            <div className="text-2xl font-semibold text-coral-500">{savedNotes.length}</div>
                        </div>
                    </div>

                    {/* Generate Button */}
                    <div className="glass rounded-xl p-5 mb-6">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 gradient-purple rounded-lg flex items-center justify-center">
                                    <Sparkles className="w-4 h-4 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-medium text-white">Generate New Notes</h3>
                                    <p className="text-xs text-gray-500">Create study notes from your sessions</p>
                                </div>
                            </div>
                            <button
                                onClick={handleOpenNotesModal}
                                className="btn-primary"
                            >
                                <Sparkles className="w-4 h-4" />
                                Generate Notes
                            </button>
                        </div>
                    </div>

                    {/* Processed Sessions */}
                    <div>
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
                                        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                                            {getScenarioIcon(session.type)}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="text-sm text-white truncate">
                                                {session.title || 'Untitled Session'}
                                            </div>
                                            <div className="text-xs text-gray-500">
                                                {session.type} • {formatDate(session.start_time)}
                                            </div>
                                        </div>
                                        {session.indexed ? (
                                            <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400">
                                                Indexed
                                            </span>
                                        ) : (
                                            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-500/20 text-gray-400">
                                                Not Indexed
                                            </span>
                                        )}
                                        <button
                                            onClick={() => handleDeleteClick('session', session.id, session.title || 'Session')}
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
                </>
            )}

            {/* Notes Modal */}
            <NotesModal
                isOpen={showNotesModal}
                onClose={() => setShowNotesModal(false)}
                onNotesGenerated={handleNotesGenerated}
                showNotification={showNotification}
            />

            {/* Delete Confirmation Dialog */}
            <ConfirmDialog
                isOpen={deleteDialog.isOpen}
                title={deleteDialog.type === 'note' ? 'Delete Note' : 'Delete Session'}
                message={`Are you sure you want to delete "${deleteDialog.title}"? This action cannot be undone.`}
                confirmLabel="Delete"
                cancelLabel="Cancel"
                onConfirm={handleConfirmDelete}
                onCancel={() => setDeleteDialog({ isOpen: false, type: 'session', id: '', title: '' })}
                destructive
            />
        </div>
    );
}
