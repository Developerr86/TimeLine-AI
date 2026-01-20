import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    ArrowLeft, FileText, Clock, Cpu, Zap,
    Save, Edit3, X, Image, RefreshCw, Loader2, Database, Check
} from 'lucide-react';
import {
    api,
    SessionDetailsResponse,
    SessionFrame,
    TranscriptChunk
} from '../services/api';

interface ScenarioDetailsViewProps {
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
}

export default function ScenarioDetailsView({ showNotification }: ScenarioDetailsViewProps) {
    const { sessionId } = useParams<{ sessionId: string }>();
    const navigate = useNavigate();

    const [loading, setLoading] = useState(true);
    const [session, setSession] = useState<SessionDetailsResponse | null>(null);
    const [frames, setFrames] = useState<SessionFrame[]>([]);
    const [transcriptChunks, setTranscriptChunks] = useState<TranscriptChunk[]>([]);
    const [transcriptContent, setTranscriptContent] = useState('');
    const [isEditing, setIsEditing] = useState(false);
    const [editedContent, setEditedContent] = useState('');
    const [saving, setSaving] = useState(false);
    const [activeTab, setActiveTab] = useState<'transcript' | 'frames'>('transcript');
    const [indexing, setIndexing] = useState(false);
    const [indexed, setIndexed] = useState(false);

    const fetchSessionData = useCallback(async () => {
        if (!sessionId) return;

        try {
            const [detailsRes, framesRes, transcriptRes] = await Promise.all([
                api.getSessionDetails(sessionId),
                api.getSessionFrames(sessionId),
                api.getSessionTranscript(sessionId)
            ]);

            setSession(detailsRes);
            setFrames(framesRes.frames);
            setTranscriptChunks(transcriptRes.chunks);
            setTranscriptContent(transcriptRes.content);
        } catch (error) {
            console.error('Failed to fetch session data:', error);
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => {
        fetchSessionData();

        // Poll for updates if session is active
        const interval = setInterval(() => {
            if (session?.is_active) {
                fetchSessionData();
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [fetchSessionData, session?.is_active]);

    const handleSaveTranscript = async () => {
        if (!sessionId) return;

        setSaving(true);
        try {
            await api.updateSessionTranscript(sessionId, editedContent);
            setTranscriptContent(editedContent);
            setIsEditing(false);
            showNotification('✅ Transcript saved', 'success');
        } catch (error) {
            showNotification('❌ Failed to save transcript', 'error');
        } finally {
            setSaving(false);
        }
    };

    const handleStartEdit = () => {
        setEditedContent(transcriptContent);
        setIsEditing(true);
    };

    const handleCancelEdit = () => {
        setIsEditing(false);
        setEditedContent('');
    };

    const handleIndexTranscript = async () => {
        if (!sessionId || !transcriptContent) return;

        setIndexing(true);
        try {
            const result = await api.indexSession(sessionId);
            if (result.status === 'success') {
                setIndexed(true);
                showNotification(`✅ Indexed ${result.chunks_indexed} chunks into vector store`, 'success');
            } else {
                showNotification(`❌ ${result.message || 'Indexing failed'}`, 'error');
            }
        } catch (error) {
            console.error('Failed to index transcript:', error);
            showNotification('❌ Failed to index transcript', 'error');
        } finally {
            setIndexing(false);
        }
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <div className="spinner w-8 h-8" />
            </div>
        );
    }

    if (!session) {
        return (
            <div className="p-8">
                <button
                    onClick={() => navigate('/')}
                    className="flex items-center gap-2 text-gray-400 hover:text-white mb-6"
                >
                    <ArrowLeft className="w-5 h-5" />
                    Back to Timeline
                </button>
                <div className="glass rounded-2xl p-12 text-center">
                    <div className="text-6xl mb-4">❌</div>
                    <h2 className="text-2xl font-medium text-white mb-2">Session not found</h2>
                    <p className="text-gray-400">The session you're looking for doesn't exist.</p>
                </div>
            </div>
        );
    }

    const { transcriber_status } = session;
    const isActive = session.is_active;
    const sessionType = session.session.type;
    const isDocSession = sessionType === 'DOC';
    const isWebSession = sessionType === 'WEB';
    const isVideoSession = sessionType === 'VIDEO';
    const modelLoading = transcriber_status?.model_loading || false;
    const modelLoaded = transcriber_status?.model_loaded || false;
    const deviceType = transcriber_status?.device_type || 'cpu';
    const liveChunks = transcriber_status?.transcript_chunks || [];

    // Use live chunks if active, otherwise use saved chunks
    const displayChunks = isActive ? liveChunks : transcriptChunks;

    return (
        <div className="animate-fade-in p-6">
            {/* Header */}
            <div className="flex items-center gap-4 mb-6">
                <button
                    onClick={() => navigate('/')}
                    className="p-2 rounded-lg glass hover:bg-white/10 transition-colors"
                >
                    <ArrowLeft className="w-5 h-5 text-gray-400" />
                </button>

                <div className="flex-1">
                    <h1 className="text-2xl font-light text-white">
                        {session.session.title || 'Untitled Session'}
                    </h1>
                    <p className="text-sm text-gray-400 truncate max-w-xl">
                        {session.session.source_url}
                    </p>
                </div>

                {/* Status Badge */}
                {isActive && isVideoSession && (
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/20 text-green-400">
                        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        Recording
                    </div>
                )}
            </div>

            {/* Model Loading Status */}
            {isActive && isVideoSession && (
                <div className="glass rounded-xl p-4 mb-6">
                    <div className="flex items-center gap-4">
                        {modelLoading ? (
                            <>
                                <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
                                <div>
                                    <p className="text-white font-medium">Loading Whisper Model...</p>
                                    <p className="text-sm text-gray-400 flex items-center gap-2">
                                        {deviceType === 'cuda' ? (
                                            <><Zap className="w-4 h-4 text-yellow-400" /> GPU (CUDA)</>
                                        ) : (
                                            <><Cpu className="w-4 h-4 text-blue-400" /> CPU</>
                                        )}
                                    </p>
                                </div>
                            </>
                        ) : modelLoaded ? (
                            <>
                                <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center">
                                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                                </div>
                                <div>
                                    <p className="text-white font-medium">🎙️ Recording & Transcribing</p>
                                    <p className="text-sm text-gray-400 flex items-center gap-2">
                                        {deviceType === 'cuda' ? (
                                            <><Zap className="w-4 h-4 text-yellow-400" /> GPU Acceleration</>
                                        ) : (
                                            <><Cpu className="w-4 h-4 text-blue-400" /> CPU Processing</>
                                        )}
                                        <span className="text-gray-500">•</span>
                                        <Clock className="w-4 h-4" />
                                        {formatTime(transcriber_status?.audio_duration_seconds || 0)}
                                    </p>
                                </div>
                            </>
                        ) : (
                            <>
                                <Clock className="w-6 h-6 text-gray-400" />
                                <div>
                                    <p className="text-white font-medium">Waiting for audio...</p>
                                    <p className="text-sm text-gray-400">The transcription model will load shortly</p>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}

            {/* Tabs */}
            <div className="flex gap-2 mb-4">
                <button
                    onClick={() => setActiveTab('transcript')}
                    className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${activeTab === 'transcript'
                        ? 'bg-purple-500/20 text-purple-400'
                        : 'glass text-gray-400 hover:text-white'
                        }`}
                >
                    <FileText className="w-4 h-4" />
                    Transcript
                    {displayChunks.length > 0 && (
                        <span className="px-2 py-0.5 bg-purple-500/30 rounded-full text-xs">
                            {displayChunks.length}
                        </span>
                    )}
                </button>
                {/* Only show Frames tab for VIDEO sessions */}
                {isVideoSession && (
                    <button
                        onClick={() => setActiveTab('frames')}
                        className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${activeTab === 'frames'
                            ? 'bg-purple-500/20 text-purple-400'
                            : 'glass text-gray-400 hover:text-white'
                            }`}
                    >
                        <Image className="w-4 h-4" />
                        Frames
                        {frames.length > 0 && (
                            <span className="px-2 py-0.5 bg-purple-500/30 rounded-full text-xs">
                                {frames.length}
                            </span>
                        )}
                    </button>
                )}
            </div>

            {/* Content */}
            {activeTab === 'transcript' && (
                <div className="glass rounded-xl">
                    {/* Transcript Header */}
                    <div className="flex items-center justify-between p-4 border-b border-white/5">
                        <h3 className="text-white font-medium">Transcript</h3>
                        <div className="flex items-center gap-2">
                            {isActive && (
                                <button
                                    onClick={fetchSessionData}
                                    className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                </button>
                            )}
                            {!isActive && !isEditing && (
                                <button
                                    onClick={handleStartEdit}
                                    className="btn-secondary text-sm"
                                >
                                    <Edit3 className="w-4 h-4" />
                                    Edit
                                </button>
                            )}
                            {/* Chunk & Index button - only show when session has transcript and is not active */}
                            {!isActive && !isEditing && transcriptContent && (
                                <button
                                    onClick={handleIndexTranscript}
                                    disabled={indexing}
                                    className={`btn-secondary text-sm ${indexed ? 'text-green-400' : ''}`}
                                    title={indexed ? 'Already indexed' : 'Chunk and index transcript for RAG search'}
                                >
                                    {indexing ? (
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                    ) : indexed ? (
                                        <Check className="w-4 h-4" />
                                    ) : (
                                        <Database className="w-4 h-4" />
                                    )}
                                    {indexed ? 'Indexed' : 'Chunk & Index'}
                                </button>
                            )}
                            {isEditing && (
                                <>
                                    <button
                                        onClick={handleCancelEdit}
                                        className="btn-secondary text-sm"
                                    >
                                        <X className="w-4 h-4" />
                                        Cancel
                                    </button>
                                    <button
                                        onClick={handleSaveTranscript}
                                        disabled={saving}
                                        className="btn-primary text-sm"
                                    >
                                        {saving ? (
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                        ) : (
                                            <Save className="w-4 h-4" />
                                        )}
                                        Save
                                    </button>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Transcript Content */}
                    <div className="p-4 max-h-[500px] overflow-y-auto">
                        {isEditing ? (
                            <textarea
                                value={editedContent}
                                onChange={(e) => setEditedContent(e.target.value)}
                                className="w-full h-80 bg-transparent border border-white/10 rounded-lg p-4 text-gray-300 font-mono text-sm resize-none focus:outline-none focus:border-purple-500"
                                placeholder="Enter transcript..."
                            />
                        ) : displayChunks.length > 0 ? (
                            <div className="space-y-3">
                                {displayChunks.map((chunk, index) => (
                                    <div key={index} className="flex gap-3">
                                        {chunk.timestamp && (
                                            <span className="text-purple-400 font-mono text-sm whitespace-nowrap">
                                                {chunk.timestamp}
                                            </span>
                                        )}
                                        <p className="text-gray-300 text-sm leading-relaxed">
                                            {chunk.text}
                                        </p>
                                    </div>
                                ))}
                                {isActive && (
                                    <div className="flex items-center gap-2 text-gray-500 text-sm">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Listening for more...
                                    </div>
                                )}
                            </div>
                        ) : (
                            <p className="text-gray-500 italic">
                                {isActive ? 'Waiting for transcript...' : 'No transcript available'}
                            </p>
                        )}
                    </div>
                </div>
            )}

            {activeTab === 'frames' && (
                <div className="glass rounded-xl p-4">
                    {frames.length > 0 ? (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                            {frames.map((frame, index) => (
                                <div
                                    key={index}
                                    className="relative group rounded-lg overflow-hidden bg-black/20"
                                >
                                    <img
                                        src={api.getFrameUrl(sessionId!, frame.filename)}
                                        alt={`Frame ${index + 1}`}
                                        className="w-full aspect-video object-cover"
                                        loading="lazy"
                                    />
                                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 p-2">
                                        <p className="text-white text-xs font-mono">
                                            {frame.video_time !== null
                                                ? formatTime(frame.video_time)
                                                : frame.filename
                                            }
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-gray-500 italic text-center py-8">
                            {isActive ? 'Waiting for frames...' : 'No frames captured'}
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}
