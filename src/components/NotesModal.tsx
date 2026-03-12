import { useState, useEffect } from 'react';
import { X, Sparkles, FileText, Video, Globe, Check, Loader2, Image, Layers, FileCheck, ChevronDown, ChevronRight } from 'lucide-react';
import { api, ProcessedSession } from '../services/api';

interface NotesModalProps {
    isOpen: boolean;
    onClose: () => void;
    onNotesGenerated: (notes: string, sessionCount: number) => void;
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
}

type ProgressStep = 'idle' | 'analyzing_contact_sheets' | 'selecting_frames' | 'processing_frames' | 'generating_notes' | 'complete';

interface StepInfo {
    key: ProgressStep;
    label: string;
    description: string;
    icon: React.ReactNode;
}

interface ContactSheetSelection {
    sessionId: string;
    sessionTitle: string;
    batchIndex: number;
    totalBatches: number;
    selectedIndices: number[];
    contactSheetPath: string;
}

interface LogEntry {
    timestamp: Date;
    message: string;
    type: 'info' | 'success' | 'error' | 'processing';
}

export default function NotesModal({ isOpen, onClose, onNotesGenerated, showNotification }: NotesModalProps) {
    const [sessions, setSessions] = useState<ProcessedSession[]>([]);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [currentStep, setCurrentStep] = useState<ProgressStep>('idle');
    const [progress, setProgress] = useState({ current: 0, total: 0, message: '' });
    const [videoSessionCount, setVideoSessionCount] = useState(0);
    const [skipFrameProcessing, setSkipFrameProcessing] = useState(false);
    
    // Frame selection state
    const [pendingSelections, setPendingSelections] = useState<ContactSheetSelection[]>([]);
    const [currentSelectionIndex, setCurrentSelectionIndex] = useState(0);
    const [showLogs, setShowLogs] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [currentBatch, setCurrentBatch] = useState(0);

    const steps: StepInfo[] = [
        {
            key: 'analyzing_contact_sheets',
            label: 'Analyzing contact sheets',
            description: 'Having a bird\'s eye view of video frames',
            icon: <Layers className="w-4 h-4" />
        },
        {
            key: 'selecting_frames',
            label: 'Selecting frames',
            description: 'Choose frames to process for each video',
            icon: <Check className="w-4 h-4" />
        },
        {
            key: 'processing_frames',
            label: 'Processing frames',
            description: 'Extracting information from selected frames',
            icon: <Image className="w-4 h-4" />
        },
        {
            key: 'generating_notes',
            label: 'Generating notes',
            description: 'Combining transcript and frame data for comprehensive notes',
            icon: <FileCheck className="w-4 h-4" />
        }
    ];

    const addLog = (message: string, type: LogEntry['type'] = 'info') => {
        setLogs(prev => [...prev, { timestamp: new Date(), message, type }]);
    };

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
            const indexedIds = new Set(
                response.sessions.filter(s => s.indexed).map(s => s.id)
            );
            setSelectedIds(indexedIds);
            const videoCount = response.sessions.filter(s => s.type === 'VIDEO').length;
            setVideoSessionCount(videoCount);
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

    const handleFrameToggle = (batchIndex: number, frameIndex: number) => {
        setPendingSelections(prev => {
            const newSelections = [...prev];
            const selection = newSelections[batchIndex];
            if (selection.selectedIndices.includes(frameIndex)) {
                selection.selectedIndices = selection.selectedIndices.filter(i => i !== frameIndex);
            } else {
                selection.selectedIndices = [...selection.selectedIndices, frameIndex];
            }
            return newSelections;
        });
    };

    const selectAllFrames = (batchIndex: number) => {
        setPendingSelections(prev => {
            const newSelections = [...prev];
            const selection = newSelections[batchIndex];
            selection.selectedIndices = Array.from({ length: 16 }, (_, i) => i + 1);
            return newSelections;
        });
    };

    const selectNoFrames = (batchIndex: number) => {
        setPendingSelections(prev => {
            const newSelections = [...prev];
            const selection = newSelections[batchIndex];
            selection.selectedIndices = [];
            return newSelections;
        });
    };

    const proceedWithSelections = async () => {
        if (currentSelectionIndex < pendingSelections.length - 1) {
            setCurrentSelectionIndex(prev => prev + 1);
        } else {
            // All selections done, proceed to process frames
            await processSelectedFrames();
        }
    };

    const processSelectedFrames = async () => {
        setCurrentStep('processing_frames');
        setPendingSelections([]);
        
        const videoSessions = sessions.filter(
            s => selectedIds.has(s.id) && s.type === 'VIDEO'
        );

        let allFrameIndices: { [sessionId: string]: number[] } = {};
        
        // Collect all selected frames from user selections
        pendingSelections.forEach(selection => {
            allFrameIndices[selection.sessionId] = selection.selectedIndices;
        });

        for (let i = 0; i < videoSessions.length; i++) {
            const session = videoSessions[i];
            const selectedFrames = allFrameIndices[session.id] || [];
            
            if (selectedFrames.length === 0) {
                addLog(`Skipping ${session.title || 'Untitled'} - no frames selected`, 'info');
                continue;
            }

            setProgress({ 
                current: i + 1, 
                total: videoSessions.length, 
                message: `Processing ${selectedFrames.length} frames for video ${i + 1}/${videoSessions.length}...`
            });

            try {
                addLog(`Processing ${selectedFrames.length} frames for: ${session.title || 'Untitled'}`, 'processing');
                await api.processVideoFrames(session.id, selectedFrames);
                addLog(`Frames processed successfully for: ${session.title || 'Untitled'}`, 'success');
            } catch (error) {
                addLog(`Failed to process frames: ${error}`, 'error');
            }
        }

        // Generate notes
        await generateNotes();
    };

    const generateNotes = async () => {
        setCurrentStep('generating_notes');
        setProgress({ 
            current: 0, 
            total: 100, 
            message: skipFrameProcessing 
                ? 'Generating notes from transcript only (frame processing skipped)...' 
                : 'Generating notes from transcript and frame data...'
        });

        const sessionIds = sessions
            .filter(s => selectedIds.has(s.id))
            .map(s => s.id);

        addLog(skipFrameProcessing 
            ? 'Starting note generation (frames skipped)...' 
            : 'Starting note generation...', 'info');

        try {
            const response = await api.generateNotes(sessionIds, skipFrameProcessing);
            
            if (response.status === 'success' && response.notes) {
                addLog('Notes generated successfully!', 'success');
                onNotesGenerated(response.notes, response.session_count || sessionIds.length);
                onClose();
            } else {
                addLog(`Failed to generate notes: ${response.message}`, 'error');
                showNotification(`❌ ${response.message || 'Failed to generate notes'}`, 'error');
            }
        } catch (error) {
            addLog(`Error generating notes: ${error}`, 'error');
            showNotification('❌ Failed to generate notes', 'error');
        } finally {
            setGenerating(false);
            setCurrentStep('idle');
        }
    };

    const handleGenerate = async () => {
        if (selectedIds.size === 0) {
            showNotification('Please select at least one session', 'error');
            return;
        }

        setGenerating(true);
        setCurrentStep('idle');
        setProgress({ current: 0, total: 0, message: '' });
        setLogs([]);
        setPendingSelections([]);
        setCurrentSelectionIndex(0);

        const videoSessions = sessions.filter(
            s => selectedIds.has(s.id) && s.type === 'VIDEO'
        );

        // If user chose to skip frame processing, go directly to note generation
        if (skipFrameProcessing || videoSessions.length === 0) {
            if (skipFrameProcessing && videoSessions.length > 0) {
                addLog('Skipping frame processing as requested', 'info');
            }
            await generateNotes();
            return;
        }

        // Step 1: Analyze contact sheets for VIDEO sessions
        if (videoSessions.length > 0) {
            setCurrentStep('analyzing_contact_sheets');
            let selections: ContactSheetSelection[] = [];

            for (let i = 0; i < videoSessions.length; i++) {
                const session = videoSessions[i];
                setCurrentBatch(i + 1);
                setProgress({ 
                    current: i + 1, 
                    total: videoSessions.length, 
                    message: `Analyzing video ${i + 1}/${videoSessions.length}: ${session.title || 'Untitled'}`
                });

                addLog(`Analyzing contact sheets for: ${session.title || 'Untitled'}`, 'processing');

                try {
                    const result = await api.analyzeContactSheets(session.id);
                    
                    if (result.selected_indices && result.selected_indices.length > 0) {
                        // AI selected some frames - convert to batch selections
                        const totalBatches = Math.ceil(result.total_frames / 16);
                        addLog(`AI suggested ${result.selected_indices.length} frames (Batch ${i + 1}/${videoSessions.length})`, 'success');
                        
                        // For now, we'll show all batches to user for confirmation
                        for (let batch = 0; batch < totalBatches; batch++) {
                            const batchStart = batch * 16 + 1;
                            const batchEnd = Math.min((batch + 1) * 16, result.total_frames);
                            const batchIndices = result.selected_indices.filter(
                                idx => idx >= batchStart && idx <= batchEnd
                            ).map(idx => idx - batchStart);
                            
                            selections.push({
                                sessionId: session.id,
                                sessionTitle: session.title || 'Untitled',
                                batchIndex: batch,
                                totalBatches: totalBatches,
                                selectedIndices: batchIndices,
                                contactSheetPath: `/api/media/video/${session.id}/contact_sheets/contact_sheet_${batch.toString().padStart(3, '0')}.jpg`
                            });
                        }
                    } else {
                        // AI selected none - show all batches to user
                        const totalBatches = Math.ceil(result.total_frames / 16);
                        addLog(`AI found no relevant frames for: ${session.title || 'Untitled'} - please select manually`, 'info');
                        
                        for (let batch = 0; batch < totalBatches; batch++) {
                            selections.push({
                                sessionId: session.id,
                                sessionTitle: session.title || 'Untitled',
                                batchIndex: batch,
                                totalBatches: totalBatches,
                                selectedIndices: [],
                                contactSheetPath: `/api/media/video/${session.id}/contact_sheets/contact_sheet_${batch.toString().padStart(3, '0')}.jpg`
                            });
                        }
                    }
                } catch (error) {
                    addLog(`Error analyzing contact sheets: ${error}`, 'error');
                }
            }

            if (selections.length > 0) {
                setPendingSelections(selections);
                setCurrentSelectionIndex(0);
                setCurrentStep('selecting_frames');
            } else {
                // No frames to process, go directly to note generation
                await generateNotes();
            }
        } else {
            // No video sessions, directly generate notes
            await generateNotes();
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

    const getStepStatus = (stepKey: ProgressStep) => {
        const stepOrder: ProgressStep[] = ['analyzing_contact_sheets', 'selecting_frames', 'processing_frames', 'generating_notes'];
        const currentIndex = stepOrder.indexOf(currentStep);
        const stepIndex = stepOrder.indexOf(stepKey);

        if (currentStep === 'idle' || currentStep === 'complete') {
            return 'pending';
        }
        if (stepIndex < currentIndex) {
            return 'completed';
        }
        if (stepIndex === currentIndex) {
            return 'active';
        }
        return 'pending';
    };

    if (!isOpen) return null;

    const currentSelection = pendingSelections[currentSelectionIndex];

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
            <div className="glass rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col mx-4">
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
                        disabled={generating}
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Progress Steps */}
                <div className="px-5 py-4 border-b border-white/10 bg-black/20">
                    <div className="flex items-center gap-2 mb-3">
                        {steps.map((step) => {
                            const status = getStepStatus(step.key);
                            return (
                                <div key={step.key} className="flex items-center gap-2">
                                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                                        status === 'completed' ? 'bg-green-500 text-white' :
                                        status === 'active' ? 'bg-purple-500 text-white animate-pulse' :
                                        'bg-gray-600 text-gray-400'
                                    }`}>
                                        {status === 'completed' ? <Check className="w-3 h-3" /> : 
                                         status === 'active' ? <Loader2 className="w-3 h-3 animate-spin" /> :
                                         step.key.charAt(0).toUpperCase()}
                                    </div>
                                    {step.key !== 'generating_notes' && (
                                        <div className="w-8 h-0.5 bg-gray-600" />
                                    )}
                                </div>
                            );
                        })}
                    </div>
                    
                    {/* Progress message */}
                    {generating && (
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-gray-400">{progress.message}</span>
                            {progress.total > 0 && (
                                <span className="text-purple-400">{progress.current}/{progress.total}</span>
                            )}
                        </div>
                    )}
                    
                    {/* Toggle Logs Button */}
                    {generating && logs.length > 0 && (
                        <button
                            onClick={() => setShowLogs(!showLogs)}
                            className="mt-2 flex items-center gap-2 text-xs text-gray-500 hover:text-gray-400"
                        >
                            {showLogs ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                            {showLogs ? 'Hide Logs' : 'Show Logs'} ({logs.length} entries)
                        </button>
                    )}
                </div>

                {/* Logs Panel */}
                {showLogs && logs.length > 0 && (
                    <div className="max-h-40 overflow-y-auto bg-black/40 border-b border-white/10 p-3">
                        <div className="space-y-1 font-mono text-xs">
                            {logs.map((log, i) => (
                                <div key={i} className={`${
                                    log.type === 'error' ? 'text-red-400' :
                                    log.type === 'success' ? 'text-green-400' :
                                    log.type === 'processing' ? 'text-yellow-400' :
                                    'text-gray-400'
                                }`}>
                                    <span className="text-gray-600">[{log.timestamp.toLocaleTimeString()}]</span> {log.message}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-5">
                    {/* Frame Selection View */}
                    {currentStep === 'selecting_frames' && currentSelection ? (
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="text-lg font-medium text-white">
                                        Select Frames for: {currentSelection.sessionTitle}
                                    </h3>
                                    <p className="text-sm text-gray-400">
                                        Batch {currentSelection.batchIndex + 1} of {currentSelection.totalBatches}
                                        {currentSelection.totalBatches > 1 && 
                                            ` (Video ${Math.floor(currentSelection.batchIndex / currentSelection.totalBatches * 
                                            (sessions.filter(s => selectedIds.has(s.id) && s.type === 'VIDEO').length)) + 1})`
                                        }
                                    </p>
                                </div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => selectAllFrames(currentSelectionIndex)}
                                        className="text-xs text-purple-400 hover:text-purple-300"
                                    >
                                        Select All
                                    </button>
                                    <span className="text-gray-600">|</span>
                                    <button
                                        onClick={() => selectNoFrames(currentSelectionIndex)}
                                        className="text-xs text-gray-400 hover:text-gray-300"
                                    >
                                        Clear All
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-4 gap-2">
                                {Array.from({ length: 16 }, (_, i) => i + 1).map((frameNum) => (
                                    <button
                                        key={frameNum}
                                        onClick={() => handleFrameToggle(currentSelectionIndex, frameNum)}
                                        className={`relative aspect-square rounded-lg border-2 transition-all overflow-hidden ${
                                            currentSelection.selectedIndices.includes(frameNum)
                                                ? 'border-purple-500 bg-purple-500/20'
                                                : 'border-gray-600 bg-gray-800/50 hover:border-gray-500'
                                        }`}
                                    >
                                        <img 
                                            src={currentSelection.contactSheetPath}
                                            alt={`Frame ${frameNum}`}
                                            className="absolute inset-0 w-full h-full object-cover"
                                            style={{
                                                objectPosition: `${((frameNum - 1) % 4) * 25}% ${Math.floor((frameNum - 1) / 4) * 25}%`
                                            }}
                                        />
                                        <div className={`absolute inset-0 flex items-center justify-center ${
                                            currentSelection.selectedIndices.includes(frameNum)
                                                ? 'bg-purple-500/30'
                                                : 'bg-black/40'
                                        }`}>
                                            <span className="text-sm font-bold text-white drop-shadow-lg">{frameNum}</span>
                                        </div>
                                        {currentSelection.selectedIndices.includes(frameNum) && (
                                            <div className="absolute top-1 right-1 w-5 h-5 bg-purple-500 rounded-full flex items-center justify-center">
                                                <Check className="w-3 h-3 text-white" />
                                            </div>
                                        )}
                                    </button>
                                ))}
                            </div>

                            <div className="flex items-center justify-between pt-4 border-t border-white/10">
                                <button
                                    onClick={() => {
                                        if (currentSelectionIndex > 0) {
                                            setCurrentSelectionIndex(prev => prev - 1);
                                        }
                                    }}
                                    disabled={currentSelectionIndex === 0}
                                    className="btn-secondary disabled:opacity-50"
                                >
                                    Previous Batch
                                </button>
                                <span className="text-sm text-gray-400">
                                    {currentSelection.selectedIndices.length} frames selected
                                </span>
                                <button
                                    onClick={proceedWithSelections}
                                    className="btn-primary"
                                >
                                    {currentSelectionIndex < pendingSelections.length - 1 
                                        ? 'Next Batch' 
                                        : 'Process Selected Frames'}
                                </button>
                            </div>
                        </div>
                    ) : generating ? (
                        <div className="flex flex-col items-center justify-center py-12">
                            <div className="w-16 h-16 mb-4 rounded-full bg-purple-500/20 flex items-center justify-center">
                                <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
                            </div>
                            <h3 className="text-lg font-medium text-white mb-2">
                                {currentStep === 'analyzing_contact_sheets' && 'Analyzing Video Frames'}
                                {currentStep === 'processing_frames' && 'Processing Selected Frames'}
                                {currentStep === 'generating_notes' && 'Generating Your Notes'}
                            </h3>
                            <p className="text-gray-400 text-sm text-center max-w-md">
                                {currentStep === 'analyzing_contact_sheets' && 
                                    `Analyzing batch ${currentBatch}... Having a bird's eye view of your video frames to identify the most relevant ones`}
                                {currentStep === 'processing_frames' && 
                                    'Extracting text and information from the selected frames...'}
                                {currentStep === 'generating_notes' && 
                                    'Combining transcript and visual data to create comprehensive notes...'}
                            </p>
                        </div>
                    ) : loading ? (
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
                                        className={`w-full flex items-center gap-3 p-3 rounded-xl border transition-all ${
                                            selectedIds.has(session.id)
                                                ? 'border-coral-500/50 bg-coral-500/10'
                                                : 'border-white/10 bg-white/5 hover:bg-white/10'
                                        }`}
                                    >
                                        <div className={`w-5 h-5 rounded flex items-center justify-center border ${
                                            selectedIds.has(session.id)
                                                ? 'bg-coral-500 border-coral-500'
                                                : 'border-gray-500'
                                        }`}>
                                            {selectedIds.has(session.id) && (
                                                <Check className="w-3 h-3 text-white" />
                                            )}
                                        </div>
                                        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                                            {getScenarioIcon(session.type)}
                                        </div>
                                        <div className="flex-1 text-left">
                                            <div className="text-sm text-white truncate">
                                                {session.title || 'Untitled Session'}
                                            </div>
                                            <div className="text-xs text-gray-500">
                                                {session.type} • {formatDate(session.start_time)}
                                            </div>
                                        </div>
                                        {session.indexed && (
                                            <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400">
                                                Indexed
                                            </span>
                                        )}
                                    </button>
                                ))}
                            </div>

                            {/* Skip Frame Processing Option */}
                            {videoSessionCount > 0 && (
                                <div className="mt-4 p-3 rounded-xl border border-purple-500/30 bg-purple-500/10">
                                    <label className="flex items-center gap-3 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={skipFrameProcessing}
                                            onChange={(e) => setSkipFrameProcessing(e.target.checked)}
                                            className="w-4 h-4 rounded border-gray-500 text-purple-500 focus:ring-purple-500 bg-gray-800"
                                        />
                                        <div className="flex-1">
                                            <div className="text-sm text-white">Skip Frame Processing</div>
                                            <div className="text-xs text-gray-400">
                                                Generate notes directly from transcript without analyzing video frames
                                            </div>
                                        </div>
                                    </label>
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* Footer */}
                <div className="p-5 border-t border-white/10 flex items-center justify-between">
                    <p className="text-xs text-gray-500">
                        {sessions.length > 0 && !loading && !generating && (
                            <>
                                {videoSessionCount > 0 && (
                                    <span className="text-purple-400">
                                        {skipFrameProcessing 
                                            ? `🎬 ${videoSessionCount} video(s) - notes will be generated from transcript only`
                                            : `🎬 ${videoSessionCount} video(s) - you'll select frames to process`
                                        }
                                    </span>
                                )}
                            </>
                        )}
                    </p>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="btn-secondary"
                            disabled={generating}
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
                                    Processing...
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
