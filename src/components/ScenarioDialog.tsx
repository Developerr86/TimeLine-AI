import { useState, useEffect } from 'react';
import type { PendingScenario } from '../vite-env.d';

interface ScenarioDialogProps {
    onConfirm: () => Promise<void>;
    onDismiss: () => Promise<void>;
}

const scenarioInfo = {
    WEB: {
        icon: '🌐',
        title: 'Web Article Detected',
        description: 'Extract and save the article content from this webpage?',
        action: 'Capture Article'
    },
    DOC: {
        icon: '📄',
        title: 'Document Detected',
        description: 'Process this document and extract text content?',
        action: 'Process Document'
    },
    VIDEO: {
        icon: '🎬',
        title: 'Video Lecture Detected',
        description: 'Start recording audio and capturing frames from this video?',
        action: 'Start Recording'
    }
};

export default function ScenarioDialog({ onConfirm, onDismiss }: ScenarioDialogProps) {
    const [scenario, setScenario] = useState<PendingScenario | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    useEffect(() => {
        // Listen for scenario dialog events from Electron main process
        if (window.electronAPI?.onShowScenarioDialog) {
            window.electronAPI.onShowScenarioDialog((pendingScenario) => {
                console.log('📋 Scenario dialog triggered:', pendingScenario);
                setScenario(pendingScenario);
            });
        }
    }, []);

    const handleConfirm = async () => {
        if (!scenario) return;
        setIsProcessing(true);
        try {
            await onConfirm();
            setScenario(null);
        } catch (error) {
            console.error('Failed to confirm scenario:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleDismiss = async () => {
        setIsProcessing(true);
        try {
            await onDismiss();
            setScenario(null);
        } catch (error) {
            console.error('Failed to dismiss scenario:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    if (!scenario) return null;

    const info = scenarioInfo[scenario.scenario_type];
    const confidence = Math.round(scenario.confidence * 100);

    return (
        <div className="scenario-dialog-overlay">
            <div className="scenario-dialog">
                {/* Header */}
                <div className="scenario-dialog-header">
                    <span className="scenario-icon">{info.icon}</span>
                    <h2>{info.title}</h2>
                </div>

                {/* Content */}
                <div className="scenario-dialog-content">
                    <p className="scenario-description">{info.description}</p>
                    <div className="scenario-confidence">
                        <span className="confidence-label">Confidence:</span>
                        <div className="confidence-bar">
                            <div
                                className="confidence-fill"
                                style={{ width: `${confidence}%` }}
                            />
                        </div>
                        <span className="confidence-value">{confidence}%</span>
                    </div>
                </div>

                {/* Actions */}
                <div className="scenario-dialog-actions">
                    <button
                        className="btn-dismiss"
                        onClick={handleDismiss}
                        disabled={isProcessing}
                    >
                        Dismiss
                    </button>
                    <button
                        className="btn-confirm"
                        onClick={handleConfirm}
                        disabled={isProcessing}
                    >
                        {isProcessing ? 'Processing...' : info.action}
                    </button>
                </div>
            </div>

            <style>{`
                .scenario-dialog-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.7);
                    backdrop-filter: blur(4px);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                    animation: fadeIn 0.2s ease-out;
                }

                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }

                @keyframes slideUp {
                    from { 
                        opacity: 0;
                        transform: translateY(20px) scale(0.95);
                    }
                    to { 
                        opacity: 1;
                        transform: translateY(0) scale(1);
                    }
                }

                .scenario-dialog {
                    background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    padding: 24px;
                    width: 400px;
                    max-width: 90vw;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                    animation: slideUp 0.3s ease-out;
                }

                .scenario-dialog-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 20px;
                }

                .scenario-icon {
                    font-size: 32px;
                }

                .scenario-dialog-header h2 {
                    margin: 0;
                    font-size: 20px;
                    font-weight: 600;
                    color: #fff;
                }

                .scenario-dialog-content {
                    margin-bottom: 24px;
                }

                .scenario-description {
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 14px;
                    line-height: 1.5;
                    margin: 0 0 16px 0;
                }

                .scenario-confidence {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .confidence-label {
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 13px;
                    white-space: nowrap;
                }

                .confidence-bar {
                    flex: 1;
                    height: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    overflow: hidden;
                }

                .confidence-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
                    border-radius: 4px;
                    transition: width 0.3s ease;
                }

                .confidence-value {
                    color: #4ade80;
                    font-weight: 600;
                    font-size: 14px;
                    min-width: 40px;
                    text-align: right;
                }

                .scenario-dialog-actions {
                    display: flex;
                    gap: 12px;
                }

                .scenario-dialog-actions button {
                    flex: 1;
                    padding: 12px 20px;
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border: none;
                }

                .btn-dismiss {
                    background: rgba(255, 255, 255, 0.1);
                    color: rgba(255, 255, 255, 0.8);
                }

                .btn-dismiss:hover:not(:disabled) {
                    background: rgba(255, 255, 255, 0.15);
                }

                .btn-confirm {
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    color: #fff;
                }

                .btn-confirm:hover:not(:disabled) {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
                }

                .scenario-dialog-actions button:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }
            `}</style>
        </div>
    );
}
