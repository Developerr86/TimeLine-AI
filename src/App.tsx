import { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TimelineView from './components/TimelineView';
import SettingsView from './components/SettingsView';
import ControlPanel from './components/ControlPanel';
import Notification from './components/Notification';
import ScenarioDialog from './components/ScenarioDialog';
import ActivityList from './components/ActivityList';
import ScenarioDetailsView from './components/ScenarioDetailsView';
import ChatView from './components/ChatView';
import { api, StatusResponse } from './services/api';

export interface NotificationState {
    message: string;
    type: 'info' | 'success' | 'error';
    show: boolean;
}

// Wrapper component to conditionally render control panel
function MainLayout({
    status,
    onStart,
    onStop,
    showNotification,
    fetchStatus,
    children
}: {
    status: StatusResponse | null;
    onStart: () => void;
    onStop: () => void;
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
    fetchStatus: () => void;
    children: React.ReactNode;
}) {
    const location = useLocation();
    const isScenarioPage = location.pathname.startsWith('/scenario/');
    const isChatPage = location.pathname === '/chat';
    const showControls = !isScenarioPage && !isChatPage;

    return (
        <>
            {showControls && (
                <>
                    <ControlPanel
                        status={status}
                        onStart={onStart}
                        onStop={onStop}
                    />
                    <ActivityList
                        showNotification={showNotification}
                        onProcess={() => fetchStatus()}
                        onDismiss={() => fetchStatus()}
                    />
                </>
            )}
            {children}
        </>
    );
}

function App() {
    const [status, setStatus] = useState<StatusResponse | null>(null);
    const [notification, setNotification] = useState<NotificationState>({
        message: '',
        type: 'info',
        show: false,
    });

    const showNotification = useCallback((message: string, type: 'info' | 'success' | 'error' = 'info') => {
        setNotification({ message, type, show: true });
        setTimeout(() => {
            setNotification(prev => ({ ...prev, show: false }));
        }, 4000);
    }, []);

    const fetchStatus = useCallback(async () => {
        try {
            const data = await api.getStatus();
            setStatus(data);
        } catch (error) {
            console.error('Failed to fetch status:', error);
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, [fetchStatus]);

    // Track previous pending activities count for notifications
    const prevActivitiesCountRef = useRef<number>(0);

    useEffect(() => {
        if (status?.orchestrator?.pending_activities_count !== undefined) {
            const currentCount = status.orchestrator.pending_activities_count;
            if (currentCount > prevActivitiesCountRef.current && prevActivitiesCountRef.current >= 0) {
                // New activity detected
                const latestActivity = status.orchestrator.pending_activities?.[0];
                if (latestActivity) {
                    showNotification(`📥 Video Detected: ${latestActivity.title}`, 'info');
                }
            }
            prevActivitiesCountRef.current = currentCount;
        }
    }, [status?.orchestrator?.pending_activities_count, status?.orchestrator?.pending_activities, showNotification]);

    const handleStart = async () => {
        try {
            await api.startCapture();
            showNotification('✅ Activity tracking started! Install the browser extension to detect scenarios.', 'success');
            fetchStatus();
        } catch (error) {
            showNotification('❌ Failed to start tracking', 'error');
        }
    };

    const handleStop = async () => {
        try {
            await api.stopCapture();
            showNotification('⏸️ Activity tracking stopped', 'info');
            fetchStatus();
        } catch (error) {
            showNotification('❌ Failed to stop tracking', 'error');
        }
    };

    // Scenario handling
    const handleConfirmScenario = async () => {
        if (window.electronAPI?.confirmScenario) {
            const result = await window.electronAPI.confirmScenario();
            if (result.status === 'confirmed') {
                showNotification('✅ Scenario confirmed! Processing started...', 'success');
            }
            fetchStatus();
        }
    };

    const handleDismissScenario = async () => {
        if (window.electronAPI?.dismissScenario) {
            await window.electronAPI.dismissScenario();
            showNotification('👋 Scenario dismissed', 'info');
            fetchStatus();
        }
    };

    return (
        <Router>
            <div className="flex min-h-screen">
                {/* Titlebar drag region */}
                <div className="fixed top-0 left-0 right-0 h-8 drag-region z-50" />

                <Sidebar />

                <main className="flex-1 ml-20 p-8 pt-12">
                    <MainLayout
                        status={status}
                        onStart={handleStart}
                        onStop={handleStop}
                        showNotification={showNotification}
                        fetchStatus={fetchStatus}
                    >
                        <Routes>
                            <Route
                                path="/"
                                element={
                                    <TimelineView
                                        showNotification={showNotification}
                                        onRefresh={fetchStatus}
                                    />
                                }
                            />
                            <Route
                                path="/settings"
                                element={
                                    <SettingsView
                                        showNotification={showNotification}
                                    />
                                }
                            />
                            <Route
                                path="/chat"
                                element={
                                    <ChatView
                                        showNotification={showNotification}
                                    />
                                }
                            />
                            <Route
                                path="/scenario/:sessionId"
                                element={
                                    <ScenarioDetailsView
                                        showNotification={showNotification}
                                    />
                                }
                            />
                        </Routes>
                    </MainLayout>
                </main>

                <Notification
                    message={notification.message}
                    type={notification.type}
                    show={notification.show}
                />

                {/* Scenario confirmation dialog */}
                <ScenarioDialog
                    onConfirm={handleConfirmScenario}
                    onDismiss={handleDismissScenario}
                />
            </div>
        </Router>
    );
}

export default App;
