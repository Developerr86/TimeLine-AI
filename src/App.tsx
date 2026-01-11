import { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TimelineView from './components/TimelineView';
import SettingsView from './components/SettingsView';
import ControlPanel from './components/ControlPanel';
import Notification from './components/Notification';
import { api, StatusResponse } from './services/api';

export interface NotificationState {
    message: string;
    type: 'info' | 'success' | 'error';
    show: boolean;
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

    const handleStart = async () => {
        try {
            await api.startCapture();
            showNotification('⏳ Starting capture in 10 seconds...', 'info');
            setTimeout(() => {
                showNotification('✅ Screenshot capture started!', 'success');
            }, 10000);
            fetchStatus();
        } catch (error) {
            showNotification('❌ Failed to start capture', 'error');
        }
    };

    const handleStop = async () => {
        try {
            await api.stopCapture();
            showNotification('⏸️ Screenshot capture stopped', 'info');
            fetchStatus();
        } catch (error) {
            showNotification('❌ Failed to stop capture', 'error');
        }
    };

    return (
        <Router>
            <div className="flex min-h-screen">
                {/* Titlebar drag region */}
                <div className="fixed top-0 left-0 right-0 h-8 drag-region z-50" />

                <Sidebar />

                <main className="flex-1 ml-20 p-8 pt-12">
                    <ControlPanel
                        status={status}
                        onStart={handleStart}
                        onStop={handleStop}
                    />

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
                    </Routes>
                </main>

                <Notification
                    message={notification.message}
                    type={notification.type}
                    show={notification.show}
                />
            </div>
        </Router>
    );
}

export default App;
