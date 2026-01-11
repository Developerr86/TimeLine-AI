import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Save, Trash2, AlertTriangle } from 'lucide-react';
import { api, Config } from '../services/api';

interface SettingsViewProps {
    showNotification: (message: string, type: 'info' | 'success' | 'error') => void;
}

export default function SettingsView({ showNotification }: SettingsViewProps) {
    const navigate = useNavigate();
    const [config, setConfig] = useState<Partial<Config>>({
        interval: 5,
        model_type: 'ollama',
        ollama_model: 'qwen2.5vl:3b',
        gemini_model: 'gemini-2.5-pro',
        remote_url: 'http://localhost:5001/predict',
        similarity_threshold: 0.95,
        notes_history_limit: 5,
        notes_model_provider: 'gemini',
        notes_ollama_model: 'llama3',
    });
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        const fetchConfig = async () => {
            try {
                const data = await api.getConfig();
                setConfig(data);
            } catch (error) {
                showNotification('Failed to load configuration', 'error');
            } finally {
                setLoading(false);
            }
        };
        fetchConfig();
    }, [showNotification]);

    const handleChange = (field: keyof Config, value: string | number) => {
        setConfig(prev => ({ ...prev, [field]: value }));
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            await api.updateConfig(config);
            showNotification('✅ Configuration saved successfully!', 'success');
            navigate('/');
        } catch (error) {
            showNotification('❌ Failed to save configuration', 'error');
        } finally {
            setSaving(false);
        }
    };

    const handleClearContext = async () => {
        if (!confirm('⚠️ ARE YOU SURE?\n\nThis will PERMANENTLY DELETE all screenshots and activity history.\n\nThis action cannot be undone.')) {
            return;
        }

        try {
            await api.clearContext();
            showNotification('✅ All data has been cleared successfully', 'success');
            navigate('/');
        } catch (error) {
            showNotification('❌ Failed to clear data', 'error');
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <div className="spinner w-8 h-8" />
            </div>
        );
    }

    return (
        <div className="animate-fade-in max-w-3xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-3xl font-light text-white">Settings</h1>
                <button
                    onClick={() => navigate('/')}
                    className="btn-secondary"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Timeline
                </button>
            </div>

            {/* Settings Form */}
            <div className="glass rounded-2xl p-6 mb-6">
                <h2 className="text-xl font-medium text-coral-400 mb-6">⚙️ Configuration</h2>

                <div className="space-y-5">
                    {/* Screenshot Interval */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Screenshot Interval (seconds)
                        </label>
                        <input
                            type="number"
                            className="input"
                            value={config.interval}
                            onChange={(e) => handleChange('interval', parseInt(e.target.value))}
                            min={1}
                            max={300}
                        />
                        <p className="text-xs text-gray-500 mt-1">How often to capture screenshots (1-300 seconds)</p>
                    </div>

                    {/* Similarity Threshold */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Similarity Threshold
                        </label>
                        <input
                            type="number"
                            className="input"
                            value={config.similarity_threshold}
                            onChange={(e) => handleChange('similarity_threshold', parseFloat(e.target.value))}
                            min={0}
                            max={1}
                            step={0.01}
                        />
                        <p className="text-xs text-gray-500 mt-1">Images above this similarity (0-1) will be discarded. Recommended: 0.95</p>
                    </div>

                    {/* Notes History Limit */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Notes History Limit
                        </label>
                        <input
                            type="number"
                            className="input"
                            value={config.notes_history_limit}
                            onChange={(e) => handleChange('notes_history_limit', parseInt(e.target.value))}
                            min={1}
                            max={100}
                        />
                        <p className="text-xs text-gray-500 mt-1">Number of recent activities to use when generating notes</p>
                    </div>

                    {/* Notes Model Provider */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Notes AI Provider
                        </label>
                        <select
                            className="select"
                            value={config.notes_model_provider}
                            onChange={(e) => handleChange('notes_model_provider', e.target.value as 'gemini' | 'ollama')}
                        >
                            <option value="gemini">Google Gemini (Cloud)</option>
                            <option value="ollama">Ollama (Local)</option>
                        </select>
                    </div>

                    {config.notes_model_provider === 'ollama' && (
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Ollama Notes Model
                            </label>
                            <input
                                type="text"
                                className="input"
                                value={config.notes_ollama_model}
                                onChange={(e) => handleChange('notes_ollama_model', e.target.value)}
                            />
                            <p className="text-xs text-gray-500 mt-1">Text model for notes (e.g., llama3, mistral)</p>
                        </div>
                    )}

                    <hr className="border-white/10 my-6" />

                    {/* Screenshot Model Type */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Screenshot AI Model Type
                        </label>
                        <select
                            className="select"
                            value={config.model_type}
                            onChange={(e) => handleChange('model_type', e.target.value as 'ollama' | 'gemini' | 'remote')}
                        >
                            <option value="ollama">Ollama (Local)</option>
                            <option value="gemini">Google Gemini (Cloud)</option>
                            <option value="remote">Remote Server (Custom URL)</option>
                        </select>
                    </div>

                    {config.model_type === 'ollama' && (
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Ollama Model Name
                            </label>
                            <input
                                type="text"
                                className="input"
                                value={config.ollama_model}
                                onChange={(e) => handleChange('ollama_model', e.target.value)}
                            />
                            <p className="text-xs text-gray-500 mt-1">e.g., qwen2.5vl:3b, llava, bakllava</p>
                        </div>
                    )}

                    {config.model_type === 'gemini' && (
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Gemini Model Name
                            </label>
                            <input
                                type="text"
                                className="input"
                                value={config.gemini_model}
                                onChange={(e) => handleChange('gemini_model', e.target.value)}
                            />
                            <p className="text-xs text-gray-500 mt-1">e.g., gemini-2.5-pro, gemini-2.0-flash</p>
                        </div>
                    )}

                    {config.model_type === 'remote' && (
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Remote Server URL
                            </label>
                            <input
                                type="text"
                                className="input"
                                value={config.remote_url}
                                onChange={(e) => handleChange('remote_url', e.target.value)}
                            />
                            <p className="text-xs text-gray-500 mt-1">Full URL to the prediction endpoint</p>
                        </div>
                    )}
                </div>

                {/* Save Button */}
                <div className="flex gap-3 mt-8">
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="btn-primary"
                    >
                        {saving ? (
                            <>
                                <span className="spinner w-4 h-4" />
                                Saving...
                            </>
                        ) : (
                            <>
                                <Save className="w-4 h-4" />
                                Save Configuration
                            </>
                        )}
                    </button>
                    <button
                        onClick={() => navigate('/')}
                        className="btn-secondary"
                    >
                        Cancel
                    </button>
                </div>
            </div>

            {/* Danger Zone */}
            <div className="glass rounded-2xl p-6 border border-red-500/30">
                <h2 className="text-xl font-medium text-red-400 mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    Danger Zone
                </h2>
                <p className="text-gray-400 text-sm mb-4">
                    This will delete all captured screenshots and clear the activity history. This action cannot be undone.
                </p>
                <button
                    onClick={handleClearContext}
                    className="btn-danger"
                >
                    <Trash2 className="w-4 h-4" />
                    Clear All Data
                </button>
            </div>

            {/* Info Panel */}
            <div className="glass rounded-2xl p-6 mt-6">
                <h3 className="text-lg font-medium text-coral-400 mb-4">💡 Tips & Information</h3>
                <ul className="space-y-2 text-sm text-gray-400">
                    <li><strong className="text-gray-300">Screenshot Interval:</strong> Lower values (3-5s) capture more frequently but use more resources.</li>
                    <li><strong className="text-gray-300">Similarity Threshold:</strong> Values between 0.90-0.99 work best. Higher = only very different screenshots are kept.</li>
                    <li><strong className="text-gray-300">Ollama:</strong> Runs completely offline. Requires Ollama installation and model download.</li>
                    <li><strong className="text-gray-300">Gemini:</strong> Cloud-based AI. Requires internet and API key in .env file.</li>
                    <li><strong className="text-gray-300">Remote:</strong> Connect to a custom VLM server running on another port/machine.</li>
                </ul>
            </div>
        </div>
    );
}
