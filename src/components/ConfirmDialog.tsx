import { X, AlertTriangle } from 'lucide-react';

interface ConfirmDialogProps {
    isOpen: boolean;
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    onConfirm: () => void;
    onCancel: () => void;
    destructive?: boolean;
}

export default function ConfirmDialog({
    isOpen,
    title,
    message,
    confirmLabel = 'Confirm',
    cancelLabel = 'Cancel',
    onConfirm,
    onCancel,
    destructive = false
}: ConfirmDialogProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
            <div className="glass rounded-2xl w-full max-w-md mx-4 p-6">
                {/* Header */}
                <div className="flex items-center gap-4 mb-4">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center border-[3px] ${destructive ? 'bg-red-500/20 border-red-500/30' : 'bg-blue-500/20 border-blue-500/30'}`}>
                        <AlertTriangle className={`w-6 h-6 ${destructive ? 'text-red-400' : 'text-blue-400'}`} />
                    </div>
                    <h2 className="text-xl font-semibold text-white">{title}</h2>
                    <button
                        onClick={onCancel}
                        className="ml-auto p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Message */}
                <p className="text-gray-400 mb-6 ml-16">{message}</p>

                {/* Actions */}
                <div className="flex gap-3 justify-end">
                    <button
                        onClick={onCancel}
                        className="px-5 py-2.5 rounded-xl bg-white/10 text-gray-300 hover:bg-white/15 transition-colors font-medium"
                    >
                        {cancelLabel}
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`px-5 py-2.5 rounded-xl text-white transition-colors font-medium ${
                            destructive 
                                ? 'bg-red-500 hover:bg-red-600' 
                                : 'bg-blue-500 hover:bg-blue-600'
                        }`}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>

            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                .animate-fade-in {
                    animation: fadeIn 0.2s ease-out;
                }
            `}</style>
        </div>
    );
}
