import { CheckCircle, AlertCircle, Info } from 'lucide-react';

interface NotificationProps {
    message: string;
    type: 'info' | 'success' | 'error';
    show: boolean;
}

export default function Notification({ message, type, show }: NotificationProps) {
    const styles = {
        info: {
            bg: 'bg-blue-500/20 border-blue-500/50',
            text: 'text-blue-300',
            icon: Info,
        },
        success: {
            bg: 'bg-emerald-500/20 border-emerald-500/50',
            text: 'text-emerald-300',
            icon: CheckCircle,
        },
        error: {
            bg: 'bg-red-500/20 border-red-500/50',
            text: 'text-red-300',
            icon: AlertCircle,
        },
    };

    const { bg, text, icon: IconComponent } = styles[type];

    return (
        <div
            className={`fixed top-12 left-1/2 -translate-x-1/2 z-50 transition-all duration-300 ${show ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'
                }`}
        >
            <div className={`flex items-center gap-3 px-5 py-3 rounded-xl border backdrop-blur-xl shadow-2xl ${bg}`}>
                <IconComponent className={`w-5 h-5 ${text}`} />
                <span className={`font-medium ${text}`}>{message}</span>
            </div>
        </div>
    );
}
