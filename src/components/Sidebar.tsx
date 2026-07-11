import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    Settings,
    FileText,
    Sparkles,
    MessageSquare
} from 'lucide-react';

const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Timeline' },
    { to: '/notes', icon: FileText, label: 'Notes' },
    { to: '/chat', icon: MessageSquare, label: 'Chat' },
    { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar() {
    return (
        <aside className="fixed left-0 top-0 bottom-0 w-20 glass flex flex-col items-center py-8 z-40">
            {/* Logo */}
            <div className="mb-8">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center shadow-lg" style={{ background: '#ff6b6b' }}>
                    <Sparkles className="w-6 h-6 text-white" />
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex flex-col items-center gap-3 flex-1">
                {navItems.map((item) => (
                    <NavLink
                        key={item.label}
                        to={item.to}
                        className={({ isActive }) =>
                            `w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 no-drag ${
                                isActive
                                    ? 'glass-light shadow-lg'
                                    : 'hover:glass-light'
                            }`
                        }
                        title={item.label}
                    >
                        <item.icon className="w-5 h-5 text-gray-300" />
                    </NavLink>
                ))}
            </nav>
        </aside>
    );
}
