import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    Settings,
    BarChart3,
    Calendar,
    Info,
    Sparkles
} from 'lucide-react';

const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Timeline' },
    { to: '#', icon: BarChart3, label: 'Analytics', disabled: true },
    { to: '#', icon: Calendar, label: 'Calendar', disabled: true },
    { to: '/settings', icon: Settings, label: 'Settings' },
    { to: '#', icon: Info, label: 'Info', disabled: true },
];

export default function Sidebar() {
    return (
        <aside className="fixed left-0 top-0 bottom-0 w-20 glass flex flex-col items-center py-8 z-40">
            {/* Logo */}
            <div className="mb-8">
                <div className="w-12 h-12 gradient-primary rounded-xl flex items-center justify-center shadow-lg">
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
                            `w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 no-drag ${item.disabled
                                ? 'opacity-40 cursor-not-allowed'
                                : isActive
                                    ? 'glass-light shadow-lg'
                                    : 'hover:glass-light'
                            }`
                        }
                        onClick={(e) => item.disabled && e.preventDefault()}
                        title={item.label}
                    >
                        <item.icon
                            className={`w-5 h-5 ${item.disabled ? 'text-gray-500' : 'text-gray-300'}`}
                        />
                    </NavLink>
                ))}
            </nav>
        </aside>
    );
}
