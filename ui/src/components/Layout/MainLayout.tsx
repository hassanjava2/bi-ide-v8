import React, { useState, useEffect } from 'react';
import { NavLink, useLocation, Outlet } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Building2, 
  Users, 
  Code2, 
  Brain,
  Settings,
  LogOut,
  Menu,
  X,
  ChevronLeft,
  User,
  Bell,
  Search,
  Home
} from 'lucide-react';
// Button component imported but not used in this file

interface MainLayoutProps {
  children?: React.ReactNode;
  user?: {
    name: string;
    email: string;
    role: string;
    avatar?: string;
  };
  onLogout?: () => void;
}

interface NavItem {
  path: string;
  label: string;
  icon: React.ElementType;
  badge?: number;
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', label: 'الرئيسية', icon: Home },
  { path: '/dashboard', label: 'لوحة التحكم', icon: LayoutDashboard },
  { path: '/erp', label: 'نظام ERP', icon: Building2 },
  { path: '/council', label: 'مجلس الحكماء', icon: Brain },
  { path: '/ide', label: 'بيئة التطوير', icon: Code2 },
  { path: '/users', label: 'المستخدمين', icon: Users },
  { path: '/settings', label: 'الإعدادات', icon: Settings },
];

const Breadcrumb: React.FC = () => {
  const location = useLocation();
  const paths = location.pathname.split('/').filter(Boolean);

  const getLabel = (path: string) => {
    const item = NAV_ITEMS.find(n => n.path === `/${path}`);
    return item?.label || path;
  };

  return (
    <nav className="flex items-center gap-2 text-sm text-gray-400">
      <NavLink to="/" className="hover:text-white transition-colors">
        الرئيسية
      </NavLink>
      {paths.map((path, index) => (
        <React.Fragment key={path}>
          <ChevronLeft className="w-4 h-4" />
          <span className={index === paths.length - 1 ? 'text-white' : ''}>
            {getLabel(path)}
          </span>
        </React.Fragment>
      ))}
    </nav>
  );
};

export const MainLayout: React.FC<MainLayoutProps> = ({ 
  children,
  user,
  onLogout 
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifications] = useState(3);
  const location = useLocation();

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const handleLogout = () => {
    if (confirm('هل أنت متأكد من تسجيل الخروج؟')) {
      onLogout?.();
    }
  };

  const SidebarContent = () => (
    <>
      {/* Logo */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-bi-primary to-bi-secondary flex items-center justify-center">
            <Brain className="w-6 h-6 text-bi-gold" />
          </div>
          <div className={sidebarOpen ? 'block' : 'hidden lg:block'}>
            <h1 className="font-bold text-lg">BI-IDE</h1>
            <p className="text-xs text-gray-400">v3.0.0</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `
              flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all
              ${isActive 
                ? 'bg-bi-accent text-white' 
                : 'text-gray-400 hover:bg-white/5 hover:text-white'
              }
              ${!sidebarOpen && 'lg:justify-center'}
            `}
            title={!sidebarOpen ? item.label : undefined}
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            <span className={`${sidebarOpen ? 'block' : 'hidden lg:hidden'} flex-1`}>
              {item.label}
            </span>
            {item.badge && sidebarOpen && (
              <span className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Info */}
      <div className="p-4 border-t border-white/10">
        <div className={`flex items-center gap-3 ${!sidebarOpen && 'lg:justify-center'}`}>
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-bi-accent to-bi-primary flex items-center justify-center flex-shrink-0">
            {user?.avatar ? (
              <img src={user.avatar} alt={user.name} className="w-full h-full rounded-full" />
            ) : (
              <User className="w-5 h-5" />
            )}
          </div>
          {sidebarOpen && (
            <div className="flex-1 min-w-0">
              <p className="font-medium text-sm truncate">{user?.name || 'المستخدم'}</p>
              <p className="text-xs text-gray-400 truncate">{user?.email || 'user@example.com'}</p>
            </div>
          )}
          {sidebarOpen && (
            <button
              onClick={handleLogout}
              className="p-2 hover:bg-white/10 rounded-lg text-gray-400 hover:text-red-400 transition-colors"
              title="تسجيل الخروج"
            >
              <LogOut className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </>
  );

  return (
    <div className="min-h-screen bg-bi-dark flex">
      {/* Desktop Sidebar */}
      <aside
        className={`
          fixed lg:sticky top-0 h-screen bg-bi-card border-l border-white/10
          transition-all duration-300 z-40
          ${sidebarOpen ? 'w-64' : 'w-20'}
          ${mobileMenuOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
          right-0
        `}
      >
        <SidebarContent />
      </aside>

      {/* Mobile Overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="sticky top-0 z-20 bg-bi-card/80 backdrop-blur border-b border-white/10">
          <div className="flex items-center justify-between p-4">
            <div className="flex items-center gap-4">
              {/* Mobile Menu Toggle */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="lg:hidden p-2 hover:bg-white/10 rounded-lg"
              >
                <Menu className="w-5 h-5" />
              </button>

              {/* Sidebar Toggle (Desktop) */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="hidden lg:flex p-2 hover:bg-white/10 rounded-lg"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>

              {/* Breadcrumb */}
              <div className="hidden md:block">
                <Breadcrumb />
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Search */}
              <div className="hidden md:flex items-center bg-white/5 rounded-lg px-3 py-1.5">
                <Search className="w-4 h-4 text-gray-400 ml-2" />
                <input
                  type="text"
                  placeholder="بحث..."
                  className="bg-transparent border-none outline-none text-sm w-48"
                />
                <span className="text-xs text-gray-500 mr-2">⌘K</span>
              </div>

              {/* Notifications */}
              <button className="relative p-2 hover:bg-white/10 rounded-lg">
                <Bell className="w-5 h-5" />
                {notifications > 0 && (
                  <span className="absolute top-1 right-1 w-4 h-4 bg-red-500 rounded-full text-xs flex items-center justify-center">
                    {notifications}
                  </span>
                )}
              </button>

              {/* Mobile User Avatar */}
              <button className="lg:hidden w-8 h-8 rounded-full bg-gradient-to-br from-bi-accent to-bi-primary flex items-center justify-center">
                <User className="w-4 h-4" />
              </button>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 p-4 lg:p-6 overflow-auto">
          <div className="max-w-7xl mx-auto">
            {children || <Outlet />}
          </div>
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
