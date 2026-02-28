import { memo, useMemo } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Users,
  Building2,
  MessageCircle,
  Code2,
  Brain,
  Download,
  Settings,
  Crown,
  Layers,
  Server
} from 'lucide-react'

// Static data moved outside component to prevent recreating on each render
const MENU_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: 'لوحة التحكم' },
  { path: '/council', icon: Users, label: 'مجلس الحكماء' },
  { path: '/erp', icon: Building2, label: 'نظام ERP' },
  { path: '/community', icon: MessageCircle, label: 'المجتمع' },
  { path: '/ide', icon: Code2, label: 'بيئة التطوير' },
  { path: '/training', icon: Brain, label: 'التدريب' },
  { path: '/nodes', icon: Server, label: 'العقد المتصلة' },
  { path: '/downloads', icon: Download, label: 'التنزيلات' },
  { path: '/meta', icon: Layers, label: 'التحكم الفوقي' },
  { path: '/settings', icon: Settings, label: 'الإعدادات' },
] as const

// Memoized menu item component with custom comparison
interface MenuItemProps {
  path: string
  icon: typeof LayoutDashboard
  label: string
}

const MenuItem = memo(function MenuItem({ path, icon: Icon, label }: MenuItemProps) {
  return (
    <NavLink
      to={path}
      className={({ isActive }) =>
        `sidebar-item ${isActive ? 'active' : ''}`
      }
    >
      <Icon className="w-5 h-5" />
      <span>{label}</span>
    </NavLink>
  )
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if path changes
  return prevProps.path === nextProps.path
})

// Memoized user profile component
const UserProfile = memo(function UserProfile() {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-yellow-500 to-yellow-600 flex items-center justify-center text-white font-bold border-2 border-yellow-400">
          ر
        </div>
        <div>
          <p className="font-medium text-white">الرئيس</p>
          <p className="text-xs text-green-400">● متصل</p>
        </div>
      </div>
      <div className="flex gap-2">
        <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded">
          الفيتو مفعل
        </span>
        <span className="px-2 py-1 bg-bi-accent/20 text-bi-accent text-xs rounded">
          24/7
        </span>
      </div>
    </div>
  )
})

// Memoized logo component
const Logo = memo(function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-bi-accent to-bi-gold flex items-center justify-center">
        <Crown className="w-6 h-6 text-white" />
      </div>
      <div>
        <h1 className="text-xl font-bold text-gradient">BI IDE</h1>
        <p className="text-xs text-gray-400">النظام الهرمي الذكي</p>
      </div>
    </div>
  )
})

function Sidebar() {
  // Memoize menu items mapping
  const menuItemsList = useMemo(() => {
    return MENU_ITEMS.map((item) => (
      <MenuItem
        key={item.path}
        path={item.path}
        icon={item.icon}
        label={item.label}
      />
    ))
  }, [])

  return (
    <aside className="fixed right-0 top-0 h-full w-64 glass-panel border-l border-white/10 z-50">
      <div className="p-6 border-b border-white/10">
        <Logo />
      </div>

      <nav className="p-4 space-y-1">
        {menuItemsList}
      </nav>

      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-white/10">
        <UserProfile />
      </div>
    </aside>
  )
}

export default memo(Sidebar)
