import { Bell, LogOut, AlertTriangle, AlertCircle, Info, ShieldAlert, Wifi, WifiOff } from 'lucide-react'
import type { SystemStatus, AlertLevel } from '../types'

interface ConnectionStatus {
  isConnected: boolean
  isFallback: boolean
}

interface HeaderProps {
  systemStatus: SystemStatus | null
  connectionStatus?: ConnectionStatus
  onLogout: () => void
}

const alertConfig: Record<AlertLevel, { icon: any; label: string; class: string }> = {
  GREEN: { icon: Info, label: 'عادي', class: 'alert-green' },
  YELLOW: { icon: AlertCircle, label: 'تنبيه', class: 'alert-yellow' },
  ORANGE: { icon: AlertTriangle, label: 'تحذير', class: 'alert-orange' },
  RED: { icon: ShieldAlert, label: 'حرج', class: 'alert-red' },
  BLACK: { icon: ShieldAlert, label: 'وجودي', class: 'alert-black' },
}

export default function Header({ systemStatus, connectionStatus, onLogout }: HeaderProps) {
  const currentAlert: AlertLevel = systemStatus?.execution?.active_crises ? 'RED' : 'GREEN'
  const alert = alertConfig[currentAlert]
  const AlertIcon = alert.icon

  const isConnected = connectionStatus?.isConnected ?? false
  const isFallback = connectionStatus?.isFallback ?? false

  return (
    <header className="h-16 glass-panel border-b border-white/10 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold text-white">
          لوحة تحكم الرئيس
        </h2>
        
        {/* مستوى التنبيه الحالي */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${alert.class}`}>
          <AlertIcon className="w-4 h-4" />
          <span className="text-sm font-medium">حالة النظام: {alert.label}</span>
        </div>

        {/* Connection Status */}
        <div 
          className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs ${
            isConnected 
              ? 'bg-green-500/20 text-green-400' 
              : isFallback 
                ? 'bg-yellow-500/20 text-yellow-400' 
                : 'bg-red-500/20 text-red-400'
          }`}
          title={isConnected ? 'متصل بـ WebSocket' : isFallback ? 'وضع الاحتياط (Polling)' : 'غير متصل'}
        >
          {isConnected ? (
            <Wifi className="w-3 h-3" />
          ) : (
            <WifiOff className="w-3 h-3" />
          )}
          <span>{isConnected ? 'مباشر' : isFallback ? 'احتياطي' : 'غير متصل'}</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* إشعارات */}
        <button className="relative p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Bell className="w-5 h-5 text-gray-300" />
          {systemStatus?.scouts?.high_priority_queue ? (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center text-white font-bold">
              {systemStatus.scouts?.high_priority_queue}
            </span>
          ) : null}
        </button>

        {/* معلومات سريعة */}
        <div className="hidden md:flex items-center gap-4 text-sm text-gray-400">
          <span>المجلس: {systemStatus?.council?.wise_men_count ?? 16} حكيم</span>
          <span>الكفاءة: {systemStatus?.meta?.performance_score ?? 85}%</span>
        </div>

        {/* تسجيل الخروج */}
        <button 
          onClick={onLogout}
          className="flex items-center gap-2 px-4 py-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
        >
          <LogOut className="w-4 h-4" />
          <span>خروج</span>
        </button>
      </div>
    </header>
  )
}
