import { useEffect, useState } from 'react'
import { 
  Users, Brain, Activity, Target, TrendingUp, 
  Shield, Zap, MessageSquare, AlertCircle,
  Layers, Lock, Scale, Globe, Infinity
} from 'lucide-react'
import { useSystemStatus, useCouncilMetrics, useHierarchyStatus } from '../hooks/queries'
import LiveMetricsPanel from '../components/LiveMetricsPanel'

const stats = [
  { label: 'الحكماء النشطون', value: '16', icon: Users, color: 'text-blue-400' },
  { label: 'الخبراء الجاهزون', value: '11', icon: Brain, color: 'text-purple-400' },
  { label: 'المهام النشطة', value: '3', icon: Target, color: 'text-green-400' },
  { label: 'كفاءة النظام', value: '85%', icon: Activity, color: 'text-yellow-400' },
]

const layerStats = [
  { label: 'طبقات الحماية', value: '5', icon: Lock, color: 'text-red-400' },
  { label: 'الامتثال القانوني', value: '4', icon: Scale, color: 'text-green-400' },
  { label: 'APIs خارجية', value: '3', icon: Globe, color: 'text-blue-400' },
  { label: 'أرشيف الأبدية', value: '1000', unit: 'سنة', icon: Infinity, color: 'text-purple-400' },
]

export default function Dashboard() {
  // React Query hooks for data fetching with caching
  const { 
    data: status, 
    isLoading: statusLoading, 
    error: statusError,
    refetch: refetchStatus 
  } = useSystemStatus({ poll: true, pollInterval: 10000 })
  
  const { 
    data: councilMetrics, 
    isLoading: councilLoading 
  } = useCouncilMetrics({ poll: true, pollInterval: 30000 })
  
  const { 
    data: hierarchyData 
  } = useHierarchyStatus({ poll: true, pollInterval: 30000 })

  const [guardianReport, setGuardianReport] = useState<any>(null)
  const [wisdom, setWisdom] = useState('')

  // Fetch additional data (wisdom and guardian report) via traditional fetch
  // These don't change often, so we'll fetch them once on mount
  useEffect(() => {
    const fetchAdditionalData = async () => {
      try {
        const [guardianRes, wisdomRes] = await Promise.all([
          fetch('/api/v1/guardian/status').then(r => r.ok ? r.json() : null).catch(() => null),
          fetch('/api/v1/wisdom?horizon=century').then(r => r.ok ? r.json() : null).catch(() => ({ wisdom: '' }))
        ])
        setGuardianReport(guardianRes)
        setWisdom(wisdomRes?.wisdom || '')
      } catch {
        // Silent fail for non-critical data
      }
    }
    fetchAdditionalData()
  }, [])

  const isLoading = statusLoading || councilLoading
  const error = statusError

  // Combined refresh function
  const handleRefresh = () => {
    refetchStatus()
  }

  if (isLoading && !status) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-bi-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">جاري تحميل بيانات النظام...</p>
        </div>
      </div>
    )
  }

  if (error && !status) {
    return (
      <div className="glass-panel p-6 text-center">
        <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <p className="text-red-400">خطأ في الاتصال بالنظام</p>
        <button onClick={handleRefresh} className="btn-primary mt-4">إعادة المحاولة</button>
      </div>
    )
  }

  const hierarchy = status?.hierarchy || hierarchyData || {}

  return (
    <div className="space-y-6">
      {/* ترحيب */}
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">
              أهلاً بك في المجلس، سيادة الرئيس
            </h1>
            <p className="text-gray-400">
              النظام الهرمي المتكامل - 15 طبقة ذكاء اصطناعي
            </p>
          </div>
          <div className="text-left">
            <div className="flex items-center gap-2 text-green-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>النظام يعمل</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">v3.0.0 - 15 Layers Active</p>
          </div>
        </div>
        {wisdom && (
          <p className="mt-4 text-bi-gold text-sm flex items-center gap-2">
            <Shield className="w-4 h-4" />
            {wisdom}
          </p>
        )}
      </div>

      {/* الإحصائيات الأساسية */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <div key={index} className="stat-card">
            <div className="flex items-center justify-between">
              <stat.icon className={`w-8 h-8 ${stat.color}`} />
              <span className="text-3xl font-bold text-white">{stat.value}</span>
            </div>
            <p className="text-sm text-gray-400">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* إحصائيات الطبقات الجديدة */}
      <div className="glass-panel p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-bi-accent" />
          الطبقات المتقدمة
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {layerStats.map((stat, index) => (
            <div key={index} className="glass-card p-4">
              <div className="flex items-center justify-between">
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
                <span className="text-2xl font-bold text-white">
                  {stat.value}
                  {stat.unit && <span className="text-sm mr-1">{stat.unit}</span>}
                </span>
              </div>
              <p className="text-sm text-gray-400 mt-2">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Guardian Status */}
      {guardianReport && (
        <div className="glass-panel p-6 border-bi-gold/30">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5 text-bi-gold" />
            طبقة الحارس الشامل
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-3xl font-bold text-bi-gold">{guardianReport.total_requests || 0}</p>
              <p className="text-sm text-gray-400">طلبات محمية</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-green-400">{guardianReport.threats_blocked || 0}</p>
              <p className="text-sm text-gray-400">تهديدات تم إيقافها</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-400">{guardianReport.violations_prevented || 0}</p>
              <p className="text-sm text-gray-400">مخالفات تم منعها</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-400">{guardianReport.current_mode || 'ACTIVE'}</p>
              <p className="text-sm text-gray-400">وضع الحماية</p>
            </div>
          </div>
        </div>
      )}

      {/* حالة المجلس والنظام */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* المجلس */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-4">
            <Users className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">حالة المجلس</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">حالة الاجتماع</span>
              <span className="flex items-center gap-2 text-green-400">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                مستمر 24/7
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">الحكماء الحاضرون</span>
              <span className="text-white font-medium">
                {hierarchy?.council?.wise_men_count || 16} / 16
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">الخبراء الجاهزون</span>
              <span className="text-white font-medium">
                {hierarchy?.experts?.total || 11} خبير
              </span>
            </div>
          </div>
        </div>

        {/* الأداء */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">أداء النظام</h2>
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-gray-400">أداء النظام</span>
                <span className="text-white">{hierarchy?.meta?.performance_score || 85}%</span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-bi-accent rounded-full transition-all"
                  style={{ width: `${hierarchy?.meta?.performance_score || 85}%` }}
                ></div>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-gray-400">جودة القرارات</span>
                <span className="text-white">{hierarchy?.meta?.quality_score || 90}%</span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all"
                  style={{ width: `${hierarchy?.meta?.quality_score || 90}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* نشاط حي */}
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">النشاط الحي</h2>
          </div>
          <button 
            onClick={handleRefresh} 
            className="text-sm text-bi-accent hover:underline"
            disabled={statusLoading}
          >
            {statusLoading ? 'جاري التحديث...' : 'تحديث'}
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass-card p-4">
            <MessageSquare className="w-6 h-6 text-blue-400 mb-2" />
            <p className="text-2xl font-bold text-white">{hierarchy?.scouts?.intel_buffer_size || 0}</p>
            <p className="text-sm text-gray-400">معلومات مخزنة</p>
          </div>
          <div className="glass-card p-4">
            <Brain className="w-6 h-6 text-purple-400 mb-2" />
            <p className="text-2xl font-bold text-white">{hierarchy?.meta?.learning_progress || 0}</p>
            <p className="text-sm text-gray-400">أنماط مكتسبة</p>
          </div>
          <div className="glass-card p-4">
            <TrendingUp className="w-6 h-6 text-green-400 mb-2" />
            <p className="text-2xl font-bold text-white">
              {Math.round((hierarchy?.execution?.quality_score || 0.95) * 100)}%
            </p>
            <p className="text-sm text-gray-400">نسبة النجاح</p>
          </div>
        </div>
      </div>

      <LiveMetricsPanel title="تطور الحكماء والطبقات (Live)" showTopWise refreshMs={3000} />
    </div>
  )
}
