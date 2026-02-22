import { memo, useMemo, useCallback } from 'react'
import { Activity, Layers, Users, Wifi, WifiOff, RefreshCw } from 'lucide-react'
import { useLiveData } from '../hooks/useLiveData'

type LiveMetricsPanelProps = {
  title?: string
  compact?: boolean
  showTopWise?: boolean
  refreshMs?: number
}

// Memoized metric card component
interface MetricCardProps {
  value: number | string
  label: string
}

const MetricCard = memo(function MetricCard({ value, label }: MetricCardProps) {
  return (
    <div className="glass-card p-3 text-center">
      <p className="text-xl font-bold text-white">{value}</p>
      <p className="text-xs text-gray-400">{label}</p>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.value === nextProps.value
})

// Memoized top wise man item component
interface TopWiseManItemProps {
  name: string
  responses: number
}

const TopWiseManItem = memo(function TopWiseManItem({ name, responses }: TopWiseManItemProps) {
  return (
    <div className="flex items-center justify-between text-xs p-2 rounded bg-white/5">
      <span className="text-gray-300 truncate">{name}</span>
      <span className="text-bi-accent font-semibold">{responses ?? 0}</span>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.name === nextProps.name && prevProps.responses === nextProps.responses
})

// Memoized daily trend item component
interface DailyTrendItemProps {
  day: string
  evidenceRatePct: number
}

const DailyTrendItem = memo(function DailyTrendItem({ day, evidenceRatePct }: DailyTrendItemProps) {
  return (
    <div className="flex items-center justify-between text-[10px] text-gray-400">
      <span>{day}</span>
      <span>{evidenceRatePct}%</span>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.day === nextProps.day && prevProps.evidenceRatePct === nextProps.evidenceRatePct
})

// Memoized layer activity card component
interface LayerActivityCardProps {
  layer: string
  value: number
}

const LayerActivityCard = memo(function LayerActivityCard({ layer, value }: LayerActivityCardProps) {
  return (
    <div className="glass-card p-2 text-center">
      <p className="text-gray-400 mb-1">{layer}</p>
      <p className="text-white font-semibold">{value ?? 0}</p>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.layer === nextProps.layer && prevProps.value === nextProps.value
})

// Static layer keys moved outside component
const LAYER_KEYS = ['council', 'scouts', 'meta', 'experts', 'execution', 'guardian'] as const

function LiveMetricsPanel({
  title = 'قياس حي',
  compact = false,
  showTopWise = false,
  refreshMs
}: LiveMetricsPanelProps) {
  const { 
    liveMetrics, 
    hierarchyMetrics, 
    lastUpdated, 
    isConnected, 
    isFallback, 
    error, 
    refresh 
  } = useLiveData(refreshMs)

  // Memoize derived data
  const layers = useMemo(() => hierarchyMetrics?.layers || {}, [hierarchyMetrics])
  
  const topWise = useMemo(() => {
    return (liveMetrics?.top_wise_men || []).slice(0, compact ? 4 : 8)
  }, [liveMetrics?.top_wise_men, compact])
  
  const quality = useMemo(() => liveMetrics?.quality || {}, [liveMetrics])
  
  const dailyTrend = useMemo(() => {
    const q = quality as any
    return Array.isArray(q?.daily_trend) ? q.daily_trend : []
  }, [quality])
  
  const latestTrend = useMemo(() => {
    return dailyTrend.length > 0 ? dailyTrend[dailyTrend.length - 1] : null
  }, [dailyTrend])

  // Memoized format time function
  const formatTime = useCallback((isoString: string | null) => {
    if (!isoString) return '-'
    try {
      return new Date(isoString).toLocaleTimeString('ar-SA')
    } catch {
      return '-'
    }
  }, [])

  // Memoized refresh handler
  const handleRefresh = useCallback(() => {
    refresh()
  }, [refresh])

  // Memoized connection status
  const connectionStatus = useMemo(() => {
    if (isConnected) {
      return { icon: Wifi, color: 'text-green-400', label: 'مباشر' }
    } else if (isFallback) {
      return { icon: WifiOff, color: 'text-yellow-400', label: 'احتياطي' }
    } else {
      return { icon: WifiOff, color: 'text-red-400', label: 'غير متصل' }
    }
  }, [isConnected, isFallback])

  // Memoized metrics values
  const councilResponses = useMemo(() => liveMetrics?.council_responses ?? 0, [liveMetrics])
  const fallbackRate = useMemo(() => liveMetrics?.fallback_rate_pct ?? 0, [liveMetrics])
  const avgLatency = useMemo(() => liveMetrics?.latency_ms?.avg ?? 0, [liveMetrics])
  const lastLatency = useMemo(() => liveMetrics?.latency_ms?.last ?? 0, [liveMetrics])
  
  const q = quality as any
  const evidenceBackedRate = useMemo(() => q?.evidence_backed_rate_pct ?? 0, [q])
  const evidenceBackedTotal = useMemo(() => q?.evidence_backed_total ?? 0, [q])
  const guardTotal = useMemo(() => q?.guard_total ?? 0, [q])

  // Memoized lists
  const topWiseList = useMemo(() => {
    if (topWise.length === 0) {
      return <p className="text-xs text-gray-500">لا توجد بيانات بعد</p>
    }
    return topWise.map((wise: any, index: number) => (
      <TopWiseManItem
        key={`${wise.name}-${index}`}
        name={wise.name}
        responses={wise.responses ?? 0}
      />
    ))
  }, [topWise])

  const dailyTrendList = useMemo(() => {
    if (dailyTrend.length === 0) return null
    return dailyTrend.slice(-5).map((day: any) => (
      <DailyTrendItem
        key={day.day}
        day={day.day}
        evidenceRatePct={day.evidence_rate_pct}
      />
    ))
  }, [dailyTrend])

  const layerActivityList = useMemo(() => {
    return LAYER_KEYS.map((layer) => (
      <LayerActivityCard
        key={layer}
        layer={layer}
        value={layers[layer] ?? 0}
      />
    ))
  }, [layers])

  const StatusIcon = connectionStatus.icon

  return (
    <div className="glass-panel p-4">
      <div className="flex items-center justify-between mb-4 gap-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Activity className="w-5 h-5 text-bi-accent" />
          {title}
        </h2>
        <div className="flex items-center gap-2">
          {/* Connection status indicator */}
          <div 
            className="flex items-center gap-1 text-xs" 
            title={isConnected ? 'متصل بـ WebSocket' : isFallback ? 'وضع الاحتياط (Polling)' : 'غير متصل'}
          >
            <StatusIcon className={`w-4 h-4 ${connectionStatus.color}`} />
            <span className={connectionStatus.color}>
              {connectionStatus.label}
            </span>
          </div>
          
          {/* Refresh button */}
          <button 
            onClick={handleRefresh}
            className="p-1 hover:bg-white/10 rounded transition-colors"
            title="تحديث"
          >
            <RefreshCw className="w-4 h-4 text-gray-400" />
          </button>
          
          {/* Last updated time */}
          <div className="text-[10px] text-gray-400 text-left">
            <p>آخر تحديث: {formatTime(lastUpdated)}</p>
            {error && <p className="text-red-400">{error}</p>}
          </div>
        </div>
      </div>

      <div className={`grid ${compact ? 'grid-cols-2' : 'grid-cols-1 md:grid-cols-3'} gap-3 mb-4`}>
        <MetricCard value={councilResponses} label="ردود المجلس" />
        <MetricCard value={`${fallbackRate}%`} label="Fallback" />
        {!compact && (
          <MetricCard value={`${avgLatency}ms`} label="متوسط الاستجابة" />
        )}
      </div>

      {compact && (
        <div className="glass-card p-3 mb-3">
          <p className="text-xs text-gray-400 mb-2">Latency</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <p className="text-gray-300">Avg: <span className="text-white">{avgLatency}ms</span></p>
            <p className="text-gray-300">Last: <span className="text-white">{lastLatency}ms</span></p>
          </div>
        </div>
      )}

      {showTopWise && (
        <div className="glass-card p-3 mb-3">
          <p className="text-xs text-gray-400 mb-2 flex items-center gap-1">
            <Users className="w-3 h-3" /> أكثر الحكماء نشاطًا
          </p>
          <div className="space-y-2 max-h-40 overflow-auto">
            {topWiseList}
          </div>
        </div>
      )}

      <div className="glass-card p-3 mb-3">
        <p className="text-xs text-gray-400 mb-2">جودة الردود (evidence-backed)</p>
        <div className={`grid ${compact ? 'grid-cols-2' : 'grid-cols-3'} gap-2 text-xs mb-2`}>
          <p className="text-gray-300">نسبة كلية: <span className="text-white">{evidenceBackedRate}%</span></p>
          <p className="text-gray-300">موثّق: <span className="text-white">{evidenceBackedTotal}</span></p>
          {!compact && <p className="text-gray-300">Guarded: <span className="text-white">{guardTotal}</span></p>}
        </div>
        {latestTrend && (
          <p className="text-[10px] text-gray-500 mb-2">
            اليوم {latestTrend.day}: {latestTrend.evidence_rate_pct}% ({latestTrend.evidence_backed}/{latestTrend.responses})
          </p>
        )}
        {dailyTrendList && (
          <div className="space-y-1 max-h-20 overflow-auto">
            {dailyTrendList}
          </div>
        )}
      </div>

      <div className={`glass-card p-3 ${compact ? '' : 'text-xs'}`}>
        <p className="text-xs text-gray-400 mb-2 flex items-center gap-1">
          <Layers className="w-3 h-3" /> نشاط الطبقات
        </p>
        <div className={`grid ${compact ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-3'} gap-2 text-xs`}>
          {layerActivityList}
        </div>
      </div>
    </div>
  )
}

export default memo(LiveMetricsPanel)
