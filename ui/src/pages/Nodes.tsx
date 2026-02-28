import { useState, useEffect, useCallback } from 'react'

interface Worker {
  worker_id: string
  hostname: string
  labels: string[]
  hardware: {
    cpu_name: string
    cpu_cores: number
    ram_gb: number
    gpu: { name: string; vram_gb: number; cuda_available: boolean }
    os_type: string
    disk_gb: number
  }
  usage?: {
    cpu_percent: number
    ram_percent: number
    gpu_percent: number
    gpu_mem_percent: number
    gpu_temp_c: number
  }
  training?: {
    is_training: boolean
    job_id?: string
    layer_name?: string
  }
  status: string
  version: string
  last_heartbeat_ago: string
  registered_at: string
  connected: boolean
}

interface OrchestratorHealth {
  workers_total: number
  workers_online: number
  primary_connected: boolean
  jobs_total: number
  jobs_running: number
}

const API = '/api/v1/orchestrator'

export default function Nodes() {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [health, setHealth] = useState<OrchestratorHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchData = useCallback(async () => {
    try {
      const [healthRes, workersRes] = await Promise.all([
        fetch(`${API}/health`),
        fetch(`${API}/workers`),
      ])
      if (healthRes.ok) setHealth(await healthRes.json())
      if (workersRes.ok) {
        const data = await workersRes.json()
        setWorkers(data.workers || [])
      }
      setError('')
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [fetchData])

  const sendCommand = async (workerId: string, command: string) => {
    try {
      await fetch(`${API}/workers/${workerId}/command?command=${command}`, {
        method: 'POST',
      })
      fetchData()
    } catch (e) {
      console.error(e)
    }
  }

  const statusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500'
      case 'training': return 'bg-blue-500 animate-pulse'
      case 'syncing': return 'bg-yellow-500'
      case 'throttled': return 'bg-orange-500'
      case 'offline': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const statusLabel = (status: string) => {
    switch (status) {
      case 'online': return 'متصل'
      case 'training': return 'يتدرب 🧠'
      case 'syncing': return 'يزامن 🔄'
      case 'throttled': return 'مقيد ⚠️'
      case 'offline': return 'غير متصل'
      default: return status
    }
  }

  const roleLabel = (labels: string[]) => {
    if (labels.includes('primary') || labels.includes('rtx5090')) return '🏆 رئيسي'
    if (labels.includes('hostinger') || labels.includes('orchestrator')) return '☁️ سيرفر'
    return '🔧 مساعد'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-gray-400">جاري التحميل...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">🖧 العقد المتصلة</h1>
          <p className="text-gray-400 text-sm mt-1">مراقبة وتحكم بكل أجهزة التدريب</p>
        </div>
        <button
          onClick={fetchData}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          🔄 تحديث
        </button>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500/50 text-red-300 px-4 py-3 rounded-lg">
          ❌ {error}
        </div>
      )}

      {/* Stats Cards */}
      {health && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-gray-700/50">
            <div className="text-3xl font-bold text-white">{health.workers_total}</div>
            <div className="text-gray-400 text-sm">إجمالي العقد</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-green-500/30">
            <div className="text-3xl font-bold text-green-400">{health.workers_online}</div>
            <div className="text-gray-400 text-sm">متصلة</div>
          </div>
          <div className={`bg-gray-800/50 backdrop-blur rounded-xl p-4 border ${health.primary_connected ? 'border-yellow-500/30' : 'border-red-500/30'}`}>
            <div className="text-3xl font-bold">{health.primary_connected ? '✅' : '❌'}</div>
            <div className="text-gray-400 text-sm">RTX 5090</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-blue-500/30">
            <div className="text-3xl font-bold text-blue-400">{health.jobs_running}</div>
            <div className="text-gray-400 text-sm">مهام تعمل</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-purple-500/30">
            <div className="text-3xl font-bold text-purple-400">{health.jobs_total}</div>
            <div className="text-gray-400 text-sm">إجمالي المهام</div>
          </div>
        </div>
      )}

      {/* Workers Grid */}
      {workers.length === 0 ? (
        <div className="bg-gray-800/50 backdrop-blur rounded-xl p-8 text-center border border-gray-700/50">
          <div className="text-5xl mb-4">🖧</div>
          <h3 className="text-xl text-white mb-2">لا توجد عقد متصلة</h3>
          <p className="text-gray-400 mb-4">نصّب Worker Agent على أي حاسبة للبدء</p>
          <div className="bg-gray-900/50 rounded-lg p-4 text-left max-w-lg mx-auto">
            <p className="text-gray-300 text-sm font-mono">
              curl -fsSL https://bi-iq.com/api/v1/orchestrator/download/linux -o install.sh<br />
              chmod +x install.sh<br />
              ./install.sh https://bi-iq.com "" my-pc "gpu,primary"
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {workers.map((w) => (
            <div
              key={w.worker_id}
              className={`bg-gray-800/50 backdrop-blur rounded-xl p-5 border transition-all ${
                w.status === 'offline' ? 'border-red-500/30 opacity-60' :
                w.status === 'training' ? 'border-blue-500/50 shadow-lg shadow-blue-500/10' :
                'border-gray-700/50'
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${statusColor(w.status)}`} />
                  <div>
                    <h3 className="text-white font-bold">{w.hostname || w.worker_id}</h3>
                    <span className="text-xs text-gray-500">{roleLabel(w.labels)}</span>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  w.status === 'online' ? 'bg-green-500/20 text-green-300' :
                  w.status === 'training' ? 'bg-blue-500/20 text-blue-300' :
                  w.status === 'throttled' ? 'bg-orange-500/20 text-orange-300' :
                  'bg-red-500/20 text-red-300'
                }`}>
                  {statusLabel(w.status)}
                </span>
              </div>

              {/* Hardware */}
              <div className="grid grid-cols-3 gap-3 mb-4 text-sm">
                <div className="bg-gray-900/50 rounded-lg p-2 text-center">
                  <div className="text-gray-400 text-xs">GPU</div>
                  <div className="text-white font-medium truncate" title={w.hardware?.gpu?.name}>
                    {w.hardware?.gpu?.name === 'none' ? '—' : (w.hardware?.gpu?.name || '—').replace('NVIDIA ', '').replace('GeForce ', '')}
                  </div>
                  {w.hardware?.gpu?.vram_gb > 0 && (
                    <div className="text-gray-500 text-xs">{w.hardware.gpu.vram_gb}GB</div>
                  )}
                </div>
                <div className="bg-gray-900/50 rounded-lg p-2 text-center">
                  <div className="text-gray-400 text-xs">RAM</div>
                  <div className="text-white font-medium">{w.hardware?.ram_gb || 0}GB</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-2 text-center">
                  <div className="text-gray-400 text-xs">CPU</div>
                  <div className="text-white font-medium">{w.hardware?.cpu_cores || 0} cores</div>
                </div>
              </div>

              {/* Usage Bars */}
              {w.usage && w.status !== 'offline' && (
                <div className="space-y-2 mb-4">
                  <UsageBar label="CPU" value={w.usage.cpu_percent} color="blue" />
                  <UsageBar label="RAM" value={w.usage.ram_percent} color="green" />
                  {w.usage.gpu_percent > 0 && (
                    <UsageBar label="GPU" value={w.usage.gpu_percent} color="purple" />
                  )}
                </div>
              )}

              {/* Training Status */}
              {w.training?.is_training && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 mb-3">
                  <div className="text-blue-300 text-sm font-medium">🧠 يتدرب الآن</div>
                  {w.training.layer_name && (
                    <div className="text-blue-200 text-xs mt-1">الطبقة: {w.training.layer_name}</div>
                  )}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => sendCommand(w.worker_id, 'start_training')}
                  className="flex-1 px-3 py-1.5 bg-green-600/20 hover:bg-green-600/40 text-green-300 rounded-lg text-xs transition-colors"
                  disabled={w.status === 'offline'}
                >
                  ▶ تدريب
                </button>
                <button
                  onClick={() => sendCommand(w.worker_id, 'stop_job')}
                  className="flex-1 px-3 py-1.5 bg-red-600/20 hover:bg-red-600/40 text-red-300 rounded-lg text-xs transition-colors"
                  disabled={w.status === 'offline'}
                >
                  ⏹ إيقاف
                </button>
                <button
                  onClick={() => sendCommand(w.worker_id, 'update')}
                  className="px-3 py-1.5 bg-gray-600/20 hover:bg-gray-600/40 text-gray-300 rounded-lg text-xs transition-colors"
                  disabled={w.status === 'offline'}
                >
                  🔄
                </button>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
                <span>{w.hardware?.os_type}</span>
                <span>آخر اتصال: {w.last_heartbeat_ago}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Install Instructions */}
      <div className="bg-gray-800/50 backdrop-blur rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-lg font-bold text-white mb-4">📥 إضافة عقدة جديدة</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="text-xl mb-2">🐧</div>
            <h4 className="text-white font-medium mb-2">Linux / Ubuntu</h4>
            <code className="text-xs text-green-300 block bg-black/30 p-2 rounded">
              curl -fsSL https://bi-iq.com/api/v1/orchestrator/download/linux | bash
            </code>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="text-xl mb-2">🪟</div>
            <h4 className="text-white font-medium mb-2">Windows</h4>
            <code className="text-xs text-blue-300 block bg-black/30 p-2 rounded">
              iwr .../download/windows | iex
            </code>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="text-xl mb-2">🍎</div>
            <h4 className="text-white font-medium mb-2">macOS</h4>
            <code className="text-xs text-purple-300 block bg-black/30 p-2 rounded">
              curl -fsSL .../download/macos | bash
            </code>
          </div>
        </div>
      </div>
    </div>
  )
}

function UsageBar({ label, value, color }: { label: string; value: number; color: string }) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    red: 'bg-red-500',
    yellow: 'bg-yellow-500',
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-8">{label}</span>
      <div className="flex-1 bg-gray-700/50 rounded-full h-2">
        <div
          className={`${colorMap[color] || 'bg-blue-500'} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
      <span className={`text-xs w-10 text-right ${value > 80 ? 'text-red-400' : 'text-gray-400'}`}>
        {value.toFixed(0)}%
      </span>
    </div>
  )
}
