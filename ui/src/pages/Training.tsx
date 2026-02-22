import { useEffect, useState } from 'react'
import { Cpu, Database, AlertCircle, Play, Square, Layers, Globe, Code, Shield, TrendingUp, BookOpen, Brain } from 'lucide-react'

const DEFAULT_RTX4090_IP = '192.168.68.111'
const DEFAULT_RTX4090_PORT = '8080'
const RTX4090_BASE_URL = (
  import.meta.env.VITE_RTX4090_BASE_URL ||
  `http://${DEFAULT_RTX4090_IP}:${DEFAULT_RTX4090_PORT}`
).replace(/\/$/, '')

const rtxUrl = (path: string) => `${RTX4090_BASE_URL}${path}`

interface LayerStatus {
  epoch: number
  loss: number
  accuracy: number
  samples: number
  api_fetches: number
  specialization: string
  vram_gb: number
}

interface TrainingStatus {
  is_training: boolean
  device: string
  mode: string
  data_sources: string[]
  gpu: {
    name: string
    utilization: number
    memory_used: number
    memory_total: number
    temperature: number
  }
  layers: Record<string, LayerStatus>
}

const getLayerIcon = (name: string) => {
  if (name.includes('code') || name.includes('builder')) return <Code className="w-4 h-4" />
  if (name.includes('guardian') || name.includes('security')) return <Shield className="w-4 h-4" />
  if (name.includes('erp') || name.includes('business')) return <TrendingUp className="w-4 h-4" />
  if (name.includes('scouts') || name.includes('research')) return <BookOpen className="w-4 h-4" />
  if (name.includes('council') || name.includes('president')) return <Brain className="w-4 h-4" />
  return <Globe className="w-4 h-4" />
}

export default function Training() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchStatus = async () => {
    try {
      const response = await fetch(rtxUrl('/status'))
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setStatus(data)
      setError('')
    } catch (err) {
      setError('لا يمكن الاتصال بـ RTX 4090')
    } finally {
      setLoading(false)
    }
  }

  const controlTraining = async (action: 'start' | 'stop') => {
    try {
      await fetch(rtxUrl(`/${action}`), { method: 'POST' })
      fetchStatus()
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">جاري تحميل نظام التعلم الذكي...</p>
          <p className="text-sm text-gray-500 mt-2">15 طبقة × 2B Parameters | تعلم من الإنترنت</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* العنوان */}
      <div className="glass-panel p-6 border-purple-500/30">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-400" />
              التعلم الذكي من الإنترنت
            </h1>
            <p className="text-purple-400 font-bold">🤖 كل طبقة تتعلم شغلها أوتوماتيكياً</p>
          </div>
          <div className="text-left">
            <p className="text-sm text-gray-400">RTX URL: {RTX4090_BASE_URL}</p>
            <p className="text-sm text-gray-400">Device: {status?.device || 'Unknown'}</p>
            {status?.mode && <p className="text-xs text-purple-400">{status.mode}</p>}
          </div>
        </div>
      </div>

      {/* مصادر البيانات */}
      {status?.data_sources && (
        <div className="glass-panel p-4">
          <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
            <Globe className="w-4 h-4" />
            مصادر التعلم
          </h3>
          <div className="flex flex-wrap gap-2">
            {status.data_sources.map((source, idx) => (
              <span key={idx} className="px-3 py-1 bg-purple-500/20 text-purple-300 text-xs rounded-full border border-purple-500/30">
                {source}
              </span>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="glass-panel p-6 border-red-500/30 bg-red-500/10">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-6 h-6 text-red-400" />
            <p className="text-red-400 font-medium">{error}</p>
          </div>
        </div>
      )}

      {status && (
        <>
          {/* أزرار التحكم */}
          <div className="glass-panel p-4 flex gap-4">
            <button
              onClick={() => controlTraining('start')}
              disabled={status.is_training}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-bold ${
                status.is_training 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-purple-600 hover:bg-purple-700'
              }`}
            >
              <Play className="w-5 h-5" />
              {status.is_training ? 'قاعد يتعلم...' : '🤖 ابدأ التعلم الذكي'}
            </button>
            <button
              onClick={() => controlTraining('stop')}
              disabled={!status.is_training}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-bold ${
                !status.is_training 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              <Square className="w-5 h-5" />
              إيقاف
            </button>
            <div className="flex-1"></div>
            <div className="text-right">
              <p className="text-sm text-gray-400">GPU: {status.gpu?.name || 'N/A'}</p>
              <p className={`text-sm font-bold ${status.is_training ? 'text-purple-400' : 'text-gray-400'}`}>
                {status.is_training ? '🤖 يتعلم الآن' : '⚪ متوقف'}
              </p>
            </div>
          </div>

          {/* GPU Stats */}
          {status.gpu && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="glass-panel p-4 border-purple-500/20">
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-6 h-6 text-purple-400" />
                  <span className="text-sm text-gray-400">GPU Usage</span>
                </div>
                <p className="text-3xl font-bold text-purple-400">{status.gpu.utilization || 0}%</p>
                <div className="w-full h-2 bg-white/10 rounded-full mt-2">
                  <div 
                    className="h-full bg-purple-500 rounded-full transition-all"
                    style={{ width: `${Math.min(100, status.gpu.utilization || 0)}%` }}
                  ></div>
                </div>
              </div>

              <div className="glass-panel p-4 border-blue-500/20">
                <div className="flex items-center justify-between mb-2">
                  <Database className="w-6 h-6 text-blue-400" />
                  <span className="text-sm text-gray-400">VRAM</span>
                </div>
                <p className="text-2xl font-bold text-blue-400">
                  {(status.gpu.memory_used || 0).toFixed(1)} <span className="text-sm">/ {(status.gpu.memory_total || 0).toFixed(1)} GB</span>
                </p>
                <p className="text-xs text-gray-500 mt-1">{status.gpu.temperature || 0}°C</p>
              </div>

              <div className="glass-panel p-4 border-green-500/20">
                <div className="flex items-center justify-between mb-2">
                  <Layers className="w-6 h-6 text-green-400" />
                  <span className="text-sm text-gray-400">الطبقات</span>
                </div>
                <p className="text-3xl font-bold text-green-400">{Object.keys(status.layers).length}</p>
                <p className="text-sm text-gray-400">طبقة ذكية</p>
              </div>

              <div className="glass-panel p-4 border-orange-500/20">
                <div className="flex items-center justify-between mb-2">
                  <Globe className="w-6 h-6 text-orange-400" />
                  <span className="text-sm text-gray-400">جلب بيانات</span>
                </div>
                <p className="text-2xl font-bold text-orange-400">
                  {Object.values(status.layers).reduce((sum, l) => sum + (l.api_fetches || 0), 0)}
                </p>
                <p className="text-sm text-gray-400">مرة من الإنترنت</p>
              </div>
            </div>
          )}

          {/* جدول الطبقات */}
          <div className="glass-panel p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-purple-400">
              <Brain className="w-5 h-5" />
              الطبقات الـ 15 - كل واحدة تتعلم شغلها (Real-time)
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-gray-400 text-sm border-b border-white/10">
                    <th className="text-right py-3">الطبقة</th>
                    <th className="text-right py-3">الاختصاص</th>
                    <th className="text-center py-3">Epoch</th>
                    <th className="text-center py-3">Loss</th>
                    <th className="text-center py-3">Accuracy</th>
                    <th className="text-center py-3">جلب بيانات</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(status.layers).map(([name, layer]) => (
                    <tr key={name} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-3">
                        <div className="flex items-center gap-2">
                          {getLayerIcon(name)}
                          <span className="text-white font-medium">{name}</span>
                        </div>
                      </td>
                      <td className="py-3 text-xs text-gray-400 max-w-xs truncate">
                        {layer.specialization}
                      </td>
                      <td className="py-3 text-center text-purple-400 font-bold">{layer.epoch}</td>
                      <td className="py-3 text-center">
                        <span className={`${layer.loss < 1.0 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {layer.loss.toFixed(4)}
                        </span>
                      </td>
                      <td className="py-3 text-center">
                        <span className={`${layer.accuracy > 80 ? 'text-green-400' : 'text-blue-400'}`}>
                          {layer.accuracy.toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3 text-center text-orange-400">
                        {layer.api_fetches} 🌐
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
