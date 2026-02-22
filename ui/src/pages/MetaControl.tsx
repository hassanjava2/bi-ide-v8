import { useState, useEffect } from 'react'
import { 
  Layers, 
  Plus, 
  Trash2, 
  Link2, 
  AlertTriangle,
  RefreshCw,
  Shield,
  Users,
  Code,
  CheckCircle
} from 'lucide-react'
import LiveMetricsPanel from '../components/LiveMetricsPanel'

const layerTypes = [
  { value: 'STRATEGIC', label: 'استراتيجي', desc: 'تخطيط واستراتيجية' },
  { value: 'OPERATIONAL', label: 'تشغيلي', desc: 'تنفيذ وتشغيل' },
  { value: 'INTELLIGENCE', label: 'استخباراتي', desc: 'جمع معلومات' },
  { value: 'EXECUTIVE', label: 'تنفيذي', desc: 'بناء وتطوير' },
  { value: 'SECURITY', label: 'أمني', desc: 'حماية وأمان' },
  { value: 'CUSTOM', label: 'مخصص', desc: 'حسب الطلب' },
]

export default function MetaControl() {
  const [controllerStatus, setControllerStatus] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  
  // نموذج إنشاء طبقة
  const [layerName, setLayerName] = useState('')
  const [layerType, setLayerType] = useState('EXECUTIVE')
  const [components, setComponents] = useState('')
  
  // نموذج تدمير
  const [destroyId, setDestroyId] = useState('')
  const [confirmDestroy, setConfirmDestroy] = useState(false)
  
  // نموذج ربط
  const [sourceLayer, setSourceLayer] = useState('')
  const [targetLayer, setTargetLayer] = useState('')

  useEffect(() => {
    fetchStatus()
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/v1/meta/controller/status')
      if (response.ok) {
        const data = await response.json()
        setControllerStatus(data)
      }
    } catch (error) {
      console.log('API not ready')
    }
  }

  const createLayer = async () => {
    if (!layerName.trim()) return
    
    setLoading(true)
    setMessage('')
    
    try {
      const response = await fetch('/api/v1/meta/layer/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: layerName,
          type: layerType,
          components: components.split(',').map(c => c.trim()).filter(Boolean)
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        setMessage(`✅ تم إنشاء الطبقة: ${data.result?.result?.layer_name || layerName}`)
        setLayerName('')
        setComponents('')
      }
    } catch (error) {
      setMessage('❌ خطأ في الإنشاء')
    } finally {
      setLoading(false)
    }
  }

  const destroyLayer = async () => {
    if (!destroyId.trim() || !confirmDestroy) return
    
    setLoading(true)
    setMessage('')
    
    try {
      const response = await fetch('/api/v1/meta/layer/destroy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layer_id: destroyId, force: true })
      })
      
      if (response.ok) {
        setMessage(`💥 تم تدمير الطبقة: ${destroyId}`)
        setDestroyId('')
        setConfirmDestroy(false)
      }
    } catch (error) {
      setMessage('❌ خطأ في التدمير')
    } finally {
      setLoading(false)
    }
  }

  const connectLayers = async () => {
    if (!sourceLayer || !targetLayer) return
    
    setLoading(true)
    try {
      const response = await fetch('/api/v1/meta/layer/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceLayer, target: targetLayer })
      })
      
      if (response.ok) {
        setMessage(`🔗 تم الربط: ${sourceLayer} ↔ ${targetLayer}`)
        setSourceLayer('')
        setTargetLayer('')
      }
    } catch (error) {
      setMessage('❌ خطأ في الربط')
    } finally {
      setLoading(false)
    }
  }

  const emergencyOverride = async () => {
    if (!confirm('هل أنت متأكد من تنفيذ الأمر الطارئ؟')) return
    
    setLoading(true)
    try {
      const response = await fetch('/api/v1/meta/emergency?action=freeze&target=all', {
        method: 'POST'
      })
      
      if (response.ok) {
        setMessage('🚨 تم تنفيذ الأمر الطارئ')
      }
    } catch (error) {
      setMessage('❌ خطأ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* العنوان */}
      <div className="glass-panel p-6 border-bi-gold/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-bi-gold to-yellow-600 flex items-center justify-center">
              <Layers className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">التحكم الفوقي</h1>
              <p className="text-bi-gold">Meta Architect Layer - بناء وهدم الطبقات</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-bi-gold" />
            <span className="text-bi-gold">حكيم التحكم الكامل</span>
          </div>
        </div>
      </div>

      {message && (
        <div className="glass-panel p-4 border-green-500/30 bg-green-500/10">
          <p className="text-green-400">{message}</p>
        </div>
      )}

      {/* حالة المتحكم */}
      {controllerStatus && (
        <div className="glass-panel p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-bi-accent" />
            فرق البناء
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(controllerStatus.builder_teams || {}).map(([team, count]) => (
              <div key={team} className="glass-card p-4 text-center">
                <p className="text-2xl font-bold text-bi-accent">{count as number}</p>
                <p className="text-sm text-gray-400">
                  {team === 'architects' ? 'مهندسين معماريين' :
                   team === 'developers' ? 'مبرمجين' :
                   team === 'engineers' ? 'مهندسين' :
                   team === 'qa_officers' ? 'ضباط جودة' :
                   'مديري مشاريع'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <LiveMetricsPanel title="قياس حي للطبقات والحكماء" showTopWise refreshMs={3000} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* إنشاء طبقة */}
        <div className="glass-panel p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Plus className="w-5 h-5 text-green-400" />
            إنشاء طبقة جديدة
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">اسم الطبقة</label>
              <input
                type="text"
                value={layerName}
                onChange={(e) => setLayerName(e.target.value)}
                placeholder="مثال: نظام الدفع الجديد"
                className="input-field w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">نوع الطبقة</label>
              <select
                value={layerType}
                onChange={(e) => setLayerType(e.target.value)}
                className="input-field w-full"
              >
                {layerTypes.map(t => (
                  <option key={t.value} value={t.value}>{t.label} - {t.desc}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">المكونات (مفصولة بفواصل)</label>
              <input
                type="text"
                value={components}
                onChange={(e) => setComponents(e.target.value)}
                placeholder="API, Database, Cache"
                className="input-field w-full"
              />
            </div>
            <button
              onClick={createLayer}
              disabled={loading || !layerName.trim()}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <Plus className="w-4 h-4" />
              )}
              إنشاء الطبقة
            </button>
          </div>
        </div>

        {/* تدمير طبقة */}
        <div className="glass-panel p-6 border-red-500/30">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-red-400">
            <Trash2 className="w-5 h-5" />
            تدمير طبقة
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">معرف الطبقة</label>
              <input
                type="text"
                value={destroyId}
                onChange={(e) => setDestroyId(e.target.value)}
                placeholder="LAYER-XXX"
                className="input-field w-full border-red-500/30"
              />
            </div>
            <div className="flex items-center gap-3 p-3 bg-red-500/10 rounded-lg">
              <input
                type="checkbox"
                id="confirm"
                checked={confirmDestroy}
                onChange={(e) => setConfirmDestroy(e.target.checked)}
                className="w-4 h-4 rounded border-red-500"
              />
              <label htmlFor="confirm" className="text-sm text-red-400">
                أؤكد أن هذا الإجراء لا يمكن التراجع عنه
              </label>
            </div>
            <button
              onClick={destroyLayer}
              disabled={loading || !destroyId.trim() || !confirmDestroy}
              className="w-full py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Trash2 className="w-4 h-4" />
              تدمير نهائي
            </button>
          </div>
        </div>

        {/* ربط طبقات */}
        <div className="glass-panel p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Link2 className="w-5 h-5 text-blue-400" />
            ربط طبقتين
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">الطبقة المصدر</label>
              <input
                type="text"
                value={sourceLayer}
                onChange={(e) => setSourceLayer(e.target.value)}
                placeholder="معرف الطبقة المصدر"
                className="input-field w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">الطبقة الهدف</label>
              <input
                type="text"
                value={targetLayer}
                onChange={(e) => setTargetLayer(e.target.value)}
                placeholder="معرف الطبقة الهدف"
                className="input-field w-full"
              />
            </div>
            <button
              onClick={connectLayers}
              disabled={loading || !sourceLayer || !targetLayer}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              <Link2 className="w-4 h-4" />
              ربط
            </button>
          </div>
        </div>

        {/* أوامر خاصة */}
        <div className="glass-panel p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Code className="w-5 h-5 text-purple-400" />
            أوامر خاصة
          </h2>
          <div className="space-y-3">
            <button
              onClick={async () => {
                setLoading(true)
                try {
                  await fetch('/api/v1/meta/hierarchy/rebuild?preserve_data=true', {method: 'POST'})
                  setMessage('🔄 تم إعادة بناء الهيكل')
                } catch (e) {}
                setLoading(false)
              }}
              disabled={loading}
              className="w-full btn-secondary flex items-center justify-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              إعادة بناء الهيكل
            </button>
            
            <button
              onClick={async () => {
                setLoading(true)
                try {
                  await fetch('/api/v1/meta/hierarchy/create', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: 'New Project', layers: 3})
                  })
                  setMessage('✨ تم إنشاء هيكل جديد')
                } catch (e) {}
                setLoading(false)
              }}
              disabled={loading}
              className="w-full btn-secondary flex items-center justify-center gap-2"
            >
              <Layers className="w-4 h-4" />
              إنشاء هيكل منفصل
            </button>

            <button
              onClick={emergencyOverride}
              disabled={loading}
              className="w-full py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <AlertTriangle className="w-4 h-4" />
              أمر طارئ - تجميد النظام
            </button>
          </div>
        </div>
      </div>

      {/* قسم المعلومات */}
      <div className="glass-panel p-6">
        <h2 className="text-lg font-semibold mb-4">🔮 القدرات المتاحة</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass-card p-4">
            <CheckCircle className="w-6 h-6 text-green-400 mb-2" />
            <p className="font-medium">إنشاء طبقات</p>
            <p className="text-sm text-gray-400">6 أنواع مختلفة من الطبقات</p>
          </div>
          <div className="glass-card p-4">
            <CheckCircle className="w-6 h-6 text-green-400 mb-2" />
            <p className="font-medium">تدمير طبقات</p>
            <p className="text-sm text-gray-400">حذف دائم مع نسخ احتياطي</p>
          </div>
          <div className="glass-card p-4">
            <CheckCircle className="w-6 h-6 text-green-400 mb-2" />
            <p className="font-medium">ربط/فك ربط</p>
            <p className="text-sm text-gray-400">تحكم كامل في الاتصالات</p>
          </div>
        </div>
      </div>
    </div>
  )
}
