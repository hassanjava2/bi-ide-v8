import { memo, useMemo, useCallback, useState, useEffect } from 'react'
import { 
  Users, 
  MessageSquare, 
  Send, 
  Shield, 
  AlertTriangle,
  AlertCircle,
  Info,
  UsersRound
} from 'lucide-react'
import type { AlertLevel } from '../types'
import LiveMetricsPanel from '../components/LiveMetricsPanel'

// Static data moved outside component
const ALERT_LEVELS: { level: AlertLevel; label: string; desc: string; color: string; icon: typeof Info }[] = [
  { level: 'GREEN', label: 'أخضر', desc: 'معلومة فقط', color: 'alert-green', icon: Info },
  { level: 'YELLOW', label: 'أصفر', desc: 'قرار عادي (5 دقائق)', color: 'alert-yellow', icon: AlertCircle },
  { level: 'ORANGE', label: 'برتقالي', desc: 'قرار مهم (1 ساعة)', color: 'alert-orange', icon: AlertTriangle },
  { level: 'RED', label: 'أحمر', desc: 'قرار حيوي (15 دقيقة)', color: 'alert-red', icon: AlertTriangle },
  { level: 'BLACK', label: 'أسود', desc: 'خطر وجودي (يتوقف)', color: 'alert-black', icon: Shield },
]

const WISE_MEN = [
  { id: 'S001', name: 'حكيم القرار', role: 'رئيس المجلس', specialty: 'الاستراتيجية', status: 'speaking' },
  { id: 'S002', name: 'حكيم المستقبل', role: 'تخطيط طويل المدى', specialty: 'الرؤية', status: 'listening' },
  { id: 'S003', name: 'حكيم البصيرة', role: 'تحليل عميق', specialty: 'التحليل', status: 'listening' },
  { id: 'S004', name: 'حكيم التوازن', role: 'العدالة', specialty: 'الموازنة', status: 'thinking' },
  { id: 'S005', name: 'حكيم الشجاعة', role: 'المبادرة', specialty: 'الجرأة', status: 'listening' },
  { id: 'S006', name: 'حكيم الضبط', role: 'مراقبة', specialty: 'النظام', status: 'listening' },
  { id: 'S007', name: 'حكيم التكيف', role: 'تطور', specialty: 'التغيير', status: 'listening' },
  { id: 'S008', name: 'حكيم الذاكرة', role: 'تاريخ', specialty: 'الخبرة', status: 'listening' },
  { id: 'O001', name: 'حكيم النظام', role: 'عمليات', specialty: 'التنفيذ', status: 'active' },
  { id: 'O002', name: 'حكيم التنفيذ', role: 'عمليات', specialty: 'الإنجاز', status: 'listening' },
  { id: 'O003', name: 'حكيم الربط', role: 'عمليات', specialty: 'التواصل', status: 'listening' },
  { id: 'O004', name: 'حكيم التقارير', role: 'عمليات', specialty: 'المعلومات', status: 'listening' },
  { id: 'O005', name: 'حكيم التنسيق', role: 'عمليات', specialty: 'التنظيم', status: 'listening' },
  { id: 'O006', name: 'حكيم المتابعة', role: 'عمليات', specialty: 'المراقبة', status: 'listening' },
  { id: 'O007', name: 'حكيم التدقيق', role: 'عمليات', specialty: 'الجودة', status: 'listening' },
  { id: 'O008', name: 'حكيم الطوارئ', role: 'عمليات', specialty: 'الاستجابة', status: 'standby' },
] as const

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const apiUrl = (path: string) => `${API_BASE_URL}${path}`
const DISCUSSIONS_STORAGE_KEY = 'bi_council_discussions_v1'

const DEFAULT_DISCUSSIONS = [
  { id: 1, speaker: 'حكيم القرار', text: 'سيادة الرئيس، المجلس في انتظار أوامرك.', time: 'الآن', type: 'system' },
  { id: 2, speaker: 'حكيم البصيرة', text: 'لدينا 3 تقارير جاهزة للمراجعة.', time: 'منذ 5 دقائق', type: 'info' },
]

type DiscussionItem = {
  id: number
  speaker: string
  text: string
  time: string
  type: string
  sourceTag?: string
  evidenceHint?: string
}

// Memoized Wise Man Card component with custom comparison by id
interface WiseManCardProps {
  id: string
  name: string
  role: string
  specialty: string
  status: string
}

const WiseManCard = memo(function WiseManCard({ name, specialty, status }: WiseManCardProps) {
  const statusColor = useMemo(() => {
    switch (status) {
      case 'speaking': return 'bg-green-400 animate-pulse'
      case 'thinking': return 'bg-yellow-400'
      case 'active': return 'bg-blue-400'
      default: return 'bg-gray-500'
    }
  }, [status])

  const containerClass = useMemo(() => {
    return status === 'speaking' 
      ? 'bg-bi-accent/20 border border-bi-accent/30' 
      : 'hover:bg-white/5'
  }, [status])

  return (
    <div 
      className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${containerClass}`}
    >
      <div className="council-avatar w-10 h-10 text-sm">
        {name.charAt(0)}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white font-medium text-sm truncate">{name}</p>
        <p className="text-xs text-gray-400">{specialty}</p>
      </div>
      <div className={`w-2 h-2 rounded-full ${statusColor}`}></div>
    </div>
  )
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if id or status changes
  return prevProps.status === nextProps.status
})

// Memoized Message Item component
interface MessageItemProps {
  speaker: string
  text: string
  time: string
  type: string
  sourceTag?: string
  evidenceHint?: string
}

const MessageItem = memo(function MessageItem({ 
  speaker, 
  text, 
  time, 
  type, 
  sourceTag, 
  evidenceHint 
}: MessageItemProps) {
  const containerClass = useMemo(() => {
    if (type === 'president') return 'bg-bi-accent/20 border-r-2 border-bi-accent mr-4'
    if (type === 'response') return 'bg-white/5 ml-4'
    return 'bg-white/5'
  }, [type])

  const speakerClass = useMemo(() => {
    return type === 'president' ? 'text-bi-accent' : 'text-gray-300'
  }, [type])

  return (
    <div 
      className={`p-3 rounded-lg ${containerClass}`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-sm font-medium ${speakerClass}`}>
          {speaker}
        </span>
        <span className="text-xs text-gray-500">{time}</span>
        {sourceTag && type === 'response' && (
          <span className="text-[10px] px-2 py-0.5 rounded border border-white/10 text-gray-400">{sourceTag}</span>
        )}
      </div>
      <p className="text-white">{text}</p>
      {evidenceHint && type === 'response' && (
        <p className="text-[10px] text-gray-500 mt-1">evidence: {evidenceHint}</p>
      )}
    </div>
  )
}, (prevProps, nextProps) => {
  // Deep comparison for message items
  return (
    prevProps.speaker === nextProps.speaker &&
    prevProps.text === nextProps.text &&
    prevProps.type === nextProps.type
  )
})

// Memoized Alert Level Button component
interface AlertLevelButtonProps {
  level: AlertLevel
  label: string
  desc: string
  color: string
  icon: typeof Info
  isSelected: boolean
  onClick: (level: AlertLevel) => void
}

const AlertLevelButton = memo(function AlertLevelButton({ 
  level, 
  label, 
  color, 
  icon: Icon, 
  isSelected, 
  onClick 
}: AlertLevelButtonProps) {
  const handleClick = useCallback(() => {
    onClick(level)
  }, [level, onClick])

  return (
    <button
      onClick={handleClick}
      className={`p-2 rounded-lg text-xs text-center transition-all ${
        isSelected ? color : 'bg-white/5 text-gray-400'
      }`}
    >
      <Icon className="w-4 h-4 mx-auto mb-1" />
      {label}
    </button>
  )
}, (prevProps, nextProps) => {
  return prevProps.level === nextProps.level && prevProps.isSelected === nextProps.isSelected
})

// Memoized Header component
const CouncilHeader = memo(function CouncilHeader() {
  return (
    <div className="glass-panel p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-bi-primary to-bi-secondary flex items-center justify-center border-2 border-bi-gold">
            <Users className="w-7 h-7 text-bi-gold" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">مجلس الحكماء</h1>
            <p className="text-gray-400">16 حكيم في اجتماع مستمر 24/7</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
          <span className="text-green-400">الاجتماع مستمر</span>
        </div>
      </div>
    </div>
  )
})

// Memoized Veto Panel component
const VetoPanel = memo(function VetoPanel() {
  return (
    <div className="glass-panel p-4 border-red-500/30">
      <h2 className="text-lg font-semibold mb-4 text-red-400 flex items-center gap-2">
        <Shield className="w-5 h-5" />
        قوة الفيتو
      </h2>
      <p className="text-sm text-gray-400 mb-4">
        لديك صلاحية إيقاف أي قرار، بما في ذلك قرارات التدمير الذاتي.
      </p>
      <button className="w-full btn-alert alert-red">
        تفعيل الفيتو على آخر قرار
      </button>
    </div>
  )
})

function Council() {
  const [command, setCommand] = useState('')
  const [selectedAlert, setSelectedAlert] = useState<AlertLevel>('GREEN')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isGroupDiscussion, setIsGroupDiscussion] = useState(false)
  const [discussions, setDiscussions] = useState<DiscussionItem[]>(() => {
    try {
      const raw = localStorage.getItem(DISCUSSIONS_STORAGE_KEY)
      if (!raw) return DEFAULT_DISCUSSIONS
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) && parsed.length > 0 ? parsed : DEFAULT_DISCUSSIONS
    } catch {
      return DEFAULT_DISCUSSIONS
    }
  })

  // Memoized alert selection handler
  const handleAlertSelect = useCallback((level: AlertLevel) => {
    setSelectedAlert(level)
  }, [])

  // Load history on mount
  useEffect(() => {
    fetch(apiUrl('/api/v1/council/history'))
      .then(res => res.json())
      .then(data => {
        if (data.history && data.history.length > 0) {
          const historyMsgs = data.history.map((msg: any, idx: number) => ({
            id: idx + 1,
            speaker: msg.role === 'user' ? 'الرئيس' : (msg.council_member || 'حكيم القرار'),
            text: msg.message,
            time: new Date(msg.timestamp).toLocaleTimeString('ar-SA'),
            type: msg.role === 'user' ? 'president' : 'response',
            sourceTag: msg.role === 'council' ? (msg.response_source || msg.source || undefined) : undefined,
            evidenceHint: msg.role === 'council' && Array.isArray(msg.evidence) && msg.evidence.length > 0
              ? `${msg.evidence[0].topic || 'training'} - ${msg.evidence[0].source || 'dataset'}`
              : undefined
          }))
          setDiscussions(historyMsgs)
        }
      })
      .catch(() => {
        // لا شيء إذا فشل API
      })
  }, [])

  // Persist discussions
  useEffect(() => {
    localStorage.setItem(DISCUSSIONS_STORAGE_KEY, JSON.stringify(discussions))
  }, [discussions])

  // Memoized send command handler
  const handleSendCommand = useCallback(async () => {
    if (!command.trim()) return
    
    setIsSubmitting(true)
    
    // إضافة أمر الرئيس للنقاش
    setDiscussions(prev => [...prev, {
      id: Date.now(),
      speaker: 'الرئيس',
      text: command,
      time: 'الآن',
      type: 'president'
    }])

    // استدعاء API الحقيقي
    try {
      const response = await fetch(apiUrl('/api/v1/council/message'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: command,
          user_id: 'president',
          alert_level: selectedAlert
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        
        setDiscussions(prev => [...prev, {
          id: Date.now() + 1,
          speaker: data.council_member || 'حكيم القرار',
          text: data.response,
          time: 'الآن',
          type: 'response',
          sourceTag: data.response_source || data.source || 'unknown',
          evidenceHint: Array.isArray(data.evidence) && data.evidence.length > 0
            ? `${data.evidence[0].topic || 'training'} - ${data.evidence[0].source || 'dataset'}`
            : (data.needs_topic_specificity ? 'needs-topic-specificity' : undefined)
        }])
      } else {
        throw new Error('API Error')
      }
    } catch (error) {
      // fallback إذا الـ API ما اشتغل
      setDiscussions(prev => [...prev, {
        id: Date.now() + 1,
        speaker: 'حكيم القرار',
        text: 'نفهم طلبك وسنعمل على تحليله.',
        time: 'الآن',
        type: 'response'
      }])
    }
    
    setCommand('')
    setIsSubmitting(false)
  }, [command, selectedAlert])

  // Memoized group discussion handler
  const handleGroupDiscussion = useCallback(async () => {
    if (!command.trim()) return
    
    setIsGroupDiscussion(true)
    setIsSubmitting(true)
    
    // إضافة موضوع النقاش
    setDiscussions(prev => [...prev, {
      id: Date.now(),
      speaker: 'الرئيس',
      text: `موضوع للنقاش: ${command}`,
      time: 'الآن',
      type: 'president'
    }])

    // استدعاء API النقاش الجماعي
    try {
      const response = await fetch(apiUrl('/api/v1/council/discuss'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          topic: command
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        
        // إضافة ردود الحكماء واحد تلو الآخر
        if (data.discussion && data.discussion.length > 0) {
          data.discussion.forEach((item: any, index: number) => {
            setTimeout(() => {
              setDiscussions(prev => [...prev, {
                id: Date.now() + index + 1,
                speaker: item.wise_man,
                text: item.response,
                time: 'الآن',
                type: 'response',
                sourceTag: item.response_source || 'unknown',
                evidenceHint: Array.isArray(item.evidence) && item.evidence.length > 0
                  ? `${item.evidence[0].topic || 'training'} - ${item.evidence[0].source || 'dataset'}`
                  : undefined
              }])
            }, index * 1000)
          })

          const filteredOut = Number(data.filtered_out || 0)
          if (filteredOut > 0) {
            setTimeout(() => {
              setDiscussions(prev => [...prev, {
                id: Date.now() + 9000,
                speaker: 'منسّق المجلس',
                text: `تم حجب ${filteredOut} رد/ردود لعدم وجود evidence تدريبي كافٍ.`,
                time: 'الآن',
                type: 'system',
                sourceTag: 'evidence-filter'
              }])
            }, data.discussion.length * 1000)
          }
        }
      }
    } catch (error) {
      console.error('Discussion error:', error)
    }
    
    setTimeout(() => {
      setCommand('')
      setIsSubmitting(false)
      setIsGroupDiscussion(false)
    }, 6000)
  }, [command])

  // Memoized command input handler
  const handleCommandChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setCommand(e.target.value)
  }, [])

  // Memoized key press handler
  const handleKeyPress = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSendCommand()
    }
  }, [handleSendCommand])

  // Memoized wise men list
  const wiseMenList = useMemo(() => {
    return WISE_MEN.map((man) => (
      <WiseManCard
        key={man.id}
        id={man.id}
        name={man.name}
        role={man.role}
        specialty={man.specialty}
        status={man.status}
      />
    ))
  }, [])

  // Memoized discussions list
  const discussionsList = useMemo(() => {
    return discussions.map((msg) => (
      <MessageItem
        key={msg.id}
        speaker={msg.speaker}
        text={msg.text}
        time={msg.time}
        type={msg.type}
        sourceTag={msg.sourceTag}
        evidenceHint={msg.evidenceHint}
      />
    ))
  }, [discussions])

  // Memoized alert level buttons
  const alertLevelButtons = useMemo(() => {
    return ALERT_LEVELS.map((level) => (
      <AlertLevelButton
        key={level.level}
        level={level.level}
        label={level.label}
        desc={level.desc}
        color={level.color}
        icon={level.icon}
        isSelected={selectedAlert === level.level}
        onClick={handleAlertSelect}
      />
    ))
  }, [selectedAlert, handleAlertSelect])

  // Memoized current alert description
  const currentAlertDesc = useMemo(() => {
    return ALERT_LEVELS.find(a => a.level === selectedAlert)?.desc
  }, [selectedAlert])

  return (
    <div className="space-y-6">
      <CouncilHeader />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* قائمة الحكماء */}
        <div className="lg:col-span-1 space-y-4">
          <div className="glass-panel p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Users className="w-5 h-5 text-bi-accent" />
              الحكماء الـ16
            </h2>
            <div className="space-y-2 max-h-96 overflow-y-auto scrollbar-thin">
              {wiseMenList}
            </div>
          </div>

          <LiveMetricsPanel title="قياس حي" compact showTopWise refreshMs={3000} />

          <VetoPanel />
        </div>

        {/* منطقة التفاعل */}
        <div className="lg:col-span-2 space-y-4">
          {/* سجل النقاش */}
          <div className="glass-panel p-4 h-96 flex flex-col">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-bi-accent" />
              سجل المجلس
            </h2>
            <div className="flex-1 overflow-y-auto scrollbar-thin space-y-3">
              {discussionsList}
            </div>
          </div>

          {/* إرسال أمر */}
          <div className="glass-panel p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">أمر جديد للمجلس</h3>
            
            {/* مستوى التنبيه */}
            <div className="grid grid-cols-5 gap-2 mb-4">
              {alertLevelButtons}
            </div>

            {/* حقل الأمر */}
            <div className="flex gap-2">
              <input
                type="text"
                value={command}
                onChange={handleCommandChange}
                placeholder="أدخل أمرك للمجلس..."
                className="input-field flex-1"
                onKeyPress={handleKeyPress}
              />
              <button
                onClick={handleSendCommand}
                disabled={isSubmitting || !command.trim()}
                className="btn-primary flex items-center gap-2 disabled:opacity-50"
              >
                {isSubmitting && !isGroupDiscussion ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <Send className="w-4 h-4" />
                )}
                إرسال
              </button>
              <button
                onClick={handleGroupDiscussion}
                disabled={isSubmitting || !command.trim()}
                className="btn-secondary flex items-center gap-2 disabled:opacity-50 bg-purple-600 hover:bg-purple-700"
                title="نقاش جماعي - 5 حكماء يتكلمون"
              >
                {isGroupDiscussion ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <UsersRound className="w-4 h-4" />
                )}
                نقاش جماعي
              </button>
            </div>
            <p className="mt-2 text-xs text-gray-500">
              {currentAlertDesc}
              {isGroupDiscussion && (
                <span className="text-purple-400 mr-2">| يتحدث 5 حكماء...</span>
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default memo(Council)
