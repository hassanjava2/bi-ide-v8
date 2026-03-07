import { useState, useEffect, useCallback } from 'react'
import { Search, TreePine, Users, Cpu, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react'

// Types
interface CapsuleStatus {
    registry_capsules: number
    tree_nodes: number
    tree_trees: number
    tree_linked: number
    tree_orphans: number
    categories: Record<string, number>
    layer_connections: Record<string, number>
}

interface SearchResult {
    capsule_id: string
    name_ar: string
    keywords: string[]
    score?: number
    level?: number
    source: string
}

interface SageInfo {
    sage: string
    capsule_count: number
    capsules: string[]
}

// Category colors
const CATEGORY_COLORS: Record<string, string> = {
    software: 'from-blue-500 to-blue-700',
    brain: 'from-purple-500 to-purple-700',
    hacking: 'from-red-500 to-red-700',
    engineering: 'from-amber-500 to-amber-700',
    science: 'from-cyan-500 to-cyan-700',
    manufacturing: 'from-orange-500 to-orange-700',
    medicine: 'from-green-500 to-green-700',
    agriculture: 'from-lime-500 to-lime-700',
    crafts: 'from-yellow-500 to-yellow-700',
    energy: 'from-emerald-500 to-emerald-700',
    water: 'from-sky-500 to-sky-700',
    communication: 'from-indigo-500 to-indigo-700',
    transport: 'from-slate-500 to-slate-700',
    business: 'from-teal-500 to-teal-700',
    governance: 'from-rose-500 to-rose-700',
    society: 'from-pink-500 to-pink-700',
    survival: 'from-stone-500 to-stone-700',
    military: 'from-zinc-600 to-zinc-800',
    wisdom: 'from-violet-500 to-violet-700',
    computing: 'from-fuchsia-500 to-fuchsia-700',
    vision: 'from-blue-400 to-indigo-600',
    robotics: 'from-gray-500 to-gray-700',
    space: 'from-purple-400 to-blue-600',
    marine: 'from-cyan-400 to-blue-600',
}

// Category Arabic names
const CATEGORY_NAMES: Record<string, string> = {
    software: 'برمجيات', brain: 'أدمغة', hacking: 'اختراق',
    engineering: 'هندسة', science: 'علوم', manufacturing: 'تصنيع',
    medicine: 'طب', agriculture: 'زراعة', crafts: 'حرف',
    energy: 'طاقة', water: 'مياه', communication: 'اتصالات',
    transport: 'نقل', business: 'أعمال', governance: 'حوكمة',
    society: 'مجتمع', survival: 'بقاء', military: 'عسكري',
    wisdom: 'حكمة', computing: 'حوسبة', vision: 'رؤية حاسوبية',
    robotics: 'روبوتات', space: 'فضاء', marine: 'بحري',
    advanced: 'متقدم', knowledge: 'معرفة', food_security: 'أمن غذائي',
    other: 'أخرى',
}

// Sage info
const SAGES = [
    { id: 'Ibn Sina', title: 'الطبيب', color: '#4CAF50' },
    { id: 'Al-Khwarizmi', title: 'عالم الرياضيات', color: '#2196F3' },
    { id: 'Ibn Rushd', title: 'الفيلسوف', color: '#9C27B0' },
    { id: 'Al-Haytham', title: 'عالم البصريات', color: '#FF9800' },
    { id: 'Al-Biruni', title: 'الموسوعي', color: '#00BCD4' },
    { id: 'Maimonides', title: 'الحكيم', color: '#795548' },
    { id: 'Al-Farabi', title: 'المعلم الثاني', color: '#E91E63' },
    { id: 'Ibn Khaldun', title: 'المؤرخ', color: '#607D8B' },
    { id: 'Al-Razi', title: 'التجريبي', color: '#8BC34A' },
    { id: 'Al-Kindi', title: 'فيلسوف العرب', color: '#3F51B5' },
    { id: 'Al-Tusi', title: 'الفلكي', color: '#009688' },
    { id: 'Ibn Battuta', title: 'الرحالة', color: '#FFC107' },
    { id: 'Al-Jazari', title: 'المهندس', color: '#FF5722' },
    { id: 'Fatima Al-Fihri', title: 'المؤسسة', color: '#673AB7' },
    { id: 'Al-Masudi', title: 'المؤرخ', color: '#795548' },
    { id: 'Al-Khazini', title: 'الميكانيكي', color: '#9E9E9E' },
]

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Capsules() {
    const [status, setStatus] = useState<CapsuleStatus | null>(null)
    const [searchQuery, setSearchQuery] = useState('')
    const [searchResults, setSearchResults] = useState<SearchResult[]>([])
    const [selectedSage, setSelectedSage] = useState<SageInfo | null>(null)
    const [isLoading, setIsLoading] = useState(true)
    const [isSyncing, setIsSyncing] = useState(false)
    const [expandedCategory, setExpandedCategory] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    // Fetch status
    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/capsules/status`)
            if (res.ok) {
                const data = await res.json()
                setStatus(data)
                setError(null)
            }
        } catch {
            setError('غير متصل بالـ API')
            // Use mock data for development
            setStatus({
                registry_capsules: 498,
                tree_nodes: 592,
                tree_trees: 6,
                tree_linked: 284,
                tree_orphans: 308,
                categories: {
                    software: 55, brain: 46, hacking: 38, engineering: 51,
                    science: 46, manufacturing: 33, medicine: 20, agriculture: 20,
                    crafts: 17, energy: 18, water: 9, communication: 7,
                    transport: 5, business: 13, governance: 7, society: 9,
                    survival: 13, military: 10, wisdom: 15, computing: 16,
                    vision: 28, robotics: 4, space: 3, marine: 3,
                    advanced: 3, knowledge: 3, food_security: 2, other: 4,
                },
                layer_connections: { hierarchy: 16, real_life: 25 },
            })
        } finally {
            setIsLoading(false)
        }
    }, [])

    useEffect(() => { fetchStatus() }, [fetchStatus])

    // Search
    const handleSearch = async (q: string) => {
        setSearchQuery(q)
        if (q.length < 2) {
            setSearchResults([])
            return
        }
        try {
            const res = await fetch(`${API_BASE}/api/v1/capsules/search?q=${encodeURIComponent(q)}&top_k=10`)
            if (res.ok) {
                const data = await res.json()
                setSearchResults(data.results || [])
            }
        } catch {
            setSearchResults([])
        }
    }

    // Sync
    const handleSync = async () => {
        setIsSyncing(true)
        try {
            await fetch(`${API_BASE}/api/v1/capsules/sync`, { method: 'POST' })
            await fetchStatus()
        } catch { /* noop */ }
        setIsSyncing(false)
    }

    // Fetch sage capsules
    const fetchSageCapsules = async (sageName: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/capsules/sage/${encodeURIComponent(sageName)}`)
            if (res.ok) {
                const data = await res.json()
                setSelectedSage(data)
            }
        } catch {
            setSelectedSage({
                sage: sageName,
                capsule_count: 0,
                capsules: [],
            })
        }
    }

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-full">
                <div className="w-12 h-12 border-4 border-bi-accent border-t-transparent rounded-full animate-spin" />
            </div>
        )
    }

    const sortedCategories = status
        ? Object.entries(status.categories).sort((a, b) => b[1] - a[1])
        : []

    return (
        <div className="space-y-6 p-6" dir="rtl">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                        <TreePine className="w-8 h-8 text-bi-accent" />
                        شجرة الكبسولات
                    </h1>
                    <p className="text-gray-400 mt-1">498 كبسولة معرفية مربوطة بنظام وراثة هرمي</p>
                </div>
                <button
                    onClick={handleSync}
                    disabled={isSyncing}
                    className="px-4 py-2 bg-bi-accent/20 text-bi-accent rounded-lg hover:bg-bi-accent/30 
                     transition-all flex items-center gap-2 disabled:opacity-50"
                >
                    <RefreshCw className={`w-4 h-4 ${isSyncing ? 'animate-spin' : ''}`} />
                    مزامنة
                </button>
            </div>

            {error && (
                <div className="px-4 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-sm">
                    ⚠️ {error} — عرض بيانات محلية
                </div>
            )}

            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                    { label: 'كبسولات', value: status?.registry_capsules || 0, icon: '🧩', color: 'from-blue-500/20 to-blue-600/10' },
                    { label: 'عقد الشجرة', value: status?.tree_nodes || 0, icon: '🌳', color: 'from-green-500/20 to-green-600/10' },
                    { label: 'حكماء مربوطين', value: status?.layer_connections?.hierarchy || 0, icon: '🏛️', color: 'from-purple-500/20 to-purple-600/10' },
                    { label: 'وكلاء واقعيين', value: status?.layer_connections?.real_life || 0, icon: '🌍', color: 'from-amber-500/20 to-amber-600/10' },
                ].map((stat) => (
                    <div key={stat.label} className={`glass-card p-4 bg-gradient-to-br ${stat.color} border border-white/5`}>
                        <div className="text-2xl mb-1">{stat.icon}</div>
                        <div className="text-2xl font-bold text-white">{stat.value}</div>
                        <div className="text-sm text-gray-400">{stat.label}</div>
                    </div>
                ))}
            </div>

            {/* Search */}
            <div className="glass-card p-4">
                <div className="relative">
                    <Search className="absolute right-3 top-3 w-5 h-5 text-gray-400" />
                    <input
                        type="text"
                        placeholder="ابحث عن كبسولة... (مثال: python, cement, surgery)"
                        value={searchQuery}
                        onChange={(e) => handleSearch(e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg py-2.5 pr-10 pl-4 
                       text-white placeholder-gray-500 focus:border-bi-accent/50 focus:outline-none"
                    />
                </div>
                {searchResults.length > 0 && (
                    <div className="mt-3 space-y-2">
                        {searchResults.map((r) => (
                            <div key={r.capsule_id} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                                <div>
                                    <span className="text-white font-mono text-sm">{r.capsule_id}</span>
                                    <span className="text-gray-400 mr-2 text-sm">— {r.name_ar}</span>
                                </div>
                                <div className="flex gap-1">
                                    {r.keywords?.slice(0, 3).map((kw) => (
                                        <span key={kw} className="px-2 py-0.5 bg-bi-accent/10 text-bi-accent text-xs rounded">
                                            {kw}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Categories Grid */}
            <div>
                <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-bi-accent" />
                    الفئات ({sortedCategories.length})
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                    {sortedCategories.map(([cat, count]) => (
                        <button
                            key={cat}
                            onClick={() => setExpandedCategory(expandedCategory === cat ? null : cat)}
                            className={`p-3 rounded-lg bg-gradient-to-br ${CATEGORY_COLORS[cat] || 'from-gray-500 to-gray-700'} 
                         bg-opacity-20 hover:scale-105 transition-all text-right relative overflow-hidden group`}
                        >
                            <div className="absolute inset-0 bg-black/40 group-hover:bg-black/30 transition-all" />
                            <div className="relative z-10">
                                <div className="text-2xl font-bold text-white">{count}</div>
                                <div className="text-xs text-white/80">{CATEGORY_NAMES[cat] || cat}</div>
                                <div className="text-[10px] text-white/50 mt-0.5">{cat}</div>
                            </div>
                            {expandedCategory === cat ? (
                                <ChevronDown className="absolute top-2 left-2 w-3 h-3 text-white/50 z-10" />
                            ) : (
                                <ChevronRight className="absolute top-2 left-2 w-3 h-3 text-white/50 z-10" />
                            )}
                        </button>
                    ))}
                </div>
            </div>

            {/* Sages Section */}
            <div>
                <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <Users className="w-5 h-5 text-bi-accent" />
                    الحكماء وكبسولاتهم
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
                    {SAGES.map((sage) => (
                        <button
                            key={sage.id}
                            onClick={() => fetchSageCapsules(sage.id)}
                            className="glass-card p-3 text-center hover:border-bi-accent/30 transition-all group"
                        >
                            <div
                                className="w-10 h-10 rounded-full mx-auto mb-2 flex items-center justify-center text-white font-bold text-lg"
                                style={{ backgroundColor: sage.color + '40', border: `2px solid ${sage.color}` }}
                            >
                                {sage.id.charAt(0)}
                            </div>
                            <div className="text-xs text-white font-medium truncate">{sage.id}</div>
                            <div className="text-[10px] text-gray-400">{sage.title}</div>
                        </button>
                    ))}
                </div>

                {/* Selected Sage Details */}
                {selectedSage && (
                    <div className="mt-4 glass-card p-4">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-lg font-bold text-white">{selectedSage.sage}</h3>
                            <span className="px-3 py-1 bg-bi-accent/20 text-bi-accent rounded-full text-sm">
                                {selectedSage.capsule_count} كبسولة
                            </span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {selectedSage.capsules?.slice(0, 30).map((cap) => (
                                <span key={cap} className="px-2 py-1 bg-white/5 text-gray-300 text-xs rounded font-mono">
                                    {cap}
                                </span>
                            ))}
                            {(selectedSage.capsule_count || 0) > 30 && (
                                <span className="px-2 py-1 text-gray-500 text-xs">
                                    +{selectedSage.capsule_count - 30} أخرى
                                </span>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Capsules
