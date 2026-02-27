import { useState } from 'react'
import { 
  MessageSquare, 
  ThumbsUp, 
  Eye, 
  Clock,
  Pin,
  Lock,
  Plus,
  Search,
  Filter
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface Topic {
  id: string
  title: string
  author: string
  avatar: string
  category: string
  replies: number
  views: number
  likes: number
  lastReply: string
  isPinned?: boolean
  isLocked?: boolean
  tags: string[]
}

const CATEGORIES = [
  { id: 'all', name: 'الكل', icon: MessageSquare },
  { id: 'general', name: 'عام', icon: MessageSquare },
  { id: 'ai', name: 'الذكاء الاصطناعي', icon: MessageSquare },
  { id: 'programming', name: 'البرمجة', icon: MessageSquare },
  { id: 'business', name: 'الأعمال', icon: MessageSquare },
  { id: 'support', name: 'الدعم الفني', icon: MessageSquare },
]

const TOPICS: Topic[] = [
  {
    id: '1',
    title: 'مرحباً بكم في مجتمع BI-IDE',
    author: 'admin',
    avatar: '/avatars/admin.png',
    category: 'general',
    replies: 42,
    views: 1250,
    likes: 89,
    lastReply: 'منذ ساعة',
    isPinned: true,
    tags: ['welcome', 'announcement']
  },
  {
    id: '2',
    title: 'كيفية استخدام النظام الهرمي للذكاء الاصطناعي',
    author: 'ai_expert',
    avatar: '/avatars/ai.png',
    category: 'ai',
    replies: 23,
    views: 567,
    likes: 45,
    lastReply: 'منذ 3 ساعات',
    tags: ['tutorial', 'ai']
  },
  {
    id: '3',
    title: 'مشكلة في ربط RTX 4090',
    author: 'developer1',
    avatar: '/avatars/dev.png',
    category: 'support',
    replies: 8,
    views: 234,
    likes: 12,
    lastReply: 'منذ 5 ساعات',
    tags: ['help', 'rtx4090']
  },
]

export default function ForumsPage() {
  const [activeCategory, setActiveCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')

  const filteredTopics = TOPICS.filter(topic => {
    const matchesCategory = activeCategory === 'all' || topic.category === activeCategory
    const matchesSearch = topic.title.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">المنتديات</h1>
          <p className="text-gray-400">نقاشات، أسئلة، ومشاركة المعرفة</p>
        </div>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          موضوع جديد
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="space-y-4">
          {/* Search */}
          <Card>
            <CardContent className="p-4">
              <div className="relative">
                <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="البحث في المنتدى..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pr-10 pl-4 py-2 bg-white/5 border border-white/10 rounded-lg text-right"
                />
              </div>
            </CardContent>
          </Card>

          {/* Categories */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">الفئات</CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <div className="space-y-1">
                {CATEGORIES.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setActiveCategory(cat.id)}
                    className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                      activeCategory === cat.id
                        ? 'bg-bi-accent text-white'
                        : 'hover:bg-white/5'
                    }`}
                  >
                    <cat.icon className="w-4 h-4" />
                    <span>{cat.name}</span>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Stats */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">إحصائيات</CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">المواضيع</span>
                <span className="font-bold">8,500</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">المشاركات</span>
                <span className="font-bold">125,000</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">المستخدمون</span>
                <span className="font-bold">15,000</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Topics List */}
        <div className="lg:col-span-3 space-y-4">
          {/* Filters */}
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Filter className="w-4 h-4 mr-2" />
              الأحدث
            </Button>
            <Button variant="ghost" size="sm">الأكثر زيارة</Button>
            <Button variant="ghost" size="sm">الأكثر ردود</Button>
          </div>

          {/* Topics */}
          {filteredTopics.map((topic) => (
            <Card key={topic.id} className="hover:border-bi-accent/50 transition-colors">
              <CardContent className="p-4">
                <div className="flex items-start gap-4">
                  {/* Avatar */}
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-bi-accent to-purple-500 flex items-center justify-center text-sm font-bold">
                    {topic.author[0].toUpperCase()}
                  </div>

                  {/* Content */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      {topic.isPinned && (
                        <Pin className="w-4 h-4 text-bi-accent" />
                      )}
                      {topic.isLocked && (
                        <Lock className="w-4 h-4 text-gray-400" />
                      )}
                      <h3 className="font-semibold hover:text-bi-accent cursor-pointer">
                        {topic.title}
                      </h3>
                    </div>

                    <div className="flex items-center gap-4 text-sm text-gray-400">
                      <span>@{topic.author}</span>
                      <span>•</span>
                      <span className="capitalize">{topic.category}</span>
                    </div>

                    {/* Tags */}
                    <div className="flex gap-2 mt-2">
                      {topic.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 text-xs bg-white/5 rounded-full"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <div className="flex items-center gap-1">
                      <ThumbsUp className="w-4 h-4" />
                      <span>{topic.likes}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MessageSquare className="w-4 h-4" />
                      <span>{topic.replies}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      <span>{topic.views}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs">
                      <Clock className="w-3 h-3" />
                      <span>{topic.lastReply}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
