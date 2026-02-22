import { MessageCircle, ShoppingBag, TrendingUp, Heart, MessageSquare, Share2 } from 'lucide-react'

const posts = [
  { id: 1, author: 'أحمد محمد', content: 'ما رأيكم في النظام الجديد؟ أنا معجب جداً بالتصميم!', likes: 24, comments: 8, time: 'منذ ساعة' },
  { id: 2, author: 'سارة علي', content: 'هل جرب أحدكم ميزة AI في الـ ERP؟ نتائج مذهلة!', likes: 45, comments: 12, time: 'منذ ساعتين' },
  { id: 3, author: 'خالد العلي', content: 'نصيحة: استخدموا التدريب المخصص للحصول على أفضل أداء', likes: 18, comments: 5, time: 'منذ 3 ساعات' },
]

const products = [
  { id: 1, name: 'كورس AI متقدم', price: 199, seller: 'BI Academy', rating: 4.9 },
  { id: 2, name: 'قالب ERP احترافي', price: 99, seller: 'Dev Store', rating: 4.7 },
  { id: 3, name: 'استشارة تقنية', price: 150, seller: 'Expert Hub', rating: 5.0 },
]

export default function Community() {
  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-600 to-purple-800 flex items-center justify-center">
            <MessageCircle className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">المجتمع</h1>
            <p className="text-gray-400">تواصل، تعلم، وتسوق</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* المنشورات */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-panel p-4">
            <textarea 
              placeholder="ماذا تريد أن تشارك؟" 
              className="input-field w-full h-24 resize-none"
            ></textarea>
            <div className="flex justify-end mt-3">
              <button className="btn-primary">نشر</button>
            </div>
          </div>

          {posts.map((post) => (
            <div key={post.id} className="glass-panel p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-bi-primary to-bi-secondary flex items-center justify-center text-white font-bold">
                  {post.author.charAt(0)}
                </div>
                <div>
                  <p className="text-white font-medium">{post.author}</p>
                  <p className="text-xs text-gray-400">{post.time}</p>
                </div>
              </div>
              <p className="text-gray-200 mb-4">{post.content}</p>
              <div className="flex items-center gap-6 text-gray-400">
                <button className="flex items-center gap-2 hover:text-red-400 transition-colors">
                  <Heart className="w-5 h-5" />
                  {post.likes}
                </button>
                <button className="flex items-center gap-2 hover:text-bi-accent transition-colors">
                  <MessageSquare className="w-5 h-5" />
                  {post.comments}
                </button>
                <button className="flex items-center gap-2 hover:text-green-400 transition-colors">
                  <Share2 className="w-5 h-5" />
                  مشاركة
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* السوق */}
        <div className="space-y-4">
          <div className="glass-panel p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <ShoppingBag className="w-5 h-5 text-bi-accent" />
              السوق
            </h2>
            <div className="space-y-3">
              {products.map((product) => (
                <div key={product.id} className="glass-card p-3">
                  <h3 className="text-white font-medium">{product.name}</h3>
                  <p className="text-sm text-gray-400">{product.seller}</p>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-bi-gold font-bold">{product.price}$</span>
                    <span className="text-xs text-yellow-400">★ {product.rating}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-bi-accent" />
              الأكثر نشاطاً
            </h2>
            <div className="space-y-2">
              {['أحمد', 'سارة', 'خالد', 'محمد', 'نورة'].map((name, i) => (
                <div key={i} className="flex items-center gap-3 p-2 hover:bg-white/5 rounded-lg">
                  <span className="text-bi-gold font-bold">#{i + 1}</span>
                  <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center text-sm">
                    {name.charAt(0)}
                  </div>
                  <span className="text-white">{name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
