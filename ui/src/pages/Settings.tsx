import { Settings as SettingsIcon, User, Shield, Bell, Database, Server } from 'lucide-react'

export default function Settings() {
  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-gray-600 to-gray-800 flex items-center justify-center">
            <SettingsIcon className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">الإعدادات</h1>
            <p className="text-gray-400">تخصيص النظام</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-6">
            <User className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">الملف الشخصي</h2>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">الاسم</label>
              <input type="text" defaultValue="الرئيس" className="input-field w-full" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">البريد الإلكتروني</label>
              <input type="email" defaultValue="president@bi-ide.com" className="input-field w-full" />
            </div>
            <button className="btn-primary">حفظ التغييرات</button>
          </div>
        </div>

        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">الأمان</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <p className="text-white">المصادقة الثنائية</p>
                <p className="text-xs text-gray-400">تأمين إضافي للحساب</p>
              </div>
              <div className="w-12 h-6 bg-green-500 rounded-full relative cursor-pointer">
                <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full"></div>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <p className="text-white">تأكيد الفيتو</p>
                <p className="text-xs text-gray-400">طلب تأكيد إضافي للقرائر الحرجة</p>
              </div>
              <div className="w-12 h-6 bg-green-500 rounded-full relative cursor-pointer">
                <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full"></div>
              </div>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-6">
            <Bell className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">الإشعارات</h2>
          </div>
          <div className="space-y-3">
            {[
              'إشعارات المجلس',
              'تقارير الكشافة',
              'تحديثات التدريب',
              'تنبيهات الأمان',
            ].map((item, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-gray-300">{item}</span>
                <div className="w-10 h-5 bg-bi-accent rounded-full relative cursor-pointer">
                  <div className="absolute left-0.5 top-0.5 w-4 h-4 bg-white rounded-full"></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-6">
            <Database className="w-6 h-6 text-bi-accent" />
            <h2 className="text-lg font-semibold">قاعدة البيانات</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <span className="text-gray-300">حجم البيانات</span>
              <span className="text-white">2.4 GB</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <span className="text-gray-300">آخر نسخة احتياطية</span>
              <span className="text-white">منذ 2 ساعة</span>
            </div>
            <button className="btn-secondary w-full">نسخ احتياطي يدوي</button>
          </div>
        </div>
      </div>

      <div className="glass-panel p-6">
        <div className="flex items-center gap-3 mb-6">
          <Server className="w-6 h-6 text-bi-accent" />
          <h2 className="text-lg font-semibold">معلومات النظام</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'الإصدار', value: 'v2.0.0' },
            { label: 'البيئة', value: 'Production' },
            { label: 'المخدم', value: '8x H200' },
            { label: 'المنطقة', value: 'US-East' },
          ].map((item, i) => (
            <div key={i} className="glass-card p-4 text-center">
              <p className="text-sm text-gray-400">{item.label}</p>
              <p className="text-lg font-bold text-white">{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
