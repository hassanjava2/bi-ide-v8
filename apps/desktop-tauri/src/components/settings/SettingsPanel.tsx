/**
 * لوحة الإعدادات - Settings Panel
 * إعدادات عامة، المظهر، اختصارات لوحة المفاتيح، الاتصالات
 */

import { useState } from "react";
import { 
  Settings, 
  Palette, 
  Keyboard, 
  Plug, 
  Bell,
  Download,
  Upload,
  Moon,
  Sun,
  Monitor,
  Globe,
  Database,
  Shield,
  Check,
  ChevronRight,
  ExternalLink
} from "lucide-react";

// أنواع التبويبات
type TabId = "general" | "appearance" | "shortcuts" | "api" | "notifications";

// نوع الإعداد
interface Setting {
  id: string;
  label: string;
  description?: string;
  type: "toggle" | "select" | "text" | "number" | "button";
  value: boolean | string | number;
  options?: { value: string; label: string }[];
  icon?: React.ReactNode;
}

// مكون التبديل
function Toggle({ checked, onChange }: { checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative w-11 h-6 rounded-full transition-colors ${
        checked ? "bg-primary-600" : "bg-dark-600"
      }`}
    >
      <span
        className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
          checked ? "translate-x-5" : "translate-x-0"
        }`}
      />
    </button>
  );
}

// مكون حقل الإدخال
function InputField({ 
  label, 
  value, 
  onChange, 
  type = "text",
  placeholder 
}: { 
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  placeholder?: string;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-dark-200">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full px-3 py-2 bg-dark-900 border border-dark-700 rounded-lg text-dark-100 placeholder-dark-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
      />
    </div>
  );
}

// مكون القائمة المنسدلة
function SelectField({ 
  label, 
  value, 
  onChange, 
  options 
}: { 
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-dark-200">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 bg-dark-900 border border-dark-700 rounded-lg text-dark-100 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 appearance-none cursor-pointer"
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

// مكون مجموعة الإعدادات
function SettingGroup({ 
  title, 
  description, 
  children 
}: { 
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-dark-800 rounded-xl p-5 space-y-4">
      <div>
        <h3 className="font-semibold text-dark-100">{title}</h3>
        {description && (
          <p className="text-sm text-dark-400 mt-1">{description}</p>
        )}
      </div>
      <div className="space-y-4">
        {children}
      </div>
    </div>
  );
}

// مكون عنصر الإعداد
function SettingItem({ 
  label, 
  description, 
  children,
  icon
}: { 
  label: string;
  description?: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-3">
        {icon && (
          <div className="p-2 bg-dark-900 rounded-lg text-dark-400">
            {icon}
          </div>
        )}
        <div>
          <div className="font-medium text-dark-200">{label}</div>
          {description && (
            <div className="text-sm text-dark-500">{description}</div>
          )}
        </div>
      </div>
      <div>{children}</div>
    </div>
  );
}

// مكون اختصار لوحة المفاتيح
function ShortcutKey({ keys }: { keys: string[] }) {
  return (
    <div className="flex items-center gap-1">
      {keys.map((key, i) => (
        <span key={i} className="flex items-center">
          <kbd className="px-2 py-1 bg-dark-900 border border-dark-600 rounded text-sm font-mono text-dark-300">
            {key}
          </kbd>
          {i < keys.length - 1 && <span className="mx-1 text-dark-500">+</span>}
        </span>
      ))}
    </div>
  );
}

// المكون الرئيسي
export function SettingsPanel() {
  const [activeTab, setActiveTab] = useState<TabId>("general");

  // إعدادات عامة
  const [generalSettings, setGeneralSettings] = useState({
    autoSave: true,
    autoSaveInterval: 5,
    wordWrap: true,
    lineNumbers: true,
    minimap: true,
    fontSize: 14,
    fontFamily: "JetBrains Mono",
    tabSize: 2,
    language: "ar",
  });

  // إعدادات المظهر
  const [appearanceSettings, setAppearanceSettings] = useState({
    theme: "dark",
    accentColor: "blue",
    sidebarPosition: "left",
    showStatusBar: true,
    compactMode: false,
    animationsEnabled: true,
  });

  // إعدادات API
  const [apiSettings, setApiSettings] = useState({
    openaiKey: "",
    anthropicKey: "",
    localEndpoint: "http://localhost:8000",
    timeout: 30,
    enableStreaming: true,
  });

  // إعدادات الإشعارات
  const [notificationSettings, setNotificationSettings] = useState({
    enableNotifications: true,
    soundEnabled: true,
    showToasts: true,
    trainingAlerts: true,
    errorAlerts: true,
    updateAlerts: true,
  });

  // اختصارات لوحة المفاتيح
  const shortcuts = [
    { action: "حفظ الملف", keys: ["Ctrl", "S"] },
    { action: "فتح الملف", keys: ["Ctrl", "O"] },
    { action: "إغلاق الملف", keys: ["Ctrl", "W"] },
    { action: "البحث", keys: ["Ctrl", "F"] },
    { action: "البحث والاستبدال", keys: ["Ctrl", "H"] },
    { action: "الأوامر", keys: ["Ctrl", "Shift", "P"] },
    { action: "Palette", keys: ["Ctrl", "P"] },
    { action: "Terminal جديد", keys: ["Ctrl", "`"] },
    { action: "تعليق السطر", keys: ["Ctrl", "/"] },
    { action: "نسخ السطر", keys: ["Ctrl", "D"] },
    { action: "نقل السطر للأعلى", keys: ["Alt", "↑"] },
    { action: "نقل السطر للأسفل", keys: ["Alt", "↓"] },
    { action: "تنسيق الكود", keys: ["Shift", "Alt", "F"] },
    { action: "AI مساعدة", keys: ["Ctrl", "I"] },
  ];

  // تبويبات التنقل
  const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
    { id: "general", label: "عام", icon: Settings },
    { id: "appearance", label: "المظهر", icon: Palette },
    { id: "shortcuts", label: "اختصارات", icon: Keyboard },
    { id: "api", label: "API", icon: Plug },
    { id: "notifications", label: "إشعارات", icon: Bell },
  ];

  // تصدير الإعدادات
  const handleExport = () => {
    const settings = {
      general: generalSettings,
      appearance: appearanceSettings,
      api: apiSettings,
      notifications: notificationSettings,
    };
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bi-ide-settings.json";
    a.click();
  };

  // استيراد الإعدادات
  const handleImport = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const settings = JSON.parse(e.target?.result as string);
            if (settings.general) setGeneralSettings(settings.general);
            if (settings.appearance) setAppearanceSettings(settings.appearance);
            if (settings.api) setApiSettings(settings.api);
            if (settings.notifications) setNotificationSettings(settings.notifications);
          } catch (err) {
            console.error("Failed to import settings:", err);
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  return (
    <div className="h-full flex bg-dark-900">
      {/* شريط جانبي للتنقل */}
      <div className="w-60 border-l border-dark-700 bg-dark-800/50 p-4">
        <h2 className="text-lg font-bold text-dark-100 mb-4">الإعدادات</h2>
        <nav className="space-y-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-primary-600 text-white"
                  : "text-dark-300 hover:bg-dark-700 hover:text-dark-100"
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {activeTab === tab.id && <ChevronRight className="w-4 h-4 mr-auto" />}
            </button>
          ))}
        </nav>

        {/* أزرار التصدير/الاستيراد */}
        <div className="mt-8 pt-4 border-t border-dark-700 space-y-2">
          <button
            onClick={handleExport}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-dark-300 hover:text-dark-100 hover:bg-dark-700 rounded-lg transition-colors"
          >
            <Download className="w-4 h-4" />
            تصدير الإعدادات
          </button>
          <button
            onClick={handleImport}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-dark-300 hover:text-dark-100 hover:bg-dark-700 rounded-lg transition-colors"
          >
            <Upload className="w-4 h-4" />
            استيراد الإعدادات
          </button>
        </div>
      </div>

      {/* المحتوى الرئيسي */}
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === "general" && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h1 className="text-2xl font-bold text-dark-100">إعدادات عامة</h1>
              <p className="text-dark-400 mt-1">إعدادات المحرر والسلوك الأساسي</p>
            </div>

            <SettingGroup title="الحفظ التلقائي" description="إدارة الحفظ التلقائي للملفات">
              <SettingItem 
                label="تفعيل الحفظ التلقائي"
                description="حفظ الملفات تلقائياً أثناء التعديل"
              >
                <Toggle 
                  checked={generalSettings.autoSave} 
                  onChange={(v) => setGeneralSettings(s => ({ ...s, autoSave: v }))}
                />
              </SettingItem>
              {generalSettings.autoSave && (
                <InputField
                  label="فترة الحفظ (ثواني)"
                  value={generalSettings.autoSaveInterval.toString()}
                  onChange={(v) => setGeneralSettings(s => ({ ...s, autoSaveInterval: parseInt(v) || 5 }))}
                  type="number"
                />
              )}
            </SettingGroup>

            <SettingGroup title="المحرر" description="إعدادات عرض الكود">
              <SettingItem label="التفاف النص">
                <Toggle 
                  checked={generalSettings.wordWrap} 
                  onChange={(v) => setGeneralSettings(s => ({ ...s, wordWrap: v }))}
                />
              </SettingItem>
              <SettingItem label="أرقام الأسطر">
                <Toggle 
                  checked={generalSettings.lineNumbers} 
                  onChange={(v) => setGeneralSettings(s => ({ ...s, lineNumbers: v }))}
                />
              </SettingItem>
              <SettingItem label="الخريطة المصغرة">
                <Toggle 
                  checked={generalSettings.minimap} 
                  onChange={(v) => setGeneralSettings(s => ({ ...s, minimap: v }))}
                />
              </SettingItem>
            </SettingGroup>

            <SettingGroup title="المظهر النصي">
              <SelectField
                label="حجم الخط"
                value={generalSettings.fontSize.toString()}
                onChange={(v) => setGeneralSettings(s => ({ ...s, fontSize: parseInt(v) }))}
                options={[
                  { value: "12", label: "12px" },
                  { value: "14", label: "14px" },
                  { value: "16", label: "16px" },
                  { value: "18", label: "18px" },
                  { value: "20", label: "20px" },
                ]}
              />
              <SelectField
                label="حجم المسافة البيضاء"
                value={generalSettings.tabSize.toString()}
                onChange={(v) => setGeneralSettings(s => ({ ...s, tabSize: parseInt(v) }))}
                options={[
                  { value: "2", label: "2 مسافات" },
                  { value: "4", label: "4 مسافات" },
                  { value: "8", label: "8 مسافات" },
                ]}
              />
            </SettingGroup>
          </div>
        )}

        {activeTab === "appearance" && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h1 className="text-2xl font-bold text-dark-100">المظهر</h1>
              <p className="text-dark-400 mt-1">تخصيص شكل ومظهر التطبيق</p>
            </div>

            <SettingGroup title="السمة" description="اختر سمة التطبيق">
              <div className="grid grid-cols-3 gap-3">
                {[
                  { id: "light", label: "فاتح", icon: Sun },
                  { id: "dark", label: "داكن", icon: Moon },
                  { id: "system", label: "النظام", icon: Monitor },
                ].map(theme => (
                  <button
                    key={theme.id}
                    onClick={() => setAppearanceSettings(s => ({ ...s, theme: theme.id }))}
                    className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-colors ${
                      appearanceSettings.theme === theme.id
                        ? "border-primary-500 bg-primary-500/10"
                        : "border-dark-700 hover:border-dark-600"
                    }`}
                  >
                    <theme.icon className={`w-6 h-6 ${
                      appearanceSettings.theme === theme.id ? "text-primary-400" : "text-dark-400"
                    }`} />
                    <span className={`text-sm ${
                      appearanceSettings.theme === theme.id ? "text-primary-400 font-medium" : "text-dark-300"
                    }`}>{theme.label}</span>
                    {appearanceSettings.theme === theme.id && (
                      <Check className="w-4 h-4 text-primary-400" />
                    )}
                  </button>
                ))}
              </div>
            </SettingGroup>

            <SettingGroup title="الألوان" description="لون التمييز الرئيسي">
              <div className="flex gap-3">
                {[
                  { id: "blue", color: "#0ea5e9" },
                  { id: "purple", color: "#8b5cf6" },
                  { id: "green", color: "#22c55e" },
                  { id: "orange", color: "#f97316" },
                  { id: "pink", color: "#ec4899" },
                ].map(color => (
                  <button
                    key={color.id}
                    onClick={() => setAppearanceSettings(s => ({ ...s, accentColor: color.id }))}
                    className={`w-10 h-10 rounded-lg transition-all ${
                      appearanceSettings.accentColor === color.id
                        ? "ring-2 ring-white ring-offset-2 ring-offset-dark-800 scale-110"
                        : "hover:scale-105"
                    }`}
                    style={{ backgroundColor: color.color }}
                  />
                ))}
              </div>
            </SettingGroup>

            <SettingGroup title="الواجهة" description="إعدادات عرض الواجهة">
              <SettingItem label="الوضع المضغوط" description="تقليل المسافات في الواجهة">
                <Toggle 
                  checked={appearanceSettings.compactMode} 
                  onChange={(v) => setAppearanceSettings(s => ({ ...s, compactMode: v }))}
                />
              </SettingItem>
              <SettingItem label="التأثيرات الحركية" description="تفعيل الحركات والانتقالات">
                <Toggle 
                  checked={appearanceSettings.animationsEnabled} 
                  onChange={(v) => setAppearanceSettings(s => ({ ...s, animationsEnabled: v }))}
                />
              </SettingItem>
              <SettingItem label="شريط الحالة" description="إظهار شريط الحالة في الأسفل">
                <Toggle 
                  checked={appearanceSettings.showStatusBar} 
                  onChange={(v) => setAppearanceSettings(s => ({ ...s, showStatusBar: v }))}
                />
              </SettingItem>
            </SettingGroup>
          </div>
        )}

        {activeTab === "shortcuts" && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h1 className="text-2xl font-bold text-dark-100">اختصارات لوحة المفاتيح</h1>
              <p className="text-dark-400 mt-1">اختصارات التنقل والتحكم السريع</p>
            </div>

            <SettingGroup title="اختصارات عامة" description="الاختصارات الأساسية في التطبيق">
              <div className="space-y-3">
                {shortcuts.map((shortcut, idx) => (
                  <div 
                    key={idx}
                    className="flex items-center justify-between py-3 px-4 bg-dark-900 rounded-lg hover:bg-dark-700/50 transition-colors"
                  >
                    <span className="text-dark-200">{shortcut.action}</span>
                    <ShortcutKey keys={shortcut.keys} />
                  </div>
                ))}
              </div>
            </SettingGroup>

            <div className="p-4 bg-dark-800/50 rounded-lg border border-dark-700">
              <p className="text-sm text-dark-400">
                <span className="text-primary-400">ملاحظة:</span> يمكنك تعديل الاختصارات من خلال ملف settings.json
              </p>
            </div>
          </div>
        )}

        {activeTab === "api" && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h1 className="text-2xl font-bold text-dark-100">اتصالات API</h1>
              <p className="text-dark-400 mt-1">إعدادات الاتصال بالخدمات الخارجية</p>
            </div>

            <SettingGroup title="OpenAI" description="إعدادات API الخاص بـ OpenAI">
              <InputField
                label="مفتاح API"
                value={apiSettings.openaiKey}
                onChange={(v) => setApiSettings(s => ({ ...s, openaiKey: v }))}
                type="password"
                placeholder="sk-..."
              />
            </SettingGroup>

            <SettingGroup title="Anthropic" description="إعدادات API الخاص بـ Anthropic Claude">
              <InputField
                label="مفتاح API"
                value={apiSettings.anthropicKey}
                onChange={(v) => setApiSettings(s => ({ ...s, anthropicKey: v }))}
                type="password"
                placeholder="sk-ant-..."
              />
            </SettingGroup>

            <SettingGroup title="نقطة النهاية المحلية" description="للنماذج المحلية">
              <InputField
                label="عنوان URL"
                value={apiSettings.localEndpoint}
                onChange={(v) => setApiSettings(s => ({ ...s, localEndpoint: v }))}
                placeholder="http://localhost:8000"
              />
              <SelectField
                label="مهلة الاتصال (ثواني)"
                value={apiSettings.timeout.toString()}
                onChange={(v) => setApiSettings(s => ({ ...s, timeout: parseInt(v) }))}
                options={[
                  { value: "10", label: "10 ثواني" },
                  { value: "30", label: "30 ثانية" },
                  { value: "60", label: "60 ثانية" },
                  { value: "120", label: "120 ثانية" },
                ]}
              />
              <SettingItem label="تفعيل البث المباشر" description="استلام الردود بشكل متدفق">
                <Toggle 
                  checked={apiSettings.enableStreaming} 
                  onChange={(v) => setApiSettings(s => ({ ...s, enableStreaming: v }))}
                />
              </SettingItem>
            </SettingGroup>

            <div className="flex items-center gap-2 p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
              <Shield className="w-5 h-5 text-yellow-400" />
              <p className="text-sm text-yellow-200">
                يتم تخزين مفاتيح API بشكل آمن محلياً ولا يتم إرسالها إلى أي خادم.
              </p>
            </div>
          </div>
        )}

        {activeTab === "notifications" && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h1 className="text-2xl font-bold text-dark-100">الإشعارات</h1>
              <p className="text-dark-400 mt-1">إدارة الإشعارات والتنبيهات</p>
            </div>

            <SettingGroup title="الإشعارات العامة" description="إعدادات الإشعارات الأساسية">
              <SettingItem label="تفعيل الإشعارات" icon={<Bell className="w-4 h-4" />}>
                <Toggle 
                  checked={notificationSettings.enableNotifications} 
                  onChange={(v) => setNotificationSettings(s => ({ ...s, enableNotifications: v }))}
                />
              </SettingItem>
              <SettingItem label="الصوت" description="تشغيل صوت عند الإشعار" icon={<Database className="w-4 h-4" />}>
                <Toggle 
                  checked={notificationSettings.soundEnabled} 
                  onChange={(v) => setNotificationSettings(s => ({ ...s, soundEnabled: v }))}
                />
              </SettingItem>
              <SettingItem label="الإشعارات المنبثقة" description="إظهار نوافذ منبثقة" icon={<Globe className="w-4 h-4" />}>
                <Toggle 
                  checked={notificationSettings.showToasts} 
                  onChange={(v) => setNotificationSettings(s => ({ ...s, showToasts: v }))}
                />
              </SettingItem>
            </SettingGroup>

            <SettingGroup title="إشعارات التدريب" description="تنبيهات تتعلق بالتدريب والعمال">
              <SettingItem label="تنبيهات التدريب">
                <Toggle 
                  checked={notificationSettings.trainingAlerts} 
                  onChange={(v) => setNotificationSettings(s => ({ ...s, trainingAlerts: v }))}
                />
              </SettingItem>
              <SettingItem label="تنبيهات الأخطاء">
                <Toggle 
                  checked={notificationSettings.errorAlerts} 
                  onChange={(v) => setNotificationSettings(s => ({ ...s, errorAlerts: v }))}
                />
              </SettingItem>
              <SettingItem label="تنبيهات التحديثات">
                <Toggle 
                  checked={notificationSettings.updateAlerts} 
                  onChange={(v) => setNotificationSettings(s => ({ ...s, updateAlerts: v }))}
                />
              </SettingItem>
            </SettingGroup>
          </div>
        )}
      </div>
    </div>
  );
}

export default SettingsPanel;
