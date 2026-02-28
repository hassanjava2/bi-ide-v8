/**
 * إشعار التحديث - Update Notification
 * نافذة منبثقة لإشعارات التحديث مع عرض سجل التغييرات وتقدم التنزيل
 */

import { useState, useEffect } from "react";
import { 
  Download, 
  X, 
  Check,
  Sparkles,
  ArrowRight,
  RefreshCw,
  AlertCircle,
  FileText,
  Clock,
  ChevronDown,
  ChevronUp,
  Zap,
  Shield,
  Bug
} from "lucide-react";

// أنواع البيانات
interface UpdateInfo {
  version: string;
  releaseDate: Date;
  size: string;
  isRequired: boolean;
  changes: ChangelogItem[];
}

interface ChangelogItem {
  type: "feature" | "fix" | "security" | "improvement";
  description: string;
}

interface DownloadProgress {
  downloaded: number;
  total: number;
  speed: string;
  timeRemaining: string;
}

// مكون شريط التقدم
function ProgressBar({ 
  progress, 
  showPercentage = true 
}: { 
  progress: number;
  showPercentage?: boolean;
}) {
  const clampedProgress = Math.min(100, Math.max(0, progress));
  
  return (
    <div className="w-full">
      <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-primary-500 to-primary-400 transition-all duration-300 ease-out"
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
      {showPercentage && (
        <div className="flex justify-between text-xs text-dark-400 mt-1">
          <span>{clampedProgress.toFixed(0)}%</span>
          <span>{clampedProgress === 100 ? "اكتمل" : "جاري التنزيل..."}</span>
        </div>
      )}
    </div>
  );
}

// مكون عنصر سجل التغييرات
function ChangelogItemComponent({ item }: { item: ChangelogItem }) {
  const icons = {
    feature: { icon: Sparkles, color: "text-yellow-400", bg: "bg-yellow-500/10", label: "ميزة جديدة" },
    fix: { icon: Bug, color: "text-green-400", bg: "bg-green-500/10", label: "إصلاح" },
    security: { icon: Shield, color: "text-red-400", bg: "bg-red-500/10", label: "أمان" },
    improvement: { icon: Zap, color: "text-blue-400", bg: "bg-blue-500/10", label: "تحسين" },
  };

  const { icon: Icon, color, bg, label } = icons[item.type];

  return (
    <div className="flex items-start gap-3 py-2">
      <div className={`p-1.5 rounded ${bg} flex-shrink-0`}>
        <Icon className={`w-3.5 h-3.5 ${color}`} />
      </div>
      <div>
        <span className={`text-xs font-medium ${color} ml-2`}>{label}</span>
        <span className="text-sm text-dark-200">{item.description}</span>
      </div>
    </div>
  );
}

// المكون الرئيسي
export function UpdateNotification({
  isOpen = true,
  onClose,
}: {
  isOpen?: boolean;
  onClose?: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress>({
    downloaded: 0,
    total: 0,
    speed: "",
    timeRemaining: "",
  });

  // معلومات التحديث
  const [updateInfo] = useState<UpdateInfo>({
    version: "2.4.0",
    releaseDate: new Date(),
    size: "156 MB",
    isRequired: false,
    changes: [
      { type: "feature", description: "إضافة دعم تدفق الردود من AI" },
      { type: "feature", description: "تحسين أداء مستكشف الملفات" },
      { type: "improvement", description: "تحسين سرعة بدء التطبيق بنسبة 40%" },
      { type: "security", description: "تحديث مكتبات الأمان" },
      { type: "fix", description: "إصلاح مشكلة تجميد الواجهة عند فتح ملفات كبيرة" },
      { type: "fix", description: "إصلاح أخطاء في مزامنة Git" },
    ],
  });

  // محاكاة التنزيل
  useEffect(() => {
    if (!isDownloading || isComplete) return;

    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          setIsDownloading(false);
          setIsComplete(true);
          return 100;
        }
        
        // تحديث معلومات التنزيل
        setDownloadProgress({
          downloaded: (prev / 100) * 156,
          total: 156,
          speed: "12.5 MB/s",
          timeRemaining: `${Math.ceil((100 - prev) / 10)} ثانية`,
        });
        
        return prev + Math.random() * 5;
      });
    }, 200);

    return () => clearInterval(interval);
  }, [isDownloading, isComplete]);

  // بدء التنزيل
  const handleDownload = () => {
    setIsDownloading(true);
    setProgress(0);
  };

  // تثبيت التحديث
  const handleInstall = () => {
    setIsInstalling(true);
    // محاكاة التثبيت
    setTimeout(() => {
      window.location.reload();
    }, 2000);
  };

  // تخطي التحديث
  const handleSkip = () => {
    onClose?.();
    // حفظ التفضيل في localStorage
    localStorage.setItem("bi-ide-update-skipped", updateInfo.version);
  };

  // إعادة المحاولة
  const handleRetry = () => {
    setIsDownloading(false);
    setIsComplete(false);
    setProgress(0);
    handleDownload();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-dark-800 rounded-2xl w-full max-w-lg shadow-2xl border border-dark-700 overflow-hidden">
        {/* رأس النافذة */}
        <div className="relative bg-gradient-to-r from-primary-600 to-primary-500 p-6">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1 text-white/70 hover:text-white hover:bg-white/10 rounded transition-colors"
            disabled={isInstalling}
          >
            <X className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-4">
            <div className="p-3 bg-white/20 rounded-xl">
              <Download className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">
                {isComplete ? "اكتمل التنزيل!" : isDownloading ? "جاري التنزيل..." : "تحديث جديد متاح"}
              </h2>
              <p className="text-white/80 text-sm">
                الإصدار {updateInfo.version} • {updateInfo.size}
              </p>
            </div>
          </div>

          {updateInfo.isRequired && (
            <div className="mt-4 flex items-center gap-2 px-3 py-2 bg-red-500/20 border border-red-500/30 rounded-lg">
              <AlertCircle className="w-4 h-4 text-red-300" />
              <span className="text-sm text-red-200">
                هذا تحديث إجباري لضمان عمل التطبيق بشكل صحيح
              </span>
            </div>
          )}
        </div>

        {/* المحتوى */}
        <div className="p-6">
          {isInstalling ? (
            <div className="text-center py-8">
              <RefreshCw className="w-12 h-12 text-primary-400 animate-spin mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-dark-100 mb-2">
                جاري تثبيت التحديث...
              </h3>
              <p className="text-dark-400">
                سيتم إعادة تشغيل التطبيق قريباً
              </p>
            </div>
          ) : isDownloading ? (
            <div className="space-y-4">
              <ProgressBar progress={progress} />
              
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-3 bg-dark-900 rounded-lg">
                  <div className="text-xs text-dark-500 mb-1">المحمل</div>
                  <div className="text-sm font-mono text-dark-200">
                    {downloadProgress.downloaded.toFixed(1)} MB
                  </div>
                </div>
                <div className="p-3 bg-dark-900 rounded-lg">
                  <div className="text-xs text-dark-500 mb-1">السرعة</div>
                  <div className="text-sm font-mono text-dark-200">
                    {downloadProgress.speed}
                  </div>
                </div>
                <div className="p-3 bg-dark-900 rounded-lg">
                  <div className="text-xs text-dark-500 mb-1">الوقت المتبقي</div>
                  <div className="text-sm font-mono text-dark-200">
                    {downloadProgress.timeRemaining}
                  </div>
                </div>
              </div>

              <button
                onClick={() => setIsDownloading(false)}
                className="w-full py-2 text-sm text-dark-400 hover:text-dark-200 transition-colors"
              >
                إلغاء التنزيل
              </button>
            </div>
          ) : isComplete ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <Check className="w-8 h-8 text-green-400" />
                <div>
                  <h3 className="font-semibold text-green-400">تم التنزيل بنجاح!</h3>
                  <p className="text-sm text-dark-400">
                    التحديث جاهز للتثبيت. سيتم إعادة تشغيل التطبيق.
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <button
                  onClick={handleInstall}
                  className="w-full flex items-center justify-center gap-2 py-3 bg-primary-600 hover:bg-primary-700 rounded-lg text-white font-medium transition-colors"
                >
                  <Zap className="w-4 h-4" />
                  تثبيت الآن
                </button>
                <button
                  onClick={() => setIsComplete(false)}
                  className="w-full py-2 text-dark-400 hover:text-dark-200 transition-colors"
                >
                  التثبيت لاحقاً
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* سجل التغييرات */}
              <div className="mb-4">
                <button
                  onClick={() => setShowDetails(!showDetails)}
                  className="flex items-center justify-between w-full py-2 text-left"
                >
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-dark-400" />
                    <span className="font-medium text-dark-200">سجل التغييرات</span>
                    <span className="px-2 py-0.5 bg-dark-700 rounded text-xs text-dark-400">
                      {updateInfo.changes.length}
                    </span>
                  </div>
                  {showDetails ? (
                    <ChevronUp className="w-4 h-4 text-dark-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-dark-400" />
                  )}
                </button>

                {showDetails && (
                  <div className="mt-3 p-3 bg-dark-900 rounded-lg max-h-48 overflow-y-auto">
                    {updateInfo.changes.map((change, idx) => (
                      <ChangelogItemComponent key={idx} item={change} />
                    ))}
                  </div>
                )}
              </div>

              {/* معلومات إضافية */}
              <div className="flex items-center gap-4 text-sm text-dark-400 mb-6">
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  تاريخ الإصدار: {updateInfo.releaseDate.toLocaleDateString('ar-SA')}
                </span>
                <span className="flex items-center gap-1">
                  <Download className="w-4 h-4" />
                  الحجم: {updateInfo.size}
                </span>
              </div>

              {/* الأزرار */}
              <div className="space-y-3">
                <button
                  onClick={handleDownload}
                  className="w-full flex items-center justify-center gap-2 py-3 bg-primary-600 hover:bg-primary-700 rounded-lg text-white font-medium transition-colors"
                >
                  <Download className="w-4 h-4" />
                  تنزيل التحديث
                </button>

                <div className="flex gap-3">
                  {!updateInfo.isRequired && (
                    <button
                      onClick={handleSkip}
                      className="flex-1 py-2.5 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors"
                    >
                      تخطي
                    </button>
                  )}
                  <button
                    onClick={() => window.open("https://bi-ide.com/changelog", "_blank")}
                    className="flex-1 flex items-center justify-center gap-2 py-2.5 text-dark-300 hover:text-dark-100 hover:bg-dark-700 rounded-lg transition-colors"
                  >
                    <span>المزيد من المعلومات</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// مكون مصغر للإشعار (للشريط العلوي)
export function UpdateBadge({
  onClick,
}: {
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-1.5 bg-primary-500/10 hover:bg-primary-500/20 border border-primary-500/30 rounded-lg text-primary-400 text-sm transition-colors animate-pulse"
    >
      <Sparkles className="w-4 h-4" />
      <span>تحديث جديد</span>
    </button>
  );
}

export default UpdateNotification;
