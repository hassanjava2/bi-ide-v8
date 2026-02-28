/**
 * Language Context - سياق اللغة
 * 
 * يدعم اللغة العربية والإنجليزية مع مبدل اللغات وتخطيط من اليمين لليسار/من اليسار لليمين
 * وقاموس الترجمة وحفظ تفضيل اللغة
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  ReactNode,
} from 'react';

/** اللغات المدعومة */
export type Language = 'ar' | 'en';

/** اتجاه النص */
export type TextDirection = 'rtl' | 'ltr';

/** ترجمة */
export interface Translation {
  [key: string]: string | Translation;
}

/** القاموس الكامل */
export type TranslationDictionary = Record<Language, Translation>;

/** إعدادات السياق */
export interface LanguageContextValue {
  /** اللغة الحالية */
  language: Language;
  /** اتجاه النص */
  direction: TextDirection;
  /** هل اللغة العربية */
  isRTL: boolean;
  /** تغيير اللغة */
  setLanguage: (lang: Language) => void;
  /** تبديل اللغة */
  toggleLanguage: () => void;
  /** ترجمة مفتاح */
  t: (key: string, params?: Record<string, string | number>) => string;
  /** اللغات المدعومة */
  availableLanguages: { code: Language; name: string; nativeName: string }[];
}

/** خصائص المزود */
export interface LanguageProviderProps {
  /** العناصر الفرعية */
  children: ReactNode;
  /** اللغة الافتراضية */
  defaultLanguage?: Language;
  /** القاموس المخصص */
  customDictionary?: Partial<TranslationDictionary>;
  /** مفتاح التخزين المحلي */
  storageKey?: string;
}

/** القاموس الافتراضي */
const DEFAULT_DICTIONARY: TranslationDictionary = {
  ar: {
    // عام
    'app.name': 'BI-IDE',
    'app.tagline': 'بيئة تطوير متكاملة للذكاء الاصطناعي',
    
    // القائمة
    'menu.file': 'ملف',
    'menu.edit': 'تحرير',
    'menu.view': 'عرض',
    'menu.tools': 'أدوات',
    'menu.help': 'مساعدة',
    'menu.new': 'جديد',
    'menu.open': 'فتح',
    'menu.save': 'حفظ',
    'menu.saveAs': 'حفظ باسم',
    'menu.close': 'إغلاق',
    'menu.exit': 'خروج',
    'menu.undo': 'تراجع',
    'menu.redo': 'إعادة',
    'menu.cut': 'قص',
    'menu.copy': 'نسخ',
    'menu.paste': 'لصق',
    'menu.find': 'بحث',
    'menu.replace': 'استبدال',
    'menu.preferences': 'تفضيلات',
    'menu.language': 'اللغة',
    'menu.theme': 'المظهر',
    
    // المحرر
    'editor.newFile': 'ملف جديد',
    'editor.untitled': 'بدون عنوان',
    'editor.modified': 'معدل',
    'editor.saved': 'تم الحفظ',
    'editor.errors': 'أخطاء',
    'editor.warnings': 'تحذيرات',
    'editor.line': 'سطر',
    'editor.column': 'عمود',
    'editor.selected': 'محدد',
    
    // Git
    'git.commit': 'تثبيت',
    'git.push': 'دفع',
    'git.pull': 'سحب',
    'git.branch': 'فرع',
    'git.merge': 'دمج',
    'git.conflict': 'تعارض',
    'git.staged': 'مُجهز',
    'git.changes': 'تغييرات',
    'git.noChanges': 'لا توجد تغييرات',
    'git.messagePlaceholder': 'رسالة التثبيت...',
    
    // الذكاء الاصطناعي
    'ai.generate': 'توليد',
    'ai.explain': 'شرح',
    'ai.refactor': 'إعادة بناء',
    'ai.complete': 'إكمال',
    'ai.chat': 'دردشة',
    'ai.promptPlaceholder': 'اكتب طلبك هنا...',
    'ai.thinking': 'جاري التفكير...',
    'ai.model': 'النموذج',
    'ai.local': 'محلي',
    'ai.remote': 'بعيد',
    
    // الإعدادات
    'settings.title': 'الإعدادات',
    'settings.general': 'عام',
    'settings.editor': 'المحرر',
    'settings.ai': 'الذكاء الاصطناعي',
    'settings.git': 'Git',
    'settings.shortcuts': 'اختصارات',
    'settings.appearance': 'المظهر',
    
    // أزرار
    'button.ok': 'موافق',
    'button.cancel': 'إلغاء',
    'button.apply': 'تطبيق',
    'button.save': 'حفظ',
    'button.delete': 'حذف',
    'button.edit': 'تحرير',
    'button.create': 'إنشاء',
    'button.close': 'إغلاق',
    'button.confirm': 'تأكيد',
    'button.yes': 'نعم',
    'button.no': 'لا',
    
    // رسائل
    'message.success': 'تم بنجاح',
    'message.error': 'حدث خطأ',
    'message.warning': 'تحذير',
    'message.info': 'معلومات',
    'message.loading': 'جاري التحميل...',
    'message.confirm': 'هل أنت متأكد؟',
    'message.unsavedChanges': 'هناك تغييرات غير محفوظة',
    
    // تحديث
    'update.available': 'تحديث متاح',
    'update.downloading': 'جاري التنزيل...',
    'update.ready': 'جاهز للتثبيت',
    'update.install': 'تثبيت الآن',
    'update.skip': 'تخطي',
    'update.later': 'لاحقاً',
    
    // وضع عدم الاتصال
    'offline.status': 'غير متصل',
    'offline.sync': 'مزامنة',
    'offline.pending': 'معلق',
    'offline.queue': 'قائمة الانتظار',
  },
  en: {
    // General
    'app.name': 'BI-IDE',
    'app.tagline': 'Integrated AI Development Environment',
    
    // Menu
    'menu.file': 'File',
    'menu.edit': 'Edit',
    'menu.view': 'View',
    'menu.tools': 'Tools',
    'menu.help': 'Help',
    'menu.new': 'New',
    'menu.open': 'Open',
    'menu.save': 'Save',
    'menu.saveAs': 'Save As',
    'menu.close': 'Close',
    'menu.exit': 'Exit',
    'menu.undo': 'Undo',
    'menu.redo': 'Redo',
    'menu.cut': 'Cut',
    'menu.copy': 'Copy',
    'menu.paste': 'Paste',
    'menu.find': 'Find',
    'menu.replace': 'Replace',
    'menu.preferences': 'Preferences',
    'menu.language': 'Language',
    'menu.theme': 'Theme',
    
    // Editor
    'editor.newFile': 'New File',
    'editor.untitled': 'Untitled',
    'editor.modified': 'Modified',
    'editor.saved': 'Saved',
    'editor.errors': 'Errors',
    'editor.warnings': 'Warnings',
    'editor.line': 'Line',
    'editor.column': 'Column',
    'editor.selected': 'Selected',
    
    // Git
    'git.commit': 'Commit',
    'git.push': 'Push',
    'git.pull': 'Pull',
    'git.branch': 'Branch',
    'git.merge': 'Merge',
    'git.conflict': 'Conflict',
    'git.staged': 'Staged',
    'git.changes': 'Changes',
    'git.noChanges': 'No Changes',
    'git.messagePlaceholder': 'Commit message...',
    
    // AI
    'ai.generate': 'Generate',
    'ai.explain': 'Explain',
    'ai.refactor': 'Refactor',
    'ai.complete': 'Complete',
    'ai.chat': 'Chat',
    'ai.promptPlaceholder': 'Type your request here...',
    'ai.thinking': 'Thinking...',
    'ai.model': 'Model',
    'ai.local': 'Local',
    'ai.remote': 'Remote',
    
    // Settings
    'settings.title': 'Settings',
    'settings.general': 'General',
    'settings.editor': 'Editor',
    'settings.ai': 'AI',
    'settings.git': 'Git',
    'settings.shortcuts': 'Shortcuts',
    'settings.appearance': 'Appearance',
    
    // Buttons
    'button.ok': 'OK',
    'button.cancel': 'Cancel',
    'button.apply': 'Apply',
    'button.save': 'Save',
    'button.delete': 'Delete',
    'button.edit': 'Edit',
    'button.create': 'Create',
    'button.close': 'Close',
    'button.confirm': 'Confirm',
    'button.yes': 'Yes',
    'button.no': 'No',
    
    // Messages
    'message.success': 'Success',
    'message.error': 'Error',
    'message.warning': 'Warning',
    'message.info': 'Information',
    'message.loading': 'Loading...',
    'message.confirm': 'Are you sure?',
    'message.unsavedChanges': 'There are unsaved changes',
    
    // Update
    'update.available': 'Update Available',
    'update.downloading': 'Downloading...',
    'update.ready': 'Ready to Install',
    'update.install': 'Install Now',
    'update.skip': 'Skip',
    'update.later': 'Later',
    
    // Offline
    'offline.status': 'Offline',
    'offline.sync': 'Sync',
    'offline.pending': 'Pending',
    'offline.queue': 'Queue',
  },
};

/** اللغات المتاحة */
const AVAILABLE_LANGUAGES = [
  { code: 'ar' as Language, name: 'Arabic', nativeName: 'العربية' },
  { code: 'en' as Language, name: 'English', nativeName: 'English' },
];

/** المفتاح الافتراضي للتخزين */
const DEFAULT_STORAGE_KEY = 'bi_ide_language';

/** إنشاء السياق */
const LanguageContext = createContext<LanguageContextValue | undefined>(undefined);

/**
 * مزود سياق اللغة
 */
export function LanguageProvider({
  children,
  defaultLanguage = 'ar',
  customDictionary,
  storageKey = DEFAULT_STORAGE_KEY,
}: LanguageProviderProps): JSX.Element {
  const [language, setLanguageState] = useState<Language>(defaultLanguage);

  // دمج القاموس الافتراضي مع المخصص
  const dictionary = useMemo<TranslationDictionary>(() => {
    if (!customDictionary) return DEFAULT_DICTIONARY;
    
    return {
      ar: { ...DEFAULT_DICTIONARY.ar, ...(customDictionary.ar || {}) },
      en: { ...DEFAULT_DICTIONARY.en, ...(customDictionary.en || {}) },
    };
  }, [customDictionary]);

  // تحميل اللغة المحفوظة
  useEffect(() => {
    const saved = localStorage.getItem(storageKey) as Language | null;
    if (saved && (saved === 'ar' || saved === 'en')) {
      setLanguageState(saved);
    }
  }, [storageKey]);

  // تحديث اتجاه الصفحة
  useEffect(() => {
    const dir = language === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.setAttribute('dir', dir);
    document.documentElement.setAttribute('lang', language);
    
    // تحديث عنوان الصفحة
    const title = dictionary[language]['app.name'];
    if (title) {
      document.title = title as string;
    }
  }, [language, dictionary]);

  /**
   * تغيير اللغة
   */
  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem(storageKey, lang);
  }, [storageKey]);

  /**
   * تبديل اللغة
   */
  const toggleLanguage = useCallback(() => {
    setLanguage(prev => {
      const newLang = prev === 'ar' ? 'en' : 'ar';
      localStorage.setItem(storageKey, newLang);
      return newLang;
    });
  }, [storageKey]);

  /**
   * الحصول على الترجمة
   */
  const t = useCallback((
    key: string,
    params?: Record<string, string | number>
  ): string => {
    const keys = key.split('.');
    let translation: Translation | string | undefined = dictionary[language];

    for (const k of keys) {
      if (typeof translation === 'object' && translation !== null) {
        translation = translation[k];
      } else {
        translation = undefined;
        break;
      }
    }

    let result = (typeof translation === 'string' ? translation : key);

    // استبدال المعاملات
    if (params) {
      Object.entries(params).forEach(([paramKey, paramValue]) => {
        result = result.replace(`{{${paramKey}}}`, String(paramValue));
      });
    }

    return result;
  }, [language, dictionary]);

  const value = useMemo<LanguageContextValue>(() => ({
    language,
    direction: language === 'ar' ? 'rtl' : 'ltr',
    isRTL: language === 'ar',
    setLanguage,
    toggleLanguage,
    t,
    availableLanguages: AVAILABLE_LANGUAGES,
  }), [language, setLanguage, toggleLanguage, t]);

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

/**
 * هوك استخدام سياق اللغة
 */
export function useLanguage(): LanguageContextValue {
  const context = useContext(LanguageContext);
  
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  
  return context;
}

/**
 * مكون HOC لدعم اللغة
 */
export function withLanguage<P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> {
  return function WithLanguageComponent(props: P) {
    return (
      <LanguageProvider>
        <Component {...props} />
      </LanguageProvider>
    );
  };
}

export default LanguageContext;
