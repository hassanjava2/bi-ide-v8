/**
 * Keyboard Shortcuts - اختصارات لوحة المفاتيح
 * 
 * يحدد خريطة الاختصارات مع اختصارات افتراضية (Ctrl+S للحفظ، Ctrl+O للفتح، Ctrl+N لجديد،
 * Ctrl+Shift+P للوحة الأوامر) ودعم تخصيص الاختصارات واكتشاف التعارضات
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/** المنصة */
type Platform = 'mac' | 'windows' | 'linux';

/** معدّل المفتاح */
type Modifier = 'ctrl' | 'alt' | 'shift' | 'meta' | 'cmd';

/** تعريف الاختصار */
export interface ShortcutDefinition {
  /** معرف الاختصار */
  id: string;
  /** الاسم المعروض */
  name: string;
  /** الفئة */
  category: string;
  /** مفاتيح الاختصار */
  keys: string[];
  /** معدّلات الاختصار */
  modifiers: Modifier[];
  /** وصف الاختصار */
  description?: string;
  /** هل الاختصار قابل للتخصيص */
  customizable?: boolean;
  /** هل الاختصار معطل */
  disabled?: boolean;
}

/** اختصار مُسجّل */
export interface RegisteredShortcut extends ShortcutDefinition {
  /** رد الاتصال عند تنفيذ الاختصار */
  handler: (event: KeyboardEvent) => void | Promise<void>;
}

/** تعارض الاختصار */
export interface ShortcutConflict {
  /** الاختصار الأول */
  shortcut1: ShortcutDefinition;
  /** الاختصار الثاني */
  shortcut2: ShortcutDefinition;
  /** التركيب المتعارض */
  combination: string;
}

/** إعدادات مدير الاختصارات */
export interface KeyboardShortcutsOptions {
  /** الاختصارات الافتراضية */
  defaultShortcuts?: ShortcutDefinition[];
  /** اختصارات مخصصة من المستخدم */
  customShortcuts?: Partial<Record<string, Partial<ShortcutDefinition>>>;
  /** مفتاح التخزين المحلي */
  storageKey?: string;
  /** تمكين الاختصارات */
  enabled?: boolean;
  /** تجاهل العناصر المُعدّة */
  ignoreInputs?: boolean;
  /** رد الاتصال عند تنفيذ اختصار */
  onShortcutExecuted?: (shortcut: ShortcutDefinition) => void;
  /** رد الاتصال عند اكتشاف تعارض */
  onConflictDetected?: (conflicts: ShortcutConflict[]) => void;
}

/** نتيجة مدير الاختصارات */
export interface UseKeyboardShortcutsResult {
  /** الاختصارات المسجلة */
  shortcuts: Map<string, RegisteredShortcut>;
  /** إضافة اختصار */
  register: (shortcut: ShortcutDefinition, handler: (event: KeyboardEvent) => void | Promise<void>) => void;
  /** إزالة اختصار */
  unregister: (id: string) => void;
  /** تحديث اختصار */
  update: (id: string, updates: Partial<ShortcutDefinition>) => boolean;
  /** تمكين/تعطيل اختصار */
  setEnabled: (id: string, enabled: boolean) => void;
  /** إعادة تعيين إلى الافتراضي */
  resetToDefault: (id?: string) => void;
  /** اكتشاف التعارضات */
  detectConflicts: () => ShortcutConflict[];
  /** الحصول على الاختصار بالمعرف */
  getShortcut: (id: string) => RegisteredShortcut | undefined;
  /** الحصول على الاختصار بالتركيب */
  getByCombination: (combination: string) => RegisteredShortcut | undefined;
  /** التركيب الحالي كـ string */
  currentCombination: string;
  /** تحميل من التخزين المحلي */
  loadFromStorage: () => void;
  /** حفظ في التخزين المحلي */
  saveToStorage: () => void;
}

/** الاختصارات الافتراضية */
export const DEFAULT_SHORTCUTS: ShortcutDefinition[] = [
  // ملف
  { id: 'file.new', name: 'ملف جديد', category: 'file', keys: ['n'], modifiers: ['ctrl'], description: 'إنشاء ملف جديد' },
  { id: 'file.open', name: 'فتح', category: 'file', keys: ['o'], modifiers: ['ctrl'], description: 'فتح ملف' },
  { id: 'file.save', name: 'حفظ', category: 'file', keys: ['s'], modifiers: ['ctrl'], description: 'حفظ الملف الحالي' },
  { id: 'file.saveAs', name: 'حفظ باسم', category: 'file', keys: ['s'], modifiers: ['ctrl', 'shift'], description: 'حفظ باسم جديد' },
  { id: 'file.close', name: 'إغلاق', category: 'file', keys: ['w'], modifiers: ['ctrl'], description: 'إغلاق الملف الحالي' },
  
  // تحرير
  { id: 'edit.undo', name: 'تراجع', category: 'edit', keys: ['z'], modifiers: ['ctrl'], description: 'تراجع عن آخر إجراء' },
  { id: 'edit.redo', name: 'إعادة', category: 'edit', keys: ['z'], modifiers: ['ctrl', 'shift'], description: 'إعادة الإجراء' },
  { id: 'edit.redo.alt', name: 'إعادة (بديل)', category: 'edit', keys: ['y'], modifiers: ['ctrl'], description: 'إعادة الإجراء (بديل)', disabled: true },
  { id: 'edit.cut', name: 'قص', category: 'edit', keys: ['x'], modifiers: ['ctrl'], description: 'قص المحدد' },
  { id: 'edit.copy', name: 'نسخ', category: 'edit', keys: ['c'], modifiers: ['ctrl'], description: 'نسخ المحدد' },
  { id: 'edit.paste', name: 'لصق', category: 'edit', keys: ['v'], modifiers: ['ctrl'], description: 'لصق من الحافظة' },
  { id: 'edit.selectAll', name: 'تحديد الكل', category: 'edit', keys: ['a'], modifiers: ['ctrl'], description: 'تحديد كل المحتوى' },
  { id: 'edit.find', name: 'بحث', category: 'edit', keys: ['f'], modifiers: ['ctrl'], description: 'البحث في الملف' },
  { id: 'edit.replace', name: 'استبدال', category: 'edit', keys: ['h'], modifiers: ['ctrl'], description: 'البحث والاستبدال' },
  
  // عرض
  { id: 'view.commandPalette', name: 'لوحة الأوامر', category: 'view', keys: ['p'], modifiers: ['ctrl', 'shift'], description: 'فتح لوحة الأوامر' },
  { id: 'view.quickOpen', name: 'فتح سريع', category: 'view', keys: ['p'], modifiers: ['ctrl'], description: 'فتح ملف بسرعة' },
  { id: 'view.sidebar', name: 'إظهار/إخفاء الشريط الجانبي', category: 'view', keys: ['b'], modifiers: ['ctrl'], description: 'تبديل الشريط الجانبي' },
  { id: 'view.terminal', name: 'إظهار/إخفاء الطرفية', category: 'view', keys: ['`'], modifiers: ['ctrl'], description: 'تبديل الطرفية' },
  { id: 'view.fullscreen', name: 'ملء الشاشة', category: 'view', keys: ['f11'], modifiers: [], description: 'تبديل ملء الشاشة' },
  { id: 'view.zoomIn', name: 'تكبير', category: 'view', keys: ['equal'], modifiers: ['ctrl'], description: 'تكبير الخط' },
  { id: 'view.zoomOut', name: 'تصغير', category: 'view', keys: ['minus'], modifiers: ['ctrl'], description: 'تصغير الخط' },
  { id: 'view.zoomReset', name: 'إعادة تعيين التكبير', category: 'view', keys: ['0'], modifiers: ['ctrl'], description: 'إعادة التكبير للافتراضي' },
  
  // Git
  { id: 'git.commit', name: 'Git Commit', category: 'git', keys: ['enter'], modifiers: ['ctrl', 'shift'], description: 'تثبيت التغييرات' },
  { id: 'git.push', name: 'Git Push', category: 'git', keys: ['p'], modifiers: ['ctrl', 'alt'], description: 'دفع التغييرات' },
  { id: 'git.pull', name: 'Git Pull', category: 'git', keys: ['l'], modifiers: ['ctrl', 'alt'], description: 'سحب التغييرات' },
  { id: 'git.status', name: 'Git Status', category: 'git', keys: ['g'], modifiers: ['ctrl', 'shift'], description: 'عرض حالة Git' },
  
  // AI
  { id: 'ai.generate', name: 'توليد كود', category: 'ai', keys: ['i'], modifiers: ['ctrl'], description: 'توليد كود بالذكاء الاصطناعي' },
  { id: 'ai.explain', name: 'شرح الكود', category: 'ai', keys: ['e'], modifiers: ['ctrl', 'shift'], description: 'شرح الكود المحدد' },
  { id: 'ai.chat', name: 'فتح الدردشة', category: 'ai', keys: ['l'], modifiers: ['ctrl', 'shift'], description: 'فتح دردشة الذكاء الاصطناعي' },
  
  // ملاحة
  { id: 'navigate.back', name: 'العودة', category: 'navigate', keys: ['-'], modifiers: ['alt'], description: 'الانتقال للخلف' },
  { id: 'navigate.forward', name: 'التقدم', category: 'navigate', keys: ['-'], modifiers: ['alt', 'shift'], description: 'الانتقال للأمام' },
  { id: 'navigate.toLine', name: 'الانتقال إلى سطر', category: 'navigate', keys: ['g'], modifiers: ['ctrl'], description: 'الانتقال إلى سطر محدد' },
  { id: 'navigate.nextTab', name: 'التبويب التالي', category: 'navigate', keys: ['tab'], modifiers: ['ctrl'], description: 'الانتقال للتبويب التالي' },
  { id: 'navigate.prevTab', name: 'التبويب السابق', category: 'navigate', keys: ['tab'], modifiers: ['ctrl', 'shift'], description: 'الانتقال للتبويب السابق' },
  
  // إعدادات
  { id: 'settings.open', name: 'فتح الإعدادات', category: 'settings', keys: ['comma'], modifiers: ['ctrl'], description: 'فتح الإعدادات' },
  { id: 'settings.shortcuts', name: 'اختصارات لوحة المفاتيح', category: 'settings', keys: ['k'], modifiers: ['ctrl'], description: 'فتح إعدادات الاختصارات' },
];

/** المفتاح الافتراضي للتخزين */
const DEFAULT_STORAGE_KEY = 'bi_ide_shortcuts';

/** الحصول على المنصة */
function getPlatform(): Platform {
  if (navigator.platform.toLowerCase().includes('mac')) return 'mac';
  if (navigator.platform.toLowerCase().includes('win')) return 'windows';
  return 'linux';
}

/** تحويل الاختصار لنص قابل للقراءة */
export function formatShortcut(shortcut: ShortcutDefinition, platform?: Platform): string {
  const plat = platform || getPlatform();
  const parts: string[] = [];

  // إضافة المعدّلات بالترتيب الصحيح
  if (shortcut.modifiers.includes('ctrl')) parts.push(plat === 'mac' ? '⌃' : 'Ctrl');
  if (shortcut.modifiers.includes('alt')) parts.push(plat === 'mac' ? '⌥' : 'Alt');
  if (shortcut.modifiers.includes('shift')) parts.push(plat === 'mac' ? '⇧' : 'Shift');
  if (shortcut.modifiers.includes('meta') || shortcut.modifiers.includes('cmd')) {
    parts.push(plat === 'mac' ? '⌘' : 'Win');
  }

  // إضافة المفاتيح
  parts.push(...shortcut.keys.map(k => {
    if (k === 'equal') return '=';
    if (k === 'minus') return '-';
    if (k === 'comma') return ',';
    if (k === 'period') return '.';
    if (k === 'slash') return '/';
    if (k === 'backslash') return '\\';
    if (k === 'bracketleft') return '[';
    if (k === 'bracketright') return ']';
    if (k === 'quote') return "'";
    if (k === 'semicolon') return ';';
    if (k === 'grave') return '`';
    return k.charAt(0).toUpperCase() + k.slice(1);
  }));

  return parts.join(plat === 'mac' ? '' : '+');
}

/** بناء تركيب الاختصار */
function buildCombination(event: KeyboardEvent): string {
  const parts: string[] = [];
  
  if (event.ctrlKey) parts.push('ctrl');
  if (event.altKey) parts.push('alt');
  if (event.shiftKey) parts.push('shift');
  if (event.metaKey) parts.push('meta');
  
  parts.push(event.key.toLowerCase());
  
  return parts.join('+');
}

/** بناء تركيب من تعريف */
function buildCombinationFromDefinition(shortcut: ShortcutDefinition): string {
  const parts: string[] = [...shortcut.modifiers, ...shortcut.keys.map(k => k.toLowerCase())];
  return parts.join('+');
}

/**
 * هوك إدارة اختصارات لوحة المفاتيح
 */
export function useKeyboardShortcuts(options: KeyboardShortcutsOptions = {}): UseKeyboardShortcutsResult {
  const {
    defaultShortcuts = DEFAULT_SHORTCUTS,
    customShortcuts: initialCustom,
    storageKey = DEFAULT_STORAGE_KEY,
    enabled = true,
    ignoreInputs = true,
    onShortcutExecuted,
    onConflictDetected,
  } = options;

  const [shortcuts, setShortcuts] = useState<Map<string, RegisteredShortcut>>(new Map());
  const [currentCombination, setCurrentCombination] = useState('');
  const shortcutsRef = useRef(shortcuts);

  // تحديث المرجع
  useEffect(() => {
    shortcutsRef.current = shortcuts;
  }, [shortcuts]);

  // تحميل الاختصارات من التخزين المحلي
  const loadFromStorage = useCallback(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed = JSON.parse(saved);
        // دمج مع الافتراضيات
        const merged = new Map<string, RegisteredShortcut>();
        
        defaultShortcuts.forEach(def => {
          const savedShortcut = parsed[def.id];
          merged.set(def.id, {
            ...def,
            ...savedShortcut,
            handler: () => {}, // سيتم تعيينه لاحقاً
          });
        });
        
        setShortcuts(merged);
      }
    } catch (err) {
      console.error('فشل تحميل الاختصارات:', err);
    }
  }, [storageKey, defaultShortcuts]);

  // حفظ الاختصارات في التخزين المحلي
  const saveToStorage = useCallback(() => {
    try {
      const toSave: Record<string, Partial<ShortcutDefinition>> = {};
      shortcuts.forEach((shortcut, id) => {
        toSave[id] = {
          keys: shortcut.keys,
          modifiers: shortcut.modifiers,
          disabled: shortcut.disabled,
        };
      });
      localStorage.setItem(storageKey, JSON.stringify(toSave));
    } catch (err) {
      console.error('فشل حفظ الاختصارات:', err);
    }
  }, [storageKey, shortcuts]);

  // التحميل الأولي
  useEffect(() => {
    loadFromStorage();
  }, [loadFromStorage]);

  // إضافة اختصار
  const register = useCallback((
    shortcut: ShortcutDefinition,
    handler: (event: KeyboardEvent) => void | Promise<void>
  ) => {
    setShortcuts(prev => {
      const next = new Map(prev);
      next.set(shortcut.id, { ...shortcut, handler });
      return next;
    });
  }, []);

  // إزالة اختصار
  const unregister = useCallback((id: string) => {
    setShortcuts(prev => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  // تحديث اختصار
  const update = useCallback((id: string, updates: Partial<ShortcutDefinition>): boolean => {
    const shortcut = shortcuts.get(id);
    if (!shortcut || shortcut.customizable === false) return false;

    setShortcuts(prev => {
      const next = new Map(prev);
      next.set(id, { ...shortcut, ...updates });
      return next;
    });

    return true;
  }, [shortcuts]);

  // تمكين/تعطيل اختصار
  const setEnabled = useCallback((id: string, isEnabled: boolean) => {
    update(id, { disabled: !isEnabled });
  }, [update]);

  // إعادة تعيين إلى الافتراضي
  const resetToDefault = useCallback((id?: string) => {
    if (id) {
      const defaultShortcut = defaultShortcuts.find(s => s.id === id);
      if (defaultShortcut) {
        update(id, {
          keys: defaultShortcut.keys,
          modifiers: defaultShortcut.modifiers,
          disabled: defaultShortcut.disabled,
        });
      }
    } else {
      // إعادة تعيين الكل
      setShortcuts(prev => {
        const next = new Map<string, RegisteredShortcut>();
        defaultShortcuts.forEach(def => {
          const existing = prev.get(def.id);
          next.set(def.id, {
            ...def,
            handler: existing?.handler || (() => {}),
          });
        });
        return next;
      });
    }
  }, [defaultShortcuts, update]);

  // اكتشاف التعارضات
  const detectConflicts = useCallback((): ShortcutConflict[] => {
    const conflicts: ShortcutConflict[] = [];
    const combinations = new Map<string, RegisteredShortcut>();

    shortcuts.forEach(shortcut => {
      if (shortcut.disabled) return;
      
      const combination = buildCombinationFromDefinition(shortcut);
      const existing = combinations.get(combination);
      
      if (existing) {
        conflicts.push({
          shortcut1: existing,
          shortcut2: shortcut,
          combination,
        });
      } else {
        combinations.set(combination, shortcut);
      }
    });

    if (conflicts.length > 0) {
      onConflictDetected?.(conflicts);
    }

    return conflicts;
  }, [shortcuts, onConflictDetected]);

  // الحصول على اختصار بالمعرف
  const getShortcut = useCallback((id: string): RegisteredShortcut | undefined => {
    return shortcuts.get(id);
  }, [shortcuts]);

  // الحصول على اختصار بالتركيب
  const getByCombination = useCallback((combination: string): RegisteredShortcut | undefined => {
    for (const shortcut of shortcuts.values()) {
      if (buildCombinationFromDefinition(shortcut) === combination) {
        return shortcut;
      }
    }
    return undefined;
  }, [shortcuts]);

  // معالجة أحداث لوحة المفاتيح
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // تجاهل الإدخالات إذا كان المستخدم يكتب في input
      if (ignoreInputs) {
        const target = event.target as HTMLElement;
        if (
          target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable
        ) {
          // السماح ببعض الاختصارات حتى في الإدخالات
          const combination = buildCombination(event);
          if (!combination.includes('escape') && !combination.includes('ctrl+a')) {
            return;
          }
        }
      }

      const combination = buildCombination(event);
      setCurrentCombination(combination);

      // البحث عن اختصار مطابق
      for (const shortcut of shortcutsRef.current.values()) {
        if (shortcut.disabled) continue;
        
        const shortcutCombination = buildCombinationFromDefinition(shortcut);
        
        if (shortcutCombination === combination) {
          event.preventDefault();
          event.stopPropagation();
          
          shortcut.handler(event);
          onShortcutExecuted?.(shortcut);
          break;
        }
      }
    };

    const handleKeyUp = () => {
      setCurrentCombination('');
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [enabled, ignoreInputs, onShortcutExecuted]);

  return {
    shortcuts,
    register,
    unregister,
    update,
    setEnabled,
    resetToDefault,
    detectConflicts,
    getShortcut,
    getByCombination,
    currentCombination,
    loadFromStorage,
    saveToStorage,
  };
}

export default useKeyboardShortcuts;
