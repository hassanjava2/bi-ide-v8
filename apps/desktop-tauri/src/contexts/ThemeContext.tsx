/**
 * Theme Context - سياق الثيم
 * 
 * يدعم الثيمات الداكنة/الفاتحة/المخصصة مع مبدل الثيمات ومتغيرات CSS
 * وحفظ تفضيل الثيم واكتشاف ثيم النظام
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

/** أنواع الثيمات */
export type ThemeMode = 'light' | 'dark' | 'system';

/** الثيم الفعلي */
export type ResolvedTheme = 'light' | 'dark';

/** تعريف الثيم */
export interface ThemeDefinition {
  name: string;
  mode: ResolvedTheme;
  colors: ThemeColors;
  fonts: ThemeFonts;
  spacing: ThemeSpacing;
  borderRadius: ThemeBorderRadius;
  shadows: ThemeShadows;
}

/** ألوان الثيم */
export interface ThemeColors {
  // الخلفيات
  background: string;
  surface: string;
  surfaceVariant: string;
  // النصوص
  text: string;
  textSecondary: string;
  textMuted: string;
  // الألوان الأساسية
  primary: string;
  primaryLight: string;
  primaryDark: string;
  // الألوان الثانوية
  secondary: string;
  secondaryLight: string;
  secondaryDark: string;
  // حالات
  success: string;
  warning: string;
  error: string;
  info: string;
  // الحدود
  border: string;
  borderLight: string;
  // آخرى
  overlay: string;
  scrollbar: string;
  // المحرر
  editor: {
    background: string;
    foreground: string;
    lineNumber: string;
    lineNumberActive: string;
    selection: string;
    cursor: string;
  };
  // Syntax highlighting
  syntax: {
    keyword: string;
    string: string;
    number: string;
    comment: string;
    function: string;
    variable: string;
    type: string;
    operator: string;
  };
}

/** الخطوط */
export interface ThemeFonts {
  family: {
    base: string;
    mono: string;
    heading: string;
  };
  size: {
    xs: string;
    sm: string;
    base: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
  };
  weight: {
    light: number;
    normal: number;
    medium: number;
    semibold: number;
    bold: number;
  };
  lineHeight: {
    tight: number;
    normal: number;
    relaxed: number;
  };
}

/** المسافات */
export interface ThemeSpacing {
  px: string;
  0: string;
  0.5: string;
  1: string;
  1.5: string;
  2: string;
  2.5: string;
  3: string;
  3.5: string;
  4: string;
  5: string;
  6: string;
  7: string;
  8: string;
  9: string;
  10: string;
  12: string;
  14: string;
  16: string;
  20: string;
  24: string;
  28: string;
  32: string;
  36: string;
  40: string;
  44: string;
  48: string;
  52: string;
  56: string;
  60: string;
  64: string;
  72: string;
  80: string;
  96: string;
}

/** زوايا الحدود */
export interface ThemeBorderRadius {
  none: string;
  sm: string;
  base: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
  '3xl': string;
  full: string;
}

/** الظلال */
export interface ThemeShadows {
  none: string;
  sm: string;
  base: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
  inner: string;
}

/** إعدادات السياق */
export interface ThemeContextValue {
  /** وضع الثيم المختار */
  themeMode: ThemeMode;
  /** الثيم المحلول */
  resolvedTheme: ResolvedTheme;
  /** تعريف الثيم الحالي */
  theme: ThemeDefinition;
  /** تغيير وضع الثيم */
  setThemeMode: (mode: ThemeMode) => void;
  /** تبديل الثيم */
  toggleTheme: () => void;
  /** تعيين ثيم مخصص */
  setCustomTheme: (theme: Partial<ThemeDefinition>) => void;
  /** إعادة تعيين الثيم المخصص */
  resetCustomTheme: () => void;
  /** هل النظام يفضل الداكن */
  systemPrefersDark: boolean;
}

/** خصائص المزود */
export interface ThemeProviderProps {
  /** العناصر الفرعية */
  children: ReactNode;
  /** وضع الثيم الافتراضي */
  defaultTheme?: ThemeMode;
  /** مفتاح التخزين المحلي */
  storageKey?: string;
  /** الثيم المخصص الافتراضي */
  defaultCustomTheme?: Partial<ThemeDefinition>;
}

/** الثيم الفاتح */
const lightTheme: ThemeDefinition = {
  name: 'light',
  mode: 'light',
  colors: {
    background: '#ffffff',
    surface: '#fafafa',
    surfaceVariant: '#f0f0f0',
    text: '#1a1a1a',
    textSecondary: '#4a4a4a',
    textMuted: '#6b7280',
    primary: '#2563eb',
    primaryLight: '#3b82f6',
    primaryDark: '#1d4ed8',
    secondary: '#7c3aed',
    secondaryLight: '#8b5cf6',
    secondaryDark: '#6d28d9',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    border: '#e5e7eb',
    borderLight: '#f3f4f6',
    overlay: 'rgba(0, 0, 0, 0.5)',
    scrollbar: '#d1d5db',
    editor: {
      background: '#ffffff',
      foreground: '#1a1a1a',
      lineNumber: '#9ca3af',
      lineNumberActive: '#4b5563',
      selection: '#bfdbfe',
      cursor: '#1a1a1a',
    },
    syntax: {
      keyword: '#d73a49',
      string: '#032f62',
      number: '#005cc5',
      comment: '#6a737d',
      function: '#6f42c1',
      variable: '#24292e',
      type: '#005cc5',
      operator: '#d73a49',
    },
  },
  fonts: {
    family: {
      base: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      mono: 'JetBrains Mono, Fira Code, monospace',
      heading: 'Inter, sans-serif',
    },
    size: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
    },
    weight: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.25,
      normal: 1.5,
      relaxed: 1.75,
    },
  },
  spacing: {
    px: '1px',
    0: '0',
    0.5: '0.125rem',
    1: '0.25rem',
    1.5: '0.375rem',
    2: '0.5rem',
    2.5: '0.625rem',
    3: '0.75rem',
    3.5: '0.875rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    7: '1.75rem',
    8: '2rem',
    9: '2.25rem',
    10: '2.5rem',
    12: '3rem',
    14: '3.5rem',
    16: '4rem',
    20: '5rem',
    24: '6rem',
    28: '7rem',
    32: '8rem',
    36: '9rem',
    40: '10rem',
    44: '11rem',
    48: '12rem',
    52: '13rem',
    56: '14rem',
    60: '15rem',
    64: '16rem',
    72: '18rem',
    80: '20rem',
    96: '24rem',
  },
  borderRadius: {
    none: '0',
    sm: '0.125rem',
    base: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    '3xl': '1.5rem',
    full: '9999px',
  },
  shadows: {
    none: 'none',
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
  },
};

/** الثيم الداكن */
const darkTheme: ThemeDefinition = {
  ...lightTheme,
  name: 'dark',
  mode: 'dark',
  colors: {
    background: '#0a0a0a',
    surface: '#141414',
    surfaceVariant: '#1a1a1a',
    text: '#fafafa',
    textSecondary: '#a1a1aa',
    textMuted: '#71717a',
    primary: '#60a5fa',
    primaryLight: '#93c5fd',
    primaryDark: '#3b82f6',
    secondary: '#a78bfa',
    secondaryLight: '#c4b5fd',
    secondaryDark: '#8b5cf6',
    success: '#34d399',
    warning: '#fbbf24',
    error: '#f87171',
    info: '#60a5fa',
    border: '#27272a',
    borderLight: '#3f3f46',
    overlay: 'rgba(0, 0, 0, 0.7)',
    scrollbar: '#52525b',
    editor: {
      background: '#0a0a0a',
      foreground: '#fafafa',
      lineNumber: '#52525b',
      lineNumberActive: '#a1a1aa',
      selection: '#1e3a5f',
      cursor: '#fafafa',
    },
    syntax: {
      keyword: '#ff7b72',
      string: '#a5d6ff',
      number: '#79c0ff',
      comment: '#8b949e',
      function: '#d2a8ff',
      variable: '#e6edf3',
      type: '#79c0ff',
      operator: '#ff7b72',
    },
  },
};

/** دمج الثيمات العميق */
function deepMerge<T>(target: T, source: Partial<T>): T {
  const result = { ...target };
  
  for (const key in source) {
    if (source[key] !== undefined) {
      if (
        typeof source[key] === 'object' &&
        source[key] !== null &&
        !Array.isArray(source[key])
      ) {
        result[key] = deepMerge(result[key] as unknown as Record<string, unknown>, source[key] as Record<string, unknown>) as unknown as T[Extract<keyof T, string>];
      } else {
        result[key] = source[key] as T[Extract<keyof T, string>];
      }
    }
  }
  
  return result;
}

/** تطبيق الثيم على CSS */
function applyThemeToCSS(theme: ThemeDefinition): void {
  const root = document.documentElement;
  
  // الألوان الأساسية
  root.style.setProperty('--color-background', theme.colors.background);
  root.style.setProperty('--color-surface', theme.colors.surface);
  root.style.setProperty('--color-surface-variant', theme.colors.surfaceVariant);
  root.style.setProperty('--color-text', theme.colors.text);
  root.style.setProperty('--color-text-secondary', theme.colors.textSecondary);
  root.style.setProperty('--color-text-muted', theme.colors.textMuted);
  root.style.setProperty('--color-primary', theme.colors.primary);
  root.style.setProperty('--color-primary-light', theme.colors.primaryLight);
  root.style.setProperty('--color-primary-dark', theme.colors.primaryDark);
  root.style.setProperty('--color-secondary', theme.colors.secondary);
  root.style.setProperty('--color-success', theme.colors.success);
  root.style.setProperty('--color-warning', theme.colors.warning);
  root.style.setProperty('--color-error', theme.colors.error);
  root.style.setProperty('--color-info', theme.colors.info);
  root.style.setProperty('--color-border', theme.colors.border);
  root.style.setProperty('--color-border-light', theme.colors.borderLight);
  root.style.setProperty('--color-overlay', theme.colors.overlay);
  root.style.setProperty('--color-scrollbar', theme.colors.scrollbar);
  
  // ألوان المحرر
  root.style.setProperty('--editor-background', theme.colors.editor.background);
  root.style.setProperty('--editor-foreground', theme.colors.editor.foreground);
  root.style.setProperty('--editor-line-number', theme.colors.editor.lineNumber);
  root.style.setProperty('--editor-line-number-active', theme.colors.editor.lineNumberActive);
  root.style.setProperty('--editor-selection', theme.colors.editor.selection);
  root.style.setProperty('--editor-cursor', theme.colors.editor.cursor);
  
  // Syntax highlighting
  root.style.setProperty('--syntax-keyword', theme.colors.syntax.keyword);
  root.style.setProperty('--syntax-string', theme.colors.syntax.string);
  root.style.setProperty('--syntax-number', theme.colors.syntax.number);
  root.style.setProperty('--syntax-comment', theme.colors.syntax.comment);
  root.style.setProperty('--syntax-function', theme.colors.syntax.function);
  root.style.setProperty('--syntax-variable', theme.colors.syntax.variable);
  root.style.setProperty('--syntax-type', theme.colors.syntax.type);
  root.style.setProperty('--syntax-operator', theme.colors.syntax.operator);
  
  // الخطوط
  root.style.setProperty('--font-family-base', theme.fonts.family.base);
  root.style.setProperty('--font-family-mono', theme.fonts.family.mono);
  root.style.setProperty('--font-family-heading', theme.fonts.family.heading);
  
  // الظلال
  root.style.setProperty('--shadow-sm', theme.shadows.sm);
  root.style.setProperty('--shadow-base', theme.shadows.base);
  root.style.setProperty('--shadow-md', theme.shadows.md);
  root.style.setProperty('--shadow-lg', theme.shadows.lg);
  root.style.setProperty('--shadow-xl', theme.shadows.xl);
}

/** المفتاح الافتراضي للتخزين */
const DEFAULT_STORAGE_KEY = 'bi_ide_theme';

/** إنشاء السياق */
const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

/**
 * مزود سياق الثيم
 */
export function ThemeProvider({
  children,
  defaultTheme = 'system',
  storageKey = DEFAULT_STORAGE_KEY,
  defaultCustomTheme,
}: ThemeProviderProps): JSX.Element {
  const [themeMode, setThemeModeState] = useState<ThemeMode>(defaultTheme);
  const [systemPrefersDark, setSystemPrefersDark] = useState(false);
  const [customTheme, setCustomThemeState] = useState<Partial<ThemeDefinition> | undefined>(defaultCustomTheme);

  // تحميل الثيم المحفوظ
  useEffect(() => {
    const saved = localStorage.getItem(storageKey) as ThemeMode | null;
    if (saved && ['light', 'dark', 'system'].includes(saved)) {
      setThemeModeState(saved);
    }
    
    const savedCustom = localStorage.getItem(`${storageKey}_custom`);
    if (savedCustom) {
      try {
        setCustomThemeState(JSON.parse(savedCustom));
      } catch {
        console.error('فشل تحميل الثيم المخصص');
      }
    }
  }, [storageKey]);

  // مراقبة تفضيل النظام
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemPrefersDark(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => {
      setSystemPrefersDark(e.matches);
    };

    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // حساب الثيم المحلول
  const resolvedTheme = useMemo<ResolvedTheme>(() => {
    if (themeMode === 'system') {
      return systemPrefersDark ? 'dark' : 'light';
    }
    return themeMode;
  }, [themeMode, systemPrefersDark]);

  // بناء الثيم النهائي
  const theme = useMemo<ThemeDefinition>(() => {
    const baseTheme = resolvedTheme === 'dark' ? darkTheme : lightTheme;
    if (customTheme) {
      return deepMerge(baseTheme, customTheme);
    }
    return baseTheme;
  }, [resolvedTheme, customTheme]);

  // تطبيق الثيم
  useEffect(() => {
    applyThemeToCSS(theme);
    document.documentElement.setAttribute('data-theme', resolvedTheme);
  }, [theme, resolvedTheme]);

  /**
   * تغيير وضع الثيم
   */
  const setThemeMode = useCallback((mode: ThemeMode) => {
    setThemeModeState(mode);
    localStorage.setItem(storageKey, mode);
  }, [storageKey]);

  /**
   * تبديل الثيم
   */
  const toggleTheme = useCallback(() => {
    setThemeModeState(prev => {
      const modes: ThemeMode[] = ['light', 'dark', 'system'];
      const currentIndex = modes.indexOf(prev);
      const nextMode = modes[(currentIndex + 1) % modes.length];
      localStorage.setItem(storageKey, nextMode);
      return nextMode;
    });
  }, [storageKey]);

  /**
   * تعيين ثيم مخصص
   */
  const setCustomTheme = useCallback((themeUpdate: Partial<ThemeDefinition>) => {
    setCustomThemeState(prev => {
      const next = prev ? { ...prev, ...themeUpdate } : themeUpdate;
      localStorage.setItem(`${storageKey}_custom`, JSON.stringify(next));
      return next;
    });
  }, [storageKey]);

  /**
   * إعادة تعيين الثيم المخصص
   */
  const resetCustomTheme = useCallback(() => {
    setCustomThemeState(undefined);
    localStorage.removeItem(`${storageKey}_custom`);
  }, [storageKey]);

  const value = useMemo<ThemeContextValue>(() => ({
    themeMode,
    resolvedTheme,
    theme,
    setThemeMode,
    toggleTheme,
    setCustomTheme,
    resetCustomTheme,
    systemPrefersDark,
  }), [
    themeMode,
    resolvedTheme,
    theme,
    setThemeMode,
    toggleTheme,
    setCustomTheme,
    resetCustomTheme,
    systemPrefersDark,
  ]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * هوك استخدام سياق الثيم
 */
export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  
  return context;
}

/**
 * مكون HOC لدعم الثيم
 */
export function withTheme<P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> {
  return function WithThemeComponent(props: P) {
    return (
      <ThemeProvider>
        <Component {...props} />
      </ThemeProvider>
    );
  };
}

export default ThemeContext;
