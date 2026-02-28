/**
 * Tests for LanguageContext
 * اختبارات سياق اللغة
 */

import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { render, screen, fireEvent } from '@testing-library/react';
import { LanguageProvider, useLanguage } from '../LanguageContext';

describe('LanguageContext', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('dir');
    document.documentElement.removeAttribute('lang');
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <LanguageProvider>{children}</LanguageProvider>
  );

  it('should initialize with default language (ar)', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.language).toBe('ar');
    expect(result.current.direction).toBe('rtl');
    expect(result.current.isRTL).toBe(true);
  });

  it('should change language', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    act(() => {
      result.current.setLanguage('en');
    });

    expect(result.current.language).toBe('en');
    expect(result.current.direction).toBe('ltr');
    expect(result.current.isRTL).toBe(false);
  });

  it('should toggle language', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.language).toBe('ar');

    act(() => {
      result.current.toggleLanguage();
    });

    expect(result.current.language).toBe('en');

    act(() => {
      result.current.toggleLanguage();
    });

    expect(result.current.language).toBe('ar');
  });

  it('should translate keys correctly in Arabic', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.t('app.name')).toBe('BI-IDE');
    expect(result.current.t('menu.file')).toBe('ملف');
    expect(result.current.t('menu.save')).toBe('حفظ');
    expect(result.current.t('button.ok')).toBe('موافق');
  });

  it('should translate keys correctly in English', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    act(() => {
      result.current.setLanguage('en');
    });

    expect(result.current.t('app.name')).toBe('BI-IDE');
    expect(result.current.t('menu.file')).toBe('File');
    expect(result.current.t('menu.save')).toBe('Save');
    expect(result.current.t('button.ok')).toBe('OK');
  });

  it('should handle nested translation keys', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.t('git.commit')).toBe('تثبيت');
    expect(result.current.t('ai.generate')).toBe('توليد');
  });

  it('should return key if translation not found', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.t('nonexistent.key')).toBe('nonexistent.key');
  });

  it('should replace parameters in translations', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    // Custom dictionary with parameter
    const customWrapper = ({ children }: { children: React.ReactNode }) => (
      <LanguageProvider
        customDictionary={{
          ar: {
            'test.greeting': 'مرحباً {{name}}',
            'test.count': 'لديك {{count}} رسائل',
          },
        }}
      >
        {children}
      </LanguageProvider>
    );

    const { result: customResult } = renderHook(() => useLanguage(), { wrapper: customWrapper });

    expect(customResult.current.t('test.greeting', { name: 'أحمد' })).toBe('مرحباً أحمد');
    expect(customResult.current.t('test.count', { count: 5 })).toBe('لديك 5 رسائل');
  });

  it('should persist language to localStorage', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    act(() => {
      result.current.setLanguage('en');
    });

    expect(localStorage.getItem('bi_ide_language')).toBe('en');
  });

  it('should load language from localStorage', () => {
    localStorage.setItem('bi_ide_language', 'en');

    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.language).toBe('en');
  });

  it('should update document direction and lang', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(document.documentElement.getAttribute('dir')).toBe('rtl');
    expect(document.documentElement.getAttribute('lang')).toBe('ar');

    act(() => {
      result.current.setLanguage('en');
    });

    expect(document.documentElement.getAttribute('dir')).toBe('ltr');
    expect(document.documentElement.getAttribute('lang')).toBe('en');
  });

  it('should provide available languages list', () => {
    const { result } = renderHook(() => useLanguage(), { wrapper });

    expect(result.current.availableLanguages).toHaveLength(2);
    expect(result.current.availableLanguages[0].code).toBe('ar');
    expect(result.current.availableLanguages[1].code).toBe('en');
  });

  it('should throw error if useLanguage used outside provider', () => {
    // Suppress console.error for this test
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useLanguage());
    }).toThrow('useLanguage must be used within a LanguageProvider');

    consoleSpy.mockRestore();
  });

  it('should work in a component', () => {
    function TestComponent() {
      const { t, language, toggleLanguage } = useLanguage();
      return (
        <div>
          <span data-testid="lang">{language}</span>
          <span data-testid="text">{t('menu.file')}</span>
          <button onClick={toggleLanguage}>Toggle</button>
        </div>
      );
    }

    render(
      <LanguageProvider>
        <TestComponent />
      </LanguageProvider>
    );

    expect(screen.getByTestId('lang').textContent).toBe('ar');
    expect(screen.getByTestId('text').textContent).toBe('ملف');

    fireEvent.click(screen.getByText('Toggle'));

    expect(screen.getByTestId('lang').textContent).toBe('en');
    expect(screen.getByTestId('text').textContent).toBe('File');
  });
});
