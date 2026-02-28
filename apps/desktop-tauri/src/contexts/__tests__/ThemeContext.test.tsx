/**
 * Tests for ThemeContext
 * اختبارات سياق الثيم
 */

import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../ThemeContext';

describe('ThemeContext', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider>{children}</ThemeProvider>
  );

  it('should initialize with system theme mode', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.themeMode).toBe('system');
    expect(result.current.resolvedTheme).toMatch(/^(light|dark)$/);
  });

  it('should set theme mode', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setThemeMode('dark');
    });

    expect(result.current.themeMode).toBe('dark');
    expect(result.current.resolvedTheme).toBe('dark');
  });

  it('should toggle theme through modes', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.themeMode).toBe('system');

    act(() => {
      result.current.toggleTheme();
    });

    expect(result.current.themeMode).toBe('dark');

    act(() => {
      result.current.toggleTheme();
    });

    expect(result.current.themeMode).toBe('light');

    act(() => {
      result.current.toggleTheme();
    });

    expect(result.current.themeMode).toBe('system');
  });

  it('should persist theme to localStorage', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setThemeMode('dark');
    });

    expect(localStorage.getItem('bi_ide_theme')).toBe('dark');
  });

  it('should load theme from localStorage', () => {
    localStorage.setItem('bi_ide_theme', 'light');

    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.themeMode).toBe('light');
  });

  it('should apply theme colors correctly', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setThemeMode('light');
    });

    expect(result.current.theme.colors.background).toBe('#ffffff');
    expect(result.current.theme.colors.text).toBe('#1a1a1a');

    act(() => {
      result.current.setThemeMode('dark');
    });

    expect(result.current.theme.colors.background).toBe('#0a0a0a');
    expect(result.current.theme.colors.text).toBe('#fafafa');
  });

  it('should set custom theme', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setCustomTheme({
        colors: {
          primary: '#ff0000',
        },
      });
    });

    expect(result.current.theme.colors.primary).toBe('#ff0000');
  });

  it('should reset custom theme', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setThemeMode('light');
      result.current.setCustomTheme({
        colors: {
          primary: '#ff0000',
        },
      });
    });

    expect(result.current.theme.colors.primary).toBe('#ff0000');

    act(() => {
      result.current.resetCustomTheme();
    });

    expect(result.current.theme.colors.primary).toBe('#2563eb'); // Default light primary
  });

  it('should persist custom theme to localStorage', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setCustomTheme({
        colors: {
          primary: '#ff0000',
        },
      });
    });

    const saved = localStorage.getItem('bi_ide_theme_custom');
    expect(saved).toBeTruthy();
    expect(JSON.parse(saved!)).toEqual({
      colors: { primary: '#ff0000' },
    });
  });

  it('should detect system preference', () => {
    // Mock matchMedia
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: jest.fn().mockImplementation((query: string) => ({
        matches: query === '(prefers-color-scheme: dark)',
        media: query,
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
      })),
    });

    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.systemPrefersDark).toBe(true);
    expect(result.current.resolvedTheme).toBe('dark');
  });

  it('should update CSS variables on theme change', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setThemeMode('dark');
    });

    expect(document.documentElement.style.getPropertyValue('--color-background')).toBe('#0a0a0a');
    expect(document.documentElement.style.getPropertyValue('--color-text')).toBe('#fafafa');
  });

  it('should set data-theme attribute', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => {
      result.current.setThemeMode('dark');
    });

    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('should provide complete theme definition', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.theme).toHaveProperty('name');
    expect(result.current.theme).toHaveProperty('mode');
    expect(result.current.theme).toHaveProperty('colors');
    expect(result.current.theme).toHaveProperty('fonts');
    expect(result.current.theme).toHaveProperty('spacing');
    expect(result.current.theme).toHaveProperty('borderRadius');
    expect(result.current.theme).toHaveProperty('shadows');
  });

  it('should throw error if useTheme used outside provider', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useTheme());
    }).toThrow('useTheme must be used within a ThemeProvider');

    consoleSpy.mockRestore();
  });

  it('should work in a component', () => {
    function TestComponent() {
      const { theme, resolvedTheme, toggleTheme } = useTheme();
      return (
        <div>
          <span data-testid="mode">{resolvedTheme}</span>
          <span data-testid="bg">{theme.colors.background}</span>
          <button onClick={toggleTheme}>Toggle</button>
        </div>
      );
    }

    render(
      <ThemeProvider defaultTheme="light">
        <TestComponent />
      </ThemeProvider>
    );

    expect(screen.getByTestId('mode').textContent).toBe('light');
    expect(screen.getByTestId('bg').textContent).toBe('#ffffff');

    fireEvent.click(screen.getByText('Toggle'));

    expect(screen.getByTestId('mode').textContent).toBe('dark');
    expect(screen.getByTestId('bg').textContent).toBe('#0a0a0a');
  });
});
