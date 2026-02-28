/**
 * Tests for keyboardShortcuts
 * اختبارات اختصارات لوحة المفاتيح
 */

import { renderHook, act } from '@testing-library/react';
import { useKeyboardShortcuts, formatShortcut, DEFAULT_SHORTCUTS } from '../keyboardShortcuts';

describe('keyboardShortcuts', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  describe('formatShortcut', () => {
    it('should format Windows/Linux shortcuts', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'Win32',
        configurable: true,
      });

      const shortcut = DEFAULT_SHORTCUTS.find(s => s.id === 'file.save')!;
      expect(formatShortcut(shortcut)).toBe('Ctrl+S');

      const complex = DEFAULT_SHORTCUTS.find(s => s.id === 'view.commandPalette')!;
      expect(formatShortcut(complex)).toBe('Ctrl+Shift+P');
    });

    it('should format Mac shortcuts', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'MacIntel',
        configurable: true,
      });

      const shortcut = DEFAULT_SHORTCUTS.find(s => s.id === 'file.save')!;
      expect(formatShortcut(shortcut)).toBe('⌃S');

      const complex = DEFAULT_SHORTCUTS.find(s => s.id === 'view.commandPalette')!;
      expect(formatShortcut(complex)).toBe('⌃⇧P');
    });

    it('should handle special keys', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'Win32',
        configurable: true,
      });

      const shortcut = {
        ...DEFAULT_SHORTCUTS[0],
        keys: ['equal', 'minus', 'comma', 'slash'],
        modifiers: ['ctrl'],
      };
      expect(formatShortcut(shortcut)).toBe('Ctrl+=,-,.,/');
    });
  });

  describe('useKeyboardShortcuts', () => {
    it('should initialize with empty shortcuts', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());

      expect(result.current.shortcuts.size).toBe(0);
      expect(result.current.currentCombination).toBe('');
    });

    it('should register a shortcut', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      expect(result.current.shortcuts.has(DEFAULT_SHORTCUTS[0].id)).toBe(true);
    });

    it('should unregister a shortcut', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
        result.current.unregister(DEFAULT_SHORTCUTS[0].id);
      });

      expect(result.current.shortcuts.has(DEFAULT_SHORTCUTS[0].id)).toBe(false);
    });

    it('should update a shortcut', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      const updated = act(() => {
        return result.current.update(DEFAULT_SHORTCUTS[0].id, {
          keys: ['t'],
        });
      });

      expect(updated).toBe(true);
      const shortcut = result.current.getShortcut(DEFAULT_SHORTCUTS[0].id);
      expect(shortcut?.keys).toEqual(['t']);
    });

    it('should not update non-customizable shortcut', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      const nonCustomizable = { ...DEFAULT_SHORTCUTS[0], customizable: false };

      act(() => {
        result.current.register(nonCustomizable, handler);
      });

      const updated = act(() => {
        return result.current.update(DEFAULT_SHORTCUTS[0].id, {
          keys: ['t'],
        });
      });

      expect(updated).toBe(false);
    });

    it('should enable/disable shortcuts', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      act(() => {
        result.current.setEnabled(DEFAULT_SHORTCUTS[0].id, false);
      });

      const shortcut = result.current.getShortcut(DEFAULT_SHORTCUTS[0].id);
      expect(shortcut?.disabled).toBe(true);

      act(() => {
        result.current.setEnabled(DEFAULT_SHORTCUTS[0].id, true);
      });

      expect(result.current.getShortcut(DEFAULT_SHORTCUTS[0].id)?.disabled).toBe(false);
    });

    it('should reset to default', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
        result.current.update(DEFAULT_SHORTCUTS[0].id, { keys: ['x'] });
      });

      act(() => {
        result.current.resetToDefault(DEFAULT_SHORTCUTS[0].id);
      });

      const shortcut = result.current.getShortcut(DEFAULT_SHORTCUTS[0].id);
      expect(shortcut?.keys).toEqual(DEFAULT_SHORTCUTS[0].keys);
    });

    it('should detect conflicts', () => {
      const onConflictDetected = jest.fn();
      const { result } = renderHook(() => useKeyboardShortcuts({
        onConflictDetected,
      }));
      const handler = jest.fn();

      // Register two shortcuts with the same combination
      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
        result.current.register({
          ...DEFAULT_SHORTCUTS[1],
          keys: DEFAULT_SHORTCUTS[0].keys,
          modifiers: DEFAULT_SHORTCUTS[0].modifiers,
        }, handler);
      });

      const conflicts = result.current.detectConflicts();

      expect(conflicts.length).toBeGreaterThan(0);
      expect(onConflictDetected).toHaveBeenCalled();
    });

    it('should get shortcut by combination', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      const found = result.current.getByCombination('ctrl+s');
      expect(found).toBeDefined();
      expect(found?.id).toBe(DEFAULT_SHORTCUTS[0].id);
    });

    it('should persist to localStorage', () => {
      const { result } = renderHook(() => useKeyboardShortcuts());
      const handler = jest.fn();

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      act(() => {
        result.current.saveToStorage();
      });

      expect(localStorage.getItem('bi_ide_shortcuts')).toBeTruthy();
    });

    it('should load from localStorage', () => {
      const savedShortcuts = {
        'file.save': { keys: ['t'], modifiers: ['ctrl'], disabled: false },
      };
      localStorage.setItem('bi_ide_shortcuts', JSON.stringify(savedShortcuts));

      const { result } = renderHook(() => useKeyboardShortcuts({
        defaultShortcuts: DEFAULT_SHORTCUTS,
      }));

      act(() => {
        result.current.loadFromStorage();
      });

      const shortcut = result.current.getShortcut('file.save');
      expect(shortcut?.keys).toEqual(['t']);
    });

    it('should handle keyboard events', () => {
      const handler = jest.fn();
      const { result } = renderHook(() => useKeyboardShortcuts({
        enabled: true,
        ignoreInputs: true,
      }));

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      // Simulate keydown
      const event = new KeyboardEvent('keydown', {
        key: 's',
        ctrlKey: true,
      });

      window.dispatchEvent(event);

      expect(handler).toHaveBeenCalled();
    });

    it('should not trigger disabled shortcuts', () => {
      const handler = jest.fn();
      const { result } = renderHook(() => useKeyboardShortcuts({
        enabled: true,
      }));

      act(() => {
        result.current.register({ ...DEFAULT_SHORTCUTS[0], disabled: true }, handler);
      });

      const event = new KeyboardEvent('keydown', {
        key: 's',
        ctrlKey: true,
      });

      window.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();
    });

    it('should ignore shortcuts when typing in inputs', () => {
      const handler = jest.fn();
      const { result } = renderHook(() => useKeyboardShortcuts({
        enabled: true,
        ignoreInputs: true,
      }));

      act(() => {
        result.current.register(DEFAULT_SHORTCUTS[0], handler);
      });

      // Create an input element
      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      const event = new KeyboardEvent('keydown', {
        key: 's',
        ctrlKey: true,
      });

      input.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });

    it('should provide all default shortcuts', () => {
      expect(DEFAULT_SHORTCUTS.length).toBeGreaterThan(0);
      
      // Check required shortcuts exist
      const requiredIds = ['file.save', 'file.open', 'file.new', 'view.commandPalette'];
      requiredIds.forEach(id => {
        expect(DEFAULT_SHORTCUTS.some(s => s.id === id)).toBe(true);
      });
    });
  });
});
