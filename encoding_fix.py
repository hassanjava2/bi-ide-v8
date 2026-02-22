"""
Encoding Fix - حل مشكلة Unicode على Windows
يُستورد في بداية كل ملف Python قبل أي إيموجي
"""
import sys
import io
import os

# Skip encoding fix when running under pytest (breaks capture)
_PYTEST_RUNNING = (
    "pytest" in sys.modules
    or "_pytest" in sys.modules
    or "PYTEST_RUNNING" in os.environ
    or any("pytest" in arg for arg in sys.argv[:1])
    or (hasattr(sys, '_called_from_test'))
)

# Prevent double-wrapping
_ALREADY_FIXED = getattr(sys, '_encoding_fix_applied', False)

# فقط على Windows وليس أثناء الاختبارات
if sys.platform == 'win32' and not _PYTEST_RUNNING and not _ALREADY_FIXED:
    # Mark as applied to prevent double-wrapping
    sys._encoding_fix_applied = True
    
    # تعيين متغيرات البيئة أولاً
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # تغيير ترميز الـ Console إلى UTF-8
    try:
        # Only redirect if stdout is a real terminal and not already UTF-8
        if (hasattr(sys.stdout, 'buffer') 
                and hasattr(sys.stdout, 'encoding')
                and sys.stdout.encoding != 'utf-8'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
        if (hasattr(sys.stderr, 'buffer')
                and hasattr(sys.stderr, 'encoding')
                and sys.stderr.encoding != 'utf-8'):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
    except Exception:
        pass
    
    # محاولة تغيير Code Page في Windows Console
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)  # UTF-8
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass

def safe_print(text: str, **kwargs):
    """طباعة آمنة تعمل على جميع الأنظمة"""
    try:
        # Try printing normally first
        print(text, **kwargs)
    except (UnicodeEncodeError, ValueError):
        # If that fails, try encoding to ascii
        try:
            cleaned = text.encode('ascii', 'ignore').decode('ascii')
            print(cleaned, **kwargs)
        except Exception:
            # Last resort: ignore all non-ascii
            cleaned = ''.join(c for c in text if ord(c) < 128)
            print(cleaned, **kwargs)
