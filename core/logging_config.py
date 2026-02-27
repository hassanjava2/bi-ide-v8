"""
Logging Configuration - إعدادات التسجيل المركزية
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from functools import wraps

# Ensure UTF-8 encoding on Windows (but NOT during pytest)
if sys.platform == 'win32' and 'PYTEST_RUNNING' not in os.environ:
    import io
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '') != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '') != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Only colorize if output is a terminal
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """Setup centralized logging"""
    
    logger = logging.getLogger("bi_ide")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            console_handler.setFormatter(ColoredFormatter(fmt, datefmt='%H:%M:%S'))
        
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Regular log file
        file_handler = logging.FileHandler(
            LOG_DIR / f"bi_ide_{datetime.now():%Y%m%d}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        
        # JSON log file for structured logging
        json_handler = logging.FileHandler(
            LOG_DIR / f"bi_ide_{datetime.now():%Y%m%d}.json",
            encoding='utf-8'
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


def log_execution_time(func):
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now(timezone.utc)
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.debug(
                f"Function {func.__name__} executed in {duration:.3f}s",
                extra={'extra_data': {'duration': duration, 'function': func.__name__}}
            )
            return result
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.error(
                f"Function {func.__name__} failed after {duration:.3f}s: {e}",
                exc_info=True,
                extra={'extra_data': {'duration': duration, 'function': func.__name__}}
            )
            raise
    return wrapper


def log_ai_decision(decision_type: str, context: Dict[str, Any], result: Any):
    """Log AI decisions for auditing"""
    logger.info(
        f"AI Decision: {decision_type}",
        extra={'extra_data': {
            'decision_type': decision_type,
            'context': context,
            'result': result
        }}
    )
