"""
Configuration Management - إدارة الإعدادات
"""
import os
import secrets
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # App
    APP_NAME: str = "BI IDE"
    APP_VERSION: str = "8.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/bi_ide.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # Redis/Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600
    
    # AI/ML
    RTX4090_HOST: str = "192.168.68.125"
    RTX4090_PORT: int = 8080
    
    # RTX 4090 Retry Configuration - إعدادات إعادة المحاولة
    RTX4090_MAX_RETRIES: int = 3           # Maximum retry attempts for connection failures
    RTX4090_RETRY_DELAY: float = 1.0       # Initial delay between retries (seconds)
    RTX4090_RETRY_BACKOFF: float = 2.0     # Exponential backoff multiplier
    RTX4090_RETRY_MAX_DELAY: float = 30.0  # Maximum delay between retries (seconds)
    RTX4090_RETRY_JITTER: bool = True      # Add random jitter to prevent thundering herd
    
    AI_CORE_HOST: Optional[str] = None
    AI_CORE_PORTS: str = "8080"
    
    # Security
    SECRET_KEY: str = ""
    ADMIN_PASSWORD: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    API_KEY_HEADER: str = "X-API-Key"
    
    # Paths
    DATA_DIR: str = "./data"
    LEARNING_DATA_DIR: str = "./learning_data"
    LOGS_DIR: str = "./logs"
    
    # Features
    ENABLE_SMART_COUNCIL: bool = True
    ENABLE_AUTONOMOUS_LEARNING: bool = True
    ENABLE_COUNCIL_DISCUSSIONS: bool = True
    ENABLE_CODE_ANALYSIS: bool = True
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_ENABLED: bool = True
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    @property
    def ai_core_urls(self) -> List[str]:
        """Get list of AI core base URLs"""
        if not self.AI_CORE_HOST:
            return []
        ports = [p.strip() for p in self.AI_CORE_PORTS.split(",") if p.strip()]
        return [f"http://{self.AI_CORE_HOST}:{port}" for port in ports]
    
    @property
    def rtx4090_url(self) -> str:
        """Get RTX 4090 server URL"""
        return f"http://{self.RTX4090_HOST}:{self.RTX4090_PORT}"
    
    def ensure_directories(self):
        """Create necessary directories"""
        import os
        for dir_path in [self.DATA_DIR, self.LEARNING_DATA_DIR, self.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()

    insecure_secret_values = {
        "",
        "your-secret-key-change-this",
        "bi-ide-v8-change-this-in-production-2026",
        "CHANGE_ME_IN_PRODUCTION_USE_SECRETS_TOKEN",
    }
    insecure_admin_passwords = {
        "",
        "president123",
        "CHANGE_THIS_PASSWORD",
    }

    is_production = settings.ENVIRONMENT.lower() == "production"

    if not is_production:
        if settings.SECRET_KEY in insecure_secret_values:
            settings.SECRET_KEY = secrets.token_urlsafe(64)
            print("⚠️ Development mode: generated ephemeral SECRET_KEY")
        if settings.ADMIN_PASSWORD in insecure_admin_passwords:
            settings.ADMIN_PASSWORD = "president123"
            print("⚠️ Development mode: using default ADMIN_PASSWORD")
    else:
        if settings.SECRET_KEY in insecure_secret_values:
            raise ValueError("SECRET_KEY is missing or insecure. Set a strong SECRET_KEY in .env")
        if settings.ADMIN_PASSWORD in insecure_admin_passwords:
            raise ValueError("ADMIN_PASSWORD is missing or insecure. Set a strong ADMIN_PASSWORD in .env")

    settings.ensure_directories()
    return settings


# Global settings instance
settings = get_settings()
