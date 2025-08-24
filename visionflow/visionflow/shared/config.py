"""Configuration management for VisionFlow services."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    @property
    def host(self) -> str:
        """Get database host from environment."""
        return os.getenv("DB_HOST", "localhost")
        
    @property  
    def port(self) -> int:
        """Get database port from environment."""
        return int(os.getenv("DB_PORT", "5432"))
        
    @property
    def name(self) -> str:
        """Get database name from environment."""
        return os.getenv("DB_NAME", "visionflow")
        
    @property
    def user(self) -> str:
        """Get database user from environment."""
        return os.getenv("DB_USER", "visionflow")
        
    @property
    def password(self) -> str:
        """Get database password from environment."""
        return os.getenv("DB_PASSWORD", "visionflow")
    
    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    @property
    def host(self) -> str:
        return os.getenv("REDIS_HOST", "localhost")
        
    @property
    def port(self) -> int:
        return int(os.getenv("REDIS_PORT", "6379"))
        
    @property
    def db(self) -> int:
        return int(os.getenv("REDIS_DB", "0"))
        
    @property
    def password(self) -> str:
        return os.getenv("REDIS_PASSWORD", "")
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class StorageSettings(BaseSettings):
    """Object storage configuration."""
    
    endpoint: str = Field(default="localhost:9000", env="STORAGE_ENDPOINT")
    access_key: str = Field(default="minio", env="STORAGE_ACCESS_KEY")
    secret_key: str = Field(default="minio123", env="STORAGE_SECRET_KEY")
    bucket_name: str = Field(default="visionflow", env="STORAGE_BUCKET")
    secure: bool = Field(default=False, env="STORAGE_SECURE")

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="allow")


class ModelSettings(BaseSettings):
    """Model configuration."""
    
    wan_model_path: str = Field(default="multimodalart/wan2-1-fast", env="WAN_MODEL_PATH")
    cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    device: str = Field(default="auto", env="MODEL_DEVICE")
    max_memory_gb: int = Field(default=8, env="MAX_MEMORY_GB")
    enable_cpu_offload: bool = Field(default=True, env="ENABLE_CPU_OFFLOAD")
    enable_xformers: bool = Field(default=True, env="ENABLE_XFORMERS")
    max_duration: int = Field(default=30, env="MAX_DURATION")
    default_fps: int = Field(default=24, env="DEFAULT_FPS")
    default_resolution: str = Field(default="512x512", env="DEFAULT_RESOLUTION")

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="allow")


class APISettings(BaseSettings):
    """API configuration."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # Circuit breaker settings
    failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    recovery_timeout: int = Field(default=30, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="allow")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # GCP configuration
    vertex_ai_project: str = Field(default="visionflow-gcp-project", env="VERTEX_AI_PROJECT")
    vertex_ai_region: str = Field(default="us-central1", env="VERTEX_AI_REGION")

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="allow")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Service settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    storage: StorageSettings = StorageSettings()
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # Concurrency settings
    max_concurrent_generations: int = Field(default=2, env="MAX_CONCURRENT_GENERATIONS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")
    
    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
