"""
Configuration module for SmartSchoolGo application.

This module provides comprehensive configuration management with:
- Pydantic-based validation and type checking
- Environment-specific configurations
- Dynamic updates and health checks
- Comprehensive validation for APIs and geographic data
"""

import os
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple, Union
from enum import Enum
import re
from urllib.parse import urlparse

from pydantic import (
    Field, 
    validator, 
    root_validator,
    AnyHttpUrl
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class BaseConfig(BaseSettings):
    """Base configuration with common settings."""
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        env="ENVIRONMENT",
        description="Application environment"
    )
    
    # Application
    app_name: str = Field(
        default="SmartSchoolGo",
        env="APP_NAME",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION",
        description="Application version"
    )
    
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    
    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Log message format"
    )
    
    # Database
    database_url: Optional[str] = Field(
        default=None,
        env="DATABASE_URL",
        description="PostgreSQL database connection URL"
    )
    
    database_pool_size: int = Field(
        default=10,
        env="DATABASE_POOL_SIZE",
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    
    database_pool_overflow: int = Field(
        default=20,
        env="DATABASE_POOL_OVERFLOW", 
        ge=0,
        le=100,
        description="Database connection pool overflow"
    )
    
    # Redis Cache
    redis_url: Optional[str] = Field(
        default=None,
        env="REDIS_URL",
        description="Redis connection URL"
    )
    
    redis_ttl: int = Field(
        default=3600,
        env="REDIS_TTL",
        ge=60,
        description="Default Redis cache TTL in seconds"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API server host"
    )
    
    api_port: int = Field(
        default=8000,
        env="API_PORT",
        ge=1,
        le=65535,
        description="API server port"
    )
    
    api_workers: int = Field(
        default=1,
        env="API_WORKERS",
        ge=1,
        le=32,
        description="Number of API workers"
    )
    
    # Security
    secret_key: Optional[str] = Field(
        default=None,
        env="SECRET_KEY",
        min_length=32,
        description="Application secret key"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        env="JWT_ALGORITHM",
        description="JWT signing algorithm"
    )
    
    jwt_expire_minutes: int = Field(
        default=30,
        env="JWT_EXPIRE_MINUTES",
        ge=1,
        description="JWT token expiration time in minutes"
    )
    
    # Geographic Bounds (Australia focus)
    min_latitude: float = Field(
        default=-44.0,
        env="MIN_LATITUDE",
        ge=-90.0,
        le=90.0,
        description="Minimum allowed latitude"
    )
    
    max_latitude: float = Field(
        default=-10.0,
        env="MAX_LATITUDE", 
        ge=-90.0,
        le=90.0,
        description="Maximum allowed latitude"
    )
    
    min_longitude: float = Field(
        default=113.0,
        env="MIN_LONGITUDE",
        ge=-180.0,
        le=180.0,
        description="Minimum allowed longitude"
    )
    
    max_longitude: float = Field(
        default=154.0,
        env="MAX_LONGITUDE",
        ge=-180.0,
        le=180.0,
        description="Maximum allowed longitude"
    )
    
    # External API Keys
    google_maps_api_key: Optional[str] = Field(
        default=None,
        env="GOOGLE_MAPS_API_KEY",
        description="Google Maps API key"
    )
    
    openstreetmap_api_key: Optional[str] = Field(
        default=None,
        env="OPENSTREETMAP_API_KEY",
        description="OpenStreetMap API key"
    )
    
    gtfs_realtime_url: Optional[AnyHttpUrl] = Field(
        default=None,
        env="GTFS_REALTIME_URL",
        description="GTFS Realtime feed URL"
    )
    
    # Health Check Configuration
    health_check_interval: int = Field(
        default=30,
        env="HEALTH_CHECK_INTERVAL",
        ge=5,
        description="Health check interval in seconds"
    )
    
    # Configuration Caching
    config_cache_ttl: int = Field(
        default=300,
        env="CONFIG_CACHE_TTL",
        ge=60,
        description="Configuration cache TTL in seconds"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        
    @validator("secret_key", pre=True)
    def validate_secret_key(cls, v, values):
        """Validate secret key format and strength."""
        if v is None:
            return v
        
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        
        # Check for basic complexity
        if v.isalnum() or v.islower() or v.isupper():
            raise ValueError(
                "Secret key should contain a mix of letters, numbers, and special characters"
            )
        
        return v
    
    @validator("google_maps_api_key", pre=True)
    def validate_google_maps_api_key(cls, v):
        """Validate Google Maps API key format."""
        if v is None:
            return v
        
        # Basic Google API key validation
        if not re.match(r'^[A-Za-z0-9_-]{39}$', v):
            raise ValueError("Invalid Google Maps API key format")
        
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_geographic_bounds(cls, values):
        """Validate geographic boundary consistency."""
        min_lat = values.get('min_latitude')
        max_lat = values.get('max_latitude') 
        min_lon = values.get('min_longitude')
        max_lon = values.get('max_longitude')
        
        if min_lat is not None and max_lat is not None:
            if min_lat >= max_lat:
                raise ValueError("min_latitude must be less than max_latitude")
        
        if min_lon is not None and max_lon is not None:
            if min_lon >= max_lon:
                raise ValueError("min_longitude must be less than max_longitude")
        
        return values
    
    @root_validator(skip_on_failure=True)
    def validate_required_for_production(cls, values):
        """Validate required fields for production environment."""
        env = values.get('environment')
        
        if env == Environment.PRODUCTION:
            required_fields = [
                'secret_key', 
                'database_url',
                'redis_url'
            ]
            
            missing_fields = []
            for field in required_fields:
                if not values.get(field):
                    missing_fields.append(field)
            
            if missing_fields:
                raise ValueError(
                    f"Missing required fields for production: {', '.join(missing_fields)}"
                )
        
        return values
    
    def is_geographic_point_valid(self, latitude: float, longitude: float) -> bool:
        """Check if a geographic point is within configured bounds."""
        return (
            self.min_latitude <= latitude <= self.max_latitude and
            self.min_longitude <= longitude <= self.max_longitude
        )
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        if not self.database_url:
            return {}
        
        return {
            "url": str(self.database_url),
            "pool_size": self.database_pool_size,
            "pool_overflow": self.database_pool_overflow,
            "echo": self.debug
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return {
            "level": self.log_level.value,
            "format": self.log_format,
            "handlers": ["console"] if self.environment != Environment.PRODUCTION else ["file"]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform configuration health check."""
        health = {
            "status": "healthy",
            "environment": self.environment.value,
            "version": self.app_version,
            "checks": {}
        }
        
        # Check database connectivity
        if self.database_url:
            try:
                parsed_url = urlparse(str(self.database_url))
                health["checks"]["database"] = {
                    "status": "configured",
                    "host": parsed_url.hostname,
                    "port": parsed_url.port
                }
            except Exception as e:
                health["checks"]["database"] = {
                    "status": "error",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        # Check Redis connectivity
        if self.redis_url:
            try:
                parsed_url = urlparse(str(self.redis_url))
                health["checks"]["redis"] = {
                    "status": "configured",
                    "host": parsed_url.hostname,
                    "port": parsed_url.port
                }
            except Exception as e:
                health["checks"]["redis"] = {
                    "status": "error", 
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        # Check API keys
        api_keys = {
            "google_maps": self.google_maps_api_key,
            "openstreetmap": self.openstreetmap_api_key
        }
        
        for key_name, key_value in api_keys.items():
            health["checks"][f"{key_name}_api_key"] = {
                "status": "configured" if key_value else "missing"
            }
            if not key_value and self.environment == Environment.PRODUCTION:
                health["status"] = "degraded"
        
        return health


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    
    # Development-specific defaults
    database_url: Optional[str] = Field(
        default="postgresql://user:pass@localhost:5432/smartschoolgo_dev",
        env="DATABASE_URL"
    )
    
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    secret_key: Optional[str] = Field(
        default="dev-secret-key-change-in-production-32chars-min",
        env="SECRET_KEY"
    )


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    
    environment: Environment = Environment.TESTING
    debug: bool = True
    log_level: LogLevel = LogLevel.WARNING
    
    # Testing-specific defaults
    database_url: Optional[str] = Field(
        default="postgresql://user:pass@localhost:5432/smartschoolgo_test",
        env="DATABASE_URL"
    )
    
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379/1",
        env="REDIS_URL"
    )
    
    secret_key: Optional[str] = Field(
        default="test-secret-key-for-testing-only-32chars",
        env="SECRET_KEY"
    )
    
    # Faster intervals for testing
    health_check_interval: int = 5
    config_cache_ttl: int = 60


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # Production requires all sensitive values from environment
    # No defaults for security-critical settings


# Configuration mapping
CONFIG_MAPPING = {
    Environment.DEVELOPMENT: DevelopmentConfig,
    Environment.TESTING: TestingConfig,
    Environment.PRODUCTION: ProductionConfig,
}


# Configuration cache
_config_cache: Optional[BaseConfig] = None
_cache_timestamp: Optional[float] = None


def _load_environment_file() -> None:
    """Load environment variables from .env file."""
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)


def _get_environment() -> Environment:
    """Get current environment from environment variable."""
    env_str = os.getenv("ENVIRONMENT", "development").lower()
    
    try:
        return Environment(env_str)
    except ValueError:
        logging.warning(f"Invalid environment '{env_str}', defaulting to development")
        return Environment.DEVELOPMENT


def _create_config(environment: Optional[Environment] = None) -> BaseConfig:
    """Create configuration instance for specified environment."""
    if environment is None:
        environment = _get_environment()
    
    config_class = CONFIG_MAPPING.get(environment, DevelopmentConfig)
    
    try:
        return config_class()
    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration: {e}")


@lru_cache(maxsize=1)
def get_settings(environment: Optional[Environment] = None, force_reload: bool = False) -> BaseConfig:
    """
    Get application settings with caching.
    
    Args:
        environment: Specific environment to load (optional)
        force_reload: Force reload configuration ignoring cache
        
    Returns:
        BaseConfig: Configuration instance
        
    Raises:
        ConfigurationError: If configuration validation fails
    """
    global _config_cache, _cache_timestamp
    
    # Load environment file first
    _load_environment_file()
    
    current_time = os.path.getmtime(os.path.join(os.getcwd(), ".env")) if os.path.exists(".env") else 0
    
    # Check if cache is valid
    if (
        not force_reload and 
        _config_cache is not None and 
        _cache_timestamp is not None and
        current_time <= _cache_timestamp + _config_cache.config_cache_ttl
    ):
        return _config_cache
    
    # Create new configuration
    try:
        config = _create_config(environment)
        _config_cache = config
        _cache_timestamp = current_time
        
        # Setup logging based on configuration
        logging.basicConfig(
            level=getattr(logging, config.log_level.value),
            format=config.log_format
        )
        
        logging.info(f"Configuration loaded for {config.environment.value} environment")
        return config
        
    except Exception as e:
        logging.error(f"Configuration loading failed: {e}")
        raise ConfigurationError(f"Configuration loading failed: {e}")


def update_config_cache(new_config: BaseConfig) -> None:
    """Update configuration cache with new instance."""
    global _config_cache, _cache_timestamp
    
    _config_cache = new_config
    _cache_timestamp = os.path.getmtime(os.path.join(os.getcwd(), ".env")) if os.path.exists(".env") else 0
    
    logging.info("Configuration cache updated")


def clear_config_cache() -> None:
    """Clear configuration cache."""
    global _config_cache, _cache_timestamp
    
    _config_cache = None
    _cache_timestamp = None
    get_settings.cache_clear()
    
    logging.info("Configuration cache cleared")


def validate_config(config: Optional[BaseConfig] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate configuration and return health status.
    
    Args:
        config: Configuration to validate (uses current if None)
        
    Returns:
        Tuple[bool, Dict]: (is_valid, health_report)
    """
    if config is None:
        try:
            config = get_settings()
        except ConfigurationError as e:
            return False, {"error": str(e), "status": "invalid"}
    
    try:
        health_report = config.health_check()
        is_healthy = health_report["status"] in ["healthy", "degraded"]
        return is_healthy, health_report
    except Exception as e:
        return False, {"error": str(e), "status": "validation_failed"}


# Export commonly used functions and classes
__all__ = [
    "BaseConfig",
    "DevelopmentConfig", 
    "TestingConfig",
    "ProductionConfig",
    "Environment",
    "LogLevel",
    "ConfigurationError",
    "get_settings",
    "update_config_cache",
    "clear_config_cache", 
    "validate_config"
]