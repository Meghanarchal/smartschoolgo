"""
Configuration package for SmartSchoolGo application.

This package provides comprehensive configuration management with pydantic-based
validation, environment-specific settings, and dynamic configuration updates.

Usage:
    from config import get_settings
    
    settings = get_settings()
    print(f"Running in {settings.environment} mode")
    
    # Health check
    is_valid, health = validate_config()
    
    # Force reload configuration
    settings = get_settings(force_reload=True)
"""

from .settings import (
    BaseConfig,
    DevelopmentConfig,
    TestingConfig,
    ProductionConfig,
    Environment,
    LogLevel,
    ConfigurationError,
    get_settings,
    update_config_cache,
    clear_config_cache,
    validate_config
)

__version__ = "1.0.0"

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