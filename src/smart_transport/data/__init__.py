"""
Smart Transport Data Module

This module provides comprehensive data fetching capabilities for the SmartSchoolGo application,
including APIs for transportation, geographic, weather, and school data.

Usage:
    from smart_transport.data import ACTTransportAPI, OpenStreetMapAPI
    
    # Initialize API clients
    async with ACTTransportAPI(api_key="your_key") as transport_api:
        stops = await transport_api.fetch_stops()
        routes = await transport_api.fetch_routes()
    
    # Geocoding
    async with OpenStreetMapAPI() as geo_api:
        result = await geo_api.geocode_address("123 Main St, Canberra ACT")
        if result:
            print(f"Coordinates: {result.latitude}, {result.longitude}")

Features:
- Async/await support for concurrent operations
- Exponential backoff retry logic
- Rate limiting with time-based throttling
- Redis caching with configurable TTL
- Pydantic data validation
- Comprehensive error handling and logging
- Bulk operations with progress tracking
"""

from .fetcher import (
    # Exception classes
    APIError,
    RateLimitError,
    ValidationError,
    CacheError,
    
    # Utility classes
    RateLimiter,
    CacheManager,
    BaseAPIClient,
    
    # API clients
    ACTTransportAPI,
    OpenStreetMapAPI,
    WeatherAPI,
    SchoolDataAPI
)

from .models import (
    # Base classes
    BaseDataModel,
    
    # Geographic models
    Coordinate, 
    BoundingBox, 
    RouteGeometry,
    
    # GTFS models
    GTFSRouteType, 
    GTFSWheelchairAccessible, 
    GTFSPickupDropoffType,
    Route, 
    Stop, 
    Trip, 
    StopTime, 
    Vehicle,
    
    # School models
    SchoolType, 
    SchoolSector, 
    School, 
    Student, 
    Enrollment, 
    Catchment,
    
    # Transport network models
    NetworkNodeType, 
    NetworkEdgeType, 
    NetworkNode, 
    NetworkEdge, 
    TransportGraph,
    
    # Real-time models
    VehiclePositionStatus, 
    OccupancyStatus, 
    VehiclePosition,
    AlertCause, 
    AlertEffect, 
    Alert, 
    ServiceUpdate,
    
    # Analysis models
    OptimizationObjective, 
    RouteOptimization,
    SafetyRiskLevel, 
    SafetyAnalysis,
    DemandForecast,
    
    # Utility models
    GeocodeResult,
    WeatherData,
    ProgressInfo,
    
    # Utility functions
    export_to_csv, 
    export_to_json
)

from .processor import (
    # Exception classes
    ProcessingError, 
    ValidationError, 
    GeospatialError, 
    NetworkError,
    
    # Quality and validation classes
    QualityLevel, 
    ValidationResult, 
    ProcessingStats,
    
    # Processor classes
    BaseProcessor, 
    GTFSProcessor, 
    GeospatialProcessor,
    StudentDataProcessor, 
    NetworkProcessor, 
    TimeSeriesProcessor,
    
    # Export functions
    export_to_geojson, 
    export_network_to_graphml
)

from .database import (
    # Database exception classes
    DatabaseError,
    ConnectionError as DatabaseConnectionError,
    MigrationError,
    BackupError,
    
    # Database configuration and metrics
    OperationMode,
    DatabaseConfig,
    PerformanceMetrics,
    
    # SQLAlchemy table definitions
    SchoolTable,
    StudentTable,
    EnrollmentTable,
    CatchmentTable,
    RouteTable,
    StopTable,
    TripTable,
    StopTimeTable,
    VehiclePositionTable,
    AlertTable,
    NetworkNodeTable,
    NetworkEdgeTable,
    
    # Database manager
    DatabaseManager
)

__version__ = "1.0.0"
__author__ = "SmartSchoolGo Team"

__all__ = [
    # Exception classes
    "APIError",
    "RateLimitError", 
    "ValidationError",
    "CacheError",
    
    # Base classes
    "BaseDataModel",
    
    # Geographic models
    "Coordinate", 
    "BoundingBox", 
    "RouteGeometry",
    
    # GTFS models
    "GTFSRouteType", 
    "GTFSWheelchairAccessible", 
    "GTFSPickupDropoffType",
    "Route", 
    "Stop", 
    "Trip", 
    "StopTime", 
    "Vehicle",
    
    # School models
    "SchoolType", 
    "SchoolSector", 
    "School", 
    "Student", 
    "Enrollment", 
    "Catchment",
    
    # Transport network models
    "NetworkNodeType", 
    "NetworkEdgeType", 
    "NetworkNode", 
    "NetworkEdge", 
    "TransportGraph",
    
    # Real-time models
    "VehiclePositionStatus", 
    "OccupancyStatus", 
    "VehiclePosition",
    "AlertCause", 
    "AlertEffect", 
    "Alert", 
    "ServiceUpdate",
    
    # Analysis models
    "OptimizationObjective", 
    "RouteOptimization",
    "SafetyRiskLevel", 
    "SafetyAnalysis",
    "DemandForecast",
    
    # Utility models
    "GeocodeResult",
    "WeatherData",
    "ProgressInfo",
    
    # Utility classes
    "RateLimiter",
    "CacheManager", 
    "BaseAPIClient",
    
    # API clients
    "ACTTransportAPI",
    "OpenStreetMapAPI",
    "WeatherAPI", 
    "SchoolDataAPI",
    
    # Utility functions
    "export_to_csv", 
    "export_to_json",
    
    # Processing exception classes
    "ProcessingError", 
    "ValidationError", 
    "GeospatialError", 
    "NetworkError",
    
    # Quality and validation classes
    "QualityLevel", 
    "ValidationResult", 
    "ProcessingStats",
    
    # Processor classes
    "BaseProcessor", 
    "GTFSProcessor", 
    "GeospatialProcessor",
    "StudentDataProcessor", 
    "NetworkProcessor", 
    "TimeSeriesProcessor",
    
    # Processing export functions
    "export_to_geojson", 
    "export_network_to_graphml",
    
    # Database exception classes
    "DatabaseError",
    "DatabaseConnectionError",
    "MigrationError",
    "BackupError",
    
    # Database configuration and metrics
    "OperationMode",
    "DatabaseConfig",
    "PerformanceMetrics",
    
    # SQLAlchemy table definitions
    "SchoolTable",
    "StudentTable",
    "EnrollmentTable",
    "CatchmentTable",
    "RouteTable",
    "StopTable",
    "TripTable",
    "StopTimeTable",
    "VehiclePositionTable",
    "AlertTable",
    "NetworkNodeTable",
    "NetworkEdgeTable",
    
    # Database manager
    "DatabaseManager"
]