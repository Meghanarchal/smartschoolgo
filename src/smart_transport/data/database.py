"""
Database management module for SmartSchoolGo application.

This module provides comprehensive database management capabilities including:
- SQLAlchemy ORM operations and table definitions
- Migration scripts and schema management
- Bulk data operations with performance optimization
- Spatial queries using PostGIS extensions
- Connection pooling and transaction management
- Backup, recovery, and maintenance procedures
- Data synchronization and archival policies

Features:
- Efficient bulk data insertion with batch processing
- Complex spatial queries for geographic analysis
- Time-series data optimization
- Performance monitoring and health checks
- Automated backup and recovery procedures
"""

import asyncio
import json
import csv
import gzip
import shutil
import logging
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Optional, Any, Union, Tuple, Type, Iterator
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
import subprocess

import sqlalchemy as sa
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float, Boolean,
    DateTime, Date, Time, Text, ForeignKey, Index, CheckConstraint,
    UniqueConstraint, event, func, select, insert, update, delete,
    and_, or_, not_, case, cast, extract, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    sessionmaker, relationship, Session, Query, selectinload, joinedload,
    declarative_base, mapped_column, Mapped
)
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.engine import Engine
from sqlalchemy.sql import func as sql_func
import alembic
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.operations import Operations
import asyncpg
import asyncio
import structlog
from geoalchemy2 import Geometry, Geography
from geoalchemy2.functions import ST_Distance, ST_Within, ST_DWithin, ST_Area, ST_Buffer
import pandas as pd

from config import get_settings
from .models import (
    Coordinate, BoundingBox, RouteGeometry,
    Route, Stop, Trip, StopTime, Vehicle,
    School, Student, Enrollment, Catchment,
    NetworkNode, NetworkEdge, TransportGraph,
    VehiclePosition, Alert, ServiceUpdate,
    RouteOptimization, SafetyAnalysis, DemandForecast
)

# Configure logging
logger = structlog.get_logger(__name__)

# SQLAlchemy base
Base = declarative_base()


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class MigrationError(DatabaseError):
    """Raised when database migration fails."""
    pass


class BackupError(DatabaseError):
    """Raised when backup operations fail."""
    pass


class OperationMode(Enum):
    """Database operation modes."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    MAINTENANCE = "maintenance"


@dataclass
class DatabaseConfig:
    """Database configuration parameters."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    enable_logging: bool = True
    backup_retention_days: int = 30
    archive_retention_months: int = 12


@dataclass
class PerformanceMetrics:
    """Database performance metrics."""
    connection_pool_size: int
    active_connections: int
    total_queries: int
    slow_queries: int
    avg_query_time: float
    cache_hit_ratio: float
    table_sizes: Dict[str, int]
    index_usage: Dict[str, Dict[str, Any]]


# ==================== TABLE DEFINITIONS ====================

class SchoolTable(Base):
    """SQLAlchemy table for schools."""
    __tablename__ = 'schools'
    
    school_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    address = Column(Text, nullable=False)
    suburb = Column(String(100), nullable=False, index=True)
    postcode = Column(String(4), nullable=False, index=True)
    
    # PostGIS geometry column for spatial queries
    location = Column(Geography('POINT', srid=4326), nullable=False, index=True)
    
    school_type = Column(String(20), nullable=False, index=True)
    sector = Column(String(20), nullable=False, index=True)
    grades = Column(ARRAY(String), nullable=True)
    enrollment = Column(Integer, nullable=True)
    capacity = Column(Integer, nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    website = Column(String(500), nullable=True)
    principal = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    students = relationship("StudentTable", back_populates="school")
    catchments = relationship("CatchmentTable", back_populates="school")
    enrollments = relationship("EnrollmentTable", back_populates="school")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('enrollment >= 0', name='check_enrollment_positive'),
        CheckConstraint('capacity >= 0', name='check_capacity_positive'),
        CheckConstraint("school_type IN ('primary', 'secondary', 'combined', 'special')", name='check_school_type'),
        CheckConstraint("sector IN ('government', 'catholic', 'independent')", name='check_sector'),
        Index('idx_schools_location', 'location', postgresql_using='gist'),
        Index('idx_schools_type_sector', 'school_type', 'sector'),
    )


class StudentTable(Base):
    """SQLAlchemy table for students."""
    __tablename__ = 'students'
    
    student_id = Column(String(50), primary_key=True, index=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False, index=True)
    date_of_birth = Column(Date, nullable=False)
    grade = Column(String(10), nullable=False, index=True)
    home_address = Column(Text, nullable=False)
    
    # PostGIS geometry for home location
    home_location = Column(Geography('POINT', srid=4326), nullable=False, index=True)
    
    school_id = Column(String(50), ForeignKey('schools.school_id'), nullable=False, index=True)
    transport_eligible = Column(Boolean, default=True, nullable=False)
    special_needs = Column(Boolean, default=False, nullable=False, index=True)
    guardian_phone = Column(String(20), nullable=True)
    guardian_email = Column(String(255), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    school = relationship("SchoolTable", back_populates="students")
    enrollments = relationship("EnrollmentTable", back_populates="student")
    
    # Constraints
    __table_args__ = (
        Index('idx_students_home_location', 'home_location', postgresql_using='gist'),
        Index('idx_students_school_grade', 'school_id', 'grade'),
        Index('idx_students_transport', 'transport_eligible', 'special_needs'),
    )


class EnrollmentTable(Base):
    """SQLAlchemy table for student enrollments."""
    __tablename__ = 'enrollments'
    
    enrollment_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    student_id = Column(String(50), ForeignKey('students.student_id'), nullable=False, index=True)
    school_id = Column(String(50), ForeignKey('schools.school_id'), nullable=False, index=True)
    enrollment_date = Column(Date, nullable=False, default=date.today)
    grade_at_enrollment = Column(String(10), nullable=False)
    status = Column(String(20), default='active', nullable=False, index=True)
    distance_to_school_km = Column(Float, nullable=False)
    travel_time_minutes = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    student = relationship("StudentTable", back_populates="enrollments")
    school = relationship("SchoolTable", back_populates="enrollments")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('distance_to_school_km >= 0', name='check_distance_positive'),
        CheckConstraint('travel_time_minutes >= 0', name='check_travel_time_positive'),
        CheckConstraint("status IN ('active', 'inactive', 'graduated', 'transferred')", name='check_enrollment_status'),
        UniqueConstraint('student_id', 'school_id', 'enrollment_date', name='unique_student_school_enrollment'),
        Index('idx_enrollments_date_status', 'enrollment_date', 'status'),
    )


class CatchmentTable(Base):
    """SQLAlchemy table for school catchment areas."""
    __tablename__ = 'catchments'
    
    catchment_id = Column(String(50), primary_key=True, index=True)
    school_id = Column(String(50), ForeignKey('schools.school_id'), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    
    # PostGIS polygon for catchment boundary
    boundary = Column(Geography('POLYGON', srid=4326), nullable=False, index=True)
    
    priority_level = Column(Integer, default=1, nullable=False)
    transport_provided = Column(Boolean, default=True, nullable=False)
    max_walking_distance_km = Column(Float, default=1.6, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    school = relationship("SchoolTable", back_populates="catchments")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('priority_level >= 1 AND priority_level <= 5', name='check_priority_level'),
        CheckConstraint('max_walking_distance_km >= 0', name='check_walking_distance_positive'),
        Index('idx_catchments_boundary', 'boundary', postgresql_using='gist'),
        Index('idx_catchments_priority', 'school_id', 'priority_level'),
    )


class RouteTable(Base):
    """SQLAlchemy table for GTFS routes."""
    __tablename__ = 'routes'
    
    route_id = Column(String(255), primary_key=True, index=True)
    agency_id = Column(String(255), nullable=True)
    route_short_name = Column(String(50), nullable=True, index=True)
    route_long_name = Column(String(255), nullable=True, index=True)
    route_desc = Column(Text, nullable=True)
    route_type = Column(Integer, nullable=False, index=True)
    route_url = Column(String(500), nullable=True)
    route_color = Column(String(6), default='FFFFFF')
    route_text_color = Column(String(6), default='000000')
    route_sort_order = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    trips = relationship("TripTable", back_populates="route")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('route_type >= 0 AND route_type <= 12', name='check_route_type'),
        CheckConstraint('route_sort_order >= 0', name='check_sort_order_positive'),
        Index('idx_routes_type_name', 'route_type', 'route_short_name'),
    )


class StopTable(Base):
    """SQLAlchemy table for GTFS stops."""
    __tablename__ = 'stops'
    
    stop_id = Column(String(255), primary_key=True, index=True)
    stop_code = Column(String(50), nullable=True, index=True)
    stop_name = Column(String(255), nullable=False, index=True)
    stop_desc = Column(Text, nullable=True)
    
    # PostGIS geometry for stop location
    location = Column(Geography('POINT', srid=4326), nullable=False, index=True)
    
    zone_id = Column(String(50), nullable=True, index=True)
    stop_url = Column(String(500), nullable=True)
    location_type = Column(Integer, default=0, nullable=False)
    parent_station = Column(String(255), nullable=True, index=True)
    stop_timezone = Column(String(50), nullable=True)
    wheelchair_boarding = Column(Integer, default=0, nullable=False)
    level_id = Column(String(255), nullable=True)
    platform_code = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    stop_times = relationship("StopTimeTable", back_populates="stop")
    vehicle_positions = relationship("VehiclePositionTable", back_populates="stop")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('location_type >= 0 AND location_type <= 4', name='check_location_type'),
        CheckConstraint('wheelchair_boarding >= 0 AND wheelchair_boarding <= 2', name='check_wheelchair_boarding'),
        Index('idx_stops_location', 'location', postgresql_using='gist'),
        Index('idx_stops_zone_type', 'zone_id', 'location_type'),
    )


class TripTable(Base):
    """SQLAlchemy table for GTFS trips."""
    __tablename__ = 'trips'
    
    trip_id = Column(String(255), primary_key=True, index=True)
    route_id = Column(String(255), ForeignKey('routes.route_id'), nullable=False, index=True)
    service_id = Column(String(255), nullable=False, index=True)
    trip_headsign = Column(String(255), nullable=True)
    trip_short_name = Column(String(50), nullable=True)
    direction_id = Column(Integer, nullable=True)
    block_id = Column(String(255), nullable=True, index=True)
    shape_id = Column(String(255), nullable=True, index=True)
    wheelchair_accessible = Column(Integer, default=0, nullable=False)
    bikes_allowed = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    route = relationship("RouteTable", back_populates="trips")
    stop_times = relationship("StopTimeTable", back_populates="trip")
    vehicle_positions = relationship("VehiclePositionTable", back_populates="trip")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('direction_id IS NULL OR direction_id IN (0, 1)', name='check_direction_id'),
        CheckConstraint('wheelchair_accessible >= 0 AND wheelchair_accessible <= 2', name='check_wheelchair_accessible'),
        CheckConstraint('bikes_allowed IS NULL OR bikes_allowed IN (0, 1, 2)', name='check_bikes_allowed'),
        Index('idx_trips_route_service', 'route_id', 'service_id'),
    )


class StopTimeTable(Base):
    """SQLAlchemy table for GTFS stop times."""
    __tablename__ = 'stop_times'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trip_id = Column(String(255), ForeignKey('trips.trip_id'), nullable=False, index=True)
    arrival_time = Column(Time, nullable=True)
    departure_time = Column(Time, nullable=True)
    stop_id = Column(String(255), ForeignKey('stops.stop_id'), nullable=False, index=True)
    stop_sequence = Column(Integer, nullable=False)
    stop_headsign = Column(String(255), nullable=True)
    pickup_type = Column(Integer, default=0, nullable=False)
    drop_off_type = Column(Integer, default=0, nullable=False)
    continuous_pickup = Column(Integer, nullable=True)
    continuous_drop_off = Column(Integer, nullable=True)
    shape_dist_traveled = Column(Float, nullable=True)
    timepoint = Column(Integer, default=1, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    trip = relationship("TripTable", back_populates="stop_times")
    stop = relationship("StopTable", back_populates="stop_times")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('stop_sequence >= 0', name='check_stop_sequence_positive'),
        CheckConstraint('pickup_type >= 0 AND pickup_type <= 3', name='check_pickup_type'),
        CheckConstraint('drop_off_type >= 0 AND drop_off_type <= 3', name='check_drop_off_type'),
        CheckConstraint('timepoint IN (0, 1)', name='check_timepoint'),
        UniqueConstraint('trip_id', 'stop_sequence', name='unique_trip_stop_sequence'),
        Index('idx_stop_times_trip_sequence', 'trip_id', 'stop_sequence'),
        Index('idx_stop_times_stop_time', 'stop_id', 'arrival_time'),
    )


class VehiclePositionTable(Base):
    """SQLAlchemy table for real-time vehicle positions."""
    __tablename__ = 'vehicle_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(255), nullable=False, index=True)
    trip_id = Column(String(255), ForeignKey('trips.trip_id'), nullable=True, index=True)
    route_id = Column(String(255), nullable=True, index=True)
    
    # PostGIS geometry for vehicle location
    location = Column(Geography('POINT', srid=4326), nullable=False, index=True)
    
    bearing = Column(Float, nullable=True)
    speed_kmh = Column(Float, nullable=True)
    status = Column(String(50), nullable=True, index=True)
    occupancy_status = Column(String(50), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    stop_id = Column(String(255), ForeignKey('stops.stop_id'), nullable=True, index=True)
    congestion_level = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    trip = relationship("TripTable", back_populates="vehicle_positions")
    stop = relationship("StopTable", back_populates="vehicle_positions")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('bearing IS NULL OR (bearing >= 0 AND bearing < 360)', name='check_bearing_range'),
        CheckConstraint('speed_kmh IS NULL OR speed_kmh >= 0', name='check_speed_positive'),
        CheckConstraint('congestion_level IS NULL OR (congestion_level >= 0 AND congestion_level <= 4)', name='check_congestion_level'),
        Index('idx_vehicle_positions_location', 'location', postgresql_using='gist'),
        Index('idx_vehicle_positions_timestamp', 'timestamp'),
        Index('idx_vehicle_positions_vehicle_time', 'vehicle_id', 'timestamp'),
    )


class AlertTable(Base):
    """SQLAlchemy table for service alerts."""
    __tablename__ = 'alerts'
    
    alert_id = Column(String(255), primary_key=True, index=True)
    cause = Column(String(50), default='UNKNOWN_CAUSE', nullable=False)
    effect = Column(String(50), default='UNKNOWN_EFFECT', nullable=False)
    header_text = Column(String(255), nullable=False)
    description_text = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)
    active_period_start = Column(DateTime, nullable=False, index=True)
    active_period_end = Column(DateTime, nullable=True, index=True)
    affected_routes = Column(ARRAY(String), nullable=True)
    affected_stops = Column(ARRAY(String), nullable=True)
    severity_level = Column(Integer, default=3, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('severity_level >= 1 AND severity_level <= 5', name='check_severity_level'),
        Index('idx_alerts_active_period', 'active_period_start', 'active_period_end'),
        Index('idx_alerts_severity', 'severity_level'),
    )


class NetworkNodeTable(Base):
    """SQLAlchemy table for transport network nodes."""
    __tablename__ = 'network_nodes'
    
    node_id = Column(String(100), primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    node_type = Column(String(20), nullable=False, index=True)
    
    # PostGIS geometry for node location
    location = Column(Geography('POINT', srid=4326), nullable=False, index=True)
    
    capacity = Column(Integer, nullable=True)
    accessibility_features = Column(ARRAY(String), nullable=True)
    operating_hours = Column(JSONB, nullable=True)
    facilities = Column(ARRAY(String), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    edges_from = relationship("NetworkEdgeTable", foreign_keys="NetworkEdgeTable.from_node_id", back_populates="from_node")
    edges_to = relationship("NetworkEdgeTable", foreign_keys="NetworkEdgeTable.to_node_id", back_populates="to_node")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("node_type IN ('stop', 'station', 'interchange', 'depot', 'school', 'junction')", name='check_node_type'),
        CheckConstraint('capacity IS NULL OR capacity >= 0', name='check_capacity_positive'),
        Index('idx_network_nodes_location', 'location', postgresql_using='gist'),
        Index('idx_network_nodes_type', 'node_type'),
    )


class NetworkEdgeTable(Base):
    """SQLAlchemy table for transport network edges."""
    __tablename__ = 'network_edges'
    
    edge_id = Column(String(100), primary_key=True, index=True)
    from_node_id = Column(String(100), ForeignKey('network_nodes.node_id'), nullable=False, index=True)
    to_node_id = Column(String(100), ForeignKey('network_nodes.node_id'), nullable=False, index=True)
    edge_type = Column(String(20), nullable=False, index=True)
    distance_km = Column(Float, nullable=False)
    travel_time_minutes = Column(Float, nullable=False)
    capacity_passengers = Column(Integer, nullable=True)
    
    # PostGIS geometry for edge path
    geometry = Column(Geography('LINESTRING', srid=4326), nullable=True, index=True)
    
    frequency_per_hour = Column(Float, nullable=True)
    operating_cost_per_km = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    from_node = relationship("NetworkNodeTable", foreign_keys=[from_node_id], back_populates="edges_from")
    to_node = relationship("NetworkNodeTable", foreign_keys=[to_node_id], back_populates="edges_to")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('distance_km >= 0', name='check_distance_positive'),
        CheckConstraint('travel_time_minutes >= 0', name='check_travel_time_positive'),
        CheckConstraint('capacity_passengers IS NULL OR capacity_passengers >= 0', name='check_capacity_positive'),
        CheckConstraint("edge_type IN ('bus_route', 'walking_path', 'cycling_path', 'transfer_link', 'feeder_route')", name='check_edge_type'),
        Index('idx_network_edges_geometry', 'geometry', postgresql_using='gist'),
        Index('idx_network_edges_nodes', 'from_node_id', 'to_node_id'),
        Index('idx_network_edges_type', 'edge_type'),
    )


# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Comprehensive database management class."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.Session: Optional[sessionmaker] = None
        self.metadata = MetaData()
        self.logger = structlog.get_logger(__name__ + ".DatabaseManager")
        self._operation_mode = OperationMode.READ_WRITE
    
    def initialize(self) -> None:
        """Initialize database connection and session factory."""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.config.url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.enable_logging,
                future=True
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine, future=True)
            
            # Enable PostGIS extension if not exists
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;"))
                conn.commit()
            
            self.logger.info("Database initialized successfully", url=self.config.url)
            
        except Exception as e:
            self.logger.error("Failed to initialize database", error=str(e))
            raise ConnectionError(f"Database initialization failed: {e}")
    
    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get database session with automatic cleanup."""
        if not self.Session:
            raise ConnectionError("Database not initialized")
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_transaction(self) -> Iterator[Session]:
        """Get database session with explicit transaction control."""
        if not self.Session:
            raise ConnectionError("Database not initialized")
        
        session = self.Session()
        trans = session.begin()
        try:
            yield session
            trans.commit()
        except Exception:
            trans.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self, drop_existing: bool = False) -> None:
        """Create all database tables."""
        try:
            if drop_existing:
                Base.metadata.drop_all(self.engine)
                self.logger.warning("Dropped existing tables")
            
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error("Failed to create tables", error=str(e))
            raise DatabaseError(f"Table creation failed: {e}")
    
    # ==================== BULK OPERATIONS ====================
    
    def bulk_insert_schools(self, schools: List[School], batch_size: int = 1000) -> int:
        """Bulk insert schools with batch processing."""
        self.logger.info("Starting bulk school insert", count=len(schools), batch_size=batch_size)
        
        total_inserted = 0
        
        try:
            with self.get_transaction() as session:
                for i in range(0, len(schools), batch_size):
                    batch = schools[i:i + batch_size]
                    
                    # Convert to database format
                    db_schools = []
                    for school in batch:
                        db_school = {
                            'school_id': school.school_id,
                            'name': school.name,
                            'address': school.address,
                            'suburb': school.suburb,
                            'postcode': school.postcode,
                            'location': f'POINT({school.coordinate.longitude} {school.coordinate.latitude})',
                            'school_type': school.school_type,
                            'sector': school.sector,
                            'grades': school.grades,
                            'enrollment': school.enrollment,
                            'capacity': school.capacity,
                            'phone': school.phone,
                            'email': school.email,
                            'website': str(school.website) if school.website else None,
                            'principal': school.principal
                        }
                        db_schools.append(db_school)
                    
                    # Bulk insert using SQLAlchemy core
                    result = session.execute(
                        insert(SchoolTable).values(db_schools)
                        .on_conflict_do_update(
                            index_elements=['school_id'],
                            set_={
                                'name': insert(SchoolTable).excluded.name,
                                'address': insert(SchoolTable).excluded.address,
                                'updated_at': datetime.utcnow()
                            }
                        )
                    )
                    
                    batch_inserted = result.rowcount
                    total_inserted += batch_inserted
                    
                    self.logger.debug("Batch inserted", 
                                     batch_number=i//batch_size + 1,
                                     batch_size=len(batch),
                                     inserted=batch_inserted)
        
        except Exception as e:
            self.logger.error("Bulk school insert failed", error=str(e))
            raise DatabaseError(f"Bulk school insert failed: {e}")
        
        self.logger.info("Bulk school insert completed", total_inserted=total_inserted)
        return total_inserted
    
    def bulk_insert_students(self, students: List[Student], batch_size: int = 1000) -> int:
        """Bulk insert students with batch processing."""
        self.logger.info("Starting bulk student insert", count=len(students), batch_size=batch_size)
        
        total_inserted = 0
        
        try:
            with self.get_transaction() as session:
                for i in range(0, len(students), batch_size):
                    batch = students[i:i + batch_size]
                    
                    db_students = []
                    for student in batch:
                        db_student = {
                            'student_id': student.student_id,
                            'first_name': student.first_name,
                            'last_name': student.last_name,
                            'date_of_birth': student.date_of_birth,
                            'grade': student.grade,
                            'home_address': student.home_address,
                            'home_location': f'POINT({student.home_coordinate.longitude} {student.home_coordinate.latitude})',
                            'school_id': student.school_id,
                            'transport_eligible': student.transport_eligible,
                            'special_needs': student.special_needs,
                            'guardian_phone': student.guardian_phone,
                            'guardian_email': student.guardian_email
                        }
                        db_students.append(db_student)
                    
                    result = session.execute(
                        insert(StudentTable).values(db_students)
                        .on_conflict_do_update(
                            index_elements=['student_id'],
                            set_={
                                'first_name': insert(StudentTable).excluded.first_name,
                                'last_name': insert(StudentTable).excluded.last_name,
                                'grade': insert(StudentTable).excluded.grade,
                                'updated_at': datetime.utcnow()
                            }
                        )
                    )
                    
                    batch_inserted = result.rowcount
                    total_inserted += batch_inserted
        
        except Exception as e:
            self.logger.error("Bulk student insert failed", error=str(e))
            raise DatabaseError(f"Bulk student insert failed: {e}")
        
        self.logger.info("Bulk student insert completed", total_inserted=total_inserted)
        return total_inserted
    
    def bulk_insert_gtfs_data(
        self,
        routes: List[Route],
        stops: List[Stop], 
        trips: List[Trip],
        stop_times: List[StopTime],
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """Bulk insert GTFS data with referential integrity."""
        self.logger.info("Starting bulk GTFS insert", 
                        routes=len(routes), stops=len(stops), 
                        trips=len(trips), stop_times=len(stop_times))
        
        results = {'routes': 0, 'stops': 0, 'trips': 0, 'stop_times': 0}
        
        try:
            with self.get_transaction() as session:
                # Insert routes first
                if routes:
                    route_data = []
                    for route in routes:
                        route_data.append({
                            'route_id': route.route_id,
                            'agency_id': route.agency_id,
                            'route_short_name': route.route_short_name,
                            'route_long_name': route.route_long_name,
                            'route_desc': route.route_desc,
                            'route_type': route.route_type,
                            'route_url': str(route.route_url) if route.route_url else None,
                            'route_color': route.route_color,
                            'route_text_color': route.route_text_color,
                            'route_sort_order': route.route_sort_order
                        })
                    
                    result = session.execute(
                        insert(RouteTable).values(route_data)
                        .on_conflict_do_update(
                            index_elements=['route_id'],
                            set_={'updated_at': datetime.utcnow()}
                        )
                    )
                    results['routes'] = result.rowcount
                
                # Insert stops
                if stops:
                    stop_data = []
                    for stop in stops:
                        stop_data.append({
                            'stop_id': stop.stop_id,
                            'stop_code': stop.stop_code,
                            'stop_name': stop.stop_name,
                            'stop_desc': stop.stop_desc,
                            'location': f'POINT({stop.stop_lon} {stop.stop_lat})',
                            'zone_id': stop.zone_id,
                            'stop_url': str(stop.stop_url) if stop.stop_url else None,
                            'location_type': stop.location_type,
                            'parent_station': stop.parent_station,
                            'stop_timezone': stop.stop_timezone,
                            'wheelchair_boarding': stop.wheelchair_boarding,
                            'level_id': stop.level_id,
                            'platform_code': stop.platform_code
                        })
                    
                    result = session.execute(
                        insert(StopTable).values(stop_data)
                        .on_conflict_do_update(
                            index_elements=['stop_id'],
                            set_={'updated_at': datetime.utcnow()}
                        )
                    )
                    results['stops'] = result.rowcount
                
                # Insert trips
                if trips:
                    trip_data = []
                    for trip in trips:
                        trip_data.append({
                            'trip_id': trip.trip_id,
                            'route_id': trip.route_id,
                            'service_id': trip.service_id,
                            'trip_headsign': trip.trip_headsign,
                            'trip_short_name': trip.trip_short_name,
                            'direction_id': trip.direction_id,
                            'block_id': trip.block_id,
                            'shape_id': trip.shape_id,
                            'wheelchair_accessible': trip.wheelchair_accessible,
                            'bikes_allowed': trip.bikes_allowed
                        })
                    
                    result = session.execute(
                        insert(TripTable).values(trip_data)
                        .on_conflict_do_update(
                            index_elements=['trip_id'],
                            set_={'updated_at': datetime.utcnow()}
                        )
                    )
                    results['trips'] = result.rowcount
                
                # Insert stop times in batches
                if stop_times:
                    for i in range(0, len(stop_times), batch_size):
                        batch = stop_times[i:i + batch_size]
                        
                        stop_time_data = []
                        for st in batch:
                            stop_time_data.append({
                                'trip_id': st.trip_id,
                                'arrival_time': st.arrival_time,
                                'departure_time': st.departure_time,
                                'stop_id': st.stop_id,
                                'stop_sequence': st.stop_sequence,
                                'stop_headsign': st.stop_headsign,
                                'pickup_type': st.pickup_type,
                                'drop_off_type': st.drop_off_type,
                                'continuous_pickup': st.continuous_pickup,
                                'continuous_drop_off': st.continuous_drop_off,
                                'shape_dist_traveled': st.shape_dist_traveled,
                                'timepoint': st.timepoint
                            })
                        
                        result = session.execute(insert(StopTimeTable).values(stop_time_data))
                        results['stop_times'] += result.rowcount
        
        except Exception as e:
            self.logger.error("Bulk GTFS insert failed", error=str(e))
            raise DatabaseError(f"Bulk GTFS insert failed: {e}")
        
        self.logger.info("Bulk GTFS insert completed", results=results)
        return results
    
    # ==================== SPATIAL QUERIES ====================
    
    def find_schools_near_point(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        school_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find schools within radius of a point using PostGIS."""
        self.logger.debug("Finding schools near point", 
                         lat=latitude, lon=longitude, radius=radius_km)
        
        try:
            with self.get_session() as session:
                # Create point geometry
                point = func.ST_GeogFromText(f'POINT({longitude} {latitude})')
                
                # Build query
                query = session.query(
                    SchoolTable,
                    ST_Distance(SchoolTable.location, point).label('distance_m')
                ).filter(
                    ST_DWithin(SchoolTable.location, point, radius_km * 1000)
                )
                
                if school_type:
                    query = query.filter(SchoolTable.school_type == school_type)
                
                query = query.order_by('distance_m').limit(limit)
                
                results = []
                for school, distance in query.all():
                    result = {
                        'school_id': school.school_id,
                        'name': school.name,
                        'address': school.address,
                        'school_type': school.school_type,
                        'sector': school.sector,
                        'enrollment': school.enrollment,
                        'distance_m': float(distance),
                        'distance_km': float(distance) / 1000.0
                    }
                    results.append(result)
                
                return results
        
        except Exception as e:
            self.logger.error("Spatial query failed", error=str(e))
            raise DatabaseError(f"Spatial query failed: {e}")
    
    def find_students_in_catchment(
        self,
        catchment_id: str,
        include_location: bool = False
    ) -> List[Dict[str, Any]]:
        """Find students within a school catchment area."""
        self.logger.debug("Finding students in catchment", catchment_id=catchment_id)
        
        try:
            with self.get_session() as session:
                # Get catchment boundary
                catchment = session.query(CatchmentTable).filter(
                    CatchmentTable.catchment_id == catchment_id
                ).first()
                
                if not catchment:
                    raise DatabaseError(f"Catchment {catchment_id} not found")
                
                # Find students within boundary
                query = session.query(StudentTable).filter(
                    ST_Within(StudentTable.home_location, catchment.boundary)
                )
                
                results = []
                for student in query.all():
                    result = {
                        'student_id': student.student_id,
                        'full_name': f"{student.first_name} {student.last_name}",
                        'grade': student.grade,
                        'school_id': student.school_id,
                        'transport_eligible': student.transport_eligible,
                        'special_needs': student.special_needs
                    }
                    
                    if include_location:
                        # Extract coordinates from PostGIS geometry
                        location_wkt = session.scalar(
                            func.ST_AsText(student.home_location)
                        )
                        # Parse "POINT(lon lat)" format
                        coords = location_wkt.replace('POINT(', '').replace(')', '').split()
                        result['home_coordinate'] = {
                            'latitude': float(coords[1]),
                            'longitude': float(coords[0])
                        }
                    
                    results.append(result)
                
                return results
        
        except Exception as e:
            self.logger.error("Catchment query failed", error=str(e))
            raise DatabaseError(f"Catchment query failed: {e}")
    
    def find_stops_near_schools(
        self,
        max_distance_km: float = 1.0,
        school_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find transport stops near schools for walking connections."""
        self.logger.debug("Finding stops near schools", max_distance=max_distance_km)
        
        try:
            with self.get_session() as session:
                # Build query with spatial join
                query = session.query(
                    SchoolTable.school_id,
                    SchoolTable.name.label('school_name'),
                    StopTable.stop_id,
                    StopTable.stop_name,
                    ST_Distance(SchoolTable.location, StopTable.location).label('distance_m')
                ).join(
                    StopTable,
                    ST_DWithin(SchoolTable.location, StopTable.location, max_distance_km * 1000)
                )
                
                if school_ids:
                    query = query.filter(SchoolTable.school_id.in_(school_ids))
                
                query = query.order_by(SchoolTable.school_id, 'distance_m')
                
                results = []
                for row in query.all():
                    result = {
                        'school_id': row.school_id,
                        'school_name': row.school_name,
                        'stop_id': row.stop_id,
                        'stop_name': row.stop_name,
                        'distance_m': float(row.distance_m),
                        'distance_km': float(row.distance_m) / 1000.0,
                        'walking_time_minutes': int((float(row.distance_m) / 1000.0) / 5.0 * 60)  # 5 km/h walking speed
                    }
                    results.append(result)
                
                return results
        
        except Exception as e:
            self.logger.error("Stop-school proximity query failed", error=str(e))
            raise DatabaseError(f"Stop-school proximity query failed: {e}")
    
    # ==================== TIME SERIES OPERATIONS ====================
    
    def insert_vehicle_positions(
        self,
        positions: List[VehiclePosition],
        batch_size: int = 1000
    ) -> int:
        """Insert vehicle positions with time-series optimization."""
        self.logger.debug("Inserting vehicle positions", count=len(positions))
        
        total_inserted = 0
        
        try:
            with self.get_transaction() as session:
                for i in range(0, len(positions), batch_size):
                    batch = positions[i:i + batch_size]
                    
                    position_data = []
                    for pos in batch:
                        position_data.append({
                            'vehicle_id': pos.vehicle_id,
                            'trip_id': pos.trip_id,
                            'route_id': pos.route_id,
                            'location': f'POINT({pos.coordinate.longitude} {pos.coordinate.latitude})',
                            'bearing': pos.bearing,
                            'speed_kmh': pos.speed_kmh,
                            'status': pos.status.value if pos.status else None,
                            'occupancy_status': pos.occupancy_status.value if pos.occupancy_status else None,
                            'timestamp': pos.timestamp,
                            'stop_id': pos.stop_id,
                            'congestion_level': pos.congestion_level
                        })
                    
                    result = session.execute(
                        insert(VehiclePositionTable).values(position_data)
                    )
                    total_inserted += result.rowcount
        
        except Exception as e:
            self.logger.error("Vehicle position insert failed", error=str(e))
            raise DatabaseError(f"Vehicle position insert failed: {e}")
        
        return total_inserted
    
    def get_vehicle_positions_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        vehicle_ids: Optional[List[str]] = None,
        route_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get vehicle positions within time range with filtering."""
        self.logger.debug("Querying vehicle positions", 
                         start_time=start_time, end_time=end_time)
        
        try:
            with self.get_session() as session:
                query = session.query(VehiclePositionTable).filter(
                    and_(
                        VehiclePositionTable.timestamp >= start_time,
                        VehiclePositionTable.timestamp <= end_time
                    )
                )
                
                if vehicle_ids:
                    query = query.filter(VehiclePositionTable.vehicle_id.in_(vehicle_ids))
                
                if route_ids:
                    query = query.filter(VehiclePositionTable.route_id.in_(route_ids))
                
                query = query.order_by(VehiclePositionTable.timestamp)
                
                results = []
                for pos in query.all():
                    # Extract coordinates from PostGIS geometry
                    location_wkt = session.scalar(func.ST_AsText(pos.location))
                    coords = location_wkt.replace('POINT(', '').replace(')', '').split()
                    
                    result = {
                        'vehicle_id': pos.vehicle_id,
                        'trip_id': pos.trip_id,
                        'route_id': pos.route_id,
                        'coordinate': {
                            'latitude': float(coords[1]),
                            'longitude': float(coords[0])
                        },
                        'bearing': pos.bearing,
                        'speed_kmh': pos.speed_kmh,
                        'status': pos.status,
                        'occupancy_status': pos.occupancy_status,
                        'timestamp': pos.timestamp,
                        'congestion_level': pos.congestion_level
                    }
                    results.append(result)
                
                return results
        
        except Exception as e:
            self.logger.error("Time series query failed", error=str(e))
            raise DatabaseError(f"Time series query failed: {e}")
    
    # ==================== PERFORMANCE MONITORING ====================
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive database performance metrics."""
        try:
            with self.get_session() as session:
                # Connection pool info
                pool = self.engine.pool
                pool_size = pool.size()
                active_connections = pool.checkedout()
                
                # Query statistics
                query_stats = session.execute(text("""
                    SELECT
                        sum(calls) as total_queries,
                        sum(CASE WHEN mean_exec_time > 1000 THEN calls ELSE 0 END) as slow_queries,
                        avg(mean_exec_time) as avg_query_time
                    FROM pg_stat_statements
                    WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
                """)).first()
                
                # Table sizes
                table_sizes = {}
                size_results = session.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY size_bytes DESC
                """)).all()
                
                for row in size_results:
                    table_sizes[row.tablename] = row.size_bytes
                
                # Cache hit ratio
                cache_hit = session.execute(text("""
                    SELECT
                        sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
                    FROM pg_statio_user_tables
                """)).scalar() or 0.0
                
                # Index usage
                index_usage = {}
                index_results = session.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_tup_read,
                        idx_tup_fetch,
                        idx_scan
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC
                """)).all()
                
                for row in index_results:
                    if row.tablename not in index_usage:
                        index_usage[row.tablename] = {}
                    index_usage[row.tablename][row.indexname] = {
                        'reads': row.idx_tup_read,
                        'fetches': row.idx_tup_fetch,
                        'scans': row.idx_scan
                    }
                
                return PerformanceMetrics(
                    connection_pool_size=pool_size,
                    active_connections=active_connections,
                    total_queries=query_stats.total_queries or 0,
                    slow_queries=query_stats.slow_queries or 0,
                    avg_query_time=float(query_stats.avg_query_time or 0),
                    cache_hit_ratio=float(cache_hit),
                    table_sizes=table_sizes,
                    index_usage=index_usage
                )
        
        except Exception as e:
            self.logger.error("Failed to get performance metrics", error=str(e))
            raise DatabaseError(f"Performance metrics query failed: {e}")
    
    def analyze_slow_queries(self, min_duration_ms: int = 1000) -> List[Dict[str, Any]]:
        """Analyze slow queries for optimization opportunities."""
        try:
            with self.get_session() as session:
                results = session.execute(text("""
                    SELECT
                        query,
                        calls,
                        total_exec_time,
                        mean_exec_time,
                        max_exec_time,
                        stddev_exec_time,
                        rows,
                        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                    FROM pg_stat_statements
                    WHERE mean_exec_time > :min_duration
                    ORDER BY mean_exec_time DESC
                    LIMIT 20
                """), {'min_duration': min_duration_ms}).all()
                
                slow_queries = []
                for row in results:
                    slow_queries.append({
                        'query': row.query[:500] + '...' if len(row.query) > 500 else row.query,
                        'calls': row.calls,
                        'total_exec_time': float(row.total_exec_time),
                        'mean_exec_time': float(row.mean_exec_time),
                        'max_exec_time': float(row.max_exec_time),
                        'stddev_exec_time': float(row.stddev_exec_time or 0),
                        'avg_rows': float(row.rows) / float(row.calls) if row.calls > 0 else 0,
                        'hit_percent': float(row.hit_percent or 0)
                    })
                
                return slow_queries
        
        except Exception as e:
            self.logger.error("Slow query analysis failed", error=str(e))
            raise DatabaseError(f"Slow query analysis failed: {e}")
    
    def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """Optimize a specific table (VACUUM, ANALYZE, REINDEX)."""
        self.logger.info("Optimizing table", table=table_name)
        
        try:
            with self.engine.connect() as conn:
                # VACUUM and ANALYZE
                conn.execute(text(f"VACUUM ANALYZE {table_name}"))
                conn.commit()
                
                # Get table statistics
                stats = conn.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE tablename = :table_name
                """), {'table_name': table_name}).first()
                
                if stats:
                    return {
                        'table_name': stats.tablename,
                        'inserts': stats.inserts,
                        'updates': stats.updates,
                        'deletes': stats.deletes,
                        'live_tuples': stats.live_tuples,
                        'dead_tuples': stats.dead_tuples,
                        'last_vacuum': stats.last_vacuum,
                        'last_analyze': stats.last_analyze,
                        'optimization_completed': True
                    }
                else:
                    return {'error': f'Table {table_name} not found'}
        
        except Exception as e:
            self.logger.error("Table optimization failed", table=table_name, error=str(e))
            raise DatabaseError(f"Table optimization failed: {e}")
    
    # ==================== BACKUP AND RECOVERY ====================
    
    def create_backup(
        self,
        backup_path: str,
        compress: bool = True,
        schema_only: bool = False
    ) -> Dict[str, Any]:
        """Create database backup using pg_dump."""
        self.logger.info("Creating database backup", path=backup_path, compress=compress)
        
        try:
            # Parse database URL for pg_dump parameters
            from urllib.parse import urlparse
            parsed = urlparse(self.config.url)
            
            # Build pg_dump command
            cmd = [
                'pg_dump',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path[1:]}',  # Remove leading /
                '--verbose',
                '--no-password'
            ]
            
            if schema_only:
                cmd.append('--schema-only')
            
            if compress:
                cmd.extend(['--format=custom', '--compress=9'])
                backup_file = f"{backup_path}.backup"
            else:
                backup_file = f"{backup_path}.sql"
            
            cmd.extend(['--file', backup_file])
            
            # Set environment variable for password
            env = {'PGPASSWORD': parsed.password}
            
            # Execute backup
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise BackupError(f"pg_dump failed: {result.stderr}")
            
            # Get backup file size
            backup_size = Path(backup_file).stat().st_size
            
            return {
                'backup_file': backup_file,
                'backup_size_bytes': backup_size,
                'backup_size_mb': backup_size / (1024 * 1024),
                'compress': compress,
                'schema_only': schema_only,
                'created_at': datetime.utcnow(),
                'success': True
            }
        
        except Exception as e:
            self.logger.error("Backup creation failed", error=str(e))
            raise BackupError(f"Backup creation failed: {e}")
    
    def restore_backup(self, backup_path: str, drop_existing: bool = False) -> Dict[str, Any]:
        """Restore database from backup file."""
        self.logger.info("Restoring database backup", path=backup_path)
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.config.url)
            
            # Build pg_restore command
            cmd = [
                'pg_restore',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path[1:]}',
                '--verbose',
                '--no-password'
            ]
            
            if drop_existing:
                cmd.append('--clean')
            
            cmd.append(backup_path)
            
            # Set environment variable for password
            env = {'PGPASSWORD': parsed.password}
            
            # Execute restore
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise BackupError(f"pg_restore failed: {result.stderr}")
            
            return {
                'backup_file': backup_path,
                'restored_at': datetime.utcnow(),
                'success': True,
                'output': result.stdout
            }
        
        except Exception as e:
            self.logger.error("Backup restore failed", error=str(e))
            raise BackupError(f"Backup restore failed: {e}")
    
    def cleanup_old_backups(self, backup_dir: str, retention_days: int = None) -> Dict[str, Any]:
        """Clean up old backup files based on retention policy."""
        if retention_days is None:
            retention_days = self.config.backup_retention_days
        
        self.logger.info("Cleaning up old backups", dir=backup_dir, retention_days=retention_days)
        
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                return {'error': 'Backup directory does not exist'}
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            deleted_files = []
            total_size_freed = 0
            
            for backup_file in backup_path.glob('*.backup'):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    file_size = backup_file.stat().st_size
                    backup_file.unlink()
                    deleted_files.append(str(backup_file))
                    total_size_freed += file_size
            
            return {
                'deleted_files_count': len(deleted_files),
                'deleted_files': deleted_files,
                'total_size_freed_bytes': total_size_freed,
                'total_size_freed_mb': total_size_freed / (1024 * 1024),
                'retention_days': retention_days
            }
        
        except Exception as e:
            self.logger.error("Backup cleanup failed", error=str(e))
            raise BackupError(f"Backup cleanup failed: {e}")
    
    # ==================== MIGRATION MANAGEMENT ====================
    
    def init_migrations(self, migrations_dir: str = "migrations") -> None:
        """Initialize Alembic migrations."""
        try:
            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", migrations_dir)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.url)
            
            # Initialize migration environment
            from alembic.command import init
            init(alembic_cfg, migrations_dir)
            
            self.logger.info("Migration environment initialized", dir=migrations_dir)
        
        except Exception as e:
            self.logger.error("Migration initialization failed", error=str(e))
            raise MigrationError(f"Migration initialization failed: {e}")
    
    def create_migration(
        self,
        message: str,
        migrations_dir: str = "migrations",
        auto_generate: bool = True
    ) -> str:
        """Create new migration script."""
        try:
            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", migrations_dir)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.url)
            
            from alembic.command import revision
            if auto_generate:
                revision(alembic_cfg, message=message, autogenerate=True)
            else:
                revision(alembic_cfg, message=message)
            
            self.logger.info("Migration created", message=message)
            return message
        
        except Exception as e:
            self.logger.error("Migration creation failed", error=str(e))
            raise MigrationError(f"Migration creation failed: {e}")
    
    def run_migrations(self, migrations_dir: str = "migrations") -> Dict[str, Any]:
        """Run pending migrations."""
        try:
            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", migrations_dir)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.url)
            
            from alembic.command import upgrade
            from alembic.runtime.migration import MigrationContext
            
            # Get current revision
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
            
            # Run migrations
            upgrade(alembic_cfg, "head")
            
            # Get new revision
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                new_rev = context.get_current_revision()
            
            return {
                'previous_revision': current_rev,
                'current_revision': new_rev,
                'migration_completed': True
            }
        
        except Exception as e:
            self.logger.error("Migration execution failed", error=str(e))
            raise MigrationError(f"Migration execution failed: {e}")
    
    # ==================== DATA ARCHIVAL ====================
    
    def archive_old_vehicle_positions(
        self,
        archive_older_than_days: int = 30,
        batch_size: int = 10000
    ) -> Dict[str, Any]:
        """Archive old vehicle position data."""
        cutoff_date = datetime.now() - timedelta(days=archive_older_than_days)
        
        self.logger.info("Archiving old vehicle positions", cutoff_date=cutoff_date)
        
        try:
            archived_count = 0
            
            with self.get_transaction() as session:
                # Count records to archive
                total_count = session.query(VehiclePositionTable).filter(
                    VehiclePositionTable.timestamp < cutoff_date
                ).count()
                
                if total_count == 0:
                    return {'archived_count': 0, 'message': 'No records to archive'}
                
                # Archive in batches
                while True:
                    # Get batch of old records
                    batch = session.query(VehiclePositionTable).filter(
                        VehiclePositionTable.timestamp < cutoff_date
                    ).limit(batch_size).all()
                    
                    if not batch:
                        break
                    
                    # Delete batch
                    batch_ids = [record.id for record in batch]
                    session.query(VehiclePositionTable).filter(
                        VehiclePositionTable.id.in_(batch_ids)
                    ).delete(synchronize_session=False)
                    
                    archived_count += len(batch)
                    
                    self.logger.debug("Archived batch", 
                                     batch_size=len(batch), 
                                     total_archived=archived_count)
            
            return {
                'archived_count': archived_count,
                'total_available': total_count,
                'cutoff_date': cutoff_date,
                'archive_completed': True
            }
        
        except Exception as e:
            self.logger.error("Vehicle position archival failed", error=str(e))
            raise DatabaseError(f"Vehicle position archival failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.utcnow(),
            'issues': []
        }
        
        try:
            with self.get_session() as session:
                # Connection test
                session.execute(text('SELECT 1'))
                health_status['checks']['connection'] = 'ok'
                
                # Table existence check
                tables_exist = session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)).fetchall()
                
                expected_tables = [
                    'schools', 'students', 'enrollments', 'catchments',
                    'routes', 'stops', 'trips', 'stop_times',
                    'vehicle_positions', 'alerts', 'network_nodes', 'network_edges'
                ]
                
                existing_tables = [row[0] for row in tables_exist]
                missing_tables = set(expected_tables) - set(existing_tables)
                
                if missing_tables:
                    health_status['issues'].append(f"Missing tables: {list(missing_tables)}")
                    health_status['status'] = 'degraded'
                
                health_status['checks']['tables'] = {
                    'expected': len(expected_tables),
                    'found': len(existing_tables),
                    'missing': list(missing_tables)
                }
                
                # PostGIS extension check
                postgis_version = session.execute(text('SELECT PostGIS_version()')).scalar()
                health_status['checks']['postgis'] = {
                    'available': bool(postgis_version),
                    'version': postgis_version
                }
                
                # Connection pool status
                pool = self.engine.pool
                health_status['checks']['connection_pool'] = {
                    'size': pool.size(),
                    'checked_out': pool.checkedout(),
                    'overflow': pool.overflow(),
                    'status': 'ok' if pool.checkedout() < pool.size() else 'warning'
                }
                
                # Disk space check (approximate)
                db_size = session.execute(text(
                    "SELECT pg_database_size(current_database())"
                )).scalar()
                
                health_status['checks']['database_size'] = {
                    'size_bytes': db_size,
                    'size_mb': db_size / (1024 * 1024),
                    'size_gb': db_size / (1024 * 1024 * 1024)
                }
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['issues'].append(f"Health check failed: {str(e)}")
            self.logger.error("Database health check failed", error=str(e))
        
        return health_status


# Export main classes and functions
__all__ = [
    'DatabaseError', 'ConnectionError', 'MigrationError', 'BackupError',
    'OperationMode', 'DatabaseConfig', 'PerformanceMetrics',
    'SchoolTable', 'StudentTable', 'EnrollmentTable', 'CatchmentTable',
    'RouteTable', 'StopTable', 'TripTable', 'StopTimeTable',
    'VehiclePositionTable', 'AlertTable', 'NetworkNodeTable', 'NetworkEdgeTable',
    'DatabaseManager'
]