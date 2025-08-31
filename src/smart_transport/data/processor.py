"""
Data processing module for SmartSchoolGo application.

This module provides comprehensive data processing capabilities including:
- GTFS data cleaning and validation
- Geospatial coordinate transformations and spatial analysis
- Student enrollment and catchment data processing
- Transport network graph construction and validation
- Time series data analysis and pattern detection

Features:
- Data quality checks with detailed error reporting
- Duplicate removal and deduplication algorithms
- Missing data imputation using statistical methods
- Coordinate system transformations
- Spatial joins and analysis
- Network topology validation
- Export functions for multiple formats
"""

import asyncio
import json
import csv
import math
import statistics
import warnings
from collections import defaultdict, Counter
from datetime import datetime, date, time, timedelta
from typing import (
    List, Dict, Set, Tuple, Optional, Any, Union, 
    Callable, Iterator, Generator
)
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import transform
import pyproj
from pyproj import Transformer
import structlog

from .models import (
    BaseDataModel, Coordinate, BoundingBox, RouteGeometry,
    Route, Stop, Trip, StopTime, Vehicle,
    School, Student, Enrollment, Catchment,
    NetworkNode, NetworkEdge, TransportGraph,
    VehiclePosition, Alert, ServiceUpdate,
    ProgressInfo
)

# Configure logging
logger = structlog.get_logger(__name__)


class ProcessingError(Exception):
    """Base exception for data processing errors."""
    pass


class ValidationError(ProcessingError):
    """Raised when data validation fails."""
    pass


class GeospatialError(ProcessingError):
    """Raised when geospatial operations fail."""
    pass


class NetworkError(ProcessingError):
    """Raised when network operations fail."""
    pass


class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class ValidationResult:
    """Result of data validation process."""
    is_valid: bool
    quality_level: QualityLevel
    total_records: int
    valid_records: int
    invalid_records: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def validity_percentage(self) -> float:
        """Calculate percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100
    
    @property
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Validation Result: {self.quality_level.value.title()}\n"
            f"Valid: {self.valid_records}/{self.total_records} "
            f"({self.validity_percentage:.1f}%)\n"
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"
        )


@dataclass
class ProcessingStats:
    """Statistics for data processing operations."""
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_created: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    errors_encountered: int = 0
    warnings_generated: int = 0
    
    @property
    def duration(self) -> timedelta:
        """Calculate processing duration."""
        end_time = self.end_time or datetime.now()
        return end_time - self.start_time
    
    @property
    def processing_rate(self) -> float:
        """Calculate records per second."""
        duration_seconds = self.duration.total_seconds()
        if duration_seconds == 0:
            return 0.0
        return self.records_processed / duration_seconds


class BaseProcessor:
    """Base class for all data processors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        self._stats = ProcessingStats(start_time=datetime.now())
    
    @property
    def stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self._stats
    
    def _reset_stats(self):
        """Reset processing statistics."""
        self._stats = ProcessingStats(start_time=datetime.now())
    
    def _update_progress(self, progress_callback: Optional[Callable], 
                        current: int, total: int, message: str = ""):
        """Update progress if callback provided."""
        if progress_callback:
            progress = ProgressInfo(
                total=total,
                completed=current,
                failed=self._stats.errors_encountered,
                start_time=self._stats.start_time,
                current_time=datetime.now()
            )
            progress_callback(progress, message)


class GTFSProcessor(BaseProcessor):
    """Processor for GTFS data cleaning and validation."""
    
    def __init__(self):
        super().__init__("GTFSProcessor")
    
    def validate_routes(
        self, 
        routes: List[Route],
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """Validate GTFS routes data."""
        self.logger.info("Starting route validation", count=len(routes))
        self._reset_stats()
        
        result = ValidationResult(
            is_valid=True,
            quality_level=QualityLevel.EXCELLENT,
            total_records=len(routes),
            valid_records=0,
            invalid_records=0
        )
        
        route_ids = set()
        duplicate_ids = set()
        
        for i, route in enumerate(routes):
            self._update_progress(progress_callback, i + 1, len(routes), f"Validating route {route.route_id}")
            
            try:
                # Check for duplicate route IDs
                if route.route_id in route_ids:
                    duplicate_ids.add(route.route_id)
                    result.errors.append(f"Duplicate route ID: {route.route_id}")
                    result.invalid_records += 1
                    continue
                
                route_ids.add(route.route_id)
                
                # Validate route names
                if not route.route_short_name and not route.route_long_name:
                    result.errors.append(f"Route {route.route_id}: Missing both short and long names")
                    result.invalid_records += 1
                    continue
                
                # Validate route type
                if route.route_type not in [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]:
                    result.warnings.append(f"Route {route.route_id}: Unusual route type {route.route_type}")
                
                # Validate colors if provided
                if route.route_color and not route.route_color.upper() in ['FFFFFF']:
                    if len(route.route_color) != 6:
                        result.warnings.append(f"Route {route.route_id}: Invalid color format")
                
                result.valid_records += 1
                
            except Exception as e:
                result.errors.append(f"Route {route.route_id}: Validation error - {str(e)}")
                result.invalid_records += 1
                self._stats.errors_encountered += 1
        
        # Calculate quality level
        validity_pct = result.validity_percentage
        if validity_pct >= 98:
            result.quality_level = QualityLevel.EXCELLENT
        elif validity_pct >= 90:
            result.quality_level = QualityLevel.GOOD
        elif validity_pct >= 75:
            result.quality_level = QualityLevel.ACCEPTABLE
        elif validity_pct >= 50:
            result.quality_level = QualityLevel.POOR
        else:
            result.quality_level = QualityLevel.UNACCEPTABLE
        
        result.is_valid = result.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]
        
        # Generate statistics and recommendations
        result.statistics = {
            'unique_route_types': len(set(r.route_type for r in routes)),
            'routes_with_colors': len([r for r in routes if r.route_color]),
            'duplicate_count': len(duplicate_ids),
            'processing_time_seconds': self.stats.duration.total_seconds()
        }
        
        if duplicate_ids:
            result.recommendations.append("Remove duplicate route IDs")
        if result.invalid_records > 0:
            result.recommendations.append("Review and fix invalid route records")
        
        self._stats.end_time = datetime.now()
        self._stats.records_processed = len(routes)
        
        self.logger.info("Route validation completed", 
                        valid=result.valid_records, 
                        invalid=result.invalid_records,
                        quality=result.quality_level.value)
        
        return result
    
    def validate_stops(
        self, 
        stops: List[Stop],
        bounds: Optional[BoundingBox] = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """Validate GTFS stops data."""
        self.logger.info("Starting stop validation", count=len(stops))
        self._reset_stats()
        
        result = ValidationResult(
            is_valid=True,
            quality_level=QualityLevel.EXCELLENT,
            total_records=len(stops),
            valid_records=0,
            invalid_records=0
        )
        
        stop_ids = set()
        coordinates = []
        
        for i, stop in enumerate(stops):
            self._update_progress(progress_callback, i + 1, len(stops), f"Validating stop {stop.stop_id}")
            
            try:
                # Check for duplicate stop IDs
                if stop.stop_id in stop_ids:
                    result.errors.append(f"Duplicate stop ID: {stop.stop_id}")
                    result.invalid_records += 1
                    continue
                
                stop_ids.add(stop.stop_id)
                
                # Validate coordinates
                coord = stop.coordinate
                coordinates.append((coord.latitude, coord.longitude))
                
                # Check if coordinates are within bounds
                if bounds and not coord.is_within_bounds(bounds):
                    result.warnings.append(
                        f"Stop {stop.stop_id}: Coordinates outside expected bounds "
                        f"({coord.latitude}, {coord.longitude})"
                    )
                
                # Validate stop name
                if not stop.stop_name.strip():
                    result.errors.append(f"Stop {stop.stop_id}: Empty stop name")
                    result.invalid_records += 1
                    continue
                
                # Check for reasonable coordinate precision
                if len(str(coord.latitude).split('.')[-1]) > 6 or len(str(coord.longitude).split('.')[-1]) > 6:
                    result.warnings.append(f"Stop {stop.stop_id}: Excessive coordinate precision")
                
                result.valid_records += 1
                
            except Exception as e:
                result.errors.append(f"Stop {stop.stop_id}: Validation error - {str(e)}")
                result.invalid_records += 1
                self._stats.errors_encountered += 1
        
        # Detect potential duplicate locations
        if coordinates:
            duplicate_coords = self._find_duplicate_coordinates(coordinates, threshold_meters=10)
            if duplicate_coords:
                result.warnings.append(f"Found {len(duplicate_coords)} potential duplicate stop locations")
        
        # Calculate quality level
        validity_pct = result.validity_percentage
        if validity_pct >= 98:
            result.quality_level = QualityLevel.EXCELLENT
        elif validity_pct >= 90:
            result.quality_level = QualityLevel.GOOD
        elif validity_pct >= 75:
            result.quality_level = QualityLevel.ACCEPTABLE
        elif validity_pct >= 50:
            result.quality_level = QualityLevel.POOR
        else:
            result.quality_level = QualityLevel.UNACCEPTABLE
        
        result.is_valid = result.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]
        
        # Generate statistics
        if coordinates:
            latitudes = [coord[0] for coord in coordinates]
            longitudes = [coord[1] for coord in coordinates]
            
            result.statistics = {
                'coordinate_bounds': {
                    'min_lat': min(latitudes),
                    'max_lat': max(latitudes),
                    'min_lon': min(longitudes),
                    'max_lon': max(longitudes)
                },
                'coordinate_centroid': {
                    'lat': statistics.mean(latitudes),
                    'lon': statistics.mean(longitudes)
                },
                'duplicate_locations': len(duplicate_coords) if duplicate_coords else 0,
                'processing_time_seconds': self.stats.duration.total_seconds()
            }
        
        self._stats.end_time = datetime.now()
        self._stats.records_processed = len(stops)
        
        self.logger.info("Stop validation completed",
                        valid=result.valid_records,
                        invalid=result.invalid_records,
                        quality=result.quality_level.value)
        
        return result
    
    def clean_gtfs_data(
        self,
        routes: List[Route],
        stops: List[Stop],
        trips: List[Trip],
        stop_times: List[StopTime],
        remove_duplicates: bool = True,
        fix_missing_data: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[Route], List[Stop], List[Trip], List[StopTime], ValidationResult]:
        """Clean and validate complete GTFS dataset."""
        self.logger.info("Starting GTFS data cleaning")
        self._reset_stats()
        
        total_operations = 8
        current_operation = 0
        
        def update_cleaning_progress(message: str):
            nonlocal current_operation
            current_operation += 1
            if progress_callback:
                progress = ProgressInfo(
                    total=total_operations,
                    completed=current_operation,
                    failed=self._stats.errors_encountered,
                    start_time=self._stats.start_time,
                    current_time=datetime.now()
                )
                progress_callback(progress, message)
        
        # Clean routes
        update_cleaning_progress("Cleaning routes data")
        cleaned_routes = self._deduplicate_routes(routes) if remove_duplicates else routes
        
        # Clean stops
        update_cleaning_progress("Cleaning stops data")
        cleaned_stops = self._deduplicate_stops(stops) if remove_duplicates else stops
        
        # Validate referential integrity
        update_cleaning_progress("Validating referential integrity")
        route_ids = set(r.route_id for r in cleaned_routes)
        stop_ids = set(s.stop_id for s in cleaned_stops)
        
        # Clean trips
        update_cleaning_progress("Cleaning trips data")
        cleaned_trips = []
        for trip in trips:
            if trip.route_id in route_ids:
                cleaned_trips.append(trip)
            else:
                self.logger.warning("Trip references non-existent route", 
                                   trip_id=trip.trip_id, route_id=trip.route_id)
        
        # Clean stop times
        update_cleaning_progress("Cleaning stop times data")
        trip_ids = set(t.trip_id for t in cleaned_trips)
        cleaned_stop_times = []
        
        for stop_time in stop_times:
            if stop_time.trip_id in trip_ids and stop_time.stop_id in stop_ids:
                cleaned_stop_times.append(stop_time)
            else:
                self.logger.warning("Stop time references non-existent trip or stop",
                                   trip_id=stop_time.trip_id, stop_id=stop_time.stop_id)
        
        # Fix missing data if requested
        if fix_missing_data:
            update_cleaning_progress("Fixing missing data")
            cleaned_stop_times = self._impute_missing_times(cleaned_stop_times)
        
        # Final validation
        update_cleaning_progress("Performing final validation")
        route_validation = self.validate_routes(cleaned_routes)
        stop_validation = self.validate_stops(cleaned_stops)
        
        # Combine validation results
        combined_result = ValidationResult(
            is_valid=route_validation.is_valid and stop_validation.is_valid,
            quality_level=min(route_validation.quality_level, stop_validation.quality_level, key=lambda x: x.value),
            total_records=route_validation.total_records + stop_validation.total_records,
            valid_records=route_validation.valid_records + stop_validation.valid_records,
            invalid_records=route_validation.invalid_records + stop_validation.invalid_records,
            warnings=route_validation.warnings + stop_validation.warnings,
            errors=route_validation.errors + stop_validation.errors,
            statistics={
                'original_counts': {
                    'routes': len(routes),
                    'stops': len(stops),
                    'trips': len(trips),
                    'stop_times': len(stop_times)
                },
                'cleaned_counts': {
                    'routes': len(cleaned_routes),
                    'stops': len(cleaned_stops),
                    'trips': len(cleaned_trips),
                    'stop_times': len(cleaned_stop_times)
                },
                'removed_counts': {
                    'routes': len(routes) - len(cleaned_routes),
                    'stops': len(stops) - len(cleaned_stops),
                    'trips': len(trips) - len(cleaned_trips),
                    'stop_times': len(stop_times) - len(cleaned_stop_times)
                }
            }
        )
        
        update_cleaning_progress("GTFS cleaning completed")
        self._stats.end_time = datetime.now()
        
        self.logger.info("GTFS data cleaning completed",
                        routes_cleaned=len(cleaned_routes),
                        stops_cleaned=len(cleaned_stops),
                        quality=combined_result.quality_level.value)
        
        return cleaned_routes, cleaned_stops, cleaned_trips, cleaned_stop_times, combined_result
    
    def _deduplicate_routes(self, routes: List[Route]) -> List[Route]:
        """Remove duplicate routes based on route_id."""
        seen_ids = set()
        deduped_routes = []
        
        for route in routes:
            if route.route_id not in seen_ids:
                seen_ids.add(route.route_id)
                deduped_routes.append(route)
            else:
                self.logger.debug("Removing duplicate route", route_id=route.route_id)
        
        return deduped_routes
    
    def _deduplicate_stops(self, stops: List[Stop]) -> List[Stop]:
        """Remove duplicate stops based on stop_id and nearby coordinates."""
        seen_ids = set()
        coordinates_tree = None
        deduped_stops = []
        
        # Build spatial index for coordinate-based deduplication
        coordinates = [(s.stop_lat, s.stop_lon) for s in stops]
        if coordinates:
            coordinates_tree = cKDTree(coordinates)
        
        for i, stop in enumerate(stops):
            # Remove ID-based duplicates
            if stop.stop_id in seen_ids:
                self.logger.debug("Removing duplicate stop ID", stop_id=stop.stop_id)
                continue
            
            # Check for nearby stops (within 10 meters)
            if coordinates_tree:
                nearby_indices = coordinates_tree.query_ball_point(
                    [stop.stop_lat, stop.stop_lon], r=0.0001  # roughly 10 meters
                )
                
                # If there are nearby stops with different IDs, keep the first one
                has_nearby_duplicate = any(
                    j < i and stops[j].stop_id != stop.stop_id 
                    for j in nearby_indices if j != i
                )
                
                if has_nearby_duplicate:
                    self.logger.debug("Removing nearby duplicate stop", 
                                     stop_id=stop.stop_id, 
                                     lat=stop.stop_lat, lon=stop.stop_lon)
                    continue
            
            seen_ids.add(stop.stop_id)
            deduped_stops.append(stop)
        
        return deduped_stops
    
    def _find_duplicate_coordinates(self, coordinates: List[Tuple[float, float]], 
                                   threshold_meters: float = 10) -> List[List[int]]:
        """Find groups of coordinates that are within threshold distance."""
        if len(coordinates) < 2:
            return []
        
        # Convert to numpy array for efficient computation
        coords_array = np.array(coordinates)
        
        # Calculate pairwise distances using haversine approximation
        distances = pdist(coords_array, metric=self._haversine_distance)
        distance_matrix = squareform(distances)
        
        # Find pairs within threshold
        threshold_deg = threshold_meters / 111320  # rough conversion to degrees
        close_pairs = np.where((distance_matrix < threshold_deg) & (distance_matrix > 0))
        
        # Group connected components
        graph = defaultdict(set)
        for i, j in zip(close_pairs[0], close_pairs[1]):
            graph[i].add(j)
            graph[j].add(i)
        
        # Find connected components (groups of duplicates)
        visited = set()
        duplicate_groups = []
        
        for node in graph:
            if node not in visited:
                group = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        stack.extend(graph[current] - visited)
                
                if len(group) > 1:
                    duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate haversine distance between two coordinates."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        return 6371.0 * c
    
    def _impute_missing_times(self, stop_times: List[StopTime]) -> List[StopTime]:
        """Impute missing arrival and departure times using interpolation."""
        # Group stop times by trip
        trip_stop_times = defaultdict(list)
        for st in stop_times:
            trip_stop_times[st.trip_id].append(st)
        
        imputed_stop_times = []
        
        for trip_id, trip_sts in trip_stop_times.items():
            # Sort by stop sequence
            trip_sts.sort(key=lambda x: x.stop_sequence)
            
            # Find missing times and interpolate
            for i, st in enumerate(trip_sts):
                if not st.arrival_time or not st.departure_time:
                    # Try to interpolate from surrounding stops
                    interpolated_time = self._interpolate_time(trip_sts, i)
                    
                    if interpolated_time:
                        if not st.arrival_time:
                            st.arrival_time = interpolated_time
                        if not st.departure_time:
                            st.departure_time = interpolated_time
                
                imputed_stop_times.append(st)
        
        return imputed_stop_times
    
    def _interpolate_time(self, stop_times: List[StopTime], index: int) -> Optional[time]:
        """Interpolate missing time based on surrounding stops."""
        if index < 0 or index >= len(stop_times):
            return None
        
        # Find previous stop with valid time
        prev_time = None
        prev_seq = None
        for i in range(index - 1, -1, -1):
            if stop_times[i].departure_time:
                prev_time = stop_times[i].departure_time
                prev_seq = stop_times[i].stop_sequence
                break
        
        # Find next stop with valid time
        next_time = None
        next_seq = None
        for i in range(index + 1, len(stop_times)):
            if stop_times[i].arrival_time:
                next_time = stop_times[i].arrival_time
                next_seq = stop_times[i].stop_sequence
                break
        
        # Interpolate if we have both bounds
        if prev_time and next_time and prev_seq is not None and next_seq is not None:
            current_seq = stop_times[index].stop_sequence
            
            # Linear interpolation based on stop sequence
            total_seq_diff = next_seq - prev_seq
            current_seq_diff = current_seq - prev_seq
            
            if total_seq_diff > 0:
                ratio = current_seq_diff / total_seq_diff
                
                # Convert times to seconds for interpolation
                prev_seconds = prev_time.hour * 3600 + prev_time.minute * 60 + prev_time.second
                next_seconds = next_time.hour * 3600 + next_time.minute * 60 + next_time.second
                
                # Handle time wraparound (e.g., 23:59 to 01:00)
                if next_seconds < prev_seconds:
                    next_seconds += 24 * 3600
                
                interpolated_seconds = prev_seconds + ratio * (next_seconds - prev_seconds)
                
                # Convert back to time
                hours = int(interpolated_seconds // 3600) % 24
                minutes = int((interpolated_seconds % 3600) // 60)
                seconds = int(interpolated_seconds % 60)
                
                return time(hours, minutes, seconds)
        
        return None


class GeospatialProcessor(BaseProcessor):
    """Processor for geospatial data operations and coordinate transformations."""
    
    def __init__(self):
        super().__init__("GeospatialProcessor")
        
        # Common coordinate reference systems
        self.wgs84 = pyproj.CRS.from_epsg(4326)  # WGS84 Geographic
        self.gda94_mga_zone54 = pyproj.CRS.from_epsg(28354)  # GDA94 MGA Zone 54 (Canberra)
        self.gda2020_mga_zone55 = pyproj.CRS.from_epsg(7855)  # GDA2020 MGA Zone 55
    
    def transform_coordinates(
        self,
        coordinates: List[Coordinate],
        source_crs: str = "EPSG:4326",
        target_crs: str = "EPSG:28354",
        progress_callback: Optional[Callable] = None
    ) -> List[Coordinate]:
        """Transform coordinates between different coordinate reference systems."""
        self.logger.info("Starting coordinate transformation", 
                        count=len(coordinates),
                        source_crs=source_crs,
                        target_crs=target_crs)
        
        try:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            transformed_coords = []
            
            for i, coord in enumerate(coordinates):
                if progress_callback:
                    self._update_progress(progress_callback, i + 1, len(coordinates), 
                                        f"Transforming coordinate {i+1}")
                
                try:
                    # Transform longitude, latitude (note: pyproj expects x, y order)
                    x, y = transformer.transform(coord.longitude, coord.latitude)
                    
                    # Create new coordinate with transformed values
                    transformed_coord = Coordinate(
                        latitude=y if target_crs == "EPSG:4326" else coord.latitude,
                        longitude=x if target_crs == "EPSG:4326" else coord.longitude,
                        altitude=coord.altitude
                    )
                    
                    # Store original coordinates as additional attributes if transforming from WGS84
                    if source_crs == "EPSG:4326" and target_crs != "EPSG:4326":
                        # For projected coordinates, store them differently
                        transformed_coord = Coordinate(
                            latitude=y,  # In projected systems, this becomes northing
                            longitude=x,  # In projected systems, this becomes easting
                            altitude=coord.altitude
                        )
                    
                    transformed_coords.append(transformed_coord)
                    
                except Exception as e:
                    self.logger.warning("Failed to transform coordinate", 
                                       coord=f"{coord.latitude},{coord.longitude}",
                                       error=str(e))
                    # Keep original coordinate on failure
                    transformed_coords.append(coord)
                    self._stats.errors_encountered += 1
            
            self.logger.info("Coordinate transformation completed",
                           successful=len(transformed_coords) - self._stats.errors_encountered,
                           failed=self._stats.errors_encountered)
            
            return transformed_coords
            
        except Exception as e:
            self.logger.error("Coordinate transformation failed", error=str(e))
            raise GeospatialError(f"Coordinate transformation failed: {e}")
    
    def spatial_join_students_to_schools(
        self,
        students: List[Student],
        schools: List[School],
        max_distance_km: float = 50.0,
        progress_callback: Optional[Callable] = None
    ) -> List[Tuple[Student, School, float]]:
        """Perform spatial join to match students with their nearest schools."""
        self.logger.info("Starting spatial join", 
                        students=len(students),
                        schools=len(schools),
                        max_distance=max_distance_km)
        
        if not schools:
            raise GeospatialError("No schools provided for spatial join")
        
        # Build spatial index for schools
        school_coords = np.array([[s.coordinate.latitude, s.coordinate.longitude] for s in schools])
        school_tree = cKDTree(school_coords)
        
        matches = []
        
        for i, student in enumerate(students):
            if progress_callback:
                self._update_progress(progress_callback, i + 1, len(students),
                                    f"Processing student {student.student_id}")
            
            try:
                student_coord = np.array([student.home_coordinate.latitude, student.home_coordinate.longitude])
                
                # Find nearest schools within max distance
                # Convert km to approximate degrees (rough approximation)
                max_distance_deg = max_distance_km / 111.32
                
                distances, indices = school_tree.query(
                    student_coord, 
                    k=min(10, len(schools)),  # Find up to 10 nearest
                    distance_upper_bound=max_distance_deg
                )
                
                # Process results
                if not isinstance(distances, np.ndarray):
                    distances = [distances]
                    indices = [indices]
                
                for dist, idx in zip(distances, indices):
                    if idx < len(schools) and not np.isinf(dist):
                        # Calculate actual distance in km using Haversine formula
                        actual_distance = student.home_coordinate.distance_to(schools[idx].coordinate)
                        
                        if actual_distance <= max_distance_km:
                            matches.append((student, schools[idx], actual_distance))
                
            except Exception as e:
                self.logger.warning("Failed to process student for spatial join",
                                   student_id=student.student_id,
                                   error=str(e))
                self._stats.errors_encountered += 1
        
        # Sort matches by distance
        matches.sort(key=lambda x: x[2])
        
        self.logger.info("Spatial join completed",
                        total_matches=len(matches),
                        unique_students=len(set(m[0].student_id for m in matches)))
        
        return matches
    
    def create_catchment_polygons(
        self,
        schools: List[School],
        buffer_distance_km: float = 2.0,
        progress_callback: Optional[Callable] = None
    ) -> List[Catchment]:
        """Create catchment area polygons around schools."""
        self.logger.info("Creating catchment polygons", 
                        schools=len(schools),
                        buffer_distance=buffer_distance_km)
        
        catchments = []
        
        for i, school in enumerate(schools):
            if progress_callback:
                self._update_progress(progress_callback, i + 1, len(schools),
                                    f"Creating catchment for {school.name}")
            
            try:
                # Create circular buffer around school
                center = Point(school.coordinate.longitude, school.coordinate.latitude)
                
                # Convert km to degrees (approximate)
                buffer_degrees = buffer_distance_km / 111.32
                buffer_polygon = center.buffer(buffer_degrees)
                
                # Extract coordinates from polygon
                boundary_coords = []
                for coord in buffer_polygon.exterior.coords[:-1]:  # Exclude duplicate last point
                    boundary_coords.append(Coordinate(latitude=coord[1], longitude=coord[0]))
                
                catchment = Catchment(
                    catchment_id=f"catch_{school.school_id}",
                    school_id=school.school_id,
                    name=f"{school.name} Catchment",
                    boundary=boundary_coords,
                    priority_level=1,
                    transport_provided=True,
                    max_walking_distance_km=min(1.6, buffer_distance_km)
                )
                
                catchments.append(catchment)
                
            except Exception as e:
                self.logger.warning("Failed to create catchment",
                                   school_id=school.school_id,
                                   error=str(e))
                self._stats.errors_encountered += 1
        
        self.logger.info("Catchment polygon creation completed",
                        successful=len(catchments),
                        failed=self._stats.errors_encountered)
        
        return catchments
    
    def calculate_service_areas(
        self,
        stops: List[Stop],
        walking_speed_kmh: float = 5.0,
        max_walking_time_minutes: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Calculate pedestrian service areas around transport stops."""
        self.logger.info("Calculating service areas", 
                        stops=len(stops),
                        walking_speed=walking_speed_kmh,
                        max_time=max_walking_time_minutes)
        
        max_distance_km = (walking_speed_kmh * max_walking_time_minutes) / 60
        service_areas = []
        
        for i, stop in enumerate(stops):
            if progress_callback:
                self._update_progress(progress_callback, i + 1, len(stops),
                                    f"Calculating service area for {stop.stop_name}")
            
            try:
                center = Point(stop.stop_lon, stop.stop_lat)
                buffer_degrees = max_distance_km / 111.32
                service_polygon = center.buffer(buffer_degrees)
                
                service_area = {
                    'stop_id': stop.stop_id,
                    'stop_name': stop.stop_name,
                    'center_coordinate': stop.coordinate,
                    'service_radius_km': max_distance_km,
                    'walking_time_minutes': max_walking_time_minutes,
                    'polygon_area_km2': self._calculate_polygon_area(service_polygon),
                    'boundary_coordinates': [
                        Coordinate(latitude=coord[1], longitude=coord[0])
                        for coord in service_polygon.exterior.coords[:-1]
                    ]
                }
                
                service_areas.append(service_area)
                
            except Exception as e:
                self.logger.warning("Failed to calculate service area",
                                   stop_id=stop.stop_id,
                                   error=str(e))
                self._stats.errors_encountered += 1
        
        self.logger.info("Service area calculation completed",
                        successful=len(service_areas),
                        failed=self._stats.errors_encountered)
        
        return service_areas
    
    def _calculate_polygon_area(self, polygon) -> float:
        """Calculate polygon area in square kilometers (approximate)."""
        try:
            # Simple area calculation in degrees squared, then convert to km²
            area_deg_sq = polygon.area
            # Very rough conversion: 1 degree ≈ 111.32 km at equator
            area_km_sq = area_deg_sq * (111.32 ** 2)
            return area_km_sq
        except:
            return 0.0


class StudentDataProcessor(BaseProcessor):
    """Processor for student enrollment and catchment data."""
    
    def __init__(self):
        super().__init__("StudentDataProcessor")
    
    def process_enrollment_data(
        self,
        students: List[Student],
        schools: List[School],
        catchments: List[Catchment],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[Enrollment], ValidationResult]:
        """Process student enrollment data and create enrollment records."""
        self.logger.info("Processing enrollment data",
                        students=len(students),
                        schools=len(schools),
                        catchments=len(catchments))
        
        self._reset_stats()
        enrollments = []
        
        # Create lookup dictionaries for efficiency
        school_lookup = {s.school_id: s for s in schools}
        catchment_lookup = defaultdict(list)
        for catchment in catchments:
            catchment_lookup[catchment.school_id].append(catchment)
        
        validation_result = ValidationResult(
            is_valid=True,
            quality_level=QualityLevel.EXCELLENT,
            total_records=len(students),
            valid_records=0,
            invalid_records=0
        )
        
        for i, student in enumerate(students):
            if progress_callback:
                self._update_progress(progress_callback, i + 1, len(students),
                                    f"Processing student {student.student_id}")
            
            try:
                # Validate student has assigned school
                if student.school_id not in school_lookup:
                    validation_result.errors.append(
                        f"Student {student.student_id}: References non-existent school {student.school_id}"
                    )
                    validation_result.invalid_records += 1
                    continue
                
                school = school_lookup[student.school_id]
                
                # Calculate distance to school
                distance_km = student.home_coordinate.distance_to(school.coordinate)
                
                # Estimate travel time (assuming average speed of 30 km/h)
                travel_time_minutes = int((distance_km / 30) * 60)
                
                # Check if student is within catchment area
                in_catchment = False
                relevant_catchments = catchment_lookup.get(student.school_id, [])
                
                for catchment in relevant_catchments:
                    if self._point_in_polygon(student.home_coordinate, catchment.boundary):
                        in_catchment = True
                        break
                
                # Create enrollment record
                enrollment = Enrollment(
                    student_id=student.student_id,
                    school_id=student.school_id,
                    enrollment_date=date.today(),  # Default to today, should be provided
                    grade_at_enrollment=student.grade,
                    status="active",
                    distance_to_school_km=distance_km,
                    travel_time_minutes=travel_time_minutes
                )
                
                enrollments.append(enrollment)
                validation_result.valid_records += 1
                
                # Add warnings for unusual cases
                if distance_km > 20:
                    validation_result.warnings.append(
                        f"Student {student.student_id}: Very long distance to school ({distance_km:.1f}km)"
                    )
                
                if not in_catchment and distance_km > 5:
                    validation_result.warnings.append(
                        f"Student {student.student_id}: Outside catchment and far from school"
                    )
                
            except Exception as e:
                validation_result.errors.append(
                    f"Student {student.student_id}: Processing error - {str(e)}"
                )
                validation_result.invalid_records += 1
                self._stats.errors_encountered += 1
        
        # Calculate quality metrics
        validity_pct = validation_result.validity_percentage
        if validity_pct >= 95:
            validation_result.quality_level = QualityLevel.EXCELLENT
        elif validity_pct >= 85:
            validation_result.quality_level = QualityLevel.GOOD
        elif validity_pct >= 70:
            validation_result.quality_level = QualityLevel.ACCEPTABLE
        else:
            validation_result.quality_level = QualityLevel.POOR
        
        # Generate statistics
        if enrollments:
            distances = [e.distance_to_school_km for e in enrollments]
            travel_times = [e.travel_time_minutes for e in enrollments if e.travel_time_minutes]
            
            validation_result.statistics = {
                'total_enrollments': len(enrollments),
                'distance_statistics': {
                    'mean_km': statistics.mean(distances),
                    'median_km': statistics.median(distances),
                    'max_km': max(distances),
                    'min_km': min(distances)
                },
                'travel_time_statistics': {
                    'mean_minutes': statistics.mean(travel_times) if travel_times else 0,
                    'median_minutes': statistics.median(travel_times) if travel_times else 0,
                    'max_minutes': max(travel_times) if travel_times else 0
                },
                'schools_with_students': len(set(e.school_id for e in enrollments)),
                'processing_time_seconds': self.stats.duration.total_seconds()
            }
        
        self._stats.end_time = datetime.now()
        self._stats.records_processed = len(students)
        self._stats.records_created = len(enrollments)
        
        self.logger.info("Enrollment processing completed",
                        enrollments_created=len(enrollments),
                        quality=validation_result.quality_level.value)
        
        return enrollments, validation_result
    
    def analyze_catchment_coverage(
        self,
        students: List[Student],
        catchments: List[Catchment],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Analyze how well catchment areas cover student populations."""
        self.logger.info("Analyzing catchment coverage",
                        students=len(students),
                        catchments=len(catchments))
        
        coverage_analysis = {
            'total_students': len(students),
            'total_catchments': len(catchments),
            'students_covered': 0,
            'students_uncovered': 0,
            'coverage_by_school': {},
            'uncovered_students': [],
            'catchment_utilization': {}
        }
        
        for i, student in enumerate(students):
            if progress_callback:
                self._update_progress(progress_callback, i + 1, len(students),
                                    f"Analyzing coverage for student {student.student_id}")
            
            is_covered = False
            covering_catchments = []
            
            for catchment in catchments:
                if self._point_in_polygon(student.home_coordinate, catchment.boundary):
                    is_covered = True
                    covering_catchments.append(catchment.catchment_id)
                    
                    # Update school-specific coverage
                    if catchment.school_id not in coverage_analysis['coverage_by_school']:
                        coverage_analysis['coverage_by_school'][catchment.school_id] = {
                            'students_in_catchment': 0,
                            'catchment_area_km2': catchment.area_km2
                        }
                    coverage_analysis['coverage_by_school'][catchment.school_id]['students_in_catchment'] += 1
            
            if is_covered:
                coverage_analysis['students_covered'] += 1
            else:
                coverage_analysis['students_uncovered'] += 1
                coverage_analysis['uncovered_students'].append({
                    'student_id': student.student_id,
                    'home_coordinate': student.home_coordinate,
                    'school_id': student.school_id
                })
        
        # Calculate coverage percentage
        if coverage_analysis['total_students'] > 0:
            coverage_analysis['coverage_percentage'] = (
                coverage_analysis['students_covered'] / coverage_analysis['total_students'] * 100
            )
        else:
            coverage_analysis['coverage_percentage'] = 0.0
        
        # Calculate catchment utilization
        for catchment in catchments:
            students_in_catchment = coverage_analysis['coverage_by_school'].get(
                catchment.school_id, {}
            ).get('students_in_catchment', 0)
            
            coverage_analysis['catchment_utilization'][catchment.catchment_id] = {
                'students_count': students_in_catchment,
                'area_km2': catchment.area_km2,
                'density_students_per_km2': students_in_catchment / catchment.area_km2 if catchment.area_km2 > 0 else 0
            }
        
        self.logger.info("Catchment coverage analysis completed",
                        coverage_percentage=coverage_analysis['coverage_percentage'],
                        uncovered_students=coverage_analysis['students_uncovered'])
        
        return coverage_analysis
    
    def _point_in_polygon(self, point: Coordinate, polygon_boundary: List[Coordinate]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        if len(polygon_boundary) < 3:
            return False
        
        x, y = point.longitude, point.latitude
        n = len(polygon_boundary)
        inside = False
        
        p1x, p1y = polygon_boundary[0].longitude, polygon_boundary[0].latitude
        
        for i in range(1, n + 1):
            p2x, p2y = polygon_boundary[i % n].longitude, polygon_boundary[i % n].latitude
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class NetworkProcessor(BaseProcessor):
    """Processor for building and validating transport network graphs."""
    
    def __init__(self):
        super().__init__("NetworkProcessor")
    
    def build_transport_network(
        self,
        stops: List[Stop],
        routes: List[Route],
        trips: List[Trip],
        stop_times: List[StopTime],
        schools: List[School],
        walking_speed_kmh: float = 5.0,
        max_walking_distance_km: float = 1.0,
        progress_callback: Optional[Callable] = None
    ) -> TransportGraph:
        """Build comprehensive transport network graph."""
        self.logger.info("Building transport network",
                        stops=len(stops),
                        routes=len(routes),
                        schools=len(schools))
        
        self._reset_stats()
        
        # Create transport graph
        graph = TransportGraph(
            graph_id="main_transport_network",
            name="SmartSchoolGo Transport Network"
        )
        
        total_steps = 5
        current_step = 0
        
        def update_build_progress(message: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress = ProgressInfo(
                    total=total_steps,
                    completed=current_step,
                    failed=self._stats.errors_encountered,
                    start_time=self._stats.start_time,
                    current_time=datetime.now()
                )
                progress_callback(progress, message)
        
        # Step 1: Add stop nodes
        update_build_progress("Adding transport stops to network")
        for stop in stops:
            try:
                node = NetworkNode(
                    node_id=f"stop_{stop.stop_id}",
                    name=stop.stop_name,
                    node_type="stop",
                    coordinate=stop.coordinate,
                    accessibility_features=["wheelchair"] if stop.wheelchair_boarding == 1 else []
                )
                graph.add_node(node)
            except Exception as e:
                self.logger.warning("Failed to add stop node", stop_id=stop.stop_id, error=str(e))
                self._stats.errors_encountered += 1
        
        # Step 2: Add school nodes
        update_build_progress("Adding schools to network")
        for school in schools:
            try:
                node = NetworkNode(
                    node_id=f"school_{school.school_id}",
                    name=school.name,
                    node_type="school",
                    coordinate=school.coordinate,
                    capacity=school.capacity or school.enrollment
                )
                graph.add_node(node)
            except Exception as e:
                self.logger.warning("Failed to add school node", school_id=school.school_id, error=str(e))
                self._stats.errors_encountered += 1
        
        # Step 3: Add route edges between stops
        update_build_progress("Adding route connections")
        route_lookup = {r.route_id: r for r in routes}
        trip_routes = defaultdict(list)
        
        # Group trips by route
        for trip in trips:
            if trip.route_id in route_lookup:
                trip_routes[trip.route_id].append(trip)
        
        # Process stop times to create route edges
        trip_stop_times = defaultdict(list)
        for st in stop_times:
            trip_stop_times[st.trip_id].append(st)
        
        for route_id, route_trips in trip_routes.items():
            route = route_lookup[route_id]
            
            # Get representative trip (first one) to build route structure
            if route_trips:
                trip = route_trips[0]
                trip_sts = sorted(trip_stop_times.get(trip.trip_id, []), 
                                key=lambda x: x.stop_sequence)
                
                # Create edges between consecutive stops
                for i in range(len(trip_sts) - 1):
                    current_st = trip_sts[i]
                    next_st = trip_sts[i + 1]
                    
                    try:
                        current_stop_node_id = f"stop_{current_st.stop_id}"
                        next_stop_node_id = f"stop_{next_st.stop_id}"
                        
                        if (current_stop_node_id in graph.nodes and 
                            next_stop_node_id in graph.nodes):
                            
                            current_stop = graph.nodes[current_stop_node_id]
                            next_stop = graph.nodes[next_stop_node_id]
                            
                            distance = current_stop.coordinate.distance_to(next_stop.coordinate)
                            
                            # Calculate travel time from schedule
                            travel_time = 5  # Default 5 minutes
                            if current_st.departure_time and next_st.arrival_time:
                                current_minutes = (current_st.departure_time.hour * 60 + 
                                                 current_st.departure_time.minute)
                                next_minutes = (next_st.arrival_time.hour * 60 + 
                                              next_st.arrival_time.minute)
                                travel_time = max(1, next_minutes - current_minutes)
                            
                            edge = NetworkEdge(
                                edge_id=f"route_{route_id}_{current_st.stop_id}_{next_st.stop_id}",
                                from_node_id=current_stop_node_id,
                                to_node_id=next_stop_node_id,
                                edge_type="bus_route",
                                distance_km=distance,
                                travel_time_minutes=travel_time,
                                capacity_passengers=50  # Default bus capacity
                            )
                            
                            graph.add_edge(edge)
                            
                    except Exception as e:
                        self.logger.warning("Failed to add route edge",
                                          route_id=route_id,
                                          from_stop=current_st.stop_id,
                                          to_stop=next_st.stop_id,
                                          error=str(e))
                        self._stats.errors_encountered += 1
        
        # Step 4: Add walking connections from schools to nearby stops
        update_build_progress("Adding walking connections to schools")
        school_nodes = [node for node in graph.nodes.values() if node.node_type == "school"]
        stop_nodes = [node for node in graph.nodes.values() if node.node_type == "stop"]
        
        for school_node in school_nodes:
            # Find nearby stops within walking distance
            nearby_stops = []
            for stop_node in stop_nodes:
                distance = school_node.coordinate.distance_to(stop_node.coordinate)
                if distance <= max_walking_distance_km:
                    nearby_stops.append((stop_node, distance))
            
            # Sort by distance and connect to closest stops
            nearby_stops.sort(key=lambda x: x[1])
            
            for stop_node, distance in nearby_stops[:5]:  # Connect to max 5 nearest stops
                try:
                    # Calculate walking time
                    walking_time = (distance / walking_speed_kmh) * 60  # Convert to minutes
                    
                    # Create bidirectional walking edges
                    edge_to_stop = NetworkEdge(
                        edge_id=f"walk_{school_node.node_id}_to_{stop_node.node_id}",
                        from_node_id=school_node.node_id,
                        to_node_id=stop_node.node_id,
                        edge_type="walking_path",
                        distance_km=distance,
                        travel_time_minutes=walking_time
                    )
                    
                    edge_from_stop = NetworkEdge(
                        edge_id=f"walk_{stop_node.node_id}_to_{school_node.node_id}",
                        from_node_id=stop_node.node_id,
                        to_node_id=school_node.node_id,
                        edge_type="walking_path",
                        distance_km=distance,
                        travel_time_minutes=walking_time
                    )
                    
                    graph.add_edge(edge_to_stop)
                    graph.add_edge(edge_from_stop)
                    
                except Exception as e:
                    self.logger.warning("Failed to add walking edge",
                                       school_id=school_node.node_id,
                                       stop_id=stop_node.node_id,
                                       error=str(e))
                    self._stats.errors_encountered += 1
        
        # Step 5: Validate network connectivity
        update_build_progress("Validating network connectivity")
        validation_result = self.validate_network_topology(graph)
        
        self._stats.end_time = datetime.now()
        self._stats.records_processed = len(stops) + len(schools)
        self._stats.records_created = graph.node_count + graph.edge_count
        
        self.logger.info("Transport network built successfully",
                        nodes=graph.node_count,
                        edges=graph.edge_count,
                        connectivity_valid=validation_result.is_valid)
        
        return graph
    
    def validate_network_topology(self, graph: TransportGraph) -> ValidationResult:
        """Validate network topology and connectivity."""
        self.logger.info("Validating network topology", 
                        nodes=graph.node_count,
                        edges=graph.edge_count)
        
        result = ValidationResult(
            is_valid=True,
            quality_level=QualityLevel.EXCELLENT,
            total_records=graph.node_count,
            valid_records=0,
            invalid_records=0
        )
        
        # Check for isolated nodes
        isolated_nodes = []
        for node_id, node in graph.nodes.items():
            adjacent_nodes = graph.get_adjacent_nodes(node_id)
            if not adjacent_nodes:
                isolated_nodes.append(node_id)
                result.invalid_records += 1
            else:
                result.valid_records += 1
        
        if isolated_nodes:
            result.warnings.append(f"Found {len(isolated_nodes)} isolated nodes")
            result.quality_level = QualityLevel.GOOD
        
        # Check edge consistency
        invalid_edges = []
        for edge_id, edge in graph.edges.items():
            if (edge.from_node_id not in graph.nodes or 
                edge.to_node_id not in graph.nodes):
                invalid_edges.append(edge_id)
                result.errors.append(f"Edge {edge_id} references non-existent nodes")
        
        if invalid_edges:
            result.quality_level = QualityLevel.POOR
            result.is_valid = False
        
        # Network connectivity analysis using NetworkX
        try:
            nx_graph = self._convert_to_networkx(graph)
            
            # Check if graph is connected
            if nx_graph.number_of_nodes() > 0:
                is_connected = nx.is_connected(nx_graph.to_undirected())
                connected_components = list(nx.connected_components(nx_graph.to_undirected()))
                
                result.statistics = {
                    'is_connected': is_connected,
                    'connected_components_count': len(connected_components),
                    'largest_component_size': len(max(connected_components, key=len)) if connected_components else 0,
                    'average_degree': sum(dict(nx_graph.degree()).values()) / nx_graph.number_of_nodes() if nx_graph.number_of_nodes() > 0 else 0,
                    'isolated_nodes_count': len(isolated_nodes),
                    'invalid_edges_count': len(invalid_edges)
                }
                
                if not is_connected:
                    result.warnings.append(f"Network has {len(connected_components)} disconnected components")
                    if result.quality_level == QualityLevel.EXCELLENT:
                        result.quality_level = QualityLevel.GOOD
        
        except Exception as e:
            result.warnings.append(f"Could not perform connectivity analysis: {str(e)}")
        
        self.logger.info("Network topology validation completed",
                        is_valid=result.is_valid,
                        quality=result.quality_level.value,
                        isolated_nodes=len(isolated_nodes))
        
        return result
    
    def _convert_to_networkx(self, graph: TransportGraph) -> nx.DiGraph:
        """Convert TransportGraph to NetworkX format for analysis."""
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for node_id, node in graph.nodes.items():
            nx_graph.add_node(node_id, 
                             name=node.name,
                             node_type=node.node_type,
                             latitude=node.coordinate.latitude,
                             longitude=node.coordinate.longitude)
        
        # Add edges
        for edge_id, edge in graph.edges.items():
            nx_graph.add_edge(edge.from_node_id, edge.to_node_id,
                             edge_id=edge_id,
                             edge_type=edge.edge_type,
                             distance=edge.distance_km,
                             travel_time=edge.travel_time_minutes)
        
        return nx_graph


class TimeSeriesProcessor(BaseProcessor):
    """Processor for handling temporal data patterns and time series analysis."""
    
    def __init__(self):
        super().__init__("TimeSeriesProcessor")
    
    def analyze_ridership_patterns(
        self,
        vehicle_positions: List[VehiclePosition],
        time_window_hours: int = 24,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Analyze ridership patterns from vehicle position data."""
        self.logger.info("Analyzing ridership patterns",
                        records=len(vehicle_positions),
                        time_window=time_window_hours)
        
        if not vehicle_positions:
            return {'error': 'No vehicle position data provided'}
        
        # Convert to pandas DataFrame for time series analysis
        df_data = []
        for i, vp in enumerate(vehicle_positions):
            if progress_callback and i % 100 == 0:
                self._update_progress(progress_callback, i + 1, len(vehicle_positions),
                                    "Processing vehicle positions")
            
            df_data.append({
                'timestamp': vp.timestamp,
                'vehicle_id': vp.vehicle_id,
                'route_id': vp.route_id,
                'occupancy_status': vp.occupancy_status.value if vp.occupancy_status else 'UNKNOWN',
                'latitude': vp.coordinate.latitude,
                'longitude': vp.coordinate.longitude,
                'speed': vp.speed_kmh or 0,
                'hour': vp.timestamp.hour,
                'day_of_week': vp.timestamp.weekday(),
                'date': vp.timestamp.date()
            })
        
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        analysis_result = {
            'data_summary': {
                'total_records': len(df),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                },
                'unique_vehicles': df['vehicle_id'].nunique(),
                'unique_routes': df['route_id'].nunique()
            },
            'temporal_patterns': {},
            'occupancy_analysis': {},
            'route_analysis': {}
        }
        
        try:
            # Hourly patterns
            hourly_counts = df.groupby('hour').size()
            analysis_result['temporal_patterns']['hourly'] = {
                'peak_hours': hourly_counts.nlargest(3).to_dict(),
                'off_peak_hours': hourly_counts.nsmallest(3).to_dict(),
                'hourly_distribution': hourly_counts.to_dict()
            }
            
            # Daily patterns
            daily_counts = df.groupby('day_of_week').size()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_named = {day_names[k]: v for k, v in daily_counts.to_dict().items()}
            analysis_result['temporal_patterns']['daily'] = {
                'busiest_days': dict(daily_counts.nlargest(3)),
                'daily_distribution': daily_named
            }
            
            # Occupancy analysis
            occupancy_counts = df['occupancy_status'].value_counts()
            analysis_result['occupancy_analysis'] = {
                'distribution': occupancy_counts.to_dict(),
                'peak_occupancy_hours': df[df['occupancy_status'].isin(['FULL', 'CRUSHED_STANDING_ROOM_ONLY'])].groupby('hour').size().to_dict()
            }
            
            # Route-specific analysis
            route_stats = df.groupby('route_id').agg({
                'vehicle_id': 'nunique',
                'speed': 'mean',
                'occupancy_status': lambda x: x.mode().iloc[0] if len(x) > 0 else 'UNKNOWN'
            }).to_dict('index')
            
            analysis_result['route_analysis'] = route_stats
            
        except Exception as e:
            self.logger.error("Error in time series analysis", error=str(e))
            analysis_result['error'] = str(e)
        
        self.logger.info("Ridership pattern analysis completed")
        return analysis_result
    
    def aggregate_temporal_data(
        self,
        data: List[Dict[str, Any]],
        timestamp_field: str,
        aggregation_level: str = 'hourly',
        value_fields: List[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Aggregate temporal data at different time intervals."""
        self.logger.info("Aggregating temporal data",
                        records=len(data),
                        level=aggregation_level)
        
        if not data:
            return {'error': 'No data provided for aggregation'}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        try:
            df[timestamp_field] = pd.to_datetime(df[timestamp_field])
            df.set_index(timestamp_field, inplace=True)
            
            # Determine aggregation frequency
            freq_map = {
                'minutely': '1T',
                'hourly': '1H', 
                'daily': '1D',
                'weekly': '1W',
                'monthly': '1M'
            }
            
            freq = freq_map.get(aggregation_level, '1H')
            
            # Aggregate data
            if value_fields:
                numeric_df = df[value_fields].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            aggregated = numeric_df.resample(freq).agg({
                col: ['count', 'mean', 'sum', 'min', 'max', 'std']
                for col in numeric_df.columns
            })
            
            # Flatten column names
            aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
            
            # Convert to dictionary with timestamps as strings
            result_data = {}
            for timestamp, row in aggregated.iterrows():
                result_data[timestamp.isoformat()] = row.fillna(0).to_dict()
            
            return {
                'aggregation_level': aggregation_level,
                'period_count': len(result_data),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                },
                'aggregated_data': result_data,
                'summary_statistics': {
                    col: {
                        'overall_mean': df[col].mean(),
                        'overall_std': df[col].std(),
                        'overall_min': df[col].min(),
                        'overall_max': df[col].max()
                    }
                    for col in numeric_df.columns
                }
            }
            
        except Exception as e:
            self.logger.error("Error in temporal aggregation", error=str(e))
            return {'error': str(e)}
    
    def detect_anomalies(
        self,
        time_series_data: Dict[str, float],
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """Detect anomalies in time series data using statistical methods."""
        self.logger.info("Detecting anomalies",
                        data_points=len(time_series_data),
                        threshold=threshold_std)
        
        if not time_series_data:
            return {'error': 'No time series data provided'}
        
        values = list(time_series_data.values())
        timestamps = list(time_series_data.keys())
        
        try:
            # Calculate statistics
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Detect outliers
            anomalies = []
            for timestamp, value in time_series_data.items():
                z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
                
                if z_score > threshold_std:
                    anomalies.append({
                        'timestamp': timestamp,
                        'value': value,
                        'z_score': z_score,
                        'severity': 'high' if z_score > threshold_std * 1.5 else 'medium'
                    })
            
            return {
                'total_points': len(values),
                'anomalies_detected': len(anomalies),
                'anomaly_rate': len(anomalies) / len(values) * 100,
                'statistics': {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min(values),
                    'max': max(values),
                    'threshold_value': mean_val + (threshold_std * std_val)
                },
                'anomalies': anomalies
            }
            
        except Exception as e:
            self.logger.error("Error in anomaly detection", error=str(e))
            return {'error': str(e)}


# ==================== EXPORT FUNCTIONS ====================

def export_to_csv(data: List[BaseDataModel], filepath: str, progress_callback: Optional[Callable] = None):
    """Export list of data models to CSV file."""
    logger.info("Exporting to CSV", filepath=filepath, record_count=len(data))
    
    if not data:
        logger.warning("No data to export")
        return
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            # Use the first record to determine headers
            headers = data[0].csv_headers()
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for i, record in enumerate(data):
                if progress_callback and i % 100 == 0:
                    progress = ProgressInfo(
                        total=len(data),
                        completed=i,
                        failed=0,
                        start_time=datetime.now(),
                        current_time=datetime.now()
                    )
                    progress_callback(progress, f"Exporting record {i+1}")
                
                writer.writerow(record.to_csv_row())
        
        logger.info("CSV export completed", filepath=filepath)
        
    except Exception as e:
        logger.error("CSV export failed", filepath=filepath, error=str(e))
        raise ProcessingError(f"CSV export failed: {e}")


def export_to_geojson(data: List[BaseDataModel], filepath: str, coordinate_field: str = "coordinate"):
    """Export spatial data to GeoJSON format."""
    logger.info("Exporting to GeoJSON", filepath=filepath, record_count=len(data))
    
    if not data:
        logger.warning("No data to export")
        return
    
    try:
        features = []
        
        for record in data:
            record_dict = record.to_dict()
            
            # Extract coordinate data
            coord_data = record_dict.get(coordinate_field)
            if coord_data and isinstance(coord_data, dict):
                geometry = {
                    "type": "Point",
                    "coordinates": [coord_data.get("longitude"), coord_data.get("latitude")]
                }
                
                # Remove coordinate from properties to avoid duplication
                properties = {k: v for k, v in record_dict.items() if k != coordinate_field}
                
                feature = {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": properties
                }
                
                features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("GeoJSON export completed", filepath=filepath)
        
    except Exception as e:
        logger.error("GeoJSON export failed", filepath=filepath, error=str(e))
        raise ProcessingError(f"GeoJSON export failed: {e}")


def export_network_to_graphml(graph: TransportGraph, filepath: str):
    """Export network graph to GraphML format."""
    logger.info("Exporting network to GraphML", filepath=filepath)
    
    try:
        nx_graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in graph.nodes.items():
            nx_graph.add_node(node_id,
                             name=node.name,
                             node_type=node.node_type,
                             latitude=node.coordinate.latitude,
                             longitude=node.coordinate.longitude,
                             capacity=node.capacity or 0)
        
        # Add edges with attributes
        for edge_id, edge in graph.edges.items():
            nx_graph.add_edge(edge.from_node_id, edge.to_node_id,
                             edge_id=edge_id,
                             edge_type=edge.edge_type,
                             distance_km=edge.distance_km,
                             travel_time_minutes=edge.travel_time_minutes,
                             capacity=edge.capacity_passengers or 0)
        
        # Export to GraphML
        nx.write_graphml(nx_graph, filepath)
        
        logger.info("GraphML export completed", filepath=filepath)
        
    except Exception as e:
        logger.error("GraphML export failed", filepath=filepath, error=str(e))
        raise ProcessingError(f"GraphML export failed: {e}")


# Export main classes and functions
__all__ = [
    'ProcessingError', 'ValidationError', 'GeospatialError', 'NetworkError',
    'QualityLevel', 'ValidationResult', 'ProcessingStats',
    'BaseProcessor', 'GTFSProcessor', 'GeospatialProcessor', 
    'StudentDataProcessor', 'NetworkProcessor', 'TimeSeriesProcessor',
    'export_to_csv', 'export_to_geojson', 'export_network_to_graphml'
]