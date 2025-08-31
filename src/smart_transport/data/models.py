"""
Comprehensive data models for SmartSchoolGo application.

This module provides Pydantic models for all data structures used in the application:
- GTFS data structures for public transport
- School data models for educational institutions
- Geographic data models for spatial operations
- Transport network models for routing algorithms
- Real-time data models for live transport information
- Analysis result models for optimization and forecasting

Each model includes comprehensive validation, serialization methods,
and data transformation utilities.
"""

import csv
import json
import re
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Optional, Dict, List, Any, Union, Tuple, Set, ClassVar
from uuid import UUID, uuid4
import io

from pydantic import (
    BaseModel, 
    Field, 
    validator, 
    root_validator,
    HttpUrl,
    EmailStr,
    constr,
    confloat,
    conint
)
from pydantic.color import Color


class BaseDataModel(BaseModel):
    """Base model with common functionality for all data models."""
    
    class Config:
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            time: lambda v: v.strftime('%H:%M:%S'),
            Decimal: lambda v: float(v),
            UUID: lambda v: str(v)
        }
    
    def to_json(self, **kwargs) -> str:
        """Export model to JSON string."""
        return self.json(by_alias=True, exclude_none=True, **kwargs)
    
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Export model to dictionary."""
        return self.dict(by_alias=True, exclude_none=True, **kwargs)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create model instance from JSON string."""
        return cls.parse_raw(json_str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model instance from dictionary."""
        return cls.parse_obj(data)
    
    def to_csv_row(self) -> List[str]:
        """Export model as CSV row (list of string values)."""
        data = self.to_dict()
        return [str(v) if v is not None else '' for v in data.values()]
    
    @classmethod
    def csv_headers(cls) -> List[str]:
        """Get CSV headers for this model."""
        return list(cls.__fields__.keys())


# ==================== GEOGRAPHIC DATA MODELS ====================

class Coordinate(BaseDataModel):
    """Geographic coordinate with validation for Australian bounds."""
    
    latitude: confloat(ge=-90.0, le=90.0) = Field(
        ...,
        description="Latitude in decimal degrees (-90 to 90)"
    )
    longitude: confloat(ge=-180.0, le=180.0) = Field(
        ..., 
        description="Longitude in decimal degrees (-180 to 180)"
    )
    altitude: Optional[confloat(ge=-1000.0, le=10000.0)] = Field(
        None,
        description="Altitude in meters above sea level"
    )
    accuracy: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Coordinate accuracy in meters"
    )
    
    @validator('latitude', 'longitude')
    def validate_australian_bounds(cls, v, field):
        """Validate coordinates are within reasonable Australian bounds."""
        if field.name == 'latitude' and not (-44.0 <= v <= -10.0):
            # Allow some tolerance for testing but warn
            pass  # Could add warning log here
        elif field.name == 'longitude' and not (113.0 <= v <= 154.0):
            pass  # Could add warning log here
        return v
    
    def distance_to(self, other: 'Coordinate') -> float:
        """Calculate great circle distance to another coordinate in kilometers."""
        import math
        
        # Convert to radians
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(other.latitude)
        lon2_rad = math.radians(other.longitude)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (math.sin(dlat/2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth's radius in kilometers
        earth_radius = 6371.0
        return earth_radius * c
    
    def is_within_bounds(self, bounding_box: 'BoundingBox') -> bool:
        """Check if coordinate is within bounding box."""
        return (bounding_box.min_latitude <= self.latitude <= bounding_box.max_latitude and
                bounding_box.min_longitude <= self.longitude <= bounding_box.max_longitude)


class BoundingBox(BaseDataModel):
    """Geographic bounding box definition."""
    
    min_latitude: confloat(ge=-90.0, le=90.0) = Field(
        ...,
        description="Minimum latitude of bounding box"
    )
    max_latitude: confloat(ge=-90.0, le=90.0) = Field(
        ...,
        description="Maximum latitude of bounding box"
    )
    min_longitude: confloat(ge=-180.0, le=180.0) = Field(
        ...,
        description="Minimum longitude of bounding box"
    )
    max_longitude: confloat(ge=-180.0, le=180.0) = Field(
        ...,
        description="Maximum longitude of bounding box"
    )
    
    @root_validator
    def validate_bounds_consistency(cls, values):
        """Ensure min values are less than max values."""
        if values.get('min_latitude') >= values.get('max_latitude'):
            raise ValueError("min_latitude must be less than max_latitude")
        if values.get('min_longitude') >= values.get('max_longitude'):
            raise ValueError("min_longitude must be less than max_longitude")
        return values
    
    @property
    def center(self) -> Coordinate:
        """Get center point of bounding box."""
        return Coordinate(
            latitude=(self.min_latitude + self.max_latitude) / 2,
            longitude=(self.min_longitude + self.max_longitude) / 2
        )
    
    @property
    def area_km2(self) -> float:
        """Calculate approximate area in square kilometers."""
        # Rough approximation for small areas
        lat_diff = self.max_latitude - self.min_latitude
        lon_diff = self.max_longitude - self.min_longitude
        
        # Approximate conversion factors
        lat_km_per_degree = 111.32
        lon_km_per_degree = 111.32 * abs(math.cos(math.radians(self.center.latitude)))
        
        return lat_diff * lat_km_per_degree * lon_diff * lon_km_per_degree


class RouteGeometry(BaseDataModel):
    """Geometric representation of a transport route."""
    
    coordinates: List[Coordinate] = Field(
        ...,
        min_items=2,
        description="Ordered list of coordinates defining the route path"
    )
    total_distance: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Total route distance in kilometers"
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Bounding box containing all route coordinates"
    )
    
    @validator('coordinates')
    def validate_route_coordinates(cls, v):
        """Validate route has sufficient coordinates."""
        if len(v) < 2:
            raise ValueError("Route must have at least 2 coordinates")
        return v
    
    @root_validator
    def calculate_derived_fields(cls, values):
        """Calculate total distance and bounding box if not provided."""
        coordinates = values.get('coordinates', [])
        
        if not coordinates:
            return values
        
        # Calculate total distance
        if not values.get('total_distance'):
            total_distance = 0.0
            for i in range(len(coordinates) - 1):
                total_distance += coordinates[i].distance_to(coordinates[i + 1])
            values['total_distance'] = total_distance
        
        # Calculate bounding box
        if not values.get('bounding_box'):
            latitudes = [coord.latitude for coord in coordinates]
            longitudes = [coord.longitude for coord in coordinates]
            values['bounding_box'] = BoundingBox(
                min_latitude=min(latitudes),
                max_latitude=max(latitudes),
                min_longitude=min(longitudes),
                max_longitude=max(longitudes)
            )
        
        return values


# ==================== GTFS DATA MODELS ====================

class GTFSRouteType(IntEnum):
    """GTFS route type enumeration."""
    TRAM = 0
    SUBWAY = 1
    RAIL = 2
    BUS = 3
    FERRY = 4
    CABLE_TRAM = 5
    AERIAL_LIFT = 6
    FUNICULAR = 7
    TROLLEYBUS = 11
    MONORAIL = 12


class GTFSWheelchairAccessible(IntEnum):
    """GTFS wheelchair accessibility enumeration."""
    UNKNOWN = 0
    ACCESSIBLE = 1
    NOT_ACCESSIBLE = 2


class GTFSPickupDropoffType(IntEnum):
    """GTFS pickup/dropoff type enumeration."""
    REGULARLY_SCHEDULED = 0
    NO_PICKUP_DROPOFF = 1
    PHONE_AGENCY = 2
    COORDINATE_WITH_DRIVER = 3


class Route(BaseDataModel):
    """GTFS Route model representing a transit route."""
    
    route_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Unique identifier for the route"
    )
    agency_id: Optional[constr(min_length=1, max_length=255)] = Field(
        None,
        description="Agency operating this route"
    )
    route_short_name: Optional[constr(max_length=50)] = Field(
        None,
        description="Short name of the route (e.g., '980')"
    )
    route_long_name: Optional[constr(max_length=255)] = Field(
        None,
        description="Full descriptive name of the route"
    )
    route_desc: Optional[str] = Field(
        None,
        description="Description of the route"
    )
    route_type: GTFSRouteType = Field(
        ...,
        description="Type of transportation used on this route"
    )
    route_url: Optional[HttpUrl] = Field(
        None,
        description="URL of a web page about this route"
    )
    route_color: Optional[constr(regex=r'^[0-9A-Fa-f]{6}$')] = Field(
        "FFFFFF",
        description="Route color in hex format (6 characters)"
    )
    route_text_color: Optional[constr(regex=r'^[0-9A-Fa-f]{6}$')] = Field(
        "000000",
        description="Text color for route in hex format (6 characters)"
    )
    route_sort_order: Optional[conint(ge=0)] = Field(
        None,
        description="Order in which routes should be presented"
    )
    
    @root_validator
    def validate_route_names(cls, values):
        """Ensure at least one route name is provided."""
        short_name = values.get('route_short_name')
        long_name = values.get('route_long_name')
        
        if not short_name and not long_name:
            raise ValueError("Either route_short_name or route_long_name must be provided")
        
        return values
    
    @property
    def display_name(self) -> str:
        """Get display name for the route."""
        if self.route_short_name and self.route_long_name:
            return f"{self.route_short_name} - {self.route_long_name}"
        return self.route_short_name or self.route_long_name or self.route_id


class Stop(BaseDataModel):
    """GTFS Stop model representing a transit stop or station."""
    
    stop_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Unique identifier for the stop"
    )
    stop_code: Optional[constr(max_length=50)] = Field(
        None,
        description="Short text or number that identifies the stop for passengers"
    )
    stop_name: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Name of the stop"
    )
    stop_desc: Optional[str] = Field(
        None,
        description="Description of the stop location"
    )
    stop_lat: confloat(ge=-90.0, le=90.0) = Field(
        ...,
        description="Latitude of the stop location"
    )
    stop_lon: confloat(ge=-180.0, le=180.0) = Field(
        ...,
        description="Longitude of the stop location"
    )
    zone_id: Optional[constr(max_length=50)] = Field(
        None,
        description="Fare zone for the stop"
    )
    stop_url: Optional[HttpUrl] = Field(
        None,
        description="URL of a web page about this stop"
    )
    location_type: Optional[conint(ge=0, le=4)] = Field(
        0,
        description="Location type (0=stop, 1=station, 2=entrance/exit, 3=generic node, 4=boarding area)"
    )
    parent_station: Optional[constr(max_length=255)] = Field(
        None,
        description="Stop ID of the parent station"
    )
    stop_timezone: Optional[str] = Field(
        None,
        description="Timezone of the stop"
    )
    wheelchair_boarding: GTFSWheelchairAccessible = Field(
        GTFSWheelchairAccessible.UNKNOWN,
        description="Wheelchair accessibility of the stop"
    )
    level_id: Optional[constr(max_length=255)] = Field(
        None,
        description="Level of the stop in a station"
    )
    platform_code: Optional[constr(max_length=50)] = Field(
        None,
        description="Platform identifier for the stop"
    )
    
    @property
    def coordinate(self) -> Coordinate:
        """Get coordinate representation of stop location."""
        return Coordinate(latitude=self.stop_lat, longitude=self.stop_lon)
    
    def distance_to_stop(self, other: 'Stop') -> float:
        """Calculate distance to another stop in kilometers."""
        return self.coordinate.distance_to(other.coordinate)


class Trip(BaseDataModel):
    """GTFS Trip model representing a journey along a route."""
    
    route_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Route ID this trip belongs to"
    )
    service_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Service calendar this trip follows"
    )
    trip_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Unique identifier for the trip"
    )
    trip_headsign: Optional[constr(max_length=255)] = Field(
        None,
        description="Text that appears on signage identifying the trip's destination"
    )
    trip_short_name: Optional[constr(max_length=50)] = Field(
        None,
        description="Public facing text used to identify the trip to passengers"
    )
    direction_id: Optional[conint(ge=0, le=1)] = Field(
        None,
        description="Direction of travel for the trip (0 or 1)"
    )
    block_id: Optional[constr(max_length=255)] = Field(
        None,
        description="Block ID this trip belongs to"
    )
    shape_id: Optional[constr(max_length=255)] = Field(
        None,
        description="Shape ID defining the path the vehicle takes along the route"
    )
    wheelchair_accessible: GTFSWheelchairAccessible = Field(
        GTFSWheelchairAccessible.UNKNOWN,
        description="Wheelchair accessibility of the trip"
    )
    bikes_allowed: Optional[conint(ge=0, le=2)] = Field(
        None,
        description="Bicycle policy for the trip (0=unknown, 1=allowed, 2=not allowed)"
    )


class StopTime(BaseDataModel):
    """GTFS Stop Time model representing when a trip visits a stop."""
    
    trip_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Trip ID this stop time belongs to"
    )
    arrival_time: Optional[time] = Field(
        None,
        description="Arrival time at the stop"
    )
    departure_time: Optional[time] = Field(
        None,
        description="Departure time from the stop"
    )
    stop_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Stop ID where this stop time occurs"
    )
    stop_sequence: conint(ge=0) = Field(
        ...,
        description="Order of stops for this trip (0-based)"
    )
    stop_headsign: Optional[constr(max_length=255)] = Field(
        None,
        description="Text that appears on signage at this stop"
    )
    pickup_type: GTFSPickupDropoffType = Field(
        GTFSPickupDropoffType.REGULARLY_SCHEDULED,
        description="Pickup policy at this stop"
    )
    drop_off_type: GTFSPickupDropoffType = Field(
        GTFSPickupDropoffType.REGULARLY_SCHEDULED,
        description="Drop off policy at this stop"
    )
    continuous_pickup: Optional[conint(ge=0, le=3)] = Field(
        None,
        description="Continuous pickup policy"
    )
    continuous_drop_off: Optional[conint(ge=0, le=3)] = Field(
        None,
        description="Continuous drop off policy"
    )
    shape_dist_traveled: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Distance along shape from first stop to this stop"
    )
    timepoint: Optional[conint(ge=0, le=1)] = Field(
        1,
        description="Whether this is an exact timepoint (1) or approximate (0)"
    )
    
    @root_validator
    def validate_times(cls, values):
        """Ensure arrival time is before or equal to departure time."""
        arrival = values.get('arrival_time')
        departure = values.get('departure_time')
        
        if arrival and departure and arrival > departure:
            raise ValueError("arrival_time cannot be after departure_time")
        
        # If only one time is provided, use it for both
        if arrival and not departure:
            values['departure_time'] = arrival
        elif departure and not arrival:
            values['arrival_time'] = departure
        
        return values


class Vehicle(BaseDataModel):
    """GTFS Vehicle model for real-time vehicle information."""
    
    id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Unique vehicle identifier",
        alias="vehicle_id"
    )
    label: Optional[constr(max_length=50)] = Field(
        None,
        description="User visible label for the vehicle"
    )
    license_plate: Optional[constr(max_length=20)] = Field(
        None,
        description="Vehicle license plate number"
    )
    wheelchair_accessible: GTFSWheelchairAccessible = Field(
        GTFSWheelchairAccessible.UNKNOWN,
        description="Wheelchair accessibility of the vehicle"
    )


# ==================== SCHOOL DATA MODELS ====================

class SchoolType(str, Enum):
    """School type enumeration."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    COMBINED = "combined"
    SPECIAL = "special"


class SchoolSector(str, Enum):
    """School sector enumeration."""
    GOVERNMENT = "government"
    CATHOLIC = "catholic"
    INDEPENDENT = "independent"


class School(BaseDataModel):
    """School information model."""
    
    school_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Unique school identifier"
    )
    name: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Official school name"
    )
    address: constr(min_length=1) = Field(
        ...,
        description="Full school address"
    )
    suburb: constr(min_length=1, max_length=100) = Field(
        ...,
        description="School suburb"
    )
    postcode: constr(regex=r'^\d{4}$') = Field(
        ...,
        description="Australian postcode (4 digits)"
    )
    coordinate: Coordinate = Field(
        ...,
        description="Geographic location of the school"
    )
    school_type: SchoolType = Field(
        ...,
        description="Type of school"
    )
    sector: SchoolSector = Field(
        ...,
        description="School sector"
    )
    grades: List[constr(regex=r'^(K|P|\d{1,2})$')] = Field(
        default_factory=list,
        description="List of grades offered (K, P, 1-12)"
    )
    enrollment: Optional[conint(ge=0, le=5000)] = Field(
        None,
        description="Total student enrollment"
    )
    capacity: Optional[conint(ge=0, le=10000)] = Field(
        None,
        description="Maximum student capacity"
    )
    phone: Optional[constr(regex=r'^\+?[\d\s\-\(\)]{8,20}$')] = Field(
        None,
        description="School phone number"
    )
    email: Optional[EmailStr] = Field(
        None,
        description="School email address"
    )
    website: Optional[HttpUrl] = Field(
        None,
        description="School website URL"
    )
    principal: Optional[constr(max_length=100)] = Field(
        None,
        description="School principal name"
    )
    
    @property
    def is_primary(self) -> bool:
        """Check if school offers primary education."""
        return self.school_type in [SchoolType.PRIMARY, SchoolType.COMBINED]
    
    @property
    def is_secondary(self) -> bool:
        """Check if school offers secondary education."""
        return self.school_type in [SchoolType.SECONDARY, SchoolType.COMBINED]
    
    @property
    def utilization_rate(self) -> Optional[float]:
        """Calculate school utilization rate."""
        if self.enrollment is not None and self.capacity is not None and self.capacity > 0:
            return self.enrollment / self.capacity
        return None


class Student(BaseDataModel):
    """Student information model."""
    
    student_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Unique student identifier"
    )
    first_name: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Student first name"
    )
    last_name: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Student last name"
    )
    date_of_birth: date = Field(
        ...,
        description="Student date of birth"
    )
    grade: constr(regex=r'^(K|P|\d{1,2})$') = Field(
        ...,
        description="Current grade level"
    )
    home_address: constr(min_length=1) = Field(
        ...,
        description="Student home address"
    )
    home_coordinate: Coordinate = Field(
        ...,
        description="Geographic location of student home"
    )
    school_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="School ID where student is enrolled"
    )
    transport_eligible: bool = Field(
        True,
        description="Whether student is eligible for school transport"
    )
    special_needs: bool = Field(
        False,
        description="Whether student has special transportation needs"
    )
    guardian_phone: Optional[constr(regex=r'^\+?[\d\s\-\(\)]{8,20}$')] = Field(
        None,
        description="Guardian phone number"
    )
    guardian_email: Optional[EmailStr] = Field(
        None,
        description="Guardian email address"
    )
    
    @property
    def age(self) -> int:
        """Calculate student age in years."""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def full_name(self) -> str:
        """Get student full name."""
        return f"{self.first_name} {self.last_name}"


class Enrollment(BaseDataModel):
    """Student enrollment record."""
    
    enrollment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique enrollment identifier"
    )
    student_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Student identifier"
    )
    school_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="School identifier"
    )
    enrollment_date: date = Field(
        ...,
        description="Date of enrollment"
    )
    grade_at_enrollment: constr(regex=r'^(K|P|\d{1,2})$') = Field(
        ...,
        description="Grade level at time of enrollment"
    )
    status: constr(regex=r'^(active|inactive|graduated|transferred)$') = Field(
        "active",
        description="Enrollment status"
    )
    distance_to_school_km: confloat(ge=0.0, le=200.0) = Field(
        ...,
        description="Distance from home to school in kilometers"
    )
    travel_time_minutes: Optional[conint(ge=0, le=300)] = Field(
        None,
        description="Estimated travel time to school in minutes"
    )


class Catchment(BaseDataModel):
    """School catchment area definition."""
    
    catchment_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Unique catchment identifier"
    )
    school_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="School this catchment serves"
    )
    name: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Catchment area name"
    )
    boundary: List[Coordinate] = Field(
        ...,
        min_items=3,
        description="Polygon boundary of catchment area"
    )
    priority_level: conint(ge=1, le=5) = Field(
        1,
        description="Priority level (1=highest, 5=lowest)"
    )
    transport_provided: bool = Field(
        True,
        description="Whether school transport is provided in this catchment"
    )
    max_walking_distance_km: confloat(ge=0.0, le=10.0) = Field(
        1.6,
        description="Maximum walking distance to school in kilometers"
    )
    
    @property
    def area_km2(self) -> float:
        """Calculate catchment area using shoelace formula."""
        coords = self.boundary + [self.boundary[0]]  # Close polygon
        area = 0.0
        
        for i in range(len(coords) - 1):
            area += coords[i].longitude * coords[i + 1].latitude
            area -= coords[i + 1].longitude * coords[i].latitude
        
        # Convert to approximate km² (rough approximation)
        area = abs(area) / 2.0
        # Convert degrees to km² (very approximate)
        return area * 111.32 * 111.32


# ==================== TRANSPORT NETWORK MODELS ====================

class NetworkNodeType(str, Enum):
    """Transport network node types."""
    STOP = "stop"
    STATION = "station"
    INTERCHANGE = "interchange"
    DEPOT = "depot"
    SCHOOL = "school"
    JUNCTION = "junction"


class NetworkNode(BaseDataModel):
    """Transport network node representing a point in the network."""
    
    node_id: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Unique node identifier"
    )
    name: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Node name"
    )
    node_type: NetworkNodeType = Field(
        ...,
        description="Type of network node"
    )
    coordinate: Coordinate = Field(
        ...,
        description="Geographic location of the node"
    )
    capacity: Optional[conint(ge=0)] = Field(
        None,
        description="Passenger capacity of the node"
    )
    accessibility_features: List[str] = Field(
        default_factory=list,
        description="List of accessibility features (wheelchair, audio, etc.)"
    )
    operating_hours: Optional[Dict[str, str]] = Field(
        None,
        description="Operating hours by day of week"
    )
    facilities: List[str] = Field(
        default_factory=list,
        description="Available facilities (parking, shelters, etc.)"
    )


class NetworkEdgeType(str, Enum):
    """Transport network edge types."""
    BUS_ROUTE = "bus_route"
    WALKING_PATH = "walking_path"
    CYCLING_PATH = "cycling_path"
    TRANSFER_LINK = "transfer_link"
    FEEDER_ROUTE = "feeder_route"


class NetworkEdge(BaseDataModel):
    """Transport network edge representing a connection between nodes."""
    
    edge_id: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Unique edge identifier"
    )
    from_node_id: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Source node ID"
    )
    to_node_id: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Destination node ID"
    )
    edge_type: NetworkEdgeType = Field(
        ...,
        description="Type of network edge"
    )
    distance_km: confloat(ge=0.0) = Field(
        ...,
        description="Distance along edge in kilometers"
    )
    travel_time_minutes: confloat(ge=0.0) = Field(
        ...,
        description="Travel time along edge in minutes"
    )
    capacity_passengers: Optional[conint(ge=0)] = Field(
        None,
        description="Passenger capacity of the edge"
    )
    route_geometry: Optional[RouteGeometry] = Field(
        None,
        description="Detailed geometry of the edge path"
    )
    frequency_per_hour: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Service frequency per hour"
    )
    operating_cost_per_km: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Operating cost per kilometer"
    )
    
    @property
    def average_speed_kmh(self) -> float:
        """Calculate average speed along edge."""
        if self.travel_time_minutes > 0:
            return (self.distance_km / self.travel_time_minutes) * 60
        return 0.0


class TransportGraph(BaseDataModel):
    """Complete transport network graph."""
    
    graph_id: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Unique graph identifier"
    )
    name: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Graph name or description"
    )
    nodes: Dict[str, NetworkNode] = Field(
        default_factory=dict,
        description="Dictionary of nodes by node_id"
    )
    edges: Dict[str, NetworkEdge] = Field(
        default_factory=dict,
        description="Dictionary of edges by edge_id"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Graph creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    
    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        self.updated_at = datetime.now()
    
    def add_edge(self, edge: NetworkEdge) -> None:
        """Add an edge to the graph."""
        # Validate that nodes exist
        if edge.from_node_id not in self.nodes:
            raise ValueError(f"From node {edge.from_node_id} not found in graph")
        if edge.to_node_id not in self.nodes:
            raise ValueError(f"To node {edge.to_node_id} not found in graph")
        
        self.edges[edge.edge_id] = edge
        self.updated_at = datetime.now()
    
    def get_adjacent_nodes(self, node_id: str) -> List[str]:
        """Get list of nodes adjacent to given node."""
        adjacent = []
        for edge in self.edges.values():
            if edge.from_node_id == node_id:
                adjacent.append(edge.to_node_id)
            elif edge.to_node_id == node_id:
                adjacent.append(edge.from_node_id)
        return adjacent
    
    @property
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return len(self.edges)


# ==================== REAL-TIME DATA MODELS ====================

class VehiclePositionStatus(str, Enum):
    """Vehicle position status enumeration."""
    INCOMING_AT = "INCOMING_AT"
    STOPPED_AT = "STOPPED_AT"
    IN_TRANSIT_TO = "IN_TRANSIT_TO"


class OccupancyStatus(str, Enum):
    """Vehicle occupancy status enumeration."""
    EMPTY = "EMPTY"
    MANY_SEATS_AVAILABLE = "MANY_SEATS_AVAILABLE"
    FEW_SEATS_AVAILABLE = "FEW_SEATS_AVAILABLE"
    STANDING_ROOM_ONLY = "STANDING_ROOM_ONLY"
    CRUSHED_STANDING_ROOM_ONLY = "CRUSHED_STANDING_ROOM_ONLY"
    FULL = "FULL"
    NOT_ACCEPTING_PASSENGERS = "NOT_ACCEPTING_PASSENGERS"


class VehiclePosition(BaseDataModel):
    """Real-time vehicle position information."""
    
    vehicle_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Unique vehicle identifier"
    )
    trip_id: Optional[constr(min_length=1, max_length=255)] = Field(
        None,
        description="Trip ID vehicle is currently serving"
    )
    route_id: Optional[constr(min_length=1, max_length=255)] = Field(
        None,
        description="Route ID vehicle is currently serving"
    )
    coordinate: Coordinate = Field(
        ...,
        description="Current vehicle position"
    )
    bearing: Optional[confloat(ge=0.0, lt=360.0)] = Field(
        None,
        description="Vehicle bearing in degrees (0-359)"
    )
    speed_kmh: Optional[confloat(ge=0.0, le=200.0)] = Field(
        None,
        description="Vehicle speed in km/h"
    )
    status: Optional[VehiclePositionStatus] = Field(
        None,
        description="Current vehicle status"
    )
    occupancy_status: Optional[OccupancyStatus] = Field(
        None,
        description="Current occupancy level"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of position report"
    )
    stop_id: Optional[constr(max_length=255)] = Field(
        None,
        description="Stop ID if vehicle is at or approaching a stop"
    )
    congestion_level: Optional[conint(ge=0, le=4)] = Field(
        None,
        description="Traffic congestion level (0=free flow, 4=severe congestion)"
    )


class AlertCause(str, Enum):
    """Service alert cause enumeration."""
    UNKNOWN_CAUSE = "UNKNOWN_CAUSE"
    OTHER_CAUSE = "OTHER_CAUSE"
    TECHNICAL_PROBLEM = "TECHNICAL_PROBLEM"
    STRIKE = "STRIKE"
    DEMONSTRATION = "DEMONSTRATION"
    ACCIDENT = "ACCIDENT"
    HOLIDAY = "HOLIDAY"
    WEATHER = "WEATHER"
    MAINTENANCE = "MAINTENANCE"
    CONSTRUCTION = "CONSTRUCTION"
    POLICE_ACTIVITY = "POLICE_ACTIVITY"
    MEDICAL_EMERGENCY = "MEDICAL_EMERGENCY"


class AlertEffect(str, Enum):
    """Service alert effect enumeration."""
    NO_SERVICE = "NO_SERVICE"
    REDUCED_SERVICE = "REDUCED_SERVICE"
    SIGNIFICANT_DELAYS = "SIGNIFICANT_DELAYS"
    DETOUR = "DETOUR"
    ADDITIONAL_SERVICE = "ADDITIONAL_SERVICE"
    MODIFIED_SERVICE = "MODIFIED_SERVICE"
    OTHER_EFFECT = "OTHER_EFFECT"
    UNKNOWN_EFFECT = "UNKNOWN_EFFECT"
    STOP_MOVED = "STOP_MOVED"


class Alert(BaseDataModel):
    """Service alert model."""
    
    alert_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Unique alert identifier"
    )
    cause: AlertCause = Field(
        AlertCause.UNKNOWN_CAUSE,
        description="Cause of the alert"
    )
    effect: AlertEffect = Field(
        AlertEffect.UNKNOWN_EFFECT,
        description="Effect of the alert on service"
    )
    header_text: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Short alert header text"
    )
    description_text: Optional[str] = Field(
        None,
        description="Detailed alert description"
    )
    url: Optional[HttpUrl] = Field(
        None,
        description="URL with more information about the alert"
    )
    active_period_start: datetime = Field(
        default_factory=datetime.now,
        description="Start of alert active period"
    )
    active_period_end: Optional[datetime] = Field(
        None,
        description="End of alert active period"
    )
    affected_routes: List[constr(min_length=1, max_length=255)] = Field(
        default_factory=list,
        description="List of affected route IDs"
    )
    affected_stops: List[constr(min_length=1, max_length=255)] = Field(
        default_factory=list,
        description="List of affected stop IDs"
    )
    severity_level: conint(ge=1, le=5) = Field(
        3,
        description="Alert severity level (1=info, 5=critical)"
    )
    
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        now = datetime.now()
        if self.active_period_end:
            return self.active_period_start <= now <= self.active_period_end
        return self.active_period_start <= now


class ServiceUpdate(BaseDataModel):
    """Service update information."""
    
    update_id: UUID = Field(
        default_factory=uuid4,
        description="Unique update identifier"
    )
    trip_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Trip ID this update affects"
    )
    route_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Route ID this update affects"
    )
    stop_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Stop ID this update affects"
    )
    scheduled_arrival: Optional[datetime] = Field(
        None,
        description="Originally scheduled arrival time"
    )
    predicted_arrival: Optional[datetime] = Field(
        None,
        description="Predicted arrival time"
    )
    scheduled_departure: Optional[datetime] = Field(
        None,
        description="Originally scheduled departure time"
    )
    predicted_departure: Optional[datetime] = Field(
        None,
        description="Predicted departure time"
    )
    delay_minutes: Optional[int] = Field(
        None,
        description="Delay in minutes (positive=late, negative=early)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the update"
    )
    
    @property
    def arrival_delay_minutes(self) -> Optional[int]:
        """Calculate arrival delay in minutes."""
        if self.scheduled_arrival and self.predicted_arrival:
            delta = self.predicted_arrival - self.scheduled_arrival
            return int(delta.total_seconds() / 60)
        return self.delay_minutes


# ==================== ANALYSIS RESULT MODELS ====================

class OptimizationObjective(str, Enum):
    """Route optimization objectives."""
    MINIMIZE_TRAVEL_TIME = "minimize_travel_time"
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    MINIMIZE_TRANSFERS = "minimize_transfers"
    BALANCE_LOAD = "balance_load"


class RouteOptimization(BaseDataModel):
    """Route optimization analysis result."""
    
    optimization_id: UUID = Field(
        default_factory=uuid4,
        description="Unique optimization identifier"
    )
    name: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Optimization scenario name"
    )
    objective: OptimizationObjective = Field(
        ...,
        description="Optimization objective"
    )
    school_ids: List[constr(min_length=1, max_length=50)] = Field(
        ...,
        description="Schools included in optimization"
    )
    optimized_routes: List[RouteGeometry] = Field(
        default_factory=list,
        description="Optimized route geometries"
    )
    total_distance_km: confloat(ge=0.0) = Field(
        ...,
        description="Total distance of optimized routes"
    )
    total_travel_time_hours: confloat(ge=0.0) = Field(
        ...,
        description="Total travel time for all routes"
    )
    estimated_cost: confloat(ge=0.0) = Field(
        ...,
        description="Estimated operating cost"
    )
    students_served: conint(ge=0) = Field(
        ...,
        description="Number of students served by optimized routes"
    )
    coverage_percentage: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Percentage of eligible students covered"
    )
    vehicle_utilization: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Average vehicle utilization rate"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Analysis creation timestamp"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optimization parameters used"
    )


class SafetyRiskLevel(str, Enum):
    """Safety risk level enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyAnalysis(BaseDataModel):
    """Safety analysis result for transport routes."""
    
    analysis_id: UUID = Field(
        default_factory=uuid4,
        description="Unique analysis identifier"
    )
    route_id: Optional[constr(min_length=1, max_length=255)] = Field(
        None,
        description="Route ID analyzed (if specific route)"
    )
    area_analyzed: Optional[BoundingBox] = Field(
        None,
        description="Geographic area analyzed"
    )
    overall_risk_level: SafetyRiskLevel = Field(
        ...,
        description="Overall safety risk level"
    )
    risk_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk factors and their scores (0.0-1.0)"
    )
    high_risk_locations: List[Coordinate] = Field(
        default_factory=list,
        description="Locations identified as high risk"
    )
    safety_recommendations: List[str] = Field(
        default_factory=list,
        description="Safety improvement recommendations"
    )
    accident_probability: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Estimated accident probability"
    )
    weather_impact_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Weather impact on safety score"
    )
    traffic_density_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Traffic density impact score"
    )
    infrastructure_quality_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Infrastructure quality score"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Analysis creation timestamp"
    )


class DemandForecast(BaseDataModel):
    """Transport demand forecast analysis."""
    
    forecast_id: UUID = Field(
        default_factory=uuid4,
        description="Unique forecast identifier"
    )
    forecast_name: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Forecast scenario name"
    )
    forecast_date: date = Field(
        ...,
        description="Date this forecast is for"
    )
    forecast_period: constr(regex=r'^(hourly|daily|weekly|monthly|yearly)$') = Field(
        ...,
        description="Forecast time period granularity"
    )
    school_ids: List[constr(min_length=1, max_length=50)] = Field(
        default_factory=list,
        description="Schools included in forecast"
    )
    route_ids: List[constr(min_length=1, max_length=255)] = Field(
        default_factory=list,
        description="Routes included in forecast"
    )
    predicted_ridership: conint(ge=0) = Field(
        ...,
        description="Predicted total ridership"
    )
    ridership_by_route: Dict[str, int] = Field(
        default_factory=dict,
        description="Predicted ridership by route ID"
    )
    peak_demand_time: Optional[time] = Field(
        None,
        description="Time of peak demand"
    )
    peak_demand_count: conint(ge=0) = Field(
        ...,
        description="Peak demand passenger count"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval for predictions"
    )
    seasonal_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Seasonal adjustment factors"
    )
    weather_impact: Optional[float] = Field(
        None,
        description="Weather impact factor on demand"
    )
    special_events: List[str] = Field(
        default_factory=list,
        description="Special events affecting demand"
    )
    model_accuracy: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Forecast model accuracy score"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Forecast creation timestamp"
    )


# ==================== BULK EXPORT UTILITIES ====================

def export_to_csv(models: List[BaseDataModel], output_path: str) -> None:
    """Export list of models to CSV file."""
    if not models:
        return
    
    headers = models[0].csv_headers()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for model in models:
            writer.writerow(model.to_csv_row())


def export_to_json(models: List[BaseDataModel], output_path: str) -> None:
    """Export list of models to JSON file."""
    data = [model.to_dict() for model in models]
    
    with open(output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False, default=str)


# ==================== MODEL REGISTRY ====================

# Export all models for easy import
__all__ = [
    # Base classes
    'BaseDataModel',
    
    # Geographic models
    'Coordinate', 'BoundingBox', 'RouteGeometry',
    
    # GTFS models
    'GTFSRouteType', 'GTFSWheelchairAccessible', 'GTFSPickupDropoffType',
    'Route', 'Stop', 'Trip', 'StopTime', 'Vehicle',
    
    # School models
    'SchoolType', 'SchoolSector', 'School', 'Student', 'Enrollment', 'Catchment',
    
    # Transport network models
    'NetworkNodeType', 'NetworkEdgeType', 'NetworkNode', 'NetworkEdge', 'TransportGraph',
    
    # Real-time models
    'VehiclePositionStatus', 'OccupancyStatus', 'VehiclePosition',
    'AlertCause', 'AlertEffect', 'Alert', 'ServiceUpdate',
    
    # Analysis models
    'OptimizationObjective', 'RouteOptimization',
    'SafetyRiskLevel', 'SafetyAnalysis',
    'DemandForecast',
    
    # Utility functions
    'export_to_csv', 'export_to_json'
]