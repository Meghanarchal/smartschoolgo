"""
Data fetcher module for SmartSchoolGo application.

This module provides comprehensive data fetching capabilities from various APIs:
- ACT Transport Canberra GTFS APIs
- OpenStreetMap geographic data
- Weather information
- School location and enrollment data

Features:
- Async/await support for concurrent operations
- Exponential backoff retry logic
- Rate limiting with time-based throttling
- Redis caching with configurable TTL
- Pydantic data validation
- Comprehensive error handling and logging
- Bulk operations with progress tracking
"""

import asyncio
import json
import time
import zipfile
import io
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Tuple, AsyncGenerator
from urllib.parse import urljoin, urlparse
from pathlib import Path
import hashlib
import logging

import aiohttp
import aioredis
from pydantic import BaseModel, Field, validator
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config import get_settings
from .models import (
    Coordinate, BoundingBox,
    Route, Stop, Trip, StopTime, Vehicle,
    VehiclePosition, Alert, ServiceUpdate,
    GeocodeResult, WeatherData, School, Student,
    ProgressInfo
)


# Configure structured logging
logger = structlog.get_logger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class ValidationError(APIError):
    """Raised when API response validation fails."""
    pass


class CacheError(APIError):
    """Raised when cache operations fail."""
    pass


# Note: Data models are now imported from models.py
# This provides centralized, comprehensive data validation


class RateLimiter:
    """Time-based rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call = 0.0
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            await asyncio.sleep(sleep_time)
        
        self.last_call = time.time()


class CacheManager:
    """Redis-based cache manager."""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise CacheError(f"Redis connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
    
    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments."""
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data."""
        if not self.redis:
            return None
        
        try:
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
        
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cached data."""
        if not self.redis:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached data."""
        if not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        if not self.redis:
            return 0
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning("Cache clear pattern failed", pattern=pattern, error=str(e))
            return 0


class BaseAPIClient:
    """Base class for API clients with common functionality."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit: float = 10.0,
        cache_ttl: int = 3600,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit)
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Optional[CacheManager] = None
        
        # Setup cache if Redis is available
        settings = get_settings()
        if settings.redis_url:
            self.cache = CacheManager(str(settings.redis_url), cache_ttl)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Initialize connections."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._get_default_headers()
        )
        
        if self.cache:
            await self.cache.connect()
        
        logger.info("API client connected", base_url=self.base_url)
    
    async def disconnect(self):
        """Close connections."""
        if self.session:
            await self.session.close()
        
        if self.cache:
            await self.cache.disconnect()
        
        logger.info("API client disconnected", base_url=self.base_url)
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            'User-Agent': 'SmartSchoolGo/1.0.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if not self.session:
            raise APIError("Client not connected")
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        request_headers = headers or {}
        
        logger.debug(
            "Making API request",
            method=method,
            url=url,
            params=params,
            headers=list(request_headers.keys())
        )
        
        try:
            async with self.session.request(
                method, url, params=params, json=data, headers=request_headers
            ) as response:
                
                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning("Rate limit hit", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    raise RateLimitError(f"Rate limit exceeded, retry after {retry_after}s")
                
                # Handle other HTTP errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        "API request failed",
                        status=response.status,
                        url=url,
                        error=error_text
                    )
                    raise APIError(f"HTTP {response.status}: {error_text}")
                
                # Parse response
                if response.content_type == 'application/json':
                    result = await response.json()
                elif response.content_type.startswith('text/'):
                    result = {"text": await response.text()}
                else:
                    result = {"data": await response.read()}
                
                logger.debug(
                    "API request successful",
                    status=response.status,
                    url=url,
                    content_type=response.content_type
                )
                
                return result
                
        except aiohttp.ClientError as e:
            logger.error("API request client error", url=url, error=str(e))
            raise APIError(f"Client error: {e}")
        except asyncio.TimeoutError:
            logger.error("API request timeout", url=url)
            raise APIError("Request timeout")
    
    async def _get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_func,
        ttl: Optional[int] = None
    ) -> Any:
        """Get data from cache or fetch if not available."""
        # Try cache first
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit", key=cache_key)
                return cached
        
        # Fetch data
        logger.debug("Cache miss, fetching", key=cache_key)
        data = await fetch_func()
        
        # Cache the result
        if self.cache and data is not None:
            await self.cache.set(cache_key, data, ttl or self.cache_ttl)
        
        return data


class ACTTransportAPI(BaseAPIClient):
    """ACT Transport Canberra GTFS API client."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        # ACT Transport Canberra API endpoints
        super().__init__(
            base_url="https://www.transport.act.gov.au/contact-us/information-for-developers",
            api_key=api_key,
            rate_limit=2.0,  # Conservative rate limit
            **kwargs
        )
        
        # Real API endpoints would be:
        # - GTFS Static: https://gtfsdownload.transport.act.gov.au/gtfs/google_transit.zip
        # - GTFS Realtime: https://gtfsrealtime.transport.act.gov.au/
        
        self.gtfs_static_url = "https://gtfsdownload.transport.act.gov.au/gtfs/google_transit.zip"
        self.gtfs_realtime_base = "https://gtfsrealtime.transport.act.gov.au"
    
    async def fetch_gtfs_static_data(self, extract_to: Optional[Path] = None) -> Dict[str, Any]:
        """Fetch and extract GTFS static data."""
        cache_key = self.cache.make_key("gtfs_static") if self.cache else "gtfs_static"
        
        async def _fetch():
            logger.info("Fetching GTFS static data")
            
            # Download ZIP file
            async with self.session.get(self.gtfs_static_url) as response:
                if response.status != 200:
                    raise APIError(f"Failed to download GTFS data: {response.status}")
                
                zip_data = await response.read()
            
            # Extract and parse
            gtfs_data = {}
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
                
                if extract_to:
                    zip_ref.extractall(extract_to)
                    logger.info("GTFS data extracted", path=str(extract_to))
                
                # Parse key files
                for filename in ['stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt']:
                    if filename in zip_ref.namelist():
                        with zip_ref.open(filename) as file:
                            content = file.read().decode('utf-8')
                            gtfs_data[filename.replace('.txt', '')] = self._parse_csv_content(content)
            
            logger.info("GTFS static data processed", files=list(gtfs_data.keys()))
            return gtfs_data
        
        return await self._get_cached_or_fetch(cache_key, _fetch, ttl=86400)  # Cache for 24h
    
    async def fetch_stops(self) -> List[Stop]:
        """Fetch all GTFS stops."""
        cache_key = self.cache.make_key("gtfs_stops") if self.cache else "gtfs_stops"
        
        async def _fetch():
            static_data = await self.fetch_gtfs_static_data()
            stops_data = static_data.get('stops', [])
            
            stops = []
            for stop_row in stops_data:
                try:
                    # Convert to new Stop model format
                    stop_dict = {
                        'stop_id': stop_row.get('stop_id'),
                        'stop_name': stop_row.get('stop_name'),
                        'stop_lat': float(stop_row.get('stop_lat')),
                        'stop_lon': float(stop_row.get('stop_lon')),
                        'stop_code': stop_row.get('stop_code'),
                        'stop_desc': stop_row.get('stop_desc'),
                        'zone_id': stop_row.get('zone_id'),
                        'stop_url': stop_row.get('stop_url'),
                        'location_type': int(stop_row.get('location_type', 0)),
                        'parent_station': stop_row.get('parent_station'),
                        'wheelchair_boarding': int(stop_row.get('wheelchair_boarding', 0))
                    }
                    stop = Stop(**stop_dict)
                    stops.append(stop)
                except Exception as e:
                    logger.warning("Invalid stop data", stop_id=stop_row.get('stop_id'), error=str(e))
            
            logger.info("Fetched GTFS stops", count=len(stops))
            return [stop.dict() for stop in stops]
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=43200)  # 12h cache
        return [Stop(**stop) for stop in cached_data]
    
    async def fetch_routes(self) -> List[Route]:
        """Fetch all GTFS routes."""
        cache_key = self.cache.make_key("gtfs_routes") if self.cache else "gtfs_routes"
        
        async def _fetch():
            static_data = await self.fetch_gtfs_static_data()
            routes_data = static_data.get('routes', [])
            
            routes = []
            for route_row in routes_data:
                try:
                    # Convert to new Route model format
                    route_dict = {
                        'route_id': route_row.get('route_id'),
                        'agency_id': route_row.get('agency_id'),
                        'route_short_name': route_row.get('route_short_name'),
                        'route_long_name': route_row.get('route_long_name'),
                        'route_desc': route_row.get('route_desc'),
                        'route_type': int(route_row.get('route_type')),
                        'route_url': route_row.get('route_url'),
                        'route_color': route_row.get('route_color'),
                        'route_text_color': route_row.get('route_text_color'),
                        'route_sort_order': int(route_row.get('route_sort_order')) if route_row.get('route_sort_order') else None
                    }
                    route = Route(**route_dict)
                    routes.append(route)
                except Exception as e:
                    logger.warning("Invalid route data", route_id=route_row.get('route_id'), error=str(e))
            
            logger.info("Fetched GTFS routes", count=len(routes))
            return [route.dict() for route in routes]
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=43200)  # 12h cache
        return [Route(**route) for route in cached_data]
    
    async def stream_vehicle_positions(self) -> AsyncGenerator[VehiclePosition, None]:
        """Stream real-time vehicle positions."""
        logger.info("Starting vehicle position stream")
        
        while True:
            try:
                # In a real implementation, this would connect to the GTFS-RT feed
                # For now, we'll simulate with a simple fetch
                endpoint = f"{self.gtfs_realtime_base}/vehicle-positions"
                
                # This would actually parse protobuf GTFS-RT data
                response = await self._make_request("GET", endpoint)
                
                # Simulate parsing vehicle positions
                positions = response.get("vehicle_positions", [])
                
                for pos_data in positions:
                    try:
                        position = VehiclePosition(**pos_data)
                        yield position
                    except Exception as e:
                        logger.warning("Invalid vehicle position", error=str(e))
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error("Vehicle position stream error", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def fetch_service_alerts(self) -> List[Alert]:
        """Fetch current service alerts."""
        cache_key = self.cache.make_key("service_alerts") if self.cache else "service_alerts"
        
        async def _fetch():
            # In reality, this would parse GTFS-RT alerts feed
            endpoint = f"{self.gtfs_realtime_base}/alerts"
            
            response = await self._make_request("GET", endpoint)
            alerts_data = response.get("alerts", [])
            
            alerts = []
            for alert_data in alerts_data:
                try:
                    # Convert to new Alert model format
                    alert_dict = {
                        'alert_id': alert_data.get('alert_id'),
                        'header_text': alert_data.get('header_text', 'Service Alert'),
                        'description_text': alert_data.get('description_text'),
                        'url': alert_data.get('url'),
                        'active_period_start': datetime.fromisoformat(alert_data['active_period_start']) if alert_data.get('active_period_start') else datetime.now(),
                        'active_period_end': datetime.fromisoformat(alert_data['active_period_end']) if alert_data.get('active_period_end') else None,
                        'affected_routes': alert_data.get('informed_entity_route_ids', []),
                        'affected_stops': alert_data.get('informed_entity_stop_ids', [])
                    }
                    alert = Alert(**alert_dict)
                    alerts.append(alert)
                except Exception as e:
                    logger.warning("Invalid alert data", error=str(e))
            
            logger.info("Fetched service alerts", count=len(alerts))
            return [alert.dict() for alert in alerts]
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=300)  # 5min cache
        return [Alert(**alert) for alert in cached_data]
    
    def _parse_csv_content(self, content: str) -> List[Dict[str, str]]:
        """Parse CSV content to list of dictionaries."""
        lines = content.strip().split('\n')
        if not lines:
            return []
        
        headers = [h.strip() for h in lines[0].split(',')]
        rows = []
        
        for line in lines[1:]:
            values = [v.strip().strip('"') for v in line.split(',')]
            if len(values) == len(headers):
                row = dict(zip(headers, values))
                rows.append(row)
        
        return rows


class OpenStreetMapAPI(BaseAPIClient):
    """OpenStreetMap API client for geocoding and geographic data."""
    
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://nominatim.openstreetmap.org",
            rate_limit=1.0,  # Respectful rate limiting for Nominatim
            **kwargs
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        headers = super()._get_default_headers()
        headers['User-Agent'] = 'SmartSchoolGo/1.0.0 (contact@smartschoolgo.com)'
        return headers
    
    async def geocode_address(self, address: str, country_code: str = "AU") -> Optional[GeocodeResult]:
        """Geocode an address to coordinates."""
        cache_key = self.cache.make_key("geocode", address, country_code) if self.cache else f"geocode:{address}"
        
        async def _fetch():
            params = {
                'q': address,
                'format': 'json',
                'countrycodes': country_code,
                'limit': 1,
                'addressdetails': 1
            }
            
            response = await self._make_request("GET", "search", params=params)
            
            if not response or not isinstance(response, list) or len(response) == 0:
                logger.warning("No geocoding results", address=address)
                return None
            
            result = response[0]
            
            geocode_result = GeocodeResult(
                formatted_address=result.get('display_name', ''),
                latitude=float(result['lat']),
                longitude=float(result['lon']),
                accuracy=result.get('class'),
                place_type=result.get('type'),
                confidence=float(result.get('importance', 0.5))
            )
            
            logger.info("Address geocoded", address=address, result=geocode_result.formatted_address)
            return geocode_result.dict()
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=86400)  # 24h cache
        return GeocodeResult(**cached_data) if cached_data else None
    
    async def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodeResult]:
        """Reverse geocode coordinates to address."""
        cache_key = self.cache.make_key("reverse_geocode", latitude, longitude) if self.cache else f"reverse:{latitude},{longitude}"
        
        async def _fetch():
            params = {
                'lat': latitude,
                'lon': longitude,
                'format': 'json',
                'zoom': 18,
                'addressdetails': 1
            }
            
            response = await self._make_request("GET", "reverse", params=params)
            
            if not response:
                logger.warning("No reverse geocoding results", lat=latitude, lon=longitude)
                return None
            
            geocode_result = GeocodeResult(
                formatted_address=response.get('display_name', ''),
                latitude=latitude,
                longitude=longitude,
                accuracy=response.get('class'),
                place_type=response.get('type'),
                confidence=float(response.get('importance', 0.5))
            )
            
            logger.info("Coordinates reverse geocoded", lat=latitude, lon=longitude)
            return geocode_result.dict()
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=86400)  # 24h cache
        return GeocodeResult(**cached_data) if cached_data else None
    
    async def bulk_geocode(
        self,
        addresses: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Optional[GeocodeResult]]:
        """Bulk geocode multiple addresses with progress tracking."""
        results = {}
        progress = ProgressInfo(
            total=len(addresses),
            completed=0,
            failed=0,
            start_time=datetime.now(),
            current_time=datetime.now()
        )
        
        logger.info("Starting bulk geocoding", total=len(addresses))
        
        for i, address in enumerate(addresses):
            try:
                result = await self.geocode_address(address)
                results[address] = result
                progress.completed += 1
                
            except Exception as e:
                logger.error("Geocoding failed", address=address, error=str(e))
                results[address] = None
                progress.failed += 1
            
            # Update progress
            progress.current_time = datetime.now()
            
            if progress.completed > 0:
                rate = progress.completed / progress.elapsed_time.total_seconds()
                remaining = progress.total - progress.completed - progress.failed
                progress.estimated_completion = progress.current_time + timedelta(seconds=remaining / rate)
            
            if progress_callback:
                progress_callback(progress)
            
            logger.debug(
                "Bulk geocoding progress",
                completed=progress.completed,
                failed=progress.failed,
                total=progress.total,
                percent=progress.progress_percent
            )
        
        logger.info(
            "Bulk geocoding completed",
            total=progress.total,
            successful=progress.completed,
            failed=progress.failed
        )
        
        return results


class WeatherAPI(BaseAPIClient):
    """Weather API client for meteorological data."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        # Using OpenWeatherMap as example
        super().__init__(
            base_url="https://api.openweathermap.org/data/2.5",
            api_key=api_key,
            rate_limit=60.0,  # 60 calls per minute
            **kwargs
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        headers = super()._get_default_headers()
        # OpenWeatherMap uses query parameter for API key
        headers.pop('Authorization', None)
        return headers
    
    async def get_current_weather(self, latitude: float, longitude: float) -> Optional[WeatherData]:
        """Get current weather for coordinates."""
        cache_key = self.cache.make_key("weather_current", latitude, longitude) if self.cache else f"weather:{latitude},{longitude}"
        
        async def _fetch():
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = await self._make_request("GET", "weather", params=params)
            
            weather_data = WeatherData(
                temperature=response['main']['temp'],
                humidity=response['main']['humidity'],
                pressure=response['main'].get('pressure'),
                wind_speed=response.get('wind', {}).get('speed'),
                wind_direction=response.get('wind', {}).get('deg'),
                visibility=response.get('visibility', 0) / 1000 if 'visibility' in response else None,
                precipitation=response.get('rain', {}).get('1h', 0) + response.get('snow', {}).get('1h', 0),
                weather_condition=response['weather'][0]['main'],
                timestamp=datetime.now(),
                location=(latitude, longitude)
            )
            
            logger.info("Weather data fetched", lat=latitude, lon=longitude, condition=weather_data.weather_condition)
            return weather_data.dict()
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=900)  # 15min cache
        return WeatherData(**cached_data) if cached_data else None
    
    async def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 5
    ) -> List[WeatherData]:
        """Get weather forecast for coordinates."""
        cache_key = self.cache.make_key("weather_forecast", latitude, longitude, days) if self.cache else f"forecast:{latitude},{longitude}:{days}"
        
        async def _fetch():
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 3-hour intervals
            }
            
            response = await self._make_request("GET", "forecast", params=params)
            
            forecasts = []
            for item in response['list']:
                forecast = WeatherData(
                    temperature=item['main']['temp'],
                    humidity=item['main']['humidity'],
                    pressure=item['main'].get('pressure'),
                    wind_speed=item.get('wind', {}).get('speed'),
                    wind_direction=item.get('wind', {}).get('deg'),
                    visibility=item.get('visibility', 0) / 1000 if 'visibility' in item else None,
                    precipitation=item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0),
                    weather_condition=item['weather'][0]['main'],
                    timestamp=datetime.fromtimestamp(item['dt']),
                    location=(latitude, longitude)
                )
                forecasts.append(forecast)
            
            logger.info("Weather forecast fetched", lat=latitude, lon=longitude, periods=len(forecasts))
            return [f.dict() for f in forecasts]
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=3600)  # 1h cache
        return [WeatherData(**f) for f in cached_data] if cached_data else []


class SchoolDataAPI(BaseAPIClient):
    """API client for school location and enrollment data."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        # This would typically be an education department API
        super().__init__(
            base_url="https://api.education.act.gov.au/v1",
            api_key=api_key,
            rate_limit=10.0,
            **kwargs
        )
    
    async def fetch_schools(self, school_type: Optional[str] = None) -> List[School]:
        """Fetch school data, optionally filtered by type."""
        cache_key = self.cache.make_key("schools", school_type or "all") if self.cache else f"schools:{school_type}"
        
        async def _fetch():
            params = {}
            if school_type:
                params['type'] = school_type
            
            response = await self._make_request("GET", "schools", params=params)
            schools_data = response.get("schools", [])
            
            schools = []
            for school_data in schools_data:
                try:
                    # Convert to new School model format
                    school_dict = {
                        'school_id': school_data.get('school_id'),
                        'name': school_data.get('name'),
                        'address': school_data.get('address'),
                        'suburb': school_data.get('suburb', 'Unknown'),
                        'postcode': school_data.get('postcode', '0000'),
                        'coordinate': Coordinate(
                            latitude=school_data.get('latitude'),
                            longitude=school_data.get('longitude')
                        ),
                        'school_type': school_data.get('school_type', 'combined'),
                        'sector': school_data.get('sector', 'government'),
                        'grades': school_data.get('grades', []),
                        'enrollment': school_data.get('enrollment'),
                        'phone': school_data.get('phone'),
                        'email': school_data.get('email'),
                        'website': school_data.get('website')
                    }
                    school = School(**school_dict)
                    schools.append(school)
                except Exception as e:
                    logger.warning("Invalid school data", school_id=school_data.get('school_id'), error=str(e))
            
            logger.info("Schools fetched", count=len(schools), type=school_type)
            return [school.dict() for school in schools]
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=86400)  # 24h cache
        return [School(**school) for school in cached_data]
    
    async def get_school_by_id(self, school_id: str) -> Optional[School]:
        """Get specific school by ID."""
        cache_key = self.cache.make_key("school", school_id) if self.cache else f"school:{school_id}"
        
        async def _fetch():
            response = await self._make_request("GET", f"schools/{school_id}")
            school_data = response.get("school")
            
            if not school_data:
                return None
            
            school = School(
                school_id=school_data.get('school_id'),
                name=school_data.get('name'),
                address=school_data.get('address'),
                suburb=school_data.get('suburb', 'Unknown'),
                postcode=school_data.get('postcode', '0000'),
                coordinate=Coordinate(
                    latitude=school_data.get('latitude'),
                    longitude=school_data.get('longitude')
                ),
                school_type=school_data.get('school_type', 'combined'),
                sector=school_data.get('sector', 'government')
            )
            logger.info("School fetched", school_id=school_id, name=school.name)
            return school.dict()
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=86400)
        return School(**cached_data) if cached_data else None
    
    async def search_schools_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0
    ) -> List[School]:
        """Search for schools near coordinates."""
        cache_key = self.cache.make_key("schools_near", latitude, longitude, radius_km) if self.cache else f"schools_near:{latitude},{longitude}:{radius_km}"
        
        async def _fetch():
            params = {
                'lat': latitude,
                'lon': longitude,
                'radius': radius_km
            }
            
            response = await self._make_request("GET", "schools/search", params=params)
            schools_data = response.get("schools", [])
            
            schools = []
            for school_data in schools_data:
                try:
                    school = School(
                        school_id=school_data.get('school_id'),
                        name=school_data.get('name'),
                        address=school_data.get('address'),
                        suburb=school_data.get('suburb', 'Unknown'),
                        postcode=school_data.get('postcode', '0000'),
                        coordinate=Coordinate(
                            latitude=school_data.get('latitude'),
                            longitude=school_data.get('longitude')
                        ),
                        school_type=school_data.get('school_type', 'combined'),
                        sector=school_data.get('sector', 'government')
                    )
                    schools.append(school)
                except Exception as e:
                    logger.warning("Invalid school data", school_id=school_data.get('school_id'), error=str(e))
            
            logger.info("Schools near location fetched", lat=latitude, lon=longitude, count=len(schools))
            return [school.dict() for school in schools]
        
        cached_data = await self._get_cached_or_fetch(cache_key, _fetch, ttl=3600)  # 1h cache
        return [School(**school) for school in cached_data]


# Export classes and functions
__all__ = [
    'APIError', 'RateLimitError', 'ValidationError', 'CacheError',
    'RateLimiter', 'CacheManager', 'BaseAPIClient',
    'ACTTransportAPI', 'OpenStreetMapAPI', 'WeatherAPI', 'SchoolDataAPI'
]