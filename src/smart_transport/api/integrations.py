"""
API Integration Layer for SmartSchoolGo
Provides unified interface for external API integrations with retry logic, synchronization, and webhooks
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import httpx
from fastapi import HTTPException, BackgroundTasks
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import boto3
from botocore.exceptions import ClientError
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from concurrent.futures import ThreadPoolExecutor
import pickle
from cryptography.fernet import Fernet
import jwt
from datetime import timezone
import hmac
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import osmnx as ox
from geopy.distance import geodesic
import requests_cache
from cachetools import TTLCache, LRUCache
import schedule
import threading
from queue import Queue, PriorityQueue
import signal
import sys
from contextlib import asynccontextmanager
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

# ============================================================================
# BASE CLASSES AND ENUMS
# ============================================================================

class IntegrationStatus(Enum):
    """Integration status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SYNCING = "syncing"

class SyncStrategy(Enum):
    """Data synchronization strategies"""
    FULL_REPLACE = "full_replace"
    INCREMENTAL = "incremental"
    MERGE = "merge"
    DELTA_ONLY = "delta_only"
    BIDIRECTIONAL = "bidirectional"

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    LATEST_WINS = "latest_wins"
    SOURCE_WINS = "source_wins"
    LOCAL_WINS = "local_wins"
    MANUAL = "manual"
    MERGE = "merge"

@dataclass
class APICredentials:
    """API credentials management"""
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expiry: Optional[datetime] = None
    base_url: str = ""
    timeout: int = 30
    max_retries: int = 3

@dataclass
class SyncResult:
    """Synchronization result tracking"""
    success: bool
    records_processed: int = 0
    records_added: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    conflicts_resolved: int = 0
    errors: List[str] = field(default_factory=list)
    sync_timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0

# ============================================================================
# BASE INTEGRATION CLIENT
# ============================================================================

class BaseIntegrationClient(ABC):
    """Base class for all external API integrations"""
    
    def __init__(self, credentials: APICredentials, cache_ttl: int = 300):
        self.credentials = credentials
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=aiohttp.ClientError
        )
        self.metrics = IntegrationMetrics()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.credentials.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def ensure_token_valid(self):
        """Ensure authentication token is valid"""
        if self.credentials.token_expiry and datetime.now() >= self.credentials.token_expiry:
            await self.refresh_token()
            
    @abstractmethod
    async def refresh_token(self):
        """Refresh authentication token"""
        pass
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        async with self.rate_limiter:
            await self.ensure_token_valid()
            
            url = urljoin(self.credentials.base_url, endpoint)
            headers = kwargs.pop('headers', {})
            
            if self.credentials.token:
                headers['Authorization'] = f'Bearer {self.credentials.token}'
            elif self.credentials.api_key:
                headers['X-API-Key'] = self.credentials.api_key
                
            async with self.circuit_breaker:
                async with self.session.request(
                    method, url, headers=headers, **kwargs
                ) as response:
                    self.metrics.record_request(method, endpoint, response.status)
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        raise aiohttp.ClientError(
                            f"API request failed: {response.status} - {error_text}"
                        )
                        
                    return await response.json()

# ============================================================================
# ACT GOVERNMENT API INTEGRATION
# ============================================================================

class ACTGovernmentAPI(BaseIntegrationClient):
    """ACT Government API integration with automatic token refresh"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials)
        self.endpoints = {
            'schools': '/api/v1/schools',
            'transport': '/api/v1/transport',
            'demographics': '/api/v1/demographics',
            'safety': '/api/v1/safety',
            'infrastructure': '/api/v1/infrastructure'
        }
        
    async def refresh_token(self):
        """Refresh OAuth2 token"""
        token_url = urljoin(self.credentials.base_url, '/oauth/token')
        
        async with self.session.post(
            token_url,
            data={
                'grant_type': 'refresh_token',
                'refresh_token': self.credentials.refresh_token,
                'client_id': self.credentials.client_id,
                'client_secret': self.credentials.client_secret
            }
        ) as response:
            if response.status == 200:
                data = await response.json()
                self.credentials.token = data['access_token']
                self.credentials.refresh_token = data.get('refresh_token')
                self.credentials.token_expiry = datetime.now() + timedelta(
                    seconds=data.get('expires_in', 3600)
                )
                logger.info("ACT Government API token refreshed successfully")
            else:
                raise Exception(f"Token refresh failed: {response.status}")
                
    async def get_schools(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Get schools data with optional filters"""
        params = filters or {}
        return await self.make_request('GET', self.endpoints['schools'], params=params)
        
    async def get_transport_routes(self, school_id: Optional[str] = None) -> List[Dict]:
        """Get transport routes for specific school or all schools"""
        params = {'school_id': school_id} if school_id else {}
        return await self.make_request('GET', self.endpoints['transport'], params=params)
        
    async def get_safety_incidents(
        self,
        start_date: datetime,
        end_date: datetime,
        severity: Optional[str] = None
    ) -> List[Dict]:
        """Get safety incidents within date range"""
        params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        if severity:
            params['severity'] = severity
            
        return await self.make_request('GET', self.endpoints['safety'], params=params)

# ============================================================================
# OPENSTREETMAP DATA SYNCHRONIZATION
# ============================================================================

class OpenStreetMapSync(BaseIntegrationClient):
    """OpenStreetMap data synchronization with change detection"""
    
    def __init__(self, credentials: APICredentials, bbox: Tuple[float, float, float, float]):
        super().__init__(credentials)
        self.bbox = bbox  # (min_lat, min_lon, max_lat, max_lon)
        self.last_sync: Optional[datetime] = None
        self.change_detector = ChangeDetector()
        
    async def refresh_token(self):
        """OSM doesn't require token refresh"""
        pass
        
    async def sync_road_network(self) -> SyncResult:
        """Synchronize road network data"""
        result = SyncResult(success=False)
        start_time = datetime.now()
        
        try:
            # Download road network
            G = ox.graph_from_bbox(
                self.bbox[2], self.bbox[0], self.bbox[3], self.bbox[1],
                network_type='drive'
            )
            
            # Convert to GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)
            
            # Detect changes
            if self.last_sync:
                changes = self.change_detector.detect_changes(
                    edges, 'osm_road_network'
                )
                result.records_added = changes['added']
                result.records_updated = changes['updated']
                result.records_deleted = changes['deleted']
            else:
                result.records_added = len(edges)
                
            # Store in database
            await self.store_road_network(nodes, edges)
            
            self.last_sync = datetime.now()
            result.success = True
            result.records_processed = len(edges)
            
        except Exception as e:
            logger.error(f"OSM sync failed: {str(e)}")
            result.errors.append(str(e))
            
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result
        
    async def sync_schools(self) -> SyncResult:
        """Synchronize school locations from OSM"""
        result = SyncResult(success=False)
        
        try:
            # Query schools in bounding box
            schools = ox.geometries_from_bbox(
                self.bbox[2], self.bbox[0], self.bbox[3], self.bbox[1],
                tags={'amenity': 'school'}
            )
            
            # Process and store schools
            for idx, school in schools.iterrows():
                school_data = {
                    'osm_id': idx,
                    'name': school.get('name', 'Unknown'),
                    'geometry': school.geometry,
                    'tags': school.to_dict()
                }
                await self.store_school(school_data)
                result.records_processed += 1
                
            result.success = True
            
        except Exception as e:
            logger.error(f"School sync failed: {str(e)}")
            result.errors.append(str(e))
            
        return result
        
    async def store_road_network(self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame):
        """Store road network in database"""
        # Implementation would store in PostGIS database
        pass
        
    async def store_school(self, school_data: Dict):
        """Store school data in database"""
        # Implementation would store in database
        pass

# ============================================================================
# WEATHER SERVICE INTEGRATION
# ============================================================================

class WeatherServiceIntegration(BaseIntegrationClient):
    """Weather service integration with alert processing"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials)
        self.alert_handlers = {}
        self.severity_levels = {
            'extreme': 4,
            'severe': 3,
            'moderate': 2,
            'minor': 1
        }
        
    async def refresh_token(self):
        """Weather API uses API key, no refresh needed"""
        pass
        
    async def get_current_weather(self, location: Tuple[float, float]) -> Dict:
        """Get current weather for location"""
        params = {
            'lat': location[0],
            'lon': location[1],
            'appid': self.credentials.api_key,
            'units': 'metric'
        }
        return await self.make_request('GET', '/weather', params=params)
        
    async def get_weather_alerts(self, region: str) -> List[Dict]:
        """Get weather alerts for region"""
        params = {
            'region': region,
            'appid': self.credentials.api_key
        }
        alerts = await self.make_request('GET', '/alerts', params=params)
        
        # Process alerts
        processed_alerts = []
        for alert in alerts:
            processed = await self.process_alert(alert)
            processed_alerts.append(processed)
            
        return processed_alerts
        
    async def process_alert(self, alert: Dict) -> Dict:
        """Process weather alert and trigger handlers"""
        severity = alert.get('severity', 'minor')
        alert_type = alert.get('type', 'unknown')
        
        # Enhance alert with severity score
        alert['severity_score'] = self.severity_levels.get(severity, 0)
        
        # Trigger registered handlers
        if alert_type in self.alert_handlers:
            await self.alert_handlers[alert_type](alert)
            
        # Check if routes need adjustment
        if alert['severity_score'] >= 3:
            alert['requires_route_adjustment'] = True
            alert['affected_schools'] = await self.identify_affected_schools(alert)
            
        return alert
        
    async def identify_affected_schools(self, alert: Dict) -> List[str]:
        """Identify schools affected by weather alert"""
        # Implementation would check alert area against school locations
        return []
        
    def register_alert_handler(self, alert_type: str, handler):
        """Register handler for specific alert type"""
        self.alert_handlers[alert_type] = handler

# ============================================================================
# SCHOOL SYSTEM INTEGRATION
# ============================================================================

class SchoolSystemIntegration(BaseIntegrationClient):
    """School system integration with enrollment updates"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials)
        self.enrollment_cache = LRUCache(maxsize=1000)
        self.sync_queue = asyncio.Queue()
        
    async def refresh_token(self):
        """Refresh school system API token"""
        # Implementation specific to school system API
        pass
        
    async def sync_enrollments(self, school_id: str) -> SyncResult:
        """Sync student enrollments for a school"""
        result = SyncResult(success=False)
        
        try:
            # Get current enrollments from school system
            enrollments = await self.make_request(
                'GET', f'/schools/{school_id}/enrollments'
            )
            
            # Get local enrollments
            local_enrollments = await self.get_local_enrollments(school_id)
            
            # Detect changes
            changes = self.detect_enrollment_changes(enrollments, local_enrollments)
            
            # Process changes
            for student_id in changes['added']:
                await self.add_student_enrollment(school_id, student_id)
                result.records_added += 1
                
            for student_id in changes['removed']:
                await self.remove_student_enrollment(school_id, student_id)
                result.records_deleted += 1
                
            for student_id in changes['updated']:
                await self.update_student_enrollment(school_id, student_id)
                result.records_updated += 1
                
            result.success = True
            result.records_processed = len(enrollments)
            
        except Exception as e:
            logger.error(f"Enrollment sync failed: {str(e)}")
            result.errors.append(str(e))
            
        return result
        
    async def get_student_details(self, student_id: str) -> Dict:
        """Get detailed student information"""
        if student_id in self.enrollment_cache:
            return self.enrollment_cache[student_id]
            
        details = await self.make_request('GET', f'/students/{student_id}')
        self.enrollment_cache[student_id] = details
        return details
        
    async def handle_enrollment_webhook(self, payload: Dict):
        """Handle enrollment update webhook"""
        event_type = payload.get('event_type')
        
        if event_type == 'enrollment.created':
            await self.sync_queue.put({
                'action': 'add',
                'student_id': payload['student_id'],
                'school_id': payload['school_id']
            })
        elif event_type == 'enrollment.updated':
            await self.sync_queue.put({
                'action': 'update',
                'student_id': payload['student_id'],
                'school_id': payload['school_id']
            })
        elif event_type == 'enrollment.deleted':
            await self.sync_queue.put({
                'action': 'remove',
                'student_id': payload['student_id'],
                'school_id': payload['school_id']
            })
            
    async def process_sync_queue(self):
        """Process enrollment sync queue"""
        while True:
            try:
                item = await self.sync_queue.get()
                
                if item['action'] == 'add':
                    await self.add_student_enrollment(
                        item['school_id'], item['student_id']
                    )
                elif item['action'] == 'update':
                    await self.update_student_enrollment(
                        item['school_id'], item['student_id']
                    )
                elif item['action'] == 'remove':
                    await self.remove_student_enrollment(
                        item['school_id'], item['student_id']
                    )
                    
            except Exception as e:
                logger.error(f"Sync queue processing error: {str(e)}")
                
    def detect_enrollment_changes(
        self,
        remote: List[Dict],
        local: List[Dict]
    ) -> Dict[str, Set[str]]:
        """Detect enrollment changes between remote and local"""
        remote_ids = {e['student_id'] for e in remote}
        local_ids = {e['student_id'] for e in local}
        
        return {
            'added': remote_ids - local_ids,
            'removed': local_ids - remote_ids,
            'updated': remote_ids & local_ids  # Simplified - would check actual changes
        }
        
    async def get_local_enrollments(self, school_id: str) -> List[Dict]:
        """Get local enrollment records"""
        # Implementation would query local database
        return []
        
    async def add_student_enrollment(self, school_id: str, student_id: str):
        """Add student enrollment"""
        # Implementation would add to local database
        pass
        
    async def update_student_enrollment(self, school_id: str, student_id: str):
        """Update student enrollment"""
        # Implementation would update local database
        pass
        
    async def remove_student_enrollment(self, school_id: str, student_id: str):
        """Remove student enrollment"""
        # Implementation would remove from local database
        pass

# ============================================================================
# EMERGENCY SERVICES INTEGRATION
# ============================================================================

class EmergencyServicesIntegration(BaseIntegrationClient):
    """Emergency services integration for incident reporting"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials)
        self.incident_queue = PriorityQueue()
        self.active_incidents = {}
        
    async def refresh_token(self):
        """Refresh emergency services API token"""
        # Implementation specific to emergency services API
        pass
        
    async def report_incident(
        self,
        incident_type: str,
        location: Tuple[float, float],
        severity: str,
        details: Dict
    ) -> str:
        """Report incident to emergency services"""
        incident_data = {
            'type': incident_type,
            'location': {
                'lat': location[0],
                'lon': location[1]
            },
            'severity': severity,
            'details': details,
            'reported_at': datetime.now().isoformat(),
            'reporter': 'SmartSchoolGo'
        }
        
        response = await self.make_request('POST', '/incidents', json=incident_data)
        incident_id = response['incident_id']
        
        # Track active incident
        self.active_incidents[incident_id] = incident_data
        
        return incident_id
        
    async def get_incident_status(self, incident_id: str) -> Dict:
        """Get status of reported incident"""
        return await self.make_request('GET', f'/incidents/{incident_id}')
        
    async def get_active_incidents(self, radius_km: float, center: Tuple[float, float]) -> List[Dict]:
        """Get active incidents within radius of center point"""
        params = {
            'lat': center[0],
            'lon': center[1],
            'radius': radius_km
        }
        return await self.make_request('GET', '/incidents/active', params=params)
        
    async def subscribe_to_incidents(self, webhook_url: str, filters: Optional[Dict] = None):
        """Subscribe to incident updates via webhook"""
        subscription_data = {
            'webhook_url': webhook_url,
            'filters': filters or {},
            'events': ['incident.created', 'incident.updated', 'incident.resolved']
        }
        return await self.make_request('POST', '/webhooks/subscribe', json=subscription_data)

# ============================================================================
# WEBHOOK HANDLERS
# ============================================================================

class WebhookHandler:
    """Central webhook handler for all integrations"""
    
    def __init__(self):
        self.handlers = {}
        self.signatures = {}
        self.event_log = []
        
    def register_handler(self, source: str, handler, secret: Optional[str] = None):
        """Register webhook handler for source"""
        self.handlers[source] = handler
        if secret:
            self.signatures[source] = secret
            
    async def handle_webhook(
        self,
        source: str,
        headers: Dict,
        payload: Union[Dict, str]
    ) -> Dict:
        """Process incoming webhook"""
        # Verify signature if required
        if source in self.signatures:
            if not self.verify_signature(source, headers, payload):
                raise HTTPException(status_code=401, detail="Invalid signature")
                
        # Log event
        self.event_log.append({
            'source': source,
            'timestamp': datetime.now(),
            'headers': headers,
            'payload': payload
        })
        
        # Process webhook
        if source in self.handlers:
            result = await self.handlers[source](payload)
            return {'status': 'success', 'result': result}
        else:
            logger.warning(f"No handler registered for source: {source}")
            return {'status': 'no_handler'}
            
    def verify_signature(self, source: str, headers: Dict, payload: Union[Dict, str]) -> bool:
        """Verify webhook signature"""
        secret = self.signatures[source]
        
        if source == 'github':
            signature = headers.get('X-Hub-Signature-256')
            if not signature:
                return False
            payload_bytes = json.dumps(payload).encode() if isinstance(payload, dict) else payload.encode()
            expected = 'sha256=' + hmac.new(
                secret.encode(), payload_bytes, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
            
        elif source == 'stripe':
            signature = headers.get('Stripe-Signature')
            # Stripe signature verification logic
            return True  # Simplified
            
        return True  # Default to true for unknown sources

# ============================================================================
# BATCH PROCESSING
# ============================================================================

class BatchProcessor:
    """Batch processing for large data operations"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.progress_callbacks = []
        
    async def process_batch(
        self,
        items: List[Any],
        processor_func,
        progress_callback=None
    ) -> List[Any]:
        """Process items in batches"""
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch
            batch_results = await asyncio.gather(
                *[processor_func(item) for item in batch],
                return_exceptions=True
            )
            
            # Filter out exceptions
            valid_results = [
                r for r in batch_results
                if not isinstance(r, Exception)
            ]
            results.extend(valid_results)
            
            # Report progress
            if progress_callback:
                progress = (i + len(batch)) / total_items * 100
                await progress_callback(progress, len(valid_results), len(batch))
                
        return results
        
    async def bulk_import(
        self,
        data_source: str,
        transformer_func,
        storage_func
    ) -> SyncResult:
        """Bulk import data from source"""
        result = SyncResult(success=False)
        
        try:
            # Read data from source
            raw_data = await self.read_data_source(data_source)
            
            # Transform data in batches
            transformed = await self.process_batch(
                raw_data,
                transformer_func
            )
            
            # Store data in batches
            stored = await self.process_batch(
                transformed,
                storage_func
            )
            
            result.success = True
            result.records_processed = len(raw_data)
            result.records_added = len(stored)
            
        except Exception as e:
            logger.error(f"Bulk import failed: {str(e)}")
            result.errors.append(str(e))
            
        return result
        
    async def read_data_source(self, source: str) -> List[Any]:
        """Read data from various sources"""
        if source.endswith('.csv'):
            return pd.read_csv(source).to_dict('records')
        elif source.endswith('.json'):
            with open(source) as f:
                return json.load(f)
        elif source.endswith('.xml'):
            tree = ET.parse(source)
            # Parse XML to list of dicts
            return []
        else:
            raise ValueError(f"Unsupported data source: {source}")

# ============================================================================
# EVENT STREAMING
# ============================================================================

class EventStreamingIntegration:
    """Integration with message queues and event streaming platforms"""
    
    def __init__(self, platform: str = 'kafka'):
        self.platform = platform
        self.producers = {}
        self.consumers = {}
        self.handlers = {}
        
    async def setup_kafka(self, bootstrap_servers: str):
        """Setup Kafka producer and consumer"""
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode()
        )
        await self.kafka_producer.start()
        
    async def setup_rabbitmq(self, connection_url: str):
        """Setup RabbitMQ connection"""
        # RabbitMQ async setup
        pass
        
    async def setup_aws_sqs(self, region: str):
        """Setup AWS SQS client"""
        self.sqs_client = boto3.client('sqs', region_name=region)
        
    async def publish_event(self, topic: str, event: Dict):
        """Publish event to stream"""
        event['timestamp'] = datetime.now().isoformat()
        event['source'] = 'SmartSchoolGo'
        
        if self.platform == 'kafka':
            await self.kafka_producer.send(topic, event)
        elif self.platform == 'rabbitmq':
            # Publish to RabbitMQ
            pass
        elif self.platform == 'sqs':
            # Send to SQS
            pass
            
    async def consume_events(self, topics: List[str]):
        """Consume events from stream"""
        if self.platform == 'kafka':
            consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers='localhost:9092',
                value_deserializer=lambda m: json.loads(m.decode())
            )
            await consumer.start()
            
            try:
                async for msg in consumer:
                    await self.handle_event(msg.topic, msg.value)
            finally:
                await consumer.stop()
                
    async def handle_event(self, topic: str, event: Dict):
        """Handle incoming event"""
        if topic in self.handlers:
            await self.handlers[topic](event)
        else:
            logger.warning(f"No handler for topic: {topic}")
            
    def register_handler(self, topic: str, handler):
        """Register event handler for topic"""
        self.handlers[topic] = handler

# ============================================================================
# DATA SYNCHRONIZATION SERVICE
# ============================================================================

class DataSynchronizationService:
    """Central service for data synchronization with conflict resolution"""
    
    def __init__(self):
        self.sync_configs = {}
        self.conflict_resolver = ConflictResolver()
        self.sync_history = []
        
    def configure_sync(
        self,
        source: str,
        target: str,
        strategy: SyncStrategy,
        conflict_resolution: ConflictResolution,
        schedule: Optional[str] = None
    ):
        """Configure synchronization between source and target"""
        self.sync_configs[f"{source}_{target}"] = {
            'source': source,
            'target': target,
            'strategy': strategy,
            'conflict_resolution': conflict_resolution,
            'schedule': schedule,
            'last_sync': None
        }
        
        if schedule:
            self.schedule_sync(source, target, schedule)
            
    async def sync_data(self, source: str, target: str) -> SyncResult:
        """Synchronize data between source and target"""
        config_key = f"{source}_{target}"
        if config_key not in self.sync_configs:
            raise ValueError(f"No sync configuration for {source} -> {target}")
            
        config = self.sync_configs[config_key]
        result = SyncResult(success=False)
        
        try:
            # Get data from source
            source_data = await self.get_data(source)
            
            # Get data from target
            target_data = await self.get_data(target)
            
            # Apply synchronization strategy
            if config['strategy'] == SyncStrategy.FULL_REPLACE:
                await self.replace_all_data(target, source_data)
                result.records_processed = len(source_data)
                
            elif config['strategy'] == SyncStrategy.INCREMENTAL:
                changes = await self.detect_changes(source_data, target_data)
                await self.apply_changes(target, changes, config['conflict_resolution'])
                result.records_added = len(changes['added'])
                result.records_updated = len(changes['updated'])
                result.records_deleted = len(changes['deleted'])
                
            elif config['strategy'] == SyncStrategy.MERGE:
                merged = await self.merge_data(
                    source_data, target_data, config['conflict_resolution']
                )
                await self.update_data(target, merged)
                result.records_processed = len(merged)
                
            result.success = True
            config['last_sync'] = datetime.now()
            
            # Record sync history
            self.sync_history.append({
                'source': source,
                'target': target,
                'result': result,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            result.errors.append(str(e))
            
        return result
        
    async def detect_changes(self, source_data: List[Dict], target_data: List[Dict]) -> Dict:
        """Detect changes between source and target data"""
        source_ids = {item['id'] for item in source_data}
        target_ids = {item['id'] for item in target_data}
        
        added = source_ids - target_ids
        deleted = target_ids - source_ids
        common = source_ids & target_ids
        
        updated = []
        for item_id in common:
            source_item = next(i for i in source_data if i['id'] == item_id)
            target_item = next(i for i in target_data if i['id'] == item_id)
            
            if self.has_changed(source_item, target_item):
                updated.append(item_id)
                
        return {
            'added': [i for i in source_data if i['id'] in added],
            'updated': [i for i in source_data if i['id'] in updated],
            'deleted': [i for i in target_data if i['id'] in deleted]
        }
        
    def has_changed(self, source: Dict, target: Dict) -> bool:
        """Check if item has changed"""
        # Compare checksums or modification timestamps
        source_checksum = hashlib.md5(json.dumps(source, sort_keys=True).encode()).hexdigest()
        target_checksum = hashlib.md5(json.dumps(target, sort_keys=True).encode()).hexdigest()
        return source_checksum != target_checksum
        
    async def apply_changes(
        self,
        target: str,
        changes: Dict,
        conflict_resolution: ConflictResolution
    ):
        """Apply changes to target with conflict resolution"""
        # Handle additions
        for item in changes['added']:
            await self.add_item(target, item)
            
        # Handle updates with conflict resolution
        for item in changes['updated']:
            existing = await self.get_item(target, item['id'])
            if existing:
                resolved = self.conflict_resolver.resolve(
                    item, existing, conflict_resolution
                )
                await self.update_item(target, resolved)
            else:
                await self.add_item(target, item)
                
        # Handle deletions
        for item in changes['deleted']:
            await self.delete_item(target, item['id'])
            
    async def get_data(self, source: str) -> List[Dict]:
        """Get data from source"""
        # Implementation would fetch from actual source
        return []
        
    async def replace_all_data(self, target: str, data: List[Dict]):
        """Replace all data in target"""
        # Implementation would replace data in target
        pass
        
    async def merge_data(
        self,
        source: List[Dict],
        target: List[Dict],
        conflict_resolution: ConflictResolution
    ) -> List[Dict]:
        """Merge source and target data"""
        # Implementation would merge datasets
        return []
        
    async def update_data(self, target: str, data: List[Dict]):
        """Update data in target"""
        # Implementation would update target
        pass
        
    async def get_item(self, target: str, item_id: str) -> Optional[Dict]:
        """Get single item from target"""
        # Implementation would fetch item
        return None
        
    async def add_item(self, target: str, item: Dict):
        """Add item to target"""
        # Implementation would add item
        pass
        
    async def update_item(self, target: str, item: Dict):
        """Update item in target"""
        # Implementation would update item
        pass
        
    async def delete_item(self, target: str, item_id: str):
        """Delete item from target"""
        # Implementation would delete item
        pass
        
    def schedule_sync(self, source: str, target: str, schedule_str: str):
        """Schedule automatic synchronization"""
        schedule.every().day.at(schedule_str).do(
            lambda: asyncio.create_task(self.sync_data(source, target))
        )

# ============================================================================
# CONFLICT RESOLVER
# ============================================================================

class ConflictResolver:
    """Resolve conflicts during data synchronization"""
    
    def resolve(
        self,
        source: Dict,
        target: Dict,
        strategy: ConflictResolution
    ) -> Dict:
        """Resolve conflict between source and target"""
        if strategy == ConflictResolution.LATEST_WINS:
            source_time = datetime.fromisoformat(source.get('modified_at', '1900-01-01'))
            target_time = datetime.fromisoformat(target.get('modified_at', '1900-01-01'))
            return source if source_time > target_time else target
            
        elif strategy == ConflictResolution.SOURCE_WINS:
            return source
            
        elif strategy == ConflictResolution.LOCAL_WINS:
            return target
            
        elif strategy == ConflictResolution.MERGE:
            # Merge non-conflicting fields
            merged = target.copy()
            for key, value in source.items():
                if key not in target or target[key] == value:
                    merged[key] = value
                elif isinstance(value, list) and isinstance(target[key], list):
                    # Merge lists
                    merged[key] = list(set(value + target[key]))
                elif isinstance(value, dict) and isinstance(target[key], dict):
                    # Recursively merge dicts
                    merged[key] = self.resolve(
                        value, target[key], ConflictResolution.MERGE
                    )
            return merged
            
        else:
            # Manual resolution required
            raise Exception("Manual conflict resolution required")

# ============================================================================
# CHANGE DETECTOR
# ============================================================================

class ChangeDetector:
    """Detect changes in data over time"""
    
    def __init__(self):
        self.checksums = {}
        
    def detect_changes(self, data: Any, key: str) -> Dict[str, int]:
        """Detect changes in data since last check"""
        current_checksum = self.calculate_checksum(data)
        
        if key not in self.checksums:
            self.checksums[key] = current_checksum
            return {'added': len(data), 'updated': 0, 'deleted': 0}
            
        previous_checksum = self.checksums[key]
        
        # Simple change detection - would be more sophisticated in production
        if current_checksum != previous_checksum:
            self.checksums[key] = current_checksum
            return {'added': 0, 'updated': len(data), 'deleted': 0}
            
        return {'added': 0, 'updated': 0, 'deleted': 0}
        
    def calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data"""
        if isinstance(data, pd.DataFrame):
            data_str = data.to_json(orient='records', sort_keys=True)
        else:
            data_str = json.dumps(data, sort_keys=True)
            
        return hashlib.sha256(data_str.encode()).hexdigest()

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
    async def __aenter__(self):
        if self.state == 'open':
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success - reset failure count
            self.failure_count = 0
            if self.state == 'half-open':
                self.state = 'closed'
        elif issubclass(exc_type, self.expected_exception):
            # Expected failure - increment count
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# ============================================================================
# INTEGRATION METRICS
# ============================================================================

class IntegrationMetrics:
    """Track integration performance metrics"""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
    def record_request(self, method: str, endpoint: str, status_code: int):
        """Record API request metrics"""
        key = f"{method}:{endpoint}"
        self.request_counts[key] += 1
        
        if status_code >= 400:
            self.error_counts[key] += 1
            
    def record_response_time(self, endpoint: str, duration: float):
        """Record response time"""
        self.response_times[endpoint].append(duration)
        
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            'request_counts': dict(self.request_counts),
            'error_counts': dict(self.error_counts),
            'avg_response_times': {
                k: sum(v) / len(v) if v else 0
                for k, v in self.response_times.items()
            }
        }

# ============================================================================
# INTEGRATION MANAGER
# ============================================================================

class IntegrationManager:
    """Central manager for all integrations"""
    
    def __init__(self):
        self.integrations = {}
        self.webhook_handler = WebhookHandler()
        self.sync_service = DataSynchronizationService()
        self.batch_processor = BatchProcessor()
        self.event_streaming = EventStreamingIntegration()
        
    async def initialize(self, config: Dict):
        """Initialize all integrations from configuration"""
        # Initialize ACT Government API
        if 'act_gov' in config:
            self.integrations['act_gov'] = ACTGovernmentAPI(
                APICredentials(**config['act_gov'])
            )
            
        # Initialize OpenStreetMap sync
        if 'osm' in config:
            self.integrations['osm'] = OpenStreetMapSync(
                APICredentials(**config['osm']['credentials']),
                tuple(config['osm']['bbox'])
            )
            
        # Initialize Weather service
        if 'weather' in config:
            self.integrations['weather'] = WeatherServiceIntegration(
                APICredentials(**config['weather'])
            )
            
        # Initialize School system
        if 'school_system' in config:
            self.integrations['school'] = SchoolSystemIntegration(
                APICredentials(**config['school_system'])
            )
            
        # Initialize Emergency services
        if 'emergency' in config:
            self.integrations['emergency'] = EmergencyServicesIntegration(
                APICredentials(**config['emergency'])
            )
            
        # Setup event streaming
        if 'event_streaming' in config:
            platform = config['event_streaming']['platform']
            self.event_streaming = EventStreamingIntegration(platform)
            
            if platform == 'kafka':
                await self.event_streaming.setup_kafka(
                    config['event_streaming']['bootstrap_servers']
                )
                
        logger.info(f"Initialized {len(self.integrations)} integrations")
        
    async def sync_all(self) -> Dict[str, SyncResult]:
        """Run all configured synchronizations"""
        results = {}
        
        for name, integration in self.integrations.items():
            if hasattr(integration, 'sync'):
                results[name] = await integration.sync()
                
        return results
        
    def get_integration(self, name: str):
        """Get specific integration"""
        return self.integrations.get(name)
        
    async def shutdown(self):
        """Shutdown all integrations gracefully"""
        for name, integration in self.integrations.items():
            if hasattr(integration, 'close'):
                await integration.close()
                
        logger.info("All integrations shut down")

# ============================================================================
# EXPORT INTEGRATION COMPONENTS
# ============================================================================

__all__ = [
    'IntegrationManager',
    'ACTGovernmentAPI',
    'OpenStreetMapSync',
    'WeatherServiceIntegration',
    'SchoolSystemIntegration',
    'EmergencyServicesIntegration',
    'WebhookHandler',
    'DataSynchronizationService',
    'BatchProcessor',
    'EventStreamingIntegration',
    'SyncResult',
    'IntegrationStatus',
    'SyncStrategy',
    'ConflictResolution'
]