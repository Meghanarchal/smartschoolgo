"""
FastAPI Backend for SmartSchoolGo

This module provides a comprehensive REST API backend with WebSocket support,
authentication, rate limiting, and comprehensive data operations for the
SmartSchoolGo transport management system.

Features:
- RESTful endpoints for all data operations (CRUD)
- Real-time WebSocket endpoints for live data streaming
- JWT-based authentication and authorization
- Request validation using Pydantic models
- Comprehensive API documentation with OpenAPI/Swagger
- Rate limiting and security middleware
- Database integration with connection pooling
- Caching layer with Redis
- Comprehensive error handling and logging
- Health checks and monitoring endpoints

Author: SmartSchoolGo Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import uuid
import os
from pathlib import Path

try:
    from fastapi import (
        FastAPI, HTTPException, Depends, status, Request, Response,
        WebSocket, WebSocketDisconnect, BackgroundTasks, Security
    )
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
    from fastapi.openapi.utils import get_openapi
    from pydantic import BaseModel, ValidationError, Field
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    import redis.asyncio as redis
    import jwt
    from passlib.context import CryptContext
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    import uvicorn
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    FASTAPI_LIBS_AVAILABLE = True
except ImportError as e:
    logging.error(f"FastAPI dependencies not available: {e}")
    FASTAPI_LIBS_AVAILABLE = False

if FASTAPI_LIBS_AVAILABLE:
    # Import local modules
    from ..data.models import *
    from ..data.database import DatabaseManager
    from ..models.ml_models import MLModelManager
    from ..models.realtime_engine import RealtimeAnalyticsEngine
    from config.settings import get_settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
settings = get_settings() if FASTAPI_LIBS_AVAILABLE else None
redis_client = None
db_manager = None
ml_manager = None
realtime_engine = None

# Metrics
if FASTAPI_LIBS_AVAILABLE:
    REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
    REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
    ACTIVE_CONNECTIONS = Gauge('websocket_connections', 'Active WebSocket connections')

# Security
if FASTAPI_LIBS_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    security = HTTPBearer()
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)


class UserRole:
    """User roles for authorization."""
    ADMIN = "admin"
    PLANNER = "planner"
    PARENT = "parent"
    DRIVER = "driver"
    GUEST = "guest"


# Pydantic models for API requests/responses
class UserLogin(BaseModel):
    """User login request model."""
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=50)


class UserCreate(BaseModel):
    """User creation request model."""
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=50)
    name: str = Field(..., min_length=2, max_length=100)
    role: str = Field(..., regex=f"^{UserRole.ADMIN}|{UserRole.PLANNER}|{UserRole.PARENT}|{UserRole.DRIVER}$")


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    name: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None


class TokenResponse(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse


class SchoolCreate(BaseModel):
    """School creation request model."""
    school_name: str = Field(..., min_length=2, max_length=200)
    school_type: str = Field(...)
    sector: str = Field(...)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    address: str = Field(..., min_length=10, max_length=500)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=100)
    capacity: int = Field(..., gt=0, le=5000)
    principal_name: Optional[str] = Field(None, max_length=100)


class StudentCreate(BaseModel):
    """Student creation request model."""
    name: str = Field(..., min_length=2, max_length=100)
    student_id: str = Field(..., min_length=1, max_length=50)
    school_id: str = Field(...)
    grade_level: int = Field(..., ge=1, le=12)
    home_latitude: float = Field(..., ge=-90, le=90)
    home_longitude: float = Field(..., ge=-180, le=180)
    home_address: str = Field(..., min_length=10, max_length=500)
    parent_name: str = Field(..., min_length=2, max_length=100)
    parent_phone: str = Field(..., min_length=10, max_length=20)
    parent_email: str = Field(..., min_length=5, max_length=100)
    special_needs: Optional[List[str]] = Field(default_factory=list)
    pickup_location: Optional[str] = Field(None, max_length=500)


class RouteCreate(BaseModel):
    """Route creation request model."""
    route_name: str = Field(..., min_length=2, max_length=100)
    route_type: str = Field(...)
    school_ids: List[str] = Field(..., min_items=1)
    capacity: int = Field(..., gt=0, le=200)
    estimated_duration: int = Field(..., gt=0, le=300)  # minutes
    departure_time: str = Field(...)  # HH:MM format
    return_time: str = Field(...)  # HH:MM format


class VehicleCreate(BaseModel):
    """Vehicle creation request model."""
    vehicle_id: str = Field(..., min_length=1, max_length=50)
    make: str = Field(..., min_length=2, max_length=50)
    model: str = Field(..., min_length=2, max_length=50)
    year: int = Field(..., ge=1990, le=2030)
    capacity: int = Field(..., gt=0, le=200)
    fuel_type: str = Field(..., regex="^diesel|electric|hybrid|cng$")
    registration: str = Field(..., min_length=5, max_length=20)
    vin: Optional[str] = Field(None, max_length=30)


class WebSocketManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}  # user_id -> [connection_ids]
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str = None):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(connection_id)
        
        ACTIVE_CONNECTIONS.inc()
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
    
    def disconnect(self, connection_id: str, user_id: str = None):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if user_id and user_id in self.user_connections:
            if connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
        
        ACTIVE_CONNECTIONS.dec()
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to all connections for a user."""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections."""
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)


# Global WebSocket manager
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    logger.info("Starting SmartSchoolGo API server...")
    
    try:
        # Initialize Redis
        global redis_client
        if settings:
            redis_client = redis.Redis.from_url(settings.redis_url)
            await redis_client.ping()
            logger.info("Redis connection established")
        
        # Initialize database
        global db_manager
        if settings:
            db_manager = DatabaseManager(settings.database_url)
            await db_manager.initialize()
            logger.info("Database connection established")
        
        # Initialize ML models
        global ml_manager
        ml_manager = MLModelManager()
        logger.info("ML model manager initialized")
        
        # Initialize real-time engine
        global realtime_engine
        realtime_engine = RealtimeAnalyticsEngine()
        # Start real-time engine in background
        asyncio.create_task(realtime_engine.start())
        logger.info("Real-time analytics engine started")
        
        logger.info("SmartSchoolGo API server started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SmartSchoolGo API server...")
    
    try:
        if realtime_engine:
            await realtime_engine.stop()
            logger.info("Real-time analytics engine stopped")
        
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
        
        if db_manager:
            await db_manager.close()
            logger.info("Database connection closed")
        
        logger.info("SmartSchoolGo API server shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
if FASTAPI_LIBS_AVAILABLE:
    app = FastAPI(
        title="SmartSchoolGo API",
        description="Intelligent School Transport Management System API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(SlowAPIMiddleware)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    if settings and settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["smartschoolgo.com", "*.smartschoolgo.com"]
        )


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current user from JWT token."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database (mock implementation)
        user_data = await get_user_by_id(user_id)
        if user_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_data
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(allowed_roles: List[str]):
    """Dependency to check user role."""
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


async def get_db_session():
    """Get database session."""
    if not db_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    async with db_manager.get_session() as session:
        yield session


async def get_redis_client():
    """Get Redis client."""
    if not redis_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not available"
        )
    return redis_client


# Helper functions
async def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID (mock implementation)."""
    # In production, this would query the database
    mock_users = {
        "admin": {
            "id": "admin",
            "email": "admin@smartschoolgo.com",
            "name": "System Administrator",
            "role": UserRole.ADMIN,
            "created_at": datetime.now(),
            "last_login": datetime.now()
        },
        "planner": {
            "id": "planner",
            "email": "planner@smartschoolgo.com",
            "name": "Transport Planner",
            "role": UserRole.PLANNER,
            "created_at": datetime.now(),
            "last_login": datetime.now()
        }
    }
    
    return mock_users.get(user_id)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and collect metrics."""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    # Calculate request duration
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Update metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path)).inc()
    REQUEST_LATENCY.observe(duration)
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response


# API Routes

# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check database
        db_status = "ok" if db_manager else "unavailable"
        
        # Check Redis
        redis_status = "ok"
        if redis_client:
            try:
                await redis_client.ping()
            except Exception:
                redis_status = "error"
        else:
            redis_status = "unavailable"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "services": {
                "database": db_status,
                "cache": redis_status,
                "ml_models": "ok" if ml_manager else "unavailable",
                "realtime_engine": "ok" if realtime_engine else "unavailable"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/metrics", tags=["Health"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Authentication endpoints
@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Authentication"])
@limiter.limit("5/minute")
async def login(request: Request, user_login: UserLogin):
    """User login endpoint."""
    # Mock authentication - replace with real authentication
    mock_users = {
        "admin@smartschoolgo.com": {
            "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
            "user": {
                "id": "admin",
                "email": "admin@smartschoolgo.com",
                "name": "System Administrator",
                "role": UserRole.ADMIN,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
        },
        "planner@smartschoolgo.com": {
            "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
            "user": {
                "id": "planner",
                "email": "planner@smartschoolgo.com",
                "name": "Transport Planner",
                "role": UserRole.PLANNER,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
        }
    }
    
    user_record = mock_users.get(user_login.email)
    if not user_record or not pwd_context.verify(user_login.password, user_record["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.jwt_expire_minutes)
    access_token = create_access_token(
        data={"sub": user_record["user"]["id"]}, 
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
        user=UserResponse(**user_record["user"])
    )


@app.post("/api/v1/auth/register", response_model=UserResponse, tags=["Authentication"])
@limiter.limit("3/minute")
async def register(request: Request, user_create: UserCreate):
    """User registration endpoint."""
    # Check if user already exists (mock check)
    existing_users = ["admin@smartschoolgo.com", "planner@smartschoolgo.com"]
    
    if user_create.email in existing_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )
    
    # Hash password
    hashed_password = pwd_context.hash(user_create.password)
    
    # Create user record (mock creation)
    user_data = {
        "id": str(uuid.uuid4()),
        "email": user_create.email,
        "name": user_create.name,
        "role": user_create.role,
        "created_at": datetime.utcnow(),
        "last_login": None
    }
    
    logger.info(f"New user registered: {user_create.email}")
    
    return UserResponse(**user_data)


# School management endpoints
@app.get("/api/v1/schools", tags=["Schools"])
async def get_schools(
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get list of schools."""
    # Mock data - replace with database query
    schools = []
    for i in range(limit):
        schools.append({
            "id": f"school_{i+offset+1}",
            "school_name": f"School {i+offset+1}",
            "school_type": "Primary" if i % 2 == 0 else "Secondary",
            "sector": "Government",
            "latitude": -35.28 + (i * 0.001),
            "longitude": 149.13 + (i * 0.001),
            "address": f"123 Education St, Canberra ACT 260{i%10}",
            "phone": f"02 612{i:02d} {i:04d}",
            "capacity": 300 + (i * 50),
            "current_enrollment": 250 + (i * 40)
        })
    
    return {
        "schools": schools,
        "total": 1000,  # Mock total
        "limit": limit,
        "offset": offset
    }


@app.post("/api/v1/schools", tags=["Schools"])
async def create_school(
    school: SchoolCreate,
    current_user: dict = Depends(require_role([UserRole.ADMIN, UserRole.PLANNER]))
):
    """Create a new school."""
    # Mock creation - replace with database insert
    school_data = {
        "id": str(uuid.uuid4()),
        **school.dict(),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    logger.info(f"New school created: {school.school_name} by {current_user['email']}")
    
    return school_data


@app.get("/api/v1/schools/{school_id}", tags=["Schools"])
async def get_school(
    school_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific school by ID."""
    # Mock data - replace with database query
    if school_id == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found"
        )
    
    school = {
        "id": school_id,
        "school_name": f"Sample School {school_id}",
        "school_type": "Primary",
        "sector": "Government",
        "latitude": -35.2809,
        "longitude": 149.1300,
        "address": "123 Education Street, Canberra ACT 2600",
        "phone": "02 6123 4567",
        "email": f"contact@school{school_id}.edu.au",
        "capacity": 500,
        "current_enrollment": 450,
        "principal_name": "Dr. Jane Smith",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    return school


# Student management endpoints
@app.get("/api/v1/students", tags=["Students"])
async def get_students(
    limit: int = 100,
    offset: int = 0,
    school_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get list of students."""
    # Mock data - replace with database query
    students = []
    for i in range(limit):
        students.append({
            "id": f"student_{i+offset+1}",
            "name": f"Student {i+offset+1}",
            "student_id": f"STU{i+offset+1:04d}",
            "school_id": school_id or f"school_{(i%10)+1}",
            "grade_level": (i % 6) + 1,
            "home_latitude": -35.28 + (i * 0.0001),
            "home_longitude": 149.13 + (i * 0.0001),
            "home_address": f"456 Residential Ave, Canberra ACT 260{i%10}",
            "parent_name": f"Parent {i+offset+1}",
            "parent_phone": f"04{i:02d}{i:02d}{i:04d}",
            "parent_email": f"parent{i+offset+1}@email.com",
            "route_id": f"route_{(i%5)+1}",
            "pickup_location": f"Stop {(i%10)+1}",
            "created_at": datetime.utcnow()
        })
    
    return {
        "students": students,
        "total": 5000,  # Mock total
        "limit": limit,
        "offset": offset
    }


@app.post("/api/v1/students", tags=["Students"])
async def create_student(
    student: StudentCreate,
    current_user: dict = Depends(require_role([UserRole.ADMIN, UserRole.PLANNER]))
):
    """Create a new student."""
    # Mock creation - replace with database insert
    student_data = {
        "id": str(uuid.uuid4()),
        **student.dict(),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    logger.info(f"New student created: {student.name} by {current_user['email']}")
    
    return student_data


# Route management endpoints
@app.get("/api/v1/routes", tags=["Routes"])
async def get_routes(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get list of routes."""
    # Mock data - replace with database query
    routes = []
    for i in range(limit):
        routes.append({
            "id": f"route_{i+offset+1}",
            "route_name": f"Route {i+offset+1}",
            "route_type": "School Bus",
            "school_ids": [f"school_{j}" for j in range(1, (i%3)+2)],
            "capacity": 50,
            "current_load": 35 + (i % 15),
            "estimated_duration": 45 + (i % 20),
            "departure_time": f"0{7+(i%2)}:30",
            "return_time": f"15:30",
            "status": "Active",
            "vehicle_id": f"vehicle_{i+1}",
            "driver_id": f"driver_{i+1}",
            "stops": [f"stop_{j}" for j in range(8 + (i%5))],
            "created_at": datetime.utcnow()
        })
    
    return {
        "routes": routes,
        "total": 200,  # Mock total
        "limit": limit,
        "offset": offset
    }


@app.post("/api/v1/routes", tags=["Routes"])
async def create_route(
    route: RouteCreate,
    current_user: dict = Depends(require_role([UserRole.ADMIN, UserRole.PLANNER]))
):
    """Create a new route."""
    # Mock creation - replace with database insert
    route_data = {
        "id": str(uuid.uuid4()),
        **route.dict(),
        "status": "Planned",
        "current_load": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    logger.info(f"New route created: {route.route_name} by {current_user['email']}")
    
    return route_data


# Vehicle management endpoints
@app.get("/api/v1/vehicles", tags=["Vehicles"])
async def get_vehicles(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get list of vehicles."""
    # Mock data - replace with database query
    vehicles = []
    for i in range(limit):
        vehicles.append({
            "id": f"vehicle_{i+offset+1}",
            "vehicle_id": f"BUS-{i+offset+1:03d}",
            "make": "Mercedes" if i % 2 == 0 else "Volvo",
            "model": "Citaro" if i % 2 == 0 else "B7RLE",
            "year": 2020 + (i % 4),
            "capacity": 50,
            "fuel_type": "diesel",
            "registration": f"ACT-{i+100:03d}",
            "status": "active" if i % 5 != 0 else "maintenance",
            "route_id": f"route_{(i%10)+1}" if i % 5 != 0 else None,
            "driver_id": f"driver_{i+1}" if i % 5 != 0 else None,
            "location": {
                "latitude": -35.28 + (i * 0.001),
                "longitude": 149.13 + (i * 0.001),
                "timestamp": datetime.utcnow()
            },
            "mileage": 50000 + (i * 1000),
            "last_maintenance": datetime.utcnow() - timedelta(days=i*5),
            "next_maintenance": datetime.utcnow() + timedelta(days=90-(i*2)),
            "created_at": datetime.utcnow()
        })
    
    return {
        "vehicles": vehicles,
        "total": 100,  # Mock total
        "limit": limit,
        "offset": offset
    }


@app.post("/api/v1/vehicles", tags=["Vehicles"])
async def create_vehicle(
    vehicle: VehicleCreate,
    current_user: dict = Depends(require_role([UserRole.ADMIN]))
):
    """Create a new vehicle."""
    # Mock creation - replace with database insert
    vehicle_data = {
        "id": str(uuid.uuid4()),
        **vehicle.dict(),
        "status": "active",
        "mileage": 0,
        "last_maintenance": datetime.utcnow(),
        "next_maintenance": datetime.utcnow() + timedelta(days=90),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    logger.info(f"New vehicle created: {vehicle.vehicle_id} by {current_user['email']}")
    
    return vehicle_data


# Analytics endpoints
@app.get("/api/v1/analytics/performance", tags=["Analytics"])
async def get_performance_analytics(
    days: int = 30,
    current_user: dict = Depends(require_role([UserRole.ADMIN, UserRole.PLANNER]))
):
    """Get performance analytics data."""
    # Mock analytics data
    import random
    import numpy as np
    
    dates = [(datetime.utcnow() - timedelta(days=i)) for i in range(days)]
    dates.reverse()
    
    analytics_data = []
    for date in dates:
        analytics_data.append({
            "date": date.date(),
            "on_time_performance": round(85 + random.uniform(-5, 10), 1),
            "fleet_utilization": round(90 + random.uniform(-8, 8), 1),
            "student_satisfaction": round(4.0 + random.uniform(-0.5, 0.8), 2),
            "fuel_efficiency": round(10 + random.uniform(-2, 3), 1),
            "safety_incidents": random.randint(0, 3),
            "total_trips": random.randint(450, 550),
            "delayed_trips": random.randint(20, 80),
            "cancelled_trips": random.randint(0, 5)
        })
    
    # Calculate summary statistics
    summary = {
        "avg_on_time_performance": round(np.mean([d["on_time_performance"] for d in analytics_data]), 1),
        "avg_fleet_utilization": round(np.mean([d["fleet_utilization"] for d in analytics_data]), 1),
        "avg_student_satisfaction": round(np.mean([d["student_satisfaction"] for d in analytics_data]), 2),
        "total_incidents": sum([d["safety_incidents"] for d in analytics_data]),
        "total_trips": sum([d["total_trips"] for d in analytics_data])
    }
    
    return {
        "summary": summary,
        "daily_data": analytics_data,
        "period_days": days
    }


@app.get("/api/v1/analytics/routes", tags=["Analytics"])
async def get_route_analytics(
    current_user: dict = Depends(require_role([UserRole.ADMIN, UserRole.PLANNER]))
):
    """Get route performance analytics."""
    # Mock route analytics
    route_analytics = []
    for i in range(10):
        route_analytics.append({
            "route_id": f"route_{i+1}",
            "route_name": f"Route {i+1}",
            "avg_occupancy": round(35 + (i * 2) + np.random.uniform(-5, 5), 1),
            "on_time_rate": round(85 + np.random.uniform(-10, 12), 1),
            "avg_delay_minutes": round(np.random.uniform(1, 8), 1),
            "fuel_consumption": round(8 + np.random.uniform(-2, 4), 2),
            "student_count": 30 + (i * 3),
            "total_distance_km": round(15 + np.random.uniform(-5, 15), 1),
            "incidents_count": np.random.randint(0, 2),
            "cost_per_km": round(0.85 + np.random.uniform(-0.15, 0.25), 2)
        })
    
    return {
        "route_analytics": route_analytics,
        "total_routes": len(route_analytics)
    }


# Real-time data endpoints
@app.get("/api/v1/realtime/vehicles", tags=["Real-time"])
async def get_realtime_vehicles(
    current_user: dict = Depends(get_current_user),
    cache: redis.Redis = Depends(get_redis_client)
):
    """Get real-time vehicle positions."""
    # Try to get from cache first
    try:
        cached_data = await cache.get("realtime:vehicles")
        if cached_data:
            return json.loads(cached_data)
    except Exception:
        pass  # Cache miss, generate fresh data
    
    # Mock real-time vehicle data
    vehicles = []
    for i in range(20):
        vehicles.append({
            "vehicle_id": f"BUS-{i+1:03d}",
            "route_id": f"route_{(i%8)+1}",
            "latitude": -35.28 + np.random.uniform(-0.02, 0.02),
            "longitude": 149.13 + np.random.uniform(-0.02, 0.02),
            "bearing": np.random.randint(0, 360),
            "speed": round(np.random.uniform(15, 45), 1),
            "occupancy": np.random.randint(10, 50),
            "status": np.random.choice(["in_transit", "at_stop", "idle"]),
            "last_update": datetime.utcnow(),
            "next_stop": f"stop_{np.random.randint(1, 20)}",
            "eta_minutes": np.random.randint(2, 15)
        })
    
    data = {
        "vehicles": vehicles,
        "timestamp": datetime.utcnow(),
        "total_active": len([v for v in vehicles if v["status"] == "in_transit"])
    }
    
    # Cache for 10 seconds
    try:
        await cache.setex("realtime:vehicles", 10, json.dumps(data, default=str))
    except Exception:
        pass  # Cache write failure, continue
    
    return data


@app.get("/api/v1/realtime/alerts", tags=["Real-time"])
async def get_realtime_alerts(
    severity: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get real-time system alerts."""
    # Mock alerts data
    alerts = [
        {
            "id": "alert_001",
            "type": "delay",
            "severity": "medium",
            "title": "Route 3 Experiencing Delays",
            "message": "Route 3 is running 12 minutes behind schedule due to traffic congestion on Northbourne Avenue",
            "affected_routes": ["route_3"],
            "affected_vehicles": ["BUS-003", "BUS-012"],
            "created_at": datetime.utcnow() - timedelta(minutes=15),
            "status": "active",
            "estimated_resolution": datetime.utcnow() + timedelta(minutes=20)
        },
        {
            "id": "alert_002",
            "type": "maintenance",
            "severity": "low",
            "title": "Scheduled Maintenance",
            "message": "Bus BUS-007 scheduled for maintenance after current route completion",
            "affected_routes": ["route_5"],
            "affected_vehicles": ["BUS-007"],
            "created_at": datetime.utcnow() - timedelta(hours=2),
            "status": "scheduled",
            "estimated_resolution": datetime.utcnow() + timedelta(hours=4)
        }
    ]
    
    # Filter by severity if provided
    if severity:
        alerts = [alert for alert in alerts if alert["severity"] == severity]
    
    return {
        "alerts": alerts,
        "total": len(alerts),
        "active_count": len([a for a in alerts if a["status"] == "active"])
    }


# WebSocket endpoints
@app.websocket("/api/v1/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for real-time updates."""
    user_id = None
    
    try:
        # Get user ID from query parameters or headers
        user_id = websocket.query_params.get("user_id")
        
        await websocket_manager.connect(websocket, connection_id, user_id)
        
        # Send welcome message
        await websocket_manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to SmartSchoolGo real-time updates",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )
        
        # Listen for incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                if message_data.get("type") == "ping":
                    await websocket_manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                        connection_id
                    )
                
                elif message_data.get("type") == "subscribe":
                    # Handle subscription to specific data streams
                    stream = message_data.get("stream", "general")
                    await websocket_manager.send_personal_message(
                        json.dumps({
                            "type": "subscription", 
                            "stream": stream,
                            "status": "subscribed",
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        connection_id
                    )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    connection_id
                )
            except Exception as e:
                logger.error(f"WebSocket error for {connection_id}: {e}")
                await websocket_manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Internal server error"}),
                    connection_id
                )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {connection_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        websocket_manager.disconnect(connection_id, user_id)


# Background task to broadcast real-time updates
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks for real-time updates."""
    if not FASTAPI_LIBS_AVAILABLE:
        return
        
    async def broadcast_vehicle_updates():
        """Broadcast vehicle position updates every 10 seconds."""
        while True:
            try:
                # Get current vehicle positions
                vehicles_response = await get_realtime_vehicles(
                    current_user={"role": UserRole.ADMIN},  # Mock admin user
                    cache=redis_client
                )
                
                # Broadcast to all connected clients
                message = json.dumps({
                    "type": "vehicle_update",
                    "data": vehicles_response,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                await websocket_manager.broadcast(message)
                
                await asyncio.sleep(10)  # Wait 10 seconds
                
            except Exception as e:
                logger.error(f"Error broadcasting vehicle updates: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    # Start the background task
    asyncio.create_task(broadcast_vehicle_updates())


# Custom exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation Error",
            "errors": exc.errors(),
            "body": exc.body if hasattr(exc, 'body') else None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="SmartSchoolGo API",
        version="1.0.0",
        description="Comprehensive API for intelligent school transport management",
        routes=app.routes,
    )
    
    # Add custom security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


if FASTAPI_LIBS_AVAILABLE:
    app.openapi = custom_openapi


# Development server
if __name__ == "__main__":
    if FASTAPI_LIBS_AVAILABLE:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    else:
        print("FastAPI dependencies not available. Please install required packages.")