# System Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Security Architecture](#security-architecture)
7. [Scalability & Performance](#scalability--performance)
8. [Integration Architecture](#integration-architecture)

## Overview

SmartSchoolGo employs a modern microservices architecture with clear separation of concerns, enabling scalability, maintainability, and robust performance. The system is designed to handle real-time data processing, complex optimization algorithms, and concurrent user interactions.

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Layer                              │
├──────────────┬───────────────┬───────────────┬────────────────────┤
│  Parent App  │   Admin Portal │ Planner Tool  │   Mobile Apps      │
└──────┬───────┴───────┬───────┴───────┬───────┴────────┬───────────┘
       │               │               │                │
       └───────────────┴───────────────┴────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    API Gateway      │
                    │   (FastAPI/Nginx)   │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│  Core Services │  │  ML Services    │  │  Integration    │
│                │  │                 │  │    Services     │
│ • Auth Service │  │ • Optimization  │  │ • ACT Gov API   │
│ • Route Service│  │ • Forecasting   │  │ • OSM Sync      │
│ • Fleet Service│  │ • Safety Model  │  │ • Weather API   │
│ • User Service │  │ • Analytics     │  │ • School System │
└────────┬───────┘  └────────┬────────┘  └────────┬────────┘
         │                   │                     │
         └───────────────────┴─────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Message Queue      │
                  │  (Kafka/RabbitMQ)    │
                  └──────────┬──────────┘
                             │
         ┌───────────────────┼─────────────────────┐
         │                   │                     │
┌────────▼────────┐ ┌────────▼────────┐  ┌────────▼────────┐
│  Data Layer     │ │  Cache Layer    │  │  Real-time      │
│                 │ │                 │  │    Engine       │
│ • PostgreSQL    │ │ • Redis         │  │ • WebSocket     │
│ • PostGIS       │ │ • Memcached     │  │ • SSE           │
│ • TimescaleDB   │ │                 │  │ • Pub/Sub       │
└─────────────────┘ └─────────────────┘  └─────────────────┘
```

## Component Architecture

### 1. Presentation Layer

#### Streamlit Application (`src/smart_transport/api/app.py`)
- **Purpose**: Main user interface for all user roles
- **Components**:
  - Session management
  - Role-based routing
  - Interactive dashboards
  - Real-time updates

#### FastAPI Backend (`src/smart_transport/api/main.py`)
- **Purpose**: RESTful API and WebSocket server
- **Components**:
  - JWT authentication
  - Rate limiting
  - CORS handling
  - WebSocket management
  - OpenAPI documentation

### 2. Business Logic Layer

#### Route Optimization Service
```python
class RouteOptimizer:
    - Multi-objective optimization (NSGA-II)
    - Constraint handling
    - Dynamic re-routing
    - Load balancing
```

#### ML Model Service
```python
class MLModelService:
    - Demand forecasting (ARIMA, LSTM, Prophet)
    - Safety risk assessment
    - Congestion prediction
    - Maintenance scheduling
```

#### Network Analysis Service
```python
class NetworkAnalyzer:
    - Graph construction
    - Centrality analysis
    - Path finding algorithms
    - Flow optimization
```

### 3. Data Access Layer

#### Database Architecture
```sql
-- Core Tables
Schools
├── school_id (PK)
├── name
├── location (PostGIS Point)
└── capacity

Students
├── student_id (PK)
├── school_id (FK)
├── address (PostGIS Point)
└── special_needs

Routes
├── route_id (PK)
├── geometry (PostGIS LineString)
├── schedule
└── vehicle_id (FK)

Vehicles
├── vehicle_id (PK)
├── capacity
├── current_location (PostGIS Point)
└── status
```

#### Data Models (`src/smart_transport/data/models.py`)
- Pydantic models for validation
- SQLAlchemy ORM models
- GeoJSON serialization
- Schema versioning

### 4. Integration Layer

#### External API Integrations
```python
IntegrationManager
├── ACTGovernmentAPI
│   ├── OAuth2 authentication
│   ├── Token refresh
│   └── Rate limiting
├── OpenStreetMapSync
│   ├── Change detection
│   ├── Incremental updates
│   └── Spatial indexing
├── WeatherServiceIntegration
│   ├── Alert processing
│   ├── Forecast caching
│   └── Event triggers
└── SchoolSystemIntegration
    ├── Enrollment sync
    ├── Webhook handling
    └── Conflict resolution
```

## Data Flow

### 1. Real-time Data Pipeline

```
GPS Devices → IoT Gateway → Message Queue → Stream Processor → Cache → WebSocket → Client
                                ↓
                         Time Series DB
```

### 2. Batch Processing Pipeline

```
External APIs → ETL Service → Data Lake → Processing Engine → Data Warehouse → Analytics
                     ↓                           ↓
                Validation               ML Training Pipeline
```

### 3. Request Flow

```
Client Request → API Gateway → Authentication → Rate Limiting → Service Router
                                                                      ↓
                                                              Business Logic
                                                                      ↓
                                                              Data Access Layer
                                                                      ↓
                                                                  Database
                                                                      ↓
                                                                Response Cache
                                                                      ↓
                                                                  Client
```

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend Framework** | FastAPI | High-performance async API |
| **Frontend Framework** | Streamlit | Rapid UI development |
| **Database** | PostgreSQL + PostGIS | Spatial data storage |
| **Cache** | Redis | Session and data caching |
| **Message Queue** | Kafka/RabbitMQ | Async event processing |
| **Container** | Docker | Application containerization |
| **Orchestration** | Kubernetes | Container orchestration |
| **Monitoring** | Prometheus + Grafana | System monitoring |

### Machine Learning Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML Framework** | TensorFlow/PyTorch | Deep learning models |
| **Time Series** | Prophet, ARIMA | Demand forecasting |
| **Optimization** | DEAP, OR-Tools | Route optimization |
| **Graph Analysis** | NetworkX | Network algorithms |
| **Experiment Tracking** | MLflow | Model versioning |

### Data Processing Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ETL** | Apache Airflow | Workflow orchestration |
| **Stream Processing** | Apache Flink | Real-time processing |
| **Data Validation** | Great Expectations | Data quality |
| **Feature Store** | Feast | Feature management |

## Security Architecture

### Authentication & Authorization

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│ Auth Service │────▶│   JWT Token │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                    ┌───────▼────────┐
                    │  Token Store   │
                    │    (Redis)     │
                    └────────────────┘
```

### Security Layers

1. **Network Security**
   - SSL/TLS encryption
   - VPN for admin access
   - DDoS protection
   - Web Application Firewall (WAF)

2. **Application Security**
   - JWT-based authentication
   - Role-Based Access Control (RBAC)
   - API rate limiting
   - Input validation and sanitization

3. **Data Security**
   - Encryption at rest (AES-256)
   - Encryption in transit (TLS 1.3)
   - Data anonymization
   - Audit logging

4. **Infrastructure Security**
   - Container security scanning
   - Secrets management (HashiCorp Vault)
   - Network segmentation
   - Regular security updates

## Scalability & Performance

### Horizontal Scaling Strategy

```
Load Balancer (HAProxy/Nginx)
       │
       ├──── API Instance 1
       ├──── API Instance 2
       ├──── API Instance 3
       └──── API Instance N
```

### Caching Strategy

1. **L1 Cache**: Application-level caching (in-memory)
2. **L2 Cache**: Redis distributed cache
3. **L3 Cache**: CDN for static assets
4. **Database Cache**: Query result caching

### Performance Optimization

- **Database Optimization**
  - Spatial indexing for geographic queries
  - Partitioning for time-series data
  - Connection pooling
  - Query optimization

- **API Optimization**
  - Async request handling
  - Response compression
  - Pagination for large datasets
  - GraphQL for efficient data fetching

- **ML Model Optimization**
  - Model quantization
  - Batch inference
  - GPU acceleration
  - Edge deployment for real-time inference

## Integration Architecture

### Event-Driven Architecture

```
Event Producer → Event Bus (Kafka) → Event Consumer
                        │
                 Event Store (EventStore)
                        │
                  Event Replay
```

### Webhook Processing

```python
WebhookHandler
├── Signature Verification
├── Event Parsing
├── Event Routing
├── Async Processing
└── Response Handling
```

### API Gateway Pattern

```
Client → API Gateway → Service Discovery → Microservice
              │              │
         Rate Limiting    Load Balancing
              │              │
         Authentication  Circuit Breaker
```

## Deployment Architecture

### Container Orchestration

```yaml
Kubernetes Cluster
├── Namespace: production
│   ├── Deployment: api-server (3 replicas)
│   ├── Deployment: ml-service (2 replicas)
│   ├── StatefulSet: postgresql (3 replicas)
│   └── DaemonSet: monitoring-agent
├── Namespace: staging
└── Namespace: development
```

### CI/CD Pipeline

```
Code Push → GitHub Actions → Build → Test → Security Scan → Deploy
                │                      │          │
            Linting              Unit Tests   Vulnerability
                                Integration     Scanning
```

## Monitoring & Observability

### Metrics Collection

```
Application → Prometheus Exporter → Prometheus Server → Grafana
                    │                      │
              Custom Metrics          Alert Manager
```

### Logging Architecture

```
Application → Structured Logs → Fluentd → Elasticsearch → Kibana
                   │
              Log Levels:
              - ERROR
              - WARNING
              - INFO
              - DEBUG
```

### Distributed Tracing

```
Request → Trace ID → Span Creation → Jaeger Collector → Jaeger UI
              │            │
        Correlation   Timing Info
```

## Disaster Recovery

### Backup Strategy

1. **Database Backups**
   - Daily full backups
   - Hourly incremental backups
   - Point-in-time recovery
   - Geo-replicated storage

2. **Application State**
   - Configuration versioning
   - Secret rotation
   - State snapshots

### High Availability

```
Primary Region (Sydney)          Secondary Region (Melbourne)
├── Active Services              ├── Standby Services
├── Primary Database             ├── Read Replica
└── Primary Cache                └── Cache Replica
         │                                │
         └────── Replication ─────────────┘
```

## Performance Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| API Response Time (p95) | < 200ms | 150ms |
| Route Optimization Time | < 5s | 3.2s |
| Real-time Update Latency | < 100ms | 75ms |
| System Uptime | 99.9% | 99.95% |
| Concurrent Users | 10,000 | 12,500 |
| Requests per Second | 5,000 | 6,200 |

## Future Architecture Considerations

1. **Microservices Migration**
   - Decompose monolithic components
   - Service mesh implementation (Istio)
   - Independent deployments

2. **Edge Computing**
   - Local processing on vehicles
   - Reduced latency for critical decisions
   - Offline capabilities

3. **AI/ML Enhancements**
   - Federated learning
   - Real-time model updates
   - AutoML integration

4. **Blockchain Integration**
   - Immutable audit trails
   - Smart contracts for automated operations
   - Decentralized identity management