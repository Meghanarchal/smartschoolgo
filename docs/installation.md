# Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Development Setup](#development-setup)
3. [Production Setup](#production-setup)
4. [Docker Installation](#docker-installation)
5. [Database Setup](#database-setup)
6. [Configuration](#configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (2.4 GHz)
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python**: 3.10 or higher
- **Node.js**: 16.0 or higher
- **PostgreSQL**: 14.0+ with PostGIS 3.0+
- **Redis**: 6.0+

### Recommended Requirements
- **CPU**: 8 cores (3.0 GHz)
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA GPU with CUDA 11.0+ (for ML models)

## Development Setup

### 1. Prerequisites Installation

#### Ubuntu/Debian
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install PostgreSQL and PostGIS
sudo apt install postgresql postgresql-contrib postgis postgresql-14-postgis-3 -y

# Install Redis
sudo apt install redis-server -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt install nodejs -y

# Install Git and development tools
sudo apt install git build-essential libssl-dev libffi-dev python3-dev -y
```

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install PostgreSQL with PostGIS
brew install postgresql@14 postgis

# Install Redis
brew install redis

# Install Node.js
brew install node@16

# Start services
brew services start postgresql@14
brew services start redis
```

#### Windows
```powershell
# Install Chocolatey package manager
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python
choco install python --version=3.10.0

# Install PostgreSQL
choco install postgresql14 --params '/Password:postgres' --params-global

# Install Redis
choco install redis-64

# Install Node.js
choco install nodejs --version=16.0.0

# Install Git
choco install git
```

### 2. Project Setup

```bash
# Clone the repository
git clone https://github.com/govhack2025/smartschoolgo.git
cd smartschoolgo

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Generate secret key
python -c "import secrets; print(f'SECRET_KEY={secrets.token_urlsafe(32)}')"
```

Edit `.env` file:
```env
# Application Settings
APP_NAME=SmartSchoolGo
ENVIRONMENT=development
DEBUG=True
SECRET_KEY=your_generated_secret_key

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/smartschoolgo
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TTL=300

# API Keys
ACT_GOV_API_KEY=your_act_gov_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
MAPBOX_API_KEY=your_mapbox_api_key

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Monitoring
PROMETHEUS_ENABLED=True
JAEGER_ENABLED=False
```

### 4. Database Setup

```bash
# Start PostgreSQL service
sudo systemctl start postgresql  # Linux
brew services start postgresql@14  # macOS

# Create database user and database
sudo -u postgres psql <<EOF
CREATE USER smartschool WITH PASSWORD 'your_password';
CREATE DATABASE smartschoolgo OWNER smartschool;
\c smartschoolgo
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
GRANT ALL PRIVILEGES ON DATABASE smartschoolgo TO smartschool;
EOF

# Run database migrations
python scripts/migrate.py

# Load initial data
python scripts/seed_database.py

# Load sample data (optional)
python scripts/load_sample_data.py
```

### 5. Redis Setup

```bash
# Start Redis service
sudo systemctl start redis  # Linux
brew services start redis  # macOS
redis-server  # Windows

# Test Redis connection
redis-cli ping
# Should return: PONG
```

## Production Setup

### 1. System Preparation

```bash
# Create application user
sudo useradd -m -s /bin/bash smartschool
sudo usermod -aG sudo smartschool

# Set up application directory
sudo mkdir -p /opt/smartschoolgo
sudo chown smartschool:smartschool /opt/smartschoolgo

# Switch to application user
sudo su - smartschool
cd /opt/smartschoolgo
```

### 2. Production Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt
pip install gunicorn uvicorn[standard] supervisor

# Install Nginx
sudo apt install nginx -y

# Install Certbot for SSL
sudo apt install certbot python3-certbot-nginx -y
```

### 3. Gunicorn Configuration

Create `/opt/smartschoolgo/gunicorn_config.py`:
```python
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
accesslog = "/var/log/smartschoolgo/access.log"
errorlog = "/var/log/smartschoolgo/error.log"
loglevel = "info"
```

### 4. Nginx Configuration

Create `/etc/nginx/sites-available/smartschoolgo`:
```nginx
upstream smartschoolgo {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Proxy settings
    location / {
        proxy_pass http://smartschoolgo;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Static files
    location /static {
        alias /opt/smartschoolgo/static;
        expires 30d;
    }
    
    # Media files
    location /media {
        alias /opt/smartschoolgo/media;
        expires 7d;
    }
}
```

### 5. Systemd Service

Create `/etc/systemd/system/smartschoolgo.service`:
```ini
[Unit]
Description=SmartSchoolGo Application
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=smartschool
Group=smartschool
WorkingDirectory=/opt/smartschoolgo
Environment="PATH=/opt/smartschoolgo/venv/bin"
ExecStart=/opt/smartschoolgo/venv/bin/gunicorn \
    -c /opt/smartschoolgo/gunicorn_config.py \
    src.smart_transport.api.main:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
KillSignal=SIGQUIT
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

## Docker Installation

### 1. Docker Setup

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Build and Run

```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Docker Compose Configuration

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  postgres:
    image: postgis/postgis:14-3.2
    environment:
      POSTGRES_DB: smartschoolgo
      POSTGRES_USER: smartschool
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    
  app:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://smartschool:${DB_PASSWORD}@postgres:5432/smartschoolgo
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./src:/app/src
      - ./static:/app/static
      - ./media:/app/media
    
  nginx:
    image: nginx:alpine
    depends_on:
      - app
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      
volumes:
  postgres_data:
  redis_data:
```

## Configuration

### Application Configuration Files

1. **Logging Configuration** (`logging.yaml`):
```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: default
    filename: logs/app.log
    maxBytes: 10485760
    backupCount: 5
root:
  level: INFO
  handlers: [console, file]
```

2. **Celery Configuration** (`celery_config.py`):
```python
broker_url = 'redis://localhost:6379/1'
result_backend = 'redis://localhost:6379/2'
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Australia/Sydney'
enable_utc = True
```

## Verification

### 1. System Health Check

```bash
# Check all services
python scripts/health_check.py

# Expected output:
# ✓ Database connection: OK
# ✓ Redis connection: OK
# ✓ API endpoints: OK
# ✓ ML models loaded: OK
# ✓ External APIs: OK
```

### 2. Run Tests

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run all tests with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 3. API Verification

```bash
# Test API health endpoint
curl http://localhost:8000/health

# Test API documentation
open http://localhost:8000/docs

# Test Streamlit app
open http://localhost:8501
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -U smartschool -d smartschoolgo -h localhost

# Reset database
python scripts/reset_database.py
```

#### 2. Redis Connection Error
```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping

# Flush Redis cache
redis-cli FLUSHALL
```

#### 3. Port Already in Use
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

#### 4. Permission Errors
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod -R 755 .
```

#### 5. Module Import Errors
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
```

### Getting Help

- Check logs: `tail -f logs/app.log`
- Run diagnostics: `python scripts/diagnostics.py`
- View documentation: `http://localhost:8000/docs`
- Contact support: support@smartschoolgo.com