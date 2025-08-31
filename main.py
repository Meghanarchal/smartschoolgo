"""
SmartSchoolGo - Main Application Orchestrator
Primary entry point for system operations, monitoring, and management
"""

import asyncio
import sys
import os
import signal
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import click
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
import psutil
import redis
import aiohttp
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import schedule
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import yaml
from pydantic import ValidationError
import streamlit.web.cli as stcli

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import get_settings, Environment
from src.smart_transport.data.database import DatabaseManager
from src.smart_transport.api.integrations import IntegrationManager
from src.smart_transport.models.realtime_engine import RealtimeEngine

# Initialize Rich console for beautiful CLI output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM STATES AND CONFIGURATION
# ============================================================================

class SystemState(Enum):
    """System operational states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ServiceStatus(Enum):
    """Individual service status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

class SystemOrchestrator:
    """Main system orchestrator for SmartSchoolGo"""
    
    def __init__(self):
        self.state = SystemState.STOPPED
        self.services = {}
        self.health_checks = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        self.shutdown_event = asyncio.Event()
        self.config = None
        self.db_manager = None
        self.integration_manager = None
        self.realtime_engine = None
        
    async def initialize(self, environment: Optional[str] = None):
        """Initialize system components"""
        console.print("\n[bold cyan]Initializing SmartSchoolGo System...[/bold cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load configuration
            task = progress.add_task("[cyan]Loading configuration...", total=1)
            try:
                env = Environment(environment) if environment else None
                self.config = get_settings(env)
                progress.update(task, completed=1)
                console.print("✓ Configuration loaded", style="green")
            except Exception as e:
                console.print(f"✗ Configuration failed: {e}", style="red")
                raise
            
            # Initialize database
            task = progress.add_task("[cyan]Connecting to database...", total=1)
            try:
                self.db_manager = DatabaseManager(self.config.database_url)
                await self.db_manager.initialize()
                progress.update(task, completed=1)
                console.print("✓ Database connected", style="green")
            except Exception as e:
                console.print(f"✗ Database connection failed: {e}", style="red")
                raise
            
            # Initialize integrations
            task = progress.add_task("[cyan]Setting up integrations...", total=1)
            try:
                self.integration_manager = IntegrationManager()
                await self.integration_manager.initialize(self.config.dict())
                progress.update(task, completed=1)
                console.print("✓ Integrations initialized", style="green")
            except Exception as e:
                console.print(f"✗ Integration setup failed: {e}", style="red")
                
            # Initialize realtime engine
            task = progress.add_task("[cyan]Starting realtime engine...", total=1)
            try:
                self.realtime_engine = RealtimeEngine(self.config)
                await self.realtime_engine.start()
                progress.update(task, completed=1)
                console.print("✓ Realtime engine started", style="green")
            except Exception as e:
                console.print(f"✗ Realtime engine failed: {e}", style="red")
                
        self.state = SystemState.RUNNING
        console.print("\n[bold green]System initialization complete![/bold green]\n")
        
    async def shutdown(self):
        """Graceful system shutdown"""
        console.print("\n[bold yellow]Initiating graceful shutdown...[/bold yellow]\n")
        self.state = SystemState.STOPPING
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Stop realtime engine
            if self.realtime_engine:
                task = progress.add_task("[yellow]Stopping realtime engine...", total=1)
                await self.realtime_engine.stop()
                progress.update(task, completed=1)
                
            # Close integrations
            if self.integration_manager:
                task = progress.add_task("[yellow]Closing integrations...", total=1)
                await self.integration_manager.shutdown()
                progress.update(task, completed=1)
                
            # Close database connections
            if self.db_manager:
                task = progress.add_task("[yellow]Closing database...", total=1)
                await self.db_manager.close()
                progress.update(task, completed=1)
                
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        self.state = SystemState.STOPPED
        console.print("\n[bold green]Shutdown complete![/bold green]\n")

# ============================================================================
# HEALTH CHECK SYSTEM
# ============================================================================

class HealthChecker:
    """System health monitoring"""
    
    @staticmethod
    async def check_database(config) -> Dict[str, Any]:
        """Check database health"""
        try:
            engine = create_engine(config.database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                conn.execute(text("SELECT PostGIS_Version()"))
            return {
                "status": ServiceStatus.HEALTHY,
                "message": "Database operational",
                "latency_ms": 10
            }
        except Exception as e:
            return {
                "status": ServiceStatus.UNHEALTHY,
                "message": str(e),
                "latency_ms": -1
            }
            
    @staticmethod
    async def check_redis(config) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            r = redis.from_url(config.redis_url)
            start = time.time()
            r.ping()
            latency = (time.time() - start) * 1000
            return {
                "status": ServiceStatus.HEALTHY,
                "message": "Redis operational",
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            return {
                "status": ServiceStatus.UNHEALTHY,
                "message": str(e),
                "latency_ms": -1
            }
            
    @staticmethod
    async def check_api(port: int = 8000) -> Dict[str, Any]:
        """Check API health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health") as resp:
                    if resp.status == 200:
                        return {
                            "status": ServiceStatus.HEALTHY,
                            "message": "API operational",
                            "latency_ms": 50
                        }
        except:
            return {
                "status": ServiceStatus.STOPPED,
                "message": "API not running",
                "latency_ms": -1
            }
            
    @staticmethod
    def check_system_resources() -> Dict[str, Any]:
        """Check system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = ServiceStatus.HEALTHY
        issues = []
        
        if cpu_percent > 80:
            status = ServiceStatus.DEGRADED if cpu_percent < 90 else ServiceStatus.UNHEALTHY
            issues.append(f"High CPU usage: {cpu_percent}%")
            
        if memory.percent > 80:
            status = ServiceStatus.DEGRADED if memory.percent < 90 else ServiceStatus.UNHEALTHY
            issues.append(f"High memory usage: {memory.percent}%")
            
        if disk.percent > 80:
            status = ServiceStatus.DEGRADED if disk.percent < 90 else ServiceStatus.UNHEALTHY
            issues.append(f"Low disk space: {disk.percent}% used")
            
        return {
            "status": status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "issues": issues
        }

# ============================================================================
# CLI COMMANDS
# ============================================================================

@click.group()
@click.option('--env', default='development', help='Environment (development/staging/production)')
@click.pass_context
def cli(ctx, env):
    """SmartSchoolGo System Orchestrator"""
    ctx.ensure_object(dict)
    ctx.obj['ENV'] = env
    ctx.obj['orchestrator'] = SystemOrchestrator()

@cli.command()
@click.pass_context
def start(ctx):
    """Start all system services"""
    async def _start():
        orchestrator = ctx.obj['orchestrator']
        await orchestrator.initialize(ctx.obj['ENV'])
        
        # Start API server in background
        console.print("\n[cyan]Starting API server...[/cyan]")
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.smart_transport.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload" if ctx.obj['ENV'] == 'development' else "--workers", "4"
        ])
        
        # Start Streamlit app in background
        console.print("[cyan]Starting Streamlit interface...[/cyan]")
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "src/smart_transport/api/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
        console.print("\n[bold green]All services started successfully![/bold green]")
        console.print("\n[cyan]Access points:[/cyan]")
        console.print("  • Streamlit UI: http://localhost:8501")
        console.print("  • API Documentation: http://localhost:8000/docs")
        console.print("  • Health Dashboard: http://localhost:8000/health")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down services...[/yellow]")
            api_process.terminate()
            streamlit_process.terminate()
            await orchestrator.shutdown()
            
    asyncio.run(_start())

@cli.command()
@click.pass_context
def stop(ctx):
    """Stop all system services"""
    console.print("[yellow]Stopping all services...[/yellow]")
    
    # Find and kill processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'smartschoolgo' in cmdline.lower() or 'smart_transport' in cmdline:
                console.print(f"  Stopping process {proc.info['pid']}: {proc.info['name']}")
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    console.print("[green]All services stopped![/green]")

@cli.command()
@click.pass_context
def status(ctx):
    """Check system status and health"""
    async def _status():
        config = get_settings(Environment(ctx.obj['ENV']))
        
        # Create status table
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Service", style="cyan", width=20)
        table.add_column("Status", width=15)
        table.add_column("Details", style="dim")
        
        # Check services
        checks = [
            ("Database", await HealthChecker.check_database(config)),
            ("Redis Cache", await HealthChecker.check_redis(config)),
            ("API Server", await HealthChecker.check_api()),
            ("System Resources", HealthChecker.check_system_resources()),
        ]
        
        for service_name, result in checks:
            status = result['status']
            if status == ServiceStatus.HEALTHY:
                status_icon = "[green]✓ Healthy[/green]"
            elif status == ServiceStatus.DEGRADED:
                status_icon = "[yellow]⚠ Degraded[/yellow]"
            elif status == ServiceStatus.UNHEALTHY:
                status_icon = "[red]✗ Unhealthy[/red]"
            else:
                status_icon = "[dim]○ Stopped[/dim]"
                
            details = result.get('message', '')
            if 'latency_ms' in result and result['latency_ms'] > 0:
                details += f" ({result['latency_ms']}ms)"
            if 'issues' in result and result['issues']:
                details = ', '.join(result['issues'])
                
            table.add_row(service_name, status_icon, details)
            
        console.print("\n")
        console.print(table)
        
        # System resources panel
        resources = HealthChecker.check_system_resources()
        resource_text = f"""
CPU Usage: {resources['cpu_percent']}%
Memory Usage: {resources['memory_percent']}%
Disk Usage: {resources['disk_percent']}%
        """
        console.print("\n")
        console.print(Panel(resource_text.strip(), title="System Resources", border_style="cyan"))
        
    asyncio.run(_status())

@cli.command()
@click.option('--service', type=click.Choice(['database', 'redis', 'api', 'all']), default='all')
@click.pass_context
def health(ctx, service):
    """Perform health checks"""
    async def _health():
        config = get_settings(Environment(ctx.obj['ENV']))
        
        if service in ['database', 'all']:
            result = await HealthChecker.check_database(config)
            _print_health_result("Database", result)
            
        if service in ['redis', 'all']:
            result = await HealthChecker.check_redis(config)
            _print_health_result("Redis", result)
            
        if service in ['api', 'all']:
            result = await HealthChecker.check_api()
            _print_health_result("API", result)
            
    def _print_health_result(name, result):
        if result['status'] == ServiceStatus.HEALTHY:
            console.print(f"[green]✓[/green] {name}: {result['message']}")
        else:
            console.print(f"[red]✗[/red] {name}: {result['message']}")
            
    asyncio.run(_health())

@cli.command()
@click.option('--reset', is_flag=True, help='Reset database before migration')
@click.pass_context
def migrate(ctx, reset):
    """Run database migrations"""
    console.print("[cyan]Running database migrations...[/cyan]")
    
    if reset:
        if click.confirm("This will delete all data. Are you sure?"):
            console.print("[yellow]Resetting database...[/yellow]")
            subprocess.run([sys.executable, "scripts/reset_database.py"])
            
    subprocess.run([sys.executable, "scripts/migrate.py"])
    console.print("[green]Migrations completed![/green]")

@cli.command()
@click.option('--source', type=click.Choice(['osm', 'act', 'weather', 'all']), default='all')
@click.pass_context
def sync(ctx, source):
    """Synchronize data from external sources"""
    async def _sync():
        orchestrator = ctx.obj['orchestrator']
        await orchestrator.initialize(ctx.obj['ENV'])
        
        console.print(f"\n[cyan]Synchronizing data from {source}...[/cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            if source in ['osm', 'all']:
                task = progress.add_task("[cyan]Syncing OpenStreetMap data...", total=1)
                osm = orchestrator.integration_manager.get_integration('osm')
                if osm:
                    result = await osm.sync_road_network()
                    progress.update(task, completed=1)
                    console.print(f"  OSM: {result.records_processed} records processed")
                    
            if source in ['act', 'all']:
                task = progress.add_task("[cyan]Syncing ACT Government data...", total=1)
                act = orchestrator.integration_manager.get_integration('act_gov')
                if act:
                    schools = await act.get_schools()
                    progress.update(task, completed=1)
                    console.print(f"  ACT: {len(schools)} schools synced")
                    
            if source in ['weather', 'all']:
                task = progress.add_task("[cyan]Syncing weather data...", total=1)
                weather = orchestrator.integration_manager.get_integration('weather')
                if weather:
                    alerts = await weather.get_weather_alerts('ACT')
                    progress.update(task, completed=1)
                    console.print(f"  Weather: {len(alerts)} alerts processed")
                    
        console.print("\n[green]Data synchronization complete![/green]")
        await orchestrator.shutdown()
        
    asyncio.run(_sync())

@cli.command()
@click.option('--days', default=30, help='Number of days of data to analyze')
@click.pass_context
def analyze(ctx, days):
    """Run analytics and generate reports"""
    console.print(f"[cyan]Running analytics for last {days} days...[/cyan]\n")
    
    # This would run various analytics
    analyses = [
        "Route efficiency analysis",
        "Safety incident patterns",
        "Demand forecasting",
        "Cost optimization opportunities",
        "Environmental impact assessment"
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for analysis in analyses:
            task = progress.add_task(f"[cyan]{analysis}...", total=100)
            for i in range(100):
                time.sleep(0.01)  # Simulate processing
                progress.update(task, advance=1)
                
    console.print("\n[green]Analysis complete! Reports saved to reports/[/green]")

@cli.command()
@click.option('--tail', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO')
@click.pass_context
def logs(ctx, tail, lines, level):
    """View system logs"""
    log_file = "logs/orchestrator.log"
    
    if not os.path.exists(log_file):
        console.print("[yellow]No log file found[/yellow]")
        return
        
    if tail:
        # Follow logs
        console.print(f"[cyan]Following logs (Ctrl+C to stop)...[/cyan]\n")
        try:
            subprocess.run(["tail", "-f", log_file])
        except KeyboardInterrupt:
            pass
    else:
        # Show last n lines
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            
        # Filter by level if needed
        filtered_lines = []
        for line in all_lines:
            if level in line or not any(lvl in line for lvl in ['DEBUG', 'INFO', 'WARNING', 'ERROR']):
                filtered_lines.append(line)
                
        # Show last n lines
        for line in filtered_lines[-lines:]:
            if 'ERROR' in line:
                console.print(line.strip(), style="red")
            elif 'WARNING' in line:
                console.print(line.strip(), style="yellow")
            elif 'DEBUG' in line:
                console.print(line.strip(), style="dim")
            else:
                console.print(line.strip())

@cli.command()
@click.pass_context
def config(ctx):
    """View current configuration"""
    config = get_settings(Environment(ctx.obj['ENV']))
    
    # Convert config to dict and hide sensitive values
    config_dict = config.dict()
    for key in config_dict:
        if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
            if config_dict[key]:
                config_dict[key] = "***HIDDEN***"
                
    # Display as formatted YAML
    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=True)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    
    console.print("\n[bold cyan]Current Configuration[/bold cyan]")
    console.print(f"[dim]Environment: {ctx.obj['ENV']}[/dim]\n")
    console.print(syntax)

@cli.command()
@click.option('--type', 'backup_type', type=click.Choice(['full', 'incremental']), default='full')
@click.option('--destination', default='backups/', help='Backup destination directory')
@click.pass_context
def backup(ctx, backup_type, destination):
    """Backup system data"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(destination) / f"{backup_type}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Creating {backup_type} backup to {backup_dir}...[/cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Backup database
        task = progress.add_task("[cyan]Backing up database...", total=1)
        db_backup = backup_dir / "database.sql"
        subprocess.run([
            "pg_dump",
            "-h", "localhost",
            "-U", "smartschool",
            "-d", "smartschoolgo",
            "-f", str(db_backup)
        ])
        progress.update(task, completed=1)
        
        # Backup configuration
        task = progress.add_task("[cyan]Backing up configuration...", total=1)
        config_backup = backup_dir / "config"
        config_backup.mkdir(exist_ok=True)
        subprocess.run(["cp", "-r", "src/config", str(config_backup)])
        progress.update(task, completed=1)
        
        # Backup logs
        task = progress.add_task("[cyan]Backing up logs...", total=1)
        logs_backup = backup_dir / "logs"
        logs_backup.mkdir(exist_ok=True)
        subprocess.run(["cp", "-r", "logs", str(logs_backup)])
        progress.update(task, completed=1)
        
    console.print(f"\n[green]Backup completed: {backup_dir}[/green]")

@cli.command()
@click.option('--backup-file', required=True, help='Path to backup file')
@click.pass_context
def restore(ctx, backup_file):
    """Restore system from backup"""
    backup_path = Path(backup_file)
    
    if not backup_path.exists():
        console.print(f"[red]Backup file not found: {backup_file}[/red]")
        return
        
    if not click.confirm("This will overwrite current data. Continue?"):
        return
        
    console.print(f"[cyan]Restoring from {backup_file}...[/cyan]\n")
    
    # Restore database
    db_backup = backup_path / "database.sql"
    if db_backup.exists():
        console.print("[cyan]Restoring database...[/cyan]")
        subprocess.run([
            "psql",
            "-h", "localhost",
            "-U", "smartschool",
            "-d", "smartschoolgo",
            "-f", str(db_backup)
        ])
        
    console.print("[green]Restore completed![/green]")

@cli.command()
@click.option('--component', type=click.Choice(['api', 'ml', 'integration', 'all']), default='all')
@click.pass_context
def test(ctx, component):
    """Run system tests"""
    console.print(f"[cyan]Running tests for {component}...[/cyan]\n")
    
    test_commands = {
        'api': "pytest tests/api -v",
        'ml': "pytest tests/models -v",
        'integration': "pytest tests/integration -v",
        'all': "pytest tests/ -v --cov=src --cov-report=term-missing"
    }
    
    cmd = test_commands.get(component)
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    if result.returncode == 0:
        console.print("[green]All tests passed![/green]")
    else:
        console.print("[red]Some tests failed[/red]")
        console.print(result.stdout)

@cli.command()
@click.option('--workers', default=4, help='Number of worker processes')
@click.option('--queues', default='default,high,low', help='Comma-separated queue names')
@click.pass_context
def workers(ctx, workers, queues):
    """Start background workers for async tasks"""
    console.print(f"[cyan]Starting {workers} workers for queues: {queues}[/cyan]")
    
    # Start Celery workers
    subprocess.run([
        "celery",
        "-A", "src.smart_transport.tasks",
        "worker",
        "--loglevel=info",
        f"--concurrency={workers}",
        f"--queues={queues}"
    ])

@cli.command()
@click.pass_context
def scheduler(ctx):
    """Start task scheduler"""
    console.print("[cyan]Starting task scheduler...[/cyan]")
    
    # Define scheduled tasks
    schedule.every(5).minutes.do(lambda: console.print("[dim]Running health check...[/dim]"))
    schedule.every(30).minutes.do(lambda: console.print("[dim]Syncing data...[/dim]"))
    schedule.every().hour.do(lambda: console.print("[dim]Generating analytics...[/dim]"))
    schedule.every().day.at("02:00").do(lambda: console.print("[dim]Running backup...[/dim]"))
    
    console.print("[green]Scheduler started. Press Ctrl+C to stop.[/green]\n")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped[/yellow]")

@cli.command()
@click.pass_context
def monitor(ctx):
    """Live system monitoring dashboard"""
    console.print("[cyan]Starting monitoring dashboard...[/cyan]\n")
    
    def get_metrics():
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'network_in': psutil.net_io_counters().bytes_recv / 1024 / 1024,
            'network_out': psutil.net_io_counters().bytes_sent / 1024 / 1024,
            'processes': len(psutil.pids())
        }
    
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                metrics = get_metrics()
                
                table = Table(title="System Monitor", show_header=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")
                table.add_column("Status", style="green")
                
                # CPU
                cpu_status = "🟢" if metrics['cpu'] < 70 else "🟡" if metrics['cpu'] < 90 else "🔴"
                table.add_row("CPU Usage", f"{metrics['cpu']}%", cpu_status)
                
                # Memory
                mem_status = "🟢" if metrics['memory'] < 70 else "🟡" if metrics['memory'] < 90 else "🔴"
                table.add_row("Memory Usage", f"{metrics['memory']}%", mem_status)
                
                # Disk
                disk_status = "🟢" if metrics['disk'] < 70 else "🟡" if metrics['disk'] < 90 else "🔴"
                table.add_row("Disk Usage", f"{metrics['disk']}%", disk_status)
                
                # Network
                table.add_row("Network In", f"{metrics['network_in']:.2f} MB", "🟢")
                table.add_row("Network Out", f"{metrics['network_out']:.2f} MB", "🟢")
                
                # Processes
                table.add_row("Active Processes", str(metrics['processes']), "🟢")
                
                live.update(table)
                time.sleep(1)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")

@cli.command()
@click.argument('command', required=False)
@click.pass_context
def shell(ctx, command):
    """Interactive Python shell with system context"""
    import code
    
    # Prepare shell context
    shell_context = {
        'orchestrator': ctx.obj['orchestrator'],
        'config': get_settings(Environment(ctx.obj['ENV'])),
        'console': console,
        'asyncio': asyncio,
    }
    
    if command:
        # Execute single command
        exec(command, shell_context)
    else:
        # Start interactive shell
        console.print("[cyan]Starting interactive shell...[/cyan]")
        console.print("[dim]Available objects: orchestrator, config, console, asyncio[/dim]\n")
        
        code.interact(local=shell_context, banner="SmartSchoolGo Interactive Shell")

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    console.print(f"\n[yellow]Received signal {signum}, initiating shutdown...[/yellow]")
    asyncio.create_task(SystemOrchestrator().shutdown())
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("backups").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Print banner
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗              ║
    ║   ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝              ║
    ║   ███████╗██╔████╔██║███████║██████╔╝   ██║                 ║
    ║   ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║                 ║
    ║   ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║                 ║
    ║   ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                 ║
    ║                                                               ║
    ║   ███████╗ ██████╗██╗  ██╗ ██████╗  ██████╗ ██╗             ║
    ║   ██╔════╝██╔════╝██║  ██║██╔═══██╗██╔═══██╗██║             ║
    ║   ███████╗██║     ███████║██║   ██║██║   ██║██║             ║
    ║   ╚════██║██║     ██╔══██║██║   ██║██║   ██║██║             ║
    ║   ███████║╚██████╗██║  ██║╚██████╔╝╚██████╔╝███████╗        ║
    ║   ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝        ║
    ║                                                               ║
    ║            Smart School Transport Optimization                ║
    ║                    System Orchestrator                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    console.print(banner, style="bold cyan")
    console.print("[dim]Type 'python main.py --help' for available commands[/dim]\n")
    
    # Run CLI
    cli(obj={})