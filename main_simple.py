#!/usr/bin/env python3
"""
SmartSchoolGo - Simple Main Application for Testing
Simplified version for initial testing and deployment
"""

import os
import sys
import time
import click
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path

console = Console()

@click.group()
def cli():
    """SmartSchoolGo System - Simple Test Version"""
    pass

@cli.command()
def test():
    """Test basic system functionality"""
    console.print("\n[bold cyan]SmartSchoolGo System Test[/bold cyan]\n")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"[green]OK[/green] Python Version: {python_version}")
    
    # Check directory structure
    required_dirs = [
        'src/config',
        'src/smart_transport/api',
        'src/smart_transport/data',
        'src/smart_transport/models',
        'docs',
        'scripts'
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            console.print(f"[green]OK[/green] Directory exists: {dir_path}")
        else:
            console.print(f"[red]MISSING[/red] Directory: {dir_path}")
    
    # Check key files
    required_files = [
        '.env.example',
        'requirements.txt',
        'docker-compose.yml',
        'README.md',
        'main.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            console.print(f"[green]OK[/green] File exists: {file_path}")
        else:
            console.print(f"[red]MISSING[/red] File: {file_path}")

@cli.command()
def status():
    """Show system status"""
    table = Table(title="System Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Status", width=15)
    table.add_column("Details", style="dim")
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    cpu_status = "[green]Normal[/green]" if cpu_percent < 80 else "[yellow]High[/yellow]"
    mem_status = "[green]Normal[/green]" if memory.percent < 80 else "[yellow]High[/yellow]"
    disk_status = "[green]Normal[/green]" if disk.percent < 80 else "[yellow]High[/yellow]"
    
    table.add_row("CPU Usage", cpu_status, f"{cpu_percent}%")
    table.add_row("Memory Usage", mem_status, f"{memory.percent}%")
    table.add_row("Disk Usage", disk_status, f"{disk.percent}%")
    
    console.print(table)

@cli.command()
def info():
    """Show project information"""
    info_text = """
    SmartSchoolGo - Smart School Transport Network Optimization
    
    * AI-powered route optimization
    * Real-time safety monitoring
    * Multi-role interfaces (Parent, Admin, Planner)
    * Interactive mapping and tracking
    * Advanced analytics and reporting
    
    Technology Stack:
    * FastAPI + Streamlit
    * PostgreSQL + PostGIS + Redis
    * Machine Learning (TensorFlow, scikit-learn)
    * Real-time WebSocket communications
    """
    console.print(Panel(info_text.strip(), title="Project Overview", border_style="cyan"))

@cli.command()
@click.option('--service', type=click.Choice(['postgres', 'redis', 'all']), default='all')
def docker(service):
    """Manage Docker services"""
    console.print(f"[cyan]Managing Docker services: {service}[/cyan]\n")
    
    if service in ['postgres', 'all']:
        console.print("PostgreSQL with PostGIS:")
        console.print("   docker run --name smartschool_postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgis/postgis:14-3.2-alpine")
        
    if service in ['redis', 'all']:
        console.print("Redis Cache:")
        console.print("   docker run --name smartschool_redis -p 6379:6379 -d redis:7-alpine")
    
    console.print(f"\n[yellow]Or use docker-compose:[/yellow]")
    console.print("   docker-compose up -d")

@cli.command()
def install():
    """Show installation commands"""
    commands = [
        "# 1. Clone repository",
        "git clone <repository-url>",
        "cd smartschoolgo",
        "",
        "# 2. Create virtual environment",
        "python -m venv venv",
        "venv\\Scripts\\activate  # Windows",
        "source venv/bin/activate  # Linux/Mac",
        "",
        "# 3. Install dependencies",
        "pip install -r requirements.txt",
        "",
        "# 4. Setup environment",
        "cp .env.example .env",
        "# Edit .env with your settings",
        "",
        "# 5. Start services",
        "docker-compose up -d",
        "python scripts/init_db.py",
        "",
        "# 6. Run application",
        "python main.py start"
    ]
    
    console.print("\n[bold cyan]Installation Steps:[/bold cyan]\n")
    for cmd in commands:
        if cmd.startswith("#"):
            console.print(cmd, style="bold green")
        elif cmd == "":
            console.print()
        else:
            console.print(f"  {cmd}", style="dim")

if __name__ == "__main__":
    # Print banner
    banner = """
    ================================================================
                            SmartSchoolGo
                    Smart School Transport System
                    
                      Bus -> School -> Home -> Data
    ================================================================
    """
    
    console.print(banner, style="bold cyan")
    console.print("[dim]Simple test version - Use 'python main_simple.py --help' for commands[/dim]\n")
    
    cli()