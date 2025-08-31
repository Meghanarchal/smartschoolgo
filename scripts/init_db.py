#!/usr/bin/env python3
"""
Database initialization script for SmartSchoolGo
Creates database, extensions, and initial schema
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings

def create_database():
    """Create database if it doesn't exist"""
    config = get_settings()
    
    # Parse database URL
    # Format: postgresql://user:password@host:port/database
    db_url = config.database_url
    parts = db_url.replace('postgresql://', '').split('@')
    user_pass = parts[0].split(':')
    host_port_db = parts[1].split('/')
    host_port = host_port_db[0].split(':')
    
    db_user = user_pass[0]
    db_pass = user_pass[1] if len(user_pass) > 1 else ''
    db_host = host_port[0]
    db_port = host_port[1] if len(host_port) > 1 else '5432'
    db_name = host_port_db[1]
    
    print(f"Connecting to PostgreSQL at {db_host}:{db_port}")
    
    try:
        # Connect to PostgreSQL server (not specific database)
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created successfully!")
        else:
            print(f"Database '{db_name}' already exists.")
        
        cursor.close()
        conn.close()
        
        # Connect to the new database and create extensions
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            database=db_name
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("Creating PostgreSQL extensions...")
        
        extensions = [
            'postgis',
            'postgis_topology',
            'uuid-ossp',
            'pg_trgm',
            'btree_gist'
        ]
        
        for ext in extensions:
            try:
                cursor.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
                print(f"  ✓ Extension '{ext}' enabled")
            except Exception as e:
                print(f"  ✗ Failed to create extension '{ext}': {e}")
        
        # Create schemas
        schemas = ['smart_transport', 'analytics', 'realtime']
        for schema in schemas:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
            print(f"  ✓ Schema '{schema}' created")
        
        cursor.close()
        conn.close()
        
        print("\nDatabase initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def create_tables():
    """Create initial database tables"""
    config = get_settings()
    
    try:
        conn = psycopg2.connect(config.database_url)
        cursor = conn.cursor()
        
        print("\nCreating database tables...")
        
        # Create tables SQL
        tables_sql = """
        -- Schools table
        CREATE TABLE IF NOT EXISTS schools (
            school_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name VARCHAR(255) NOT NULL,
            address TEXT,
            location GEOMETRY(Point, 4326),
            capacity INTEGER,
            contact_email VARCHAR(255),
            contact_phone VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Students table
        CREATE TABLE IF NOT EXISTS students (
            student_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            school_id UUID REFERENCES schools(school_id),
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            grade_level INTEGER,
            home_address TEXT,
            home_location GEOMETRY(Point, 4326),
            special_needs JSONB,
            parent_email VARCHAR(255),
            parent_phone VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Vehicles table
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            registration_number VARCHAR(50) UNIQUE NOT NULL,
            capacity INTEGER NOT NULL,
            vehicle_type VARCHAR(50),
            accessibility_features JSONB,
            current_location GEOMETRY(Point, 4326),
            status VARCHAR(50) DEFAULT 'available',
            last_maintenance DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Drivers table
        CREATE TABLE IF NOT EXISTS drivers (
            driver_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            license_number VARCHAR(50) UNIQUE,
            phone VARCHAR(50),
            email VARCHAR(255),
            vehicle_id UUID REFERENCES vehicles(vehicle_id),
            status VARCHAR(50) DEFAULT 'available',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Routes table
        CREATE TABLE IF NOT EXISTS routes (
            route_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name VARCHAR(255),
            school_id UUID REFERENCES schools(school_id),
            vehicle_id UUID REFERENCES vehicles(vehicle_id),
            driver_id UUID REFERENCES drivers(driver_id),
            route_geometry GEOMETRY(LineString, 4326),
            stops JSONB,
            schedule JSONB,
            status VARCHAR(50) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Student Route Assignments
        CREATE TABLE IF NOT EXISTS student_routes (
            assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            student_id UUID REFERENCES students(student_id),
            route_id UUID REFERENCES routes(route_id),
            pickup_stop INTEGER,
            dropoff_stop INTEGER,
            pickup_time TIME,
            dropoff_time TIME,
            active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Incidents table
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            route_id UUID REFERENCES routes(route_id),
            incident_type VARCHAR(100),
            severity VARCHAR(50),
            location GEOMETRY(Point, 4326),
            description TEXT,
            reported_by VARCHAR(255),
            status VARCHAR(50) DEFAULT 'open',
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Tracking History
        CREATE TABLE IF NOT EXISTS tracking_history (
            tracking_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            vehicle_id UUID REFERENCES vehicles(vehicle_id),
            location GEOMETRY(Point, 4326),
            speed DECIMAL(5,2),
            heading DECIMAL(5,2),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_schools_location ON schools USING GIST(location);
        CREATE INDEX IF NOT EXISTS idx_students_location ON students USING GIST(home_location);
        CREATE INDEX IF NOT EXISTS idx_vehicles_location ON vehicles USING GIST(current_location);
        CREATE INDEX IF NOT EXISTS idx_routes_geometry ON routes USING GIST(route_geometry);
        CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents USING GIST(location);
        CREATE INDEX IF NOT EXISTS idx_tracking_location ON tracking_history USING GIST(location);
        CREATE INDEX IF NOT EXISTS idx_tracking_timestamp ON tracking_history(timestamp);
        
        -- Create update trigger function
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        -- Add update triggers to all tables
        CREATE TRIGGER update_schools_updated_at BEFORE UPDATE ON schools
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
        CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
        CREATE TRIGGER update_vehicles_updated_at BEFORE UPDATE ON vehicles
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
        CREATE TRIGGER update_drivers_updated_at BEFORE UPDATE ON drivers
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
        CREATE TRIGGER update_routes_updated_at BEFORE UPDATE ON routes
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        
        cursor.execute(tables_sql)
        conn.commit()
        
        print("  ✓ Tables created successfully")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def main():
    """Main initialization function"""
    print("=" * 60)
    print("SmartSchoolGo Database Initialization")
    print("=" * 60)
    
    # Create database
    if not create_database():
        print("\nDatabase initialization failed!")
        sys.exit(1)
    
    # Create tables
    if not create_tables():
        print("\nTable creation failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Database initialization completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()