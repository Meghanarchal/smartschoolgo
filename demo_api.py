"""
SmartSchoolGo Demo API Server
Simple FastAPI server to demonstrate backend capabilities
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import random
from datetime import datetime
import uvicorn

app = FastAPI(
    title="SmartSchoolGo Demo API",
    description="Demo API for SmartSchoolGo - Smart School Transport System",
    version="1.0.0"
)

# Demo data
demo_schools = [
    {"id": "SCH_001", "name": "Canberra High School", "lat": -35.2809, "lon": 149.1300},
    {"id": "SCH_002", "name": "Belconnen High School", "lat": -35.2380, "lon": 149.0648},
    {"id": "SCH_003", "name": "Dickson College", "lat": -35.2507, "lon": 149.1394}
]

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return HTMLResponse(content="""
    <html>
        <body>
            <h1>🚌 SmartSchoolGo Demo API</h1>
            <h2>Smart School Transport Network Optimization</h2>
            <p><strong>Status:</strong> ✅ Running</p>
            <p><strong>Built for:</strong> GovHack 2025</p>
            
            <h3>Available Endpoints:</h3>
            <ul>
                <li><a href="/docs">📚 API Documentation (Swagger)</a></li>
                <li><a href="/health">🏥 Health Check</a></li>
                <li><a href="/schools">🏫 Schools Data</a></li>
                <li><a href="/routes">🚌 Routes Data</a></li>
                <li><a href="/tracking/live">📍 Live Tracking</a></li>
                <li><a href="/analytics/summary">📊 Analytics Summary</a></li>
            </ul>
            
            <h3>Features Demonstrated:</h3>
            <ul>
                <li>✅ RESTful API Design</li>
                <li>✅ Real-time Data Endpoints</li>
                <li>✅ Interactive Documentation</li>
                <li>✅ JSON Response Format</li>
                <li>✅ Error Handling</li>
            </ul>
            
            <p><em>This is a demonstration API showcasing the SmartSchoolGo system capabilities.</em></p>
        </body>
    </html>
    """)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SmartSchoolGo Demo API",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/schools")
def get_schools():
    """Get all schools data"""
    return {
        "status": "success",
        "count": len(demo_schools),
        "data": demo_schools,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/schools/{school_id}")
def get_school(school_id: str):
    """Get specific school data"""
    school = next((s for s in demo_schools if s["id"] == school_id), None)
    if school:
        return {
            "status": "success",
            "data": {
                **school,
                "students": random.randint(200, 800),
                "buses": random.randint(3, 12),
                "routes_active": random.randint(2, 8)
            }
        }
    return {"status": "error", "message": "School not found"}

@app.get("/routes")
def get_routes():
    """Get all routes data"""
    routes = []
    for i, school in enumerate(demo_schools):
        for j in range(random.randint(2, 5)):
            routes.append({
                "route_id": f"RT_{i:03d}_{j}",
                "school_id": school["id"],
                "school_name": school["name"],
                "status": random.choice(["Active", "Active", "Delayed", "Maintenance"]),
                "students": random.randint(20, 60),
                "duration_mins": random.randint(25, 55),
                "distance_km": random.randint(8, 25)
            })
    
    return {
        "status": "success",
        "count": len(routes),
        "data": routes,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/tracking/live")
def get_live_tracking():
    """Get live bus tracking data"""
    tracking = []
    for i in range(random.randint(5, 10)):
        school = random.choice(demo_schools)
        tracking.append({
            "bus_id": f"BUS_{i:03d}",
            "route_id": f"RT_{i:03d}",
            "school_name": school["name"],
            "lat": school["lat"] + random.uniform(-0.05, 0.05),
            "lon": school["lon"] + random.uniform(-0.05, 0.05),
            "speed_kmh": random.randint(15, 45),
            "heading": random.randint(0, 359),
            "students_on_board": random.randint(5, 45),
            "next_stop": f"Stop {random.randint(1, 8)}",
            "eta_mins": random.randint(3, 25),
            "status": random.choice(["On Time", "On Time", "Running Late"]),
            "last_update": datetime.now().isoformat()
        })
    
    return {
        "status": "success",
        "count": len(tracking),
        "data": tracking,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/analytics/summary")
def get_analytics_summary():
    """Get analytics summary"""
    return {
        "status": "success",
        "data": {
            "system_metrics": {
                "total_schools": len(demo_schools),
                "active_routes": random.randint(15, 25),
                "students_transported": random.randint(400, 600),
                "fleet_size": random.randint(18, 30),
                "uptime_percent": round(random.uniform(98.5, 99.9), 2)
            },
            "performance_metrics": {
                "avg_travel_time_mins": random.randint(35, 45),
                "on_time_percentage": round(random.uniform(85, 95), 1),
                "fuel_efficiency": round(random.uniform(7.5, 9.2), 1),
                "cost_per_student": round(random.uniform(45, 65), 2)
            },
            "safety_metrics": {
                "incidents_this_month": random.randint(0, 3),
                "safety_score": round(random.uniform(8.5, 9.8), 1),
                "maintenance_compliance": round(random.uniform(95, 100), 1)
            },
            "optimization_results": {
                "routes_optimized": random.randint(12, 18),
                "time_savings_percent": round(random.uniform(12, 25), 1),
                "cost_savings_monthly": random.randint(800, 1500),
                "emission_reduction_percent": round(random.uniform(8, 18), 1)
            }
        },
        "generated_at": datetime.now().isoformat()
    }

@app.get("/optimization/run")
def run_optimization():
    """Simulate running route optimization"""
    return {
        "status": "success",
        "message": "Route optimization simulation started",
        "optimization_id": f"OPT_{random.randint(1000, 9999)}",
        "estimated_completion": "3-5 minutes",
        "routes_to_optimize": random.randint(10, 20),
        "expected_improvements": {
            "travel_time_reduction": f"{random.randint(5, 15)}%",
            "cost_savings": f"${random.randint(200, 800)}/month",
            "efficiency_gain": f"{random.randint(8, 18)}%"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)