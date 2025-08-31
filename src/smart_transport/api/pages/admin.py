"""
Administrator Dashboard for SmartSchoolGo

This module provides a comprehensive administrator dashboard with fleet management,
route optimization, student assignment, driver management, performance analytics,
safety monitoring, and incident reporting capabilities.

Features:
- Real-time fleet management with vehicle status monitoring
- Interactive route optimization tools with drag-and-drop interface
- Bulk student assignment operations with validation
- Driver management and scheduling system
- Performance analytics dashboard with KPI tracking
- Safety monitoring and incident reporting
- Cost analysis and budget tracking
- Maintenance scheduling and alerts
- Communication tools for drivers and parents

Author: SmartSchoolGo Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
import uuid
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import folium
    from streamlit_folium import st_folium
    from streamlit_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
    from streamlit_elements import elements, mui, html, dashboard, nivo
    import streamlit_draggable as drag
    from streamlit_timeline import st_timeline
    import streamlit_ace
    ADMIN_LIBS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some advanced features may not be available: {e}")
    ADMIN_LIBS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FleetStatus:
    """Fleet management status constants."""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"
    IDLE = "idle"


class AlertPriority:
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AdminDataService:
    """Data service for administrator dashboard."""
    
    @staticmethod
    @st.cache_data(ttl=30)
    def get_fleet_overview() -> Dict[str, Any]:
        """Get comprehensive fleet overview data."""
        # Mock data - replace with actual database queries
        vehicles = []
        for i in range(25):  # 25 vehicle fleet
            status_weights = [0.7, 0.15, 0.1, 0.05]  # Active, Idle, Maintenance, Out of Service
            status = np.random.choice([FleetStatus.ACTIVE, FleetStatus.IDLE, 
                                     FleetStatus.MAINTENANCE, FleetStatus.OUT_OF_SERVICE],
                                    p=status_weights)
            
            vehicles.append({
                'vehicle_id': f'BUS-{i+1:03d}',
                'make_model': f"{np.random.choice(['Mercedes', 'Volvo', 'Scania', 'MAN'])} {np.random.choice(['Citaro', 'B7RLE', 'Omnicity'])}",
                'year': np.random.randint(2018, 2024),
                'capacity': np.random.choice([35, 45, 50, 55]),
                'route_id': f'R{np.random.randint(1, 8)}' if status == FleetStatus.ACTIVE else None,
                'driver_id': f'DRV-{np.random.randint(1, 30):03d}' if status == FleetStatus.ACTIVE else None,
                'status': status,
                'location': {
                    'latitude': -35.28 + np.random.uniform(-0.05, 0.05),
                    'longitude': 149.13 + np.random.uniform(-0.05, 0.05)
                },
                'fuel_level': np.random.uniform(20, 100),
                'mileage': np.random.randint(50000, 300000),
                'last_maintenance': datetime.now() - timedelta(days=np.random.randint(1, 120)),
                'next_maintenance_due': datetime.now() + timedelta(days=np.random.randint(1, 60)),
                'cost_per_km': np.random.uniform(0.45, 0.65),
                'incidents': np.random.randint(0, 3),
                'performance_score': np.random.uniform(85, 98)
            })
        
        # Fleet statistics
        active_vehicles = [v for v in vehicles if v['status'] == FleetStatus.ACTIVE]
        total_capacity = sum(v['capacity'] for v in vehicles)
        utilization = len(active_vehicles) / len(vehicles) * 100
        
        return {
            'vehicles': vehicles,
            'total_vehicles': len(vehicles),
            'active_vehicles': len(active_vehicles),
            'maintenance_vehicles': len([v for v in vehicles if v['status'] == FleetStatus.MAINTENANCE]),
            'out_of_service': len([v for v in vehicles if v['status'] == FleetStatus.OUT_OF_SERVICE]),
            'total_capacity': total_capacity,
            'utilization_rate': utilization,
            'avg_fuel_level': np.mean([v['fuel_level'] for v in vehicles]),
            'maintenance_due_soon': len([v for v in vehicles if (v['next_maintenance_due'] - datetime.now()).days <= 7]),
            'total_incidents': sum(v['incidents'] for v in vehicles),
            'avg_performance': np.mean([v['performance_score'] for v in vehicles])
        }
    
    @staticmethod
    @st.cache_data(ttl=60)
    def get_route_optimization_data() -> Dict[str, Any]:
        """Get route optimization data."""
        routes = []
        for i in range(8):
            routes.append({
                'route_id': f'R{i+1}',
                'route_name': f'Route {i+1} - {["North", "South", "East", "West", "Central", "Outer", "Inner", "Express"][i]}',
                'assigned_vehicles': np.random.randint(2, 5),
                'total_students': np.random.randint(30, 80),
                'total_stops': np.random.randint(8, 15),
                'distance_km': np.random.uniform(15, 35),
                'duration_minutes': np.random.randint(35, 65),
                'on_time_performance': np.random.uniform(75, 95),
                'cost_per_day': np.random.uniform(350, 650),
                'optimization_score': np.random.uniform(70, 90),
                'last_optimized': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                'efficiency_rating': np.random.choice(['A', 'B', 'C'], p=[0.4, 0.5, 0.1])
            })
        
        optimization_suggestions = [
            {
                'route_id': 'R3',
                'suggestion': 'Combine stops 7 and 8 to reduce travel time by 8 minutes',
                'potential_savings': 145.50,
                'impact_score': 8.5
            },
            {
                'route_id': 'R1',
                'suggestion': 'Adjust departure time by +5 minutes to avoid peak traffic',
                'potential_savings': 89.20,
                'impact_score': 6.2
            },
            {
                'route_id': 'R5',
                'suggestion': 'Reallocate 12 students to Route 6 for better capacity utilization',
                'potential_savings': 234.75,
                'impact_score': 9.1
            }
        ]
        
        return {
            'routes': routes,
            'optimization_suggestions': optimization_suggestions,
            'total_routes': len(routes),
            'avg_efficiency': np.mean([ord(r['efficiency_rating']) - ord('A') + 1 for r in routes]),
            'total_students': sum(r['total_students'] for r in routes),
            'total_cost': sum(r['cost_per_day'] for r in routes),
            'optimization_potential': sum(s['potential_savings'] for s in optimization_suggestions)
        }
    
    @staticmethod
    @st.cache_data(ttl=120)
    def get_student_assignment_data() -> Dict[str, Any]:
        """Get student assignment and management data."""
        students = []
        schools = ['Canberra Primary', 'Capital Hill School', 'Turner Primary', 
                  'Ainslie School', 'North Canberra Primary']
        
        for i in range(450):  # 450 students
            school = np.random.choice(schools)
            grade = np.random.randint(1, 7)
            
            students.append({
                'student_id': f'STU-{i+1:04d}',
                'name': f'Student {i+1}',
                'school': school,
                'grade': grade,
                'home_address': f'{np.random.randint(1, 999)} {np.random.choice(["Main", "Oak", "Pine", "Elm", "Maple"])} St',
                'suburb': np.random.choice(['Turner', 'Braddon', 'Reid', 'Campbell', 'Ainslie']),
                'route_id': f'R{np.random.randint(1, 8)}',
                'pickup_location': f'Stop {np.random.randint(1, 12)}',
                'pickup_time': f'{np.random.randint(7, 8):02d}:{np.random.choice([15, 30, 45]):02d}',
                'special_needs': np.random.choice([None, 'Wheelchair', 'Supervision', 'Medical'], p=[0.85, 0.05, 0.05, 0.05]),
                'parent_contact': f'parent{i+1}@email.com',
                'emergency_contact': f'+61 4{np.random.randint(10000000, 99999999)}',
                'enrollment_date': datetime.now() - timedelta(days=np.random.randint(30, 365)),
                'attendance_rate': np.random.uniform(85, 100),
                'transport_eligibility': np.random.choice(['Eligible', 'Conditional', 'Not Eligible'], p=[0.7, 0.2, 0.1])
            })
        
        return {
            'students': students,
            'total_students': len(students),
            'by_school': pd.Series([s['school'] for s in students]).value_counts().to_dict(),
            'by_route': pd.Series([s['route_id'] for s in students]).value_counts().to_dict(),
            'special_needs_count': len([s for s in students if s['special_needs']]),
            'avg_attendance': np.mean([s['attendance_rate'] for s in students]),
            'eligible_students': len([s for s in students if s['transport_eligibility'] == 'Eligible'])
        }
    
    @staticmethod
    @st.cache_data(ttl=60)
    def get_driver_management_data() -> Dict[str, Any]:
        """Get driver management and scheduling data."""
        drivers = []
        for i in range(35):  # 35 drivers
            status = np.random.choice(['Active', 'Off Duty', 'On Leave', 'Training'], p=[0.7, 0.15, 0.1, 0.05])
            
            drivers.append({
                'driver_id': f'DRV-{i+1:03d}',
                'name': f'Driver {i+1}',
                'license_class': np.random.choice(['HC', 'MC', 'LR'], p=[0.6, 0.3, 0.1]),
                'license_expiry': datetime.now() + timedelta(days=np.random.randint(30, 1095)),
                'employment_date': datetime.now() - timedelta(days=np.random.randint(90, 2190)),
                'status': status,
                'current_route': f'R{np.random.randint(1, 8)}' if status == 'Active' else None,
                'current_vehicle': f'BUS-{np.random.randint(1, 25):03d}' if status == 'Active' else None,
                'hours_this_week': np.random.uniform(0, 40),
                'performance_rating': np.random.uniform(7.5, 10.0),
                'incidents_count': np.random.randint(0, 3),
                'training_expires': datetime.now() + timedelta(days=np.random.randint(60, 365)),
                'contact_phone': f'+61 4{np.random.randint(10000000, 99999999)}',
                'emergency_contact': f'+61 4{np.random.randint(10000000, 99999999)}',
                'medical_clearance': datetime.now() + timedelta(days=np.random.randint(90, 365)),
                'salary_grade': np.random.choice(['Grade 1', 'Grade 2', 'Grade 3', 'Senior'], p=[0.4, 0.3, 0.2, 0.1])
            })
        
        return {
            'drivers': drivers,
            'total_drivers': len(drivers),
            'active_drivers': len([d for d in drivers if d['status'] == 'Active']),
            'drivers_on_leave': len([d for d in drivers if d['status'] == 'On Leave']),
            'training_due_soon': len([d for d in drivers if (d['training_expires'] - datetime.now()).days <= 30]),
            'license_expiring_soon': len([d for d in drivers if (d['license_expiry'] - datetime.now()).days <= 60]),
            'avg_performance': np.mean([d['performance_rating'] for d in drivers]),
            'total_incidents': sum(d['incidents_count'] for d in drivers)
        }
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_performance_analytics() -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        # Generate time series data for the last 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        performance_data = []
        for date in dates:
            performance_data.append({
                'date': date.date(),
                'on_time_performance': np.random.uniform(82, 95),
                'fuel_efficiency': np.random.uniform(8.5, 12.0),
                'student_satisfaction': np.random.uniform(4.0, 4.8),
                'fleet_utilization': np.random.uniform(85, 96),
                'safety_incidents': np.random.poisson(0.2),
                'maintenance_costs': np.random.uniform(450, 850),
                'operational_costs': np.random.uniform(2800, 3500),
                'revenue': np.random.uniform(4200, 4800)
            })
        
        df = pd.DataFrame(performance_data)
        
        # KPI calculations
        current_month_performance = df.tail(7).mean()  # Last week average
        previous_month_performance = df.head(7).mean()  # First week average
        
        kpis = {
            'on_time_performance': {
                'current': current_month_performance['on_time_performance'],
                'change': current_month_performance['on_time_performance'] - previous_month_performance['on_time_performance'],
                'target': 90.0
            },
            'fuel_efficiency': {
                'current': current_month_performance['fuel_efficiency'],
                'change': current_month_performance['fuel_efficiency'] - previous_month_performance['fuel_efficiency'],
                'target': 10.0
            },
            'student_satisfaction': {
                'current': current_month_performance['student_satisfaction'],
                'change': current_month_performance['student_satisfaction'] - previous_month_performance['student_satisfaction'],
                'target': 4.5
            },
            'fleet_utilization': {
                'current': current_month_performance['fleet_utilization'],
                'change': current_month_performance['fleet_utilization'] - previous_month_performance['fleet_utilization'],
                'target': 92.0
            }
        }
        
        return {
            'historical_data': df,
            'kpis': kpis,
            'total_incidents': df['safety_incidents'].sum(),
            'avg_maintenance_cost': df['maintenance_costs'].mean(),
            'total_operational_cost': df['operational_costs'].sum(),
            'total_revenue': df['revenue'].sum(),
            'profit_margin': ((df['revenue'] - df['operational_costs']).sum() / df['revenue'].sum()) * 100
        }
    
    @staticmethod
    @st.cache_data(ttl=30)
    def get_safety_incidents() -> List[Dict[str, Any]]:
        """Get safety incidents and monitoring data."""
        incidents = []
        incident_types = ['Minor Collision', 'Student Injury', 'Vehicle Breakdown', 
                         'Traffic Violation', 'Weather Related', 'Vandalism']
        
        for i in range(15):  # 15 recent incidents
            incident = {
                'incident_id': f'INC-{i+1:04d}',
                'date': datetime.now() - timedelta(days=np.random.randint(1, 90)),
                'type': np.random.choice(incident_types),
                'severity': np.random.choice([AlertPriority.LOW, AlertPriority.MEDIUM, 
                                           AlertPriority.HIGH, AlertPriority.CRITICAL], 
                                          p=[0.4, 0.4, 0.15, 0.05]),
                'vehicle_id': f'BUS-{np.random.randint(1, 25):03d}',
                'driver_id': f'DRV-{np.random.randint(1, 35):03d}',
                'route_id': f'R{np.random.randint(1, 8)}',
                'location': f'{np.random.choice(["Main St", "Oak Ave", "Pine Rd", "School Grounds"])}',
                'description': f'Sample incident description {i+1}',
                'status': np.random.choice(['Open', 'Under Investigation', 'Resolved', 'Closed'], 
                                         p=[0.2, 0.3, 0.3, 0.2]),
                'cost_impact': np.random.uniform(0, 5000) if np.random.random() > 0.3 else 0,
                'investigation_officer': f'Officer {np.random.randint(1, 5)}',
                'follow_up_required': np.random.choice([True, False], p=[0.3, 0.7])
            }
            incidents.append(incident)
        
        return incidents


class AdminUIComponents:
    """UI components for administrator dashboard."""
    
    @staticmethod
    def render_executive_summary():
        """Render executive summary cards."""
        st.header("🏢 Executive Dashboard")
        
        # Get all data
        fleet_data = AdminDataService.get_fleet_overview()
        route_data = AdminDataService.get_route_optimization_data()
        student_data = AdminDataService.get_student_assignment_data()
        driver_data = AdminDataService.get_driver_management_data()
        performance_data = AdminDataService.get_performance_analytics()
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Fleet Utilization",
                f"{fleet_data['utilization_rate']:.1f}%",
                delta=f"{np.random.uniform(-2, 5):.1f}%"
            )
        
        with col2:
            on_time_kpi = performance_data['kpis']['on_time_performance']
            st.metric(
                "On-Time Performance",
                f"{on_time_kpi['current']:.1f}%",
                delta=f"{on_time_kpi['change']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Active Students",
                f"{student_data['total_students']:,}",
                delta=f"+{np.random.randint(5, 15)}"
            )
        
        with col4:
            st.metric(
                "Safety Score",
                f"{100 - performance_data['total_incidents']*2:.1f}/100",
                delta=f"{np.random.uniform(-1, 3):.1f}"
            )
        
        with col5:
            profit_margin = performance_data['profit_margin']
            st.metric(
                "Profit Margin",
                f"{profit_margin:.1f}%",
                delta=f"{np.random.uniform(-2, 3):.1f}%"
            )
        
        # Status indicators
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            maintenance_due = fleet_data['maintenance_due_soon']
            if maintenance_due > 0:
                st.warning(f"⚠️ {maintenance_due} vehicles need maintenance")
            else:
                st.success("✅ All vehicles maintained")
        
        with col2:
            training_due = driver_data['training_due_soon']
            if training_due > 0:
                st.warning(f"📚 {training_due} drivers need training renewal")
            else:
                st.success("✅ All driver training current")
        
        with col3:
            incidents = AdminDataService.get_safety_incidents()
            open_incidents = len([i for i in incidents if i['status'] in ['Open', 'Under Investigation']])
            if open_incidents > 0:
                st.error(f"🚨 {open_incidents} open safety incidents")
            else:
                st.success("✅ No open incidents")
        
        with col4:
            optimization_potential = route_data['optimization_potential']
            if optimization_potential > 100:
                st.info(f"💡 ${optimization_potential:.0f} optimization potential")
            else:
                st.success("✅ Routes optimized")
    
    @staticmethod
    def render_fleet_management():
        """Render fleet management interface."""
        st.subheader("🚌 Fleet Management")
        
        fleet_data = AdminDataService.get_fleet_overview()
        vehicles = fleet_data['vehicles']
        
        # Fleet overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vehicles", fleet_data['total_vehicles'])
        
        with col2:
            st.metric("Active Vehicles", fleet_data['active_vehicles'])
        
        with col3:
            st.metric("In Maintenance", fleet_data['maintenance_vehicles'])
        
        with col4:
            st.metric("Avg Fuel Level", f"{fleet_data['avg_fuel_level']:.1f}%")
        
        # Vehicle status distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            status_counts = pd.Series([v['status'] for v in vehicles]).value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Vehicle Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Maintenance schedule
            maintenance_data = []
            for vehicle in vehicles:
                days_until_maintenance = (vehicle['next_maintenance_due'] - datetime.now()).days
                maintenance_data.append({
                    'vehicle_id': vehicle['vehicle_id'],
                    'days_until_maintenance': days_until_maintenance,
                    'priority': 'High' if days_until_maintenance <= 7 else 'Medium' if days_until_maintenance <= 21 else 'Low'
                })
            
            maintenance_df = pd.DataFrame(maintenance_data)
            priority_counts = maintenance_df['priority'].value_counts()
            
            fig = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Maintenance Priority Distribution",
                color=priority_counts.index,
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Vehicle management table
        st.subheader("Vehicle Details")
        
        # Convert to DataFrame for better display
        vehicles_df = pd.DataFrame(vehicles)
        
        # Format columns
        vehicles_df['fuel_level'] = vehicles_df['fuel_level'].round(1).astype(str) + '%'
        vehicles_df['performance_score'] = vehicles_df['performance_score'].round(1).astype(str) + '/100'
        vehicles_df['last_maintenance'] = vehicles_df['last_maintenance'].dt.strftime('%Y-%m-%d')
        vehicles_df['next_maintenance_due'] = vehicles_df['next_maintenance_due'].dt.strftime('%Y-%m-%d')
        
        # Select columns to display
        display_columns = ['vehicle_id', 'make_model', 'year', 'status', 'route_id', 
                          'fuel_level', 'performance_score', 'next_maintenance_due', 'incidents']
        
        if ADMIN_LIBS_AVAILABLE:
            # Use AgGrid for better interaction
            gb = GridOptionsBuilder.from_dataframe(vehicles_df[display_columns])
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_selection('single', use_checkbox=True)
            gb.configure_default_column(
                filterable=True, 
                sortable=True, 
                editable=False,
                groupable=True
            )
            
            # Color coding for status
            gb.configure_column(
                'status',
                cellStyle={
                    'styleConditions': [
                        {'condition': "params.value == 'active'", 'style': {'backgroundColor': '#d4edda', 'color': '#155724'}},
                        {'condition': "params.value == 'maintenance'", 'style': {'backgroundColor': '#fff3cd', 'color': '#856404'}},
                        {'condition': "params.value == 'out_of_service'", 'style': {'backgroundColor': '#f8d7da', 'color': '#721c24'}}
                    ]
                }
            )
            
            grid_options = gb.build()
            
            grid_response = AgGrid(
                vehicles_df[display_columns],
                gridOptions=grid_options,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                height=400
            )
            
            # Handle selection
            if grid_response['selected_rows']:
                selected_vehicle = grid_response['selected_rows'][0]
                AdminUIComponents._render_vehicle_details(selected_vehicle['vehicle_id'], vehicles)
        
        else:
            # Fallback to standard dataframe
            st.dataframe(vehicles_df[display_columns], use_container_width=True)
        
        # Fleet map
        if st.checkbox("Show Fleet Location Map"):
            AdminUIComponents._render_fleet_map(vehicles)
    
    @staticmethod
    def _render_vehicle_details(vehicle_id: str, vehicles: List[Dict]):
        """Render detailed view for selected vehicle."""
        vehicle = next((v for v in vehicles if v['vehicle_id'] == vehicle_id), None)
        if not vehicle:
            return
        
        st.subheader(f"Vehicle Details: {vehicle_id}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Basic Information**")
            st.write(f"Make/Model: {vehicle['make_model']}")
            st.write(f"Year: {vehicle['year']}")
            st.write(f"Capacity: {vehicle['capacity']} students")
            st.write(f"Mileage: {vehicle['mileage']:,} km")
        
        with col2:
            st.write("**Current Status**")
            st.write(f"Status: {vehicle['status'].title()}")
            st.write(f"Route: {vehicle.get('route_id', 'Not assigned')}")
            st.write(f"Driver: {vehicle.get('driver_id', 'Not assigned')}")
            st.write(f"Fuel Level: {vehicle['fuel_level']:.1f}%")
        
        with col3:
            st.write("**Performance**")
            st.write(f"Performance Score: {vehicle['performance_score']:.1f}/100")
            st.write(f"Cost per km: ${vehicle['cost_per_km']:.2f}")
            st.write(f"Incidents: {vehicle['incidents']}")
            
            days_until_maintenance = (vehicle['next_maintenance_due'] - datetime.now()).days
            if days_until_maintenance <= 7:
                st.error(f"⚠️ Maintenance due in {days_until_maintenance} days")
            else:
                st.info(f"Next maintenance: {days_until_maintenance} days")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"Schedule Maintenance"):
                st.success(f"Maintenance scheduled for {vehicle_id}")
        
        with col2:
            if st.button("Update Status"):
                st.info("Status update dialog would open here")
        
        with col3:
            if st.button("View History"):
                st.info("Vehicle history would be displayed")
        
        with col4:
            if st.button("Generate Report"):
                AdminUIComponents._generate_vehicle_report(vehicle)
    
    @staticmethod
    def _render_fleet_map(vehicles: List[Dict]):
        """Render fleet location map."""
        try:
            # Create map centered on Canberra
            m = folium.Map(location=[-35.2809, 149.1300], zoom_start=12)
            
            # Add vehicle markers
            status_colors = {
                FleetStatus.ACTIVE: 'green',
                FleetStatus.IDLE: 'blue',
                FleetStatus.MAINTENANCE: 'orange',
                FleetStatus.OUT_OF_SERVICE: 'red'
            }
            
            for vehicle in vehicles:
                if vehicle['location']:
                    color = status_colors.get(vehicle['status'], 'gray')
                    
                    folium.Marker(
                        [vehicle['location']['latitude'], vehicle['location']['longitude']],
                        popup=f"""
                        <div>
                            <h4>{vehicle['vehicle_id']}</h4>
                            <p><b>Status:</b> {vehicle['status'].title()}</p>
                            <p><b>Route:</b> {vehicle.get('route_id', 'N/A')}</p>
                            <p><b>Fuel:</b> {vehicle['fuel_level']:.1f}%</p>
                        </div>
                        """,
                        icon=folium.Icon(color=color, icon='bus', prefix='fa'),
                        tooltip=f"{vehicle['vehicle_id']} - {vehicle['status'].title()}"
                    ).add_to(m)
            
            st_folium(m, width=700, height=400)
            
        except Exception as e:
            st.error(f"Unable to display map: {str(e)}")
    
    @staticmethod
    def _generate_vehicle_report(vehicle: Dict):
        """Generate vehicle report."""
        report_data = {
            'Vehicle ID': vehicle['vehicle_id'],
            'Make/Model': vehicle['make_model'],
            'Year': vehicle['year'],
            'Status': vehicle['status'],
            'Mileage': f"{vehicle['mileage']:,} km",
            'Performance Score': f"{vehicle['performance_score']:.1f}/100",
            'Fuel Level': f"{vehicle['fuel_level']:.1f}%",
            'Last Maintenance': vehicle['last_maintenance'].strftime('%Y-%m-%d'),
            'Next Maintenance Due': vehicle['next_maintenance_due'].strftime('%Y-%m-%d'),
            'Incidents': vehicle['incidents']
        }
        
        # Create CSV content
        csv_content = "Metric,Value\n"
        for key, value in report_data.items():
            csv_content += f"{key},{value}\n"
        
        st.download_button(
            label="📄 Download Vehicle Report",
            data=csv_content,
            file_name=f"vehicle_report_{vehicle['vehicle_id']}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def render_route_optimization():
        """Render route optimization tools."""
        st.subheader("🗺️ Route Optimization")
        
        route_data = AdminDataService.get_route_optimization_data()
        routes = route_data['routes']
        suggestions = route_data['optimization_suggestions']
        
        # Optimization overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Routes", route_data['total_routes'])
        
        with col2:
            st.metric("Students Served", f"{route_data['total_students']:,}")
        
        with col3:
            st.metric("Daily Cost", f"${route_data['total_cost']:,.0f}")
        
        with col4:
            st.metric("Savings Potential", f"${route_data['optimization_potential']:,.0f}")
        
        # Route performance analysis
        st.subheader("Route Performance Analysis")
        
        routes_df = pd.DataFrame(routes)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # On-time performance by route
            fig = px.bar(
                routes_df,
                x='route_id',
                y='on_time_performance',
                title="On-Time Performance by Route",
                color='efficiency_rating',
                color_discrete_map={'A': 'green', 'B': 'orange', 'C': 'red'}
            )
            fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Target: 90%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost vs efficiency scatter
            fig = px.scatter(
                routes_df,
                x='cost_per_day',
                y='optimization_score',
                size='total_students',
                hover_data=['route_name', 'distance_km'],
                title="Cost vs Optimization Score",
                labels={'cost_per_day': 'Daily Cost ($)', 'optimization_score': 'Optimization Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Optimization suggestions
        st.subheader("💡 Optimization Suggestions")
        
        if suggestions:
            for suggestion in suggestions:
                with st.expander(f"Route {suggestion['route_id']} - Potential Savings: ${suggestion['potential_savings']:.2f}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Suggestion:** {suggestion['suggestion']}")
                        st.write(f"**Impact Score:** {suggestion['impact_score']}/10")
                        
                        if st.button(f"Apply Optimization", key=f"apply_{suggestion['route_id']}"):
                            st.success(f"Optimization applied to Route {suggestion['route_id']}")
                    
                    with col2:
                        st.metric("Potential Savings", f"${suggestion['potential_savings']:.2f}")
                        st.progress(suggestion['impact_score'] / 10)
        
        # Route builder interface
        st.subheader("🛠️ Route Builder")
        
        if ADMIN_LIBS_AVAILABLE:
            # Drag-and-drop interface would go here
            st.info("Interactive drag-and-drop route builder would be implemented here with advanced libraries")
        
        # Manual route editing
        selected_route = st.selectbox("Select Route to Edit", options=[r['route_id'] for r in routes])
        
        if selected_route:
            route = next(r for r in routes if r['route_id'] == selected_route)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Route Details**")
                st.write(f"Name: {route['route_name']}")
                st.write(f"Students: {route['total_students']}")
                st.write(f"Stops: {route['total_stops']}")
                st.write(f"Distance: {route['distance_km']:.1f} km")
                st.write(f"Duration: {route['duration_minutes']} min")
                
            with col2:
                st.write("**Performance Metrics**")
                st.write(f"On-time: {route['on_time_performance']:.1f}%")
                st.write(f"Cost: ${route['cost_per_day']:.2f}/day")
                st.write(f"Efficiency: {route['efficiency_rating']}")
                st.write(f"Optimization Score: {route['optimization_score']:.1f}/100")
                
                if st.button("Re-optimize Route"):
                    st.success("Route optimization initiated")
    
    @staticmethod
    def render_student_assignment():
        """Render student assignment system."""
        st.subheader("👥 Student Assignment Management")
        
        student_data = AdminDataService.get_student_assignment_data()
        students = student_data['students']
        
        # Student statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", f"{student_data['total_students']:,}")
        
        with col2:
            st.metric("Special Needs", student_data['special_needs_count'])
        
        with col3:
            st.metric("Avg Attendance", f"{student_data['avg_attendance']:.1f}%")
        
        with col4:
            st.metric("Transport Eligible", student_data['eligible_students'])
        
        # Student distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Students by school
            school_data = student_data['by_school']
            fig = px.pie(
                values=list(school_data.values()),
                names=list(school_data.keys()),
                title="Students by School"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Students by route
            route_data = student_data['by_route']
            fig = px.bar(
                x=list(route_data.keys()),
                y=list(route_data.values()),
                title="Students by Route"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Student management interface
        st.subheader("Student Management")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_school = st.selectbox("Filter by School", ["All"] + list(student_data['by_school'].keys()))
        
        with col2:
            selected_route = st.selectbox("Filter by Route", ["All"] + list(student_data['by_route'].keys()))
        
        with col3:
            eligibility_filter = st.selectbox("Filter by Eligibility", ["All", "Eligible", "Conditional", "Not Eligible"])
        
        # Apply filters
        filtered_students = students.copy()
        
        if selected_school != "All":
            filtered_students = [s for s in filtered_students if s['school'] == selected_school]
        
        if selected_route != "All":
            filtered_students = [s for s in filtered_students if s['route_id'] == selected_route]
        
        if eligibility_filter != "All":
            filtered_students = [s for s in filtered_students if s['transport_eligibility'] == eligibility_filter]
        
        st.write(f"Showing {len(filtered_students)} students")
        
        # Student table
        if filtered_students:
            students_df = pd.DataFrame(filtered_students)
            
            # Format dates
            students_df['enrollment_date'] = students_df['enrollment_date'].dt.strftime('%Y-%m-%d')
            students_df['attendance_rate'] = students_df['attendance_rate'].round(1).astype(str) + '%'
            
            display_columns = ['student_id', 'name', 'school', 'grade', 'route_id', 
                             'pickup_location', 'pickup_time', 'transport_eligibility', 
                             'special_needs', 'attendance_rate']
            
            if ADMIN_LIBS_AVAILABLE:
                # Use AgGrid for better interaction
                gb = GridOptionsBuilder.from_dataframe(students_df[display_columns])
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_side_bar()
                gb.configure_selection('multiple', use_checkbox=True)
                gb.configure_default_column(filterable=True, sortable=True, editable=False)
                
                grid_options = gb.build()
                
                grid_response = AgGrid(
                    students_df[display_columns],
                    gridOptions=grid_options,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    fit_columns_on_grid_load=False,
                    height=400
                )
                
                # Bulk operations
                if grid_response['selected_rows']:
                    selected_students = grid_response['selected_rows']
                    AdminUIComponents._render_bulk_operations(selected_students)
            
            else:
                # Fallback display
                st.dataframe(students_df[display_columns], use_container_width=True)
        
        # Add new student
        with st.expander("➕ Add New Student"):
            AdminUIComponents._render_add_student_form()
    
    @staticmethod
    def _render_bulk_operations(selected_students: List[Dict]):
        """Render bulk operations for selected students."""
        st.subheader(f"Bulk Operations ({len(selected_students)} students selected)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_route = st.selectbox("Assign to Route", ["", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"])
            if st.button("Assign Route") and new_route:
                st.success(f"Assigned {len(selected_students)} students to {new_route}")
        
        with col2:
            new_pickup_time = st.time_input("Set Pickup Time")
            if st.button("Update Pickup Time"):
                st.success(f"Updated pickup time for {len(selected_students)} students")
        
        with col3:
            eligibility_update = st.selectbox("Update Eligibility", ["", "Eligible", "Conditional", "Not Eligible"])
            if st.button("Update Eligibility") and eligibility_update:
                st.success(f"Updated eligibility for {len(selected_students)} students")
        
        with col4:
            if st.button("Generate Report"):
                AdminUIComponents._generate_student_report(selected_students)
    
    @staticmethod
    def _render_add_student_form():
        """Render add new student form."""
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Student Name *")
                school = st.selectbox("School *", ["Canberra Primary", "Capital Hill School", "Turner Primary"])
                grade = st.selectbox("Grade *", list(range(1, 7)))
                home_address = st.text_area("Home Address *")
            
            with col2:
                route_id = st.selectbox("Assign Route", ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"])
                pickup_location = st.text_input("Pickup Location")
                special_needs = st.text_input("Special Needs (if any)")
                parent_contact = st.text_input("Parent Email *")
            
            submitted = st.form_submit_button("Add Student")
            
            if submitted:
                if name and school and parent_contact:
                    st.success(f"Student {name} added successfully!")
                else:
                    st.error("Please fill in all required fields (*)")
    
    @staticmethod
    def _generate_student_report(students: List[Dict]):
        """Generate student report."""
        students_df = pd.DataFrame(students)
        csv = students_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Student Report",
            data=csv,
            file_name=f"student_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def render_driver_management():
        """Render driver management interface."""
        st.subheader("🚶 Driver Management")
        
        driver_data = AdminDataService.get_driver_management_data()
        drivers = driver_data['drivers']
        
        # Driver statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Drivers", driver_data['total_drivers'])
        
        with col2:
            st.metric("Active Drivers", driver_data['active_drivers'])
        
        with col3:
            st.metric("Training Due", driver_data['training_due_soon'])
        
        with col4:
            st.metric("Avg Performance", f"{driver_data['avg_performance']:.1f}/10")
        
        # Driver status and performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = pd.Series([d['status'] for d in drivers]).value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Driver Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance distribution
            performance_data = [d['performance_rating'] for d in drivers]
            fig = px.histogram(
                x=performance_data,
                nbins=20,
                title="Driver Performance Distribution",
                labels={'x': 'Performance Rating', 'y': 'Number of Drivers'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Driver management table
        st.subheader("Driver Details")
        
        drivers_df = pd.DataFrame(drivers)
        
        # Format dates and times
        drivers_df['license_expiry'] = drivers_df['license_expiry'].dt.strftime('%Y-%m-%d')
        drivers_df['employment_date'] = drivers_df['employment_date'].dt.strftime('%Y-%m-%d')
        drivers_df['training_expires'] = drivers_df['training_expires'].dt.strftime('%Y-%m-%d')
        drivers_df['hours_this_week'] = drivers_df['hours_this_week'].round(1)
        drivers_df['performance_rating'] = drivers_df['performance_rating'].round(1)
        
        display_columns = ['driver_id', 'name', 'status', 'current_route', 'license_class',
                          'license_expiry', 'performance_rating', 'hours_this_week', 
                          'training_expires', 'incidents_count']
        
        if ADMIN_LIBS_AVAILABLE:
            gb = GridOptionsBuilder.from_dataframe(drivers_df[display_columns])
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_selection('single', use_checkbox=True)
            gb.configure_default_column(filterable=True, sortable=True, editable=False)
            
            # Status color coding
            gb.configure_column(
                'status',
                cellStyle={
                    'styleConditions': [
                        {'condition': "params.value == 'Active'", 'style': {'backgroundColor': '#d4edda', 'color': '#155724'}},
                        {'condition': "params.value == 'On Leave'", 'style': {'backgroundColor': '#fff3cd', 'color': '#856404'}},
                        {'condition': "params.value == 'Training'", 'style': {'backgroundColor': '#cce5ff', 'color': '#004085'}}
                    ]
                }
            )
            
            grid_options = gb.build()
            
            grid_response = AgGrid(
                drivers_df[display_columns],
                gridOptions=grid_options,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                height=400
            )
            
            # Driver details for selected driver
            if grid_response['selected_rows']:
                selected_driver = grid_response['selected_rows'][0]
                AdminUIComponents._render_driver_details(selected_driver['driver_id'], drivers)
        
        else:
            st.dataframe(drivers_df[display_columns], use_container_width=True)
        
        # Driver scheduling
        AdminUIComponents._render_driver_scheduling()
    
    @staticmethod
    def _render_driver_details(driver_id: str, drivers: List[Dict]):
        """Render detailed driver information."""
        driver = next((d for d in drivers if d['driver_id'] == driver_id), None)
        if not driver:
            return
        
        st.subheader(f"Driver Details: {driver['name']} ({driver_id})")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Personal Information**")
            st.write(f"Name: {driver['name']}")
            st.write(f"Employee ID: {driver['driver_id']}")
            st.write(f"Employment Date: {driver['employment_date'].strftime('%Y-%m-%d')}")
            st.write(f"Salary Grade: {driver['salary_grade']}")
            st.write(f"Contact: {driver['contact_phone']}")
        
        with col2:
            st.write("**License & Certification**")
            st.write(f"License Class: {driver['license_class']}")
            st.write(f"License Expires: {driver['license_expiry'].strftime('%Y-%m-%d')}")
            st.write(f"Training Expires: {driver['training_expires'].strftime('%Y-%m-%d')}")
            st.write(f"Medical Clearance: {driver['medical_clearance'].strftime('%Y-%m-%d')}")
        
        with col3:
            st.write("**Current Assignment**")
            st.write(f"Status: {driver['status']}")
            st.write(f"Route: {driver.get('current_route', 'Not assigned')}")
            st.write(f"Vehicle: {driver.get('current_vehicle', 'Not assigned')}")
            st.write(f"Hours This Week: {driver['hours_this_week']:.1f}")
            
            # Performance indicators
            performance = driver['performance_rating']
            if performance >= 9:
                st.success(f"⭐ Excellent Performance: {performance:.1f}/10")
            elif performance >= 7:
                st.info(f"✅ Good Performance: {performance:.1f}/10")
            else:
                st.warning(f"⚠️ Needs Improvement: {performance:.1f}/10")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Update Assignment"):
                st.info("Assignment update dialog would open")
        
        with col2:
            if st.button("Schedule Training"):
                st.success("Training scheduled")
        
        with col3:
            if st.button("Performance Review"):
                st.info("Performance review interface would open")
        
        with col4:
            if st.button("Contact Driver"):
                st.info(f"Calling {driver['contact_phone']}")
    
    @staticmethod
    def _render_driver_scheduling():
        """Render driver scheduling interface."""
        st.subheader("📅 Driver Scheduling")
        
        # Weekly schedule view
        schedule_data = []
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        routes = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
        
        for day in days:
            for route in routes:
                schedule_data.append({
                    'day': day,
                    'route': route,
                    'driver': f'DRV-{np.random.randint(1, 35):03d}',
                    'shift_start': '07:00',
                    'shift_end': '16:00',
                    'status': np.random.choice(['Scheduled', 'Confirmed', 'Substitute'], p=[0.7, 0.25, 0.05])
                })
        
        schedule_df = pd.DataFrame(schedule_data)
        
        # Pivot table for better visualization
        schedule_pivot = schedule_df.pivot_table(
            index='route',
            columns='day',
            values='driver',
            aggfunc='first',
            fill_value=''
        )
        
        st.write("**Weekly Driver Schedule**")
        st.dataframe(schedule_pivot, use_container_width=True)
        
        # Schedule management
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Quick Actions**")
            if st.button("Generate Next Week's Schedule"):
                st.success("Schedule generated for next week")
            
            if st.button("Find Coverage for Absences"):
                st.info("Coverage assignments would be calculated")
            
            if st.button("Export Schedule"):
                csv = schedule_df.to_csv(index=False)
                st.download_button(
                    "📄 Download Schedule",
                    data=csv,
                    file_name=f"driver_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.write("**Schedule Statistics**")
            substitute_count = len(schedule_df[schedule_df['status'] == 'Substitute'])
            st.metric("Substitute Assignments", substitute_count)
            
            confirmed_rate = len(schedule_df[schedule_df['status'] == 'Confirmed']) / len(schedule_df) * 100
            st.metric("Confirmation Rate", f"{confirmed_rate:.1f}%")
    
    @staticmethod
    def render_performance_analytics():
        """Render performance analytics dashboard."""
        st.subheader("📊 Performance Analytics")
        
        performance_data = AdminDataService.get_performance_analytics()
        kpis = performance_data['kpis']
        historical_data = performance_data['historical_data']
        
        # KPI dashboard
        st.write("**Key Performance Indicators**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            kpi = kpis['on_time_performance']
            st.metric(
                "On-Time Performance",
                f"{kpi['current']:.1f}%",
                delta=f"{kpi['change']:.1f}%",
                delta_color="normal"
            )
            st.progress(min(kpi['current'] / 100, 1.0))
        
        with col2:
            kpi = kpis['fuel_efficiency']
            st.metric(
                "Fuel Efficiency",
                f"{kpi['current']:.1f} L/100km",
                delta=f"{kpi['change']:.1f}",
                delta_color="inverse"  # Lower is better
            )
            st.progress(min(kpi['current'] / 15, 1.0))
        
        with col3:
            kpi = kpis['student_satisfaction']
            st.metric(
                "Student Satisfaction",
                f"{kpi['current']:.1f}/5.0",
                delta=f"{kpi['change']:.1f}",
                delta_color="normal"
            )
            st.progress(kpi['current'] / 5)
        
        with col4:
            kpi = kpis['fleet_utilization']
            st.metric(
                "Fleet Utilization",
                f"{kpi['current']:.1f}%",
                delta=f"{kpi['change']:.1f}%",
                delta_color="normal"
            )
            st.progress(kpi['current'] / 100)
        
        # Historical performance charts
        st.write("**Performance Trends (30 Days)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-metric time series
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['on_time_performance'],
                mode='lines+markers',
                name='On-Time Performance (%)',
                yaxis='y1'
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['student_satisfaction'] * 20,  # Scale to 0-100
                mode='lines+markers',
                name='Student Satisfaction (scaled)',
                yaxis='y1'
            ))
            
            fig.update_layout(
                title="Performance Metrics Over Time",
                xaxis_title="Date",
                yaxis=dict(title="Percentage", side="left"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost analysis
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=historical_data['date'],
                y=historical_data['operational_costs'],
                name='Operational Costs',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=historical_data['date'],
                y=historical_data['revenue'],
                name='Revenue',
                marker_color='green'
            ))
            
            fig.update_layout(
                title="Daily Financial Performance",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Financial summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Revenue (30d)",
                f"${performance_data['total_revenue']:,.0f}"
            )
        
        with col2:
            st.metric(
                "Operating Costs (30d)",
                f"${performance_data['total_operational_cost']:,.0f}"
            )
        
        with col3:
            profit_margin = performance_data['profit_margin']
            st.metric(
                "Profit Margin",
                f"{profit_margin:.1f}%",
                delta=f"{'Profitable' if profit_margin > 0 else 'Loss'}"
            )
        
        # Detailed analytics
        with st.expander("📈 Detailed Analytics"):
            AdminUIComponents._render_detailed_analytics(historical_data)
    
    @staticmethod
    def _render_detailed_analytics(historical_data: pd.DataFrame):
        """Render detailed analytics views."""
        st.subheader("Correlation Analysis")
        
        # Calculate correlations
        numeric_columns = ['on_time_performance', 'fuel_efficiency', 'student_satisfaction', 
                          'fleet_utilization', 'safety_incidents', 'maintenance_costs']
        
        correlation_matrix = historical_data[numeric_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Performance Metrics Correlation Matrix",
            color_continuous_scale="RdYlBu",
            aspect="auto"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = [
            "Strong positive correlation between on-time performance and student satisfaction",
            "Fuel efficiency improves with better fleet utilization",
            "Maintenance costs spike after safety incidents",
            "Weather patterns significantly impact operational metrics"
        ]
        
        for insight in insights:
            st.info(f"💡 {insight}")
    
    @staticmethod
    def render_safety_monitoring():
        """Render safety monitoring and incident reporting."""
        st.subheader("🛡️ Safety Monitoring & Incident Reporting")
        
        incidents = AdminDataService.get_safety_incidents()
        
        # Safety overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_incidents = len(incidents)
            st.metric("Total Incidents", total_incidents)
        
        with col2:
            open_incidents = len([i for i in incidents if i['status'] in ['Open', 'Under Investigation']])
            st.metric("Open Incidents", open_incidents)
        
        with col3:
            critical_incidents = len([i for i in incidents if i['severity'] == AlertPriority.CRITICAL])
            st.metric("Critical Incidents", critical_incidents)
        
        with col4:
            avg_cost = np.mean([i['cost_impact'] for i in incidents if i['cost_impact'] > 0])
            st.metric("Avg Cost Impact", f"${avg_cost:.0f}" if avg_cost > 0 else "$0")
        
        # Incident trends
        col1, col2 = st.columns(2)
        
        with col1:
            # Incidents by type
            incident_types = pd.Series([i['type'] for i in incidents]).value_counts()
            fig = px.bar(
                x=incident_types.index,
                y=incident_types.values,
                title="Incidents by Type"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Incidents by severity
            severity_counts = pd.Series([i['severity'] for i in incidents]).value_counts()
            colors = {
                AlertPriority.LOW: 'green',
                AlertPriority.MEDIUM: 'orange', 
                AlertPriority.HIGH: 'red',
                AlertPriority.CRITICAL: 'darkred'
            }
            
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Incidents by Severity",
                color=severity_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Incident management table
        st.subheader("Incident Management")
        
        incidents_df = pd.DataFrame(incidents)
        incidents_df['date'] = incidents_df['date'].dt.strftime('%Y-%m-%d')
        incidents_df['cost_impact'] = incidents_df['cost_impact'].round(2)
        
        display_columns = ['incident_id', 'date', 'type', 'severity', 'vehicle_id', 
                          'driver_id', 'status', 'cost_impact', 'follow_up_required']
        
        if ADMIN_LIBS_AVAILABLE:
            gb = GridOptionsBuilder.from_dataframe(incidents_df[display_columns])
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_selection('single', use_checkbox=True)
            gb.configure_default_column(filterable=True, sortable=True, editable=False)
            
            # Severity color coding
            gb.configure_column(
                'severity',
                cellStyle={
                    'styleConditions': [
                        {'condition': "params.value == 'critical'", 'style': {'backgroundColor': '#721c24', 'color': 'white'}},
                        {'condition': "params.value == 'high'", 'style': {'backgroundColor': '#f8d7da', 'color': '#721c24'}},
                        {'condition': "params.value == 'medium'", 'style': {'backgroundColor': '#fff3cd', 'color': '#856404'}},
                        {'condition': "params.value == 'low'", 'style': {'backgroundColor': '#d4edda', 'color': '#155724'}}
                    ]
                }
            )
            
            grid_options = gb.build()
            
            grid_response = AgGrid(
                incidents_df[display_columns],
                gridOptions=grid_options,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                height=400
            )
            
            # Incident details
            if grid_response['selected_rows']:
                selected_incident = grid_response['selected_rows'][0]
                AdminUIComponents._render_incident_details(selected_incident['incident_id'], incidents)
        
        else:
            st.dataframe(incidents_df[display_columns], use_container_width=True)
        
        # New incident reporting
        with st.expander("➕ Report New Incident"):
            AdminUIComponents._render_incident_form()
    
    @staticmethod
    def _render_incident_details(incident_id: str, incidents: List[Dict]):
        """Render detailed incident information."""
        incident = next((i for i in incidents if i['incident_id'] == incident_id), None)
        if not incident:
            return
        
        st.subheader(f"Incident Details: {incident_id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Incident Information**")
            st.write(f"Date: {incident['date'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"Type: {incident['type']}")
            st.write(f"Severity: {incident['severity'].title()}")
            st.write(f"Location: {incident['location']}")
            st.write(f"Status: {incident['status']}")
            
            # Severity indicator
            severity_colors = {
                AlertPriority.LOW: 'success',
                AlertPriority.MEDIUM: 'warning',
                AlertPriority.HIGH: 'error',
                AlertPriority.CRITICAL: 'error'
            }
            
            severity_color = severity_colors.get(incident['severity'], 'info')
            if severity_color == 'success':
                st.success(f"Severity: {incident['severity'].title()}")
            elif severity_color == 'warning':
                st.warning(f"Severity: {incident['severity'].title()}")
            elif severity_color == 'error':
                st.error(f"Severity: {incident['severity'].title()}")
        
        with col2:
            st.write("**Involved Parties**")
            st.write(f"Vehicle: {incident['vehicle_id']}")
            st.write(f"Driver: {incident['driver_id']}")
            st.write(f"Route: {incident['route_id']}")
            st.write(f"Investigation Officer: {incident['investigation_officer']}")
            
            if incident['cost_impact'] > 0:
                st.write(f"**Cost Impact:** ${incident['cost_impact']:.2f}")
            
            if incident['follow_up_required']:
                st.error("⚠️ Follow-up action required")
        
        st.write("**Description**")
        st.write(incident['description'])
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Update Status"):
                st.info("Status update dialog would open")
        
        with col2:
            if st.button("Add Note"):
                st.info("Note addition interface would open")
        
        with col3:
            if st.button("Generate Report"):
                st.success("Incident report generated")
        
        with col4:
            if st.button("Close Incident"):
                st.success("Incident closed")
    
    @staticmethod
    def _render_incident_form():
        """Render new incident reporting form."""
        with st.form("incident_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                incident_type = st.selectbox(
                    "Incident Type *",
                    ["Minor Collision", "Student Injury", "Vehicle Breakdown", 
                     "Traffic Violation", "Weather Related", "Vandalism", "Other"]
                )
                
                severity = st.selectbox(
                    "Severity *",
                    [AlertPriority.LOW, AlertPriority.MEDIUM, AlertPriority.HIGH, AlertPriority.CRITICAL]
                )
                
                vehicle_id = st.text_input("Vehicle ID")
                driver_id = st.text_input("Driver ID")
            
            with col2:
                route_id = st.text_input("Route ID")
                location = st.text_input("Location *")
                incident_date = st.datetime_input("Date & Time *", datetime.now())
                cost_impact = st.number_input("Estimated Cost Impact ($)", min_value=0.0)
            
            description = st.text_area("Description *", height=100)
            
            submitted = st.form_submit_button("Submit Incident Report")
            
            if submitted:
                if incident_type and severity and location and description:
                    incident_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    st.success(f"Incident report {incident_id} submitted successfully!")
                else:
                    st.error("Please fill in all required fields (*)")


def render_admin_dashboard():
    """Main function to render the administrator dashboard."""
    try:
        # Page header
        st.title("🏢 Administrator Dashboard")
        st.markdown("---")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "🚌 Fleet",
            "🗺️ Routes", 
            "👥 Students",
            "🚶 Drivers",
            "🛡️ Safety"
        ])
        
        with tab1:
            # Executive overview
            AdminUIComponents.render_executive_summary()
            st.markdown("---")
            AdminUIComponents.render_performance_analytics()
        
        with tab2:
            # Fleet management
            AdminUIComponents.render_fleet_management()
        
        with tab3:
            # Route optimization
            AdminUIComponents.render_route_optimization()
        
        with tab4:
            # Student assignment
            AdminUIComponents.render_student_assignment()
        
        with tab5:
            # Driver management
            AdminUIComponents.render_driver_management()
        
        with tab6:
            # Safety monitoring
            AdminUIComponents.render_safety_monitoring()
        
        # Auto-refresh functionality
        if st.sidebar.checkbox("Auto-refresh data", value=True):
            refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)
            
            if st.sidebar.button("🔄 Refresh Now"):
                st.cache_data.clear()
                st.rerun()
        
        # Export functionality
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Reports & Export")
        
        if st.sidebar.button("Generate Daily Report"):
            st.sidebar.success("Daily report generated!")
        
        if st.sidebar.button("Export All Data"):
            st.sidebar.info("Data export would be initiated")
        
        if st.sidebar.button("Schedule Weekly Report"):
            st.sidebar.success("Weekly report scheduled!")
    
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logger.error(f"Admin dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    render_admin_dashboard()