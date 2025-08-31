"""
SmartSchoolGo Demo Application
Simplified version to showcase key features and functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Configure page
st.set_page_config(
    page_title="SmartSchoolGo Demo",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Demo data generation functions
@st.cache_data
def generate_demo_data():
    """Generate demo data for the application"""
    # ACT region coordinates (Canberra)
    center_lat, center_lon = -35.2809, 149.1300
    
    # Generate schools
    schools = []
    school_names = [
        "Canberra High School", "Belconnen High School", "Dickson College",
        "Hawker College", "Lake Ginninderra College", "Tuggeranong College",
        "Erindale College", "Narrabundah College", "Campbell High School",
        "Karabar High School"
    ]
    
    for i, name in enumerate(school_names):
        schools.append({
            'school_id': f'SCH_{i:03d}',
            'name': name,
            'lat': center_lat + random.uniform(-0.2, 0.2),
            'lon': center_lon + random.uniform(-0.3, 0.3),
            'students': random.randint(200, 800),
            'buses': random.randint(3, 12)
        })
    
    # Generate bus routes
    routes = []
    route_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    for i in range(15):
        school = random.choice(schools)
        route_points = []
        
        # Generate route points
        for j in range(random.randint(5, 10)):
            route_points.append([
                school['lat'] + random.uniform(-0.1, 0.1),
                school['lon'] + random.uniform(-0.1, 0.1)
            ])
        
        routes.append({
            'route_id': f'RT_{i:03d}',
            'school_id': school['school_id'],
            'school_name': school['name'],
            'color': route_colors[i % len(route_colors)],
            'points': route_points,
            'students': random.randint(20, 60),
            'duration_mins': random.randint(25, 55),
            'distance_km': random.randint(8, 25),
            'status': random.choice(['Active', 'Active', 'Active', 'Delayed', 'Maintenance'])
        })
    
    # Generate student data
    students = []
    first_names = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia', 'Mason', 'Isabella', 'William']
    last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin']
    
    for i in range(500):
        school = random.choice(schools)
        route = random.choice([r for r in routes if r['school_id'] == school['school_id']] or routes[:1])
        
        students.append({
            'student_id': f'STU_{i:04d}',
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'school_id': school['school_id'],
            'school_name': school['name'],
            'route_id': route['route_id'],
            'grade': random.randint(7, 12),
            'pickup_time': f"{random.randint(6,8)}:{random.randint(0,59):02d}",
            'special_needs': random.choice([None, None, None, 'Wheelchair', 'Hearing Aid', 'Medical Alert'])
        })
    
    # Generate real-time tracking data
    tracking_data = []
    for route in routes[:8]:  # Only some routes active
        if route['status'] == 'Active':
            # Simulate bus moving along route
            progress = random.uniform(0.1, 0.9)
            point_idx = int(progress * (len(route['points']) - 1))
            current_point = route['points'][point_idx]
            
            tracking_data.append({
                'route_id': route['route_id'],
                'school_name': route['school_name'],
                'lat': current_point[0] + random.uniform(-0.01, 0.01),
                'lon': current_point[1] + random.uniform(-0.01, 0.01),
                'speed_kmh': random.randint(15, 45),
                'students_on_board': random.randint(10, route['students']),
                'next_stop': f"Stop {point_idx + 1}",
                'eta_mins': random.randint(5, 25),
                'status': random.choice(['On Time', 'On Time', 'Running Late'])
            })
    
    return schools, routes, students, tracking_data

def create_map(schools, routes, tracking_data, show_routes=True, show_tracking=True):
    """Create an interactive map with schools, routes, and real-time tracking"""
    # Center on Canberra
    m = folium.Map(
        location=[-35.2809, 149.1300],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add schools
    for school in schools:
        folium.Marker(
            [school['lat'], school['lon']],
            popup=f"""
            <b>{school['name']}</b><br>
            Students: {school['students']}<br>
            Buses: {school['buses']}
            """,
            tooltip=school['name'],
            icon=folium.Icon(color='red', icon='graduation-cap', prefix='fa')
        ).add_to(m)
    
    # Add routes
    if show_routes:
        for route in routes:
            if route['status'] == 'Active':
                folium.PolyLine(
                    route['points'],
                    color=route['color'],
                    weight=3,
                    opacity=0.7,
                    popup=f"""
                    <b>Route {route['route_id']}</b><br>
                    School: {route['school_name']}<br>
                    Students: {route['students']}<br>
                    Duration: {route['duration_mins']} mins<br>
                    Distance: {route['distance_km']} km
                    """
                ).add_to(m)
    
    # Add real-time tracking
    if show_tracking:
        for bus in tracking_data:
            status_color = 'green' if bus['status'] == 'On Time' else 'orange'
            folium.Marker(
                [bus['lat'], bus['lon']],
                popup=f"""
                <b>Bus - {bus['route_id']}</b><br>
                School: {bus['school_name']}<br>
                Speed: {bus['speed_kmh']} km/h<br>
                Students: {bus['students_on_board']}<br>
                Next Stop: {bus['next_stop']}<br>
                ETA: {bus['eta_mins']} mins<br>
                Status: {bus['status']}
                """,
                tooltip=f"Bus {bus['route_id']} - {bus['status']}",
                icon=folium.Icon(color=status_color, icon='bus', prefix='fa')
            ).add_to(m)
    
    return m

def render_parent_dashboard(students, tracking_data):
    """Render parent dashboard"""
    st.header("🏠 Parent Portal")
    
    # Select child
    child_names = [s['name'] for s in students[:20]]  # First 20 for demo
    selected_child = st.selectbox("Select your child:", child_names)
    
    if selected_child:
        child_data = next(s for s in students if s['name'] == selected_child)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("School", child_data['school_name'])
        with col2:
            st.metric("Route", child_data['route_id'])
        with col3:
            st.metric("Pickup Time", child_data['pickup_time'])
        
        # Find bus status
        route_tracking = next((t for t in tracking_data if t['route_id'] == child_data['route_id']), None)
        
        if route_tracking:
            st.success(f"🚌 Bus Status: {route_tracking['status']}")
            st.info(f"📍 Current Location: {route_tracking['next_stop']}")
            st.info(f"🕒 Estimated Arrival: {route_tracking['eta_mins']} minutes")
            
            # Progress bar
            progress = random.uniform(0.3, 0.8)
            st.progress(progress, f"Route Progress: {int(progress*100)}%")
        else:
            st.warning("Bus not currently tracked (Route may be completed or not started)")
        
        # Recent notifications
        st.subheader("📱 Recent Notifications")
        notifications = [
            "Bus departed school on time - 3:45 PM",
            "Approaching your stop - 4:12 PM",
            "Child safely boarded - 7:35 AM",
            "Route running 5 minutes late due to traffic - 7:40 AM"
        ]
        
        for notification in notifications:
            st.write(f"• {notification}")

def render_admin_dashboard(schools, routes, students, tracking_data):
    """Render administrator dashboard"""
    st.header("🏫 Administrator Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Schools", len(schools))
    with col2:
        st.metric("Active Routes", len([r for r in routes if r['status'] == 'Active']))
    with col3:
        st.metric("Students Transported", len(students))
    with col4:
        total_buses = sum(s['buses'] for s in schools)
        st.metric("Fleet Size", total_buses)
    
    # Fleet status
    st.subheader("🚌 Fleet Status")
    status_counts = {}
    for route in routes:
        status_counts[route['status']] = status_counts.get(route['status'], 0) + 1
    
    status_df = pd.DataFrame(list(status_counts.items()), columns=['Status', 'Count'])
    fig = px.bar(status_df, x='Status', y='Count', color='Status',
                title="Route Status Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Route Efficiency")
        route_data = pd.DataFrame([{
            'Route': r['route_id'],
            'Students': r['students'],
            'Duration': r['duration_mins'],
            'Distance': r['distance_km'],
            'Efficiency': r['students'] / r['duration_mins']
        } for r in routes])
        
        fig = px.scatter(route_data, x='Duration', y='Students', 
                        size='Distance', color='Efficiency',
                        title="Route Performance Analysis",
                        labels={'Duration': 'Duration (minutes)', 'Students': 'Students Transported'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎓 School Distribution")
        school_data = pd.DataFrame(schools)
        fig = px.bar(school_data, x='name', y='students',
                    title="Students per School")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Live tracking table
    st.subheader("🔴 Live Bus Tracking")
    if tracking_data:
        tracking_df = pd.DataFrame(tracking_data)
        st.dataframe(tracking_df[['route_id', 'school_name', 'speed_kmh', 'students_on_board', 'status', 'eta_mins']], 
                    use_container_width=True)
    else:
        st.info("No active buses currently tracked")

def render_planner_dashboard(schools, routes, students):
    """Render transport planner dashboard"""
    st.header("📋 Transport Planner")
    
    # Route optimization
    st.subheader("🎯 Route Optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Optimization parameters
        st.write("**Optimization Parameters:**")
        max_travel_time = st.slider("Max Travel Time (minutes)", 20, 60, 45)
        max_capacity = st.slider("Max Bus Capacity", 30, 80, 60)
        priority = st.selectbox("Priority", ["Minimize Travel Time", "Minimize Cost", "Maximize Safety"])
        
        if st.button("🚀 Run Optimization"):
            with st.spinner("Optimizing routes..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                st.success("✅ Route optimization completed!")
                
                # Show optimization results
                st.write("**Optimization Results:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Routes Optimized", "15", "↓ 3")
                with col2:
                    st.metric("Average Travel Time", "38 mins", "↓ 7 mins")
                with col3:
                    st.metric("Cost Savings", "$1,240/month", "↓ 15%")
    
    with col2:
        st.write("**Current Challenges:**")
        challenges = [
            "🔴 Route RT_003: High travel time",
            "🟡 Route RT_007: Low capacity utilization",
            "🔴 Route RT_011: Safety concerns reported",
            "🟡 School zone congestion at 3:30 PM",
            "🟢 Overall efficiency: Good"
        ]
        
        for challenge in challenges:
            st.write(challenge)
    
    # Demand forecasting
    st.subheader("📈 Demand Forecasting")
    
    # Generate forecast data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    demand = np.random.normal(450, 50, len(dates))
    demand = np.maximum(demand, 300)  # Minimum demand
    
    forecast_df = pd.DataFrame({
        'Date': dates,
        'Demand': demand,
        'Type': 'Historical'
    })
    
    # Add future forecast
    future_dates = pd.date_range(start='2025-01-01', end='2025-06-30', freq='M')
    future_demand = np.random.normal(480, 40, len(future_dates))
    future_forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Demand': future_demand,
        'Type': 'Forecast'
    })
    
    combined_df = pd.concat([forecast_df, future_forecast_df])
    
    fig = px.line(combined_df, x='Date', y='Demand', color='Type',
                 title="Student Transport Demand Forecast")
    st.plotly_chart(fig, use_container_width=True)
    
    # Network analysis
    st.subheader("🕸️ Network Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Network Statistics:**")
        st.metric("Total Network Coverage", "285 km²")
        st.metric("Average Route Length", "16.8 km")
        st.metric("Network Efficiency Score", "8.2/10")
        st.metric("Coverage Gaps", "3 areas identified")
    
    with col2:
        # Create a simple network visualization
        network_data = pd.DataFrame({
            'Metric': ['Connectivity', 'Efficiency', 'Reliability', 'Safety', 'Cost'],
            'Score': [8.5, 7.8, 9.1, 8.9, 7.2]
        })
        
        fig = px.bar(network_data, x='Metric', y='Score',
                    title="Network Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)

def render_real_time_demo():
    """Render real-time simulation demo"""
    st.header("🔴 Real-Time Demo Simulation")
    
    st.info("🎬 This demo simulates real-time bus tracking and updates every few seconds")
    
    # Create placeholder for live updates
    map_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (updates every 3 seconds)", value=True)
    
    if auto_refresh:
        # Simulate real-time updates
        if 'demo_iteration' not in st.session_state:
            st.session_state.demo_iteration = 0
        
        schools, routes, students, tracking_data = generate_demo_data()
        
        # Update tracking data to simulate movement
        for bus in tracking_data:
            bus['lat'] += random.uniform(-0.005, 0.005)
            bus['lon'] += random.uniform(-0.005, 0.005)
            bus['speed_kmh'] = random.randint(15, 45)
            bus['eta_mins'] = max(1, bus['eta_mins'] + random.randint(-2, 1))
        
        # Update map
        with map_placeholder.container():
            demo_map = create_map(schools, routes, tracking_data)
            st_folium(demo_map, width=1200, height=500)
        
        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active Buses", len(tracking_data))
            with col2:
                avg_speed = np.mean([b['speed_kmh'] for b in tracking_data])
                st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
            with col3:
                on_time = len([b for b in tracking_data if b['status'] == 'On Time'])
                st.metric("On Time", f"{on_time}/{len(tracking_data)}")
            with col4:
                total_students = sum(b['students_on_board'] for b in tracking_data)
                st.metric("Students Transported", total_students)
        
        if auto_refresh:
            time.sleep(3)
            st.rerun()

def main():
    """Main demo application"""
    
    # Sidebar navigation
    st.sidebar.title("🚌 SmartSchoolGo Demo")
    st.sidebar.write("Navigate between different user interfaces:")
    
    page = st.sidebar.selectbox(
        "Select Demo View:",
        [
            "🏠 Parent Portal", 
            "🏫 Admin Dashboard", 
            "📋 Transport Planner",
            "🗺️ Interactive Map",
            "🔴 Real-Time Demo"
        ]
    )
    
    # Generate demo data
    schools, routes, students, tracking_data = generate_demo_data()
    
    # Main content
    st.title("🚌 SmartSchoolGo - Smart School Transport System")
    st.markdown("### *Optimizing school transport through AI-powered analytics and real-time monitoring*")
    
    # Render selected page
    if page == "🏠 Parent Portal":
        render_parent_dashboard(students, tracking_data)
        
    elif page == "🏫 Admin Dashboard":
        render_admin_dashboard(schools, routes, students, tracking_data)
        
    elif page == "📋 Transport Planner":
        render_planner_dashboard(schools, routes, students)
        
    elif page == "🗺️ Interactive Map":
        st.header("🗺️ Interactive Transport Map")
        
        # Map controls
        col1, col2 = st.columns(2)
        with col1:
            show_routes = st.checkbox("Show Routes", value=True)
        with col2:
            show_tracking = st.checkbox("Show Live Tracking", value=True)
        
        # Create and display map
        demo_map = create_map(schools, routes, tracking_data, show_routes, show_tracking)
        st_folium(demo_map, width=1200, height=600)
        
        # Map legend
        st.subheader("Map Legend")
        st.write("🎓 **Red markers**: Schools")
        st.write("🚌 **Green markers**: Buses running on time")  
        st.write("🚌 **Orange markers**: Buses running late")
        st.write("**Colored lines**: Bus routes")
        
    elif page == "🔴 Real-Time Demo":
        render_real_time_demo()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Demo Statistics")
    st.sidebar.write(f"Schools: {len(schools)}")
    st.sidebar.write(f"Routes: {len(routes)}")
    st.sidebar.write(f"Students: {len(students)}")
    st.sidebar.write(f"Active Buses: {len(tracking_data)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built for GovHack 2025**")
    st.sidebar.markdown("*Demonstrating AI-powered transport optimization*")

if __name__ == "__main__":
    main()