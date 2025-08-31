"""
SmartSchoolGo Main Streamlit Application

This is the main Streamlit application providing a multi-page architecture with
role-based navigation for parents, administrators, and planners. The application
includes real-time data updates, authentication, responsive design, and comprehensive
transport management functionality.

Features:
- Multi-page architecture with role-based access
- Real-time data updates using WebSocket connections
- User authentication and session management
- Responsive design with mobile-friendly layouts
- Interactive maps and visualizations
- Performance optimization with caching
- Professional styling and user experience

Author: SmartSchoolGo Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import folium
    from streamlit_folium import st_folium
    import websocket
    import requests
    STREAMLIT_LIBS_AVAILABLE = True
except ImportError as e:
    st.error(f"Required libraries not available: {e}")
    STREAMLIT_LIBS_AVAILABLE = False

# Configure page settings
st.set_page_config(
    page_title="SmartSchoolGo",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/SmartSchoolGo/help',
        'Report a bug': 'https://github.com/SmartSchoolGo/issues',
        'About': "SmartSchoolGo - Intelligent School Transport Management"
    }
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole:
    """User role definitions."""
    PARENT = "parent"
    ADMIN = "admin"
    PLANNER = "planner"
    DRIVER = "driver"
    GUEST = "guest"


class SessionManager:
    """Manage user sessions and authentication."""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        
        if 'user_role' not in st.session_state:
            st.session_state.user_role = UserRole.GUEST
        
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'login_time' not in st.session_state:
            st.session_state.login_time = None
        
        if 'websocket_connection' not in st.session_state:
            st.session_state.websocket_connection = None
        
        if 'real_time_data' not in st.session_state:
            st.session_state.real_time_data = {}
        
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[bool, str, str]:
        """Authenticate user credentials."""
        # Simplified authentication - replace with proper authentication
        users_db = {
            "parent@demo.com": {"password": "parent123", "role": UserRole.PARENT, "name": "John Smith"},
            "admin@demo.com": {"password": "admin123", "role": UserRole.ADMIN, "name": "Jane Wilson"},
            "planner@demo.com": {"password": "planner123", "role": UserRole.PLANNER, "name": "Mike Johnson"},
            "driver@demo.com": {"password": "driver123", "role": UserRole.DRIVER, "name": "Sarah Davis"}
        }
        
        user = users_db.get(username)
        if user and user["password"] == password:
            return True, user["role"], user["name"]
        
        return False, UserRole.GUEST, ""
    
    @staticmethod
    def login(username: str, password: str) -> bool:
        """Log in user."""
        success, role, name = SessionManager.authenticate_user(username, password)
        
        if success:
            st.session_state.authenticated = True
            st.session_state.user_id = username
            st.session_state.user_role = role
            st.session_state.user_name = name
            st.session_state.login_time = datetime.now()
            
            # Initialize real-time connection
            RealtimeManager.initialize_connection()
            
            st.success(f"Welcome, {name}!")
            return True
        else:
            st.error("Invalid credentials. Please try again.")
            return False
    
    @staticmethod
    def logout():
        """Log out current user."""
        # Cleanup real-time connection
        RealtimeManager.disconnect()
        
        # Reset session state
        for key in list(st.session_state.keys()):
            if key not in ['session_id']:  # Keep session_id for tracking
                del st.session_state[key]
        
        SessionManager.initialize_session()
        st.success("Logged out successfully.")
        st.rerun()
    
    @staticmethod
    def require_authentication():
        """Require user to be authenticated."""
        if not st.session_state.authenticated:
            st.warning("Please log in to access this page.")
            return False
        return True
    
    @staticmethod
    def require_role(required_roles: List[str]):
        """Require user to have specific role."""
        if not SessionManager.require_authentication():
            return False
        
        if st.session_state.user_role not in required_roles:
            st.error("You don't have permission to access this page.")
            return False
        
        return True


class RealtimeManager:
    """Manage real-time data connections and updates."""
    
    @staticmethod
    def initialize_connection():
        """Initialize WebSocket connection for real-time updates."""
        try:
            # In production, this would connect to actual WebSocket server
            st.session_state.websocket_connection = {
                "status": "connected",
                "last_update": datetime.now(),
                "server_url": "ws://localhost:8765"
            }
            
            # Start background data fetching
            RealtimeManager.fetch_initial_data()
            
        except Exception as e:
            logger.error(f"Failed to initialize real-time connection: {e}")
            st.session_state.websocket_connection = {"status": "disconnected"}
    
    @staticmethod
    def disconnect():
        """Disconnect from real-time services."""
        if st.session_state.websocket_connection:
            st.session_state.websocket_connection["status"] = "disconnected"
    
    @staticmethod
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def fetch_initial_data():
        """Fetch initial data for the application."""
        try:
            # Mock real-time data - replace with actual API calls
            current_time = datetime.now()
            
            # Vehicle positions
            vehicles = []
            for i in range(10):
                vehicles.append({
                    "vehicle_id": f"BUS-{i+1:03d}",
                    "route_id": f"R{(i % 5) + 1}",
                    "latitude": -35.28 + (i * 0.002),
                    "longitude": 149.13 + (i * 0.003),
                    "bearing": (i * 36) % 360,
                    "speed": 25 + (i % 10),
                    "occupancy": min(50, 5 + (i * 3)),
                    "status": "In Transit",
                    "last_update": current_time - timedelta(seconds=i*10)
                })
            
            # Routes
            routes = []
            for i in range(5):
                routes.append({
                    "route_id": f"R{i+1}",
                    "route_name": f"Route {i+1} - {['North', 'South', 'East', 'West', 'Central'][i]}",
                    "active_vehicles": len([v for v in vehicles if v["route_id"] == f"R{i+1}"]),
                    "avg_delay": (i * 2) % 8,
                    "total_students": 35 + (i * 8),
                    "status": "Active"
                })
            
            # Alerts
            alerts = [
                {
                    "alert_id": "ALERT-001",
                    "severity": "High",
                    "type": "Delay",
                    "message": "Route R1 experiencing 15-minute delays due to traffic",
                    "affected_routes": ["R1"],
                    "created_at": current_time - timedelta(minutes=45),
                    "status": "Active"
                },
                {
                    "alert_id": "ALERT-002", 
                    "severity": "Medium",
                    "type": "Maintenance",
                    "message": "Bus BUS-003 scheduled for maintenance after current route",
                    "affected_routes": ["R2"],
                    "created_at": current_time - timedelta(minutes=120),
                    "status": "Scheduled"
                }
            ]
            
            # Performance metrics
            metrics = {
                "on_time_performance": 87.5,
                "average_delay": 3.2,
                "fleet_utilization": 92.1,
                "student_satisfaction": 4.2,
                "fuel_efficiency": 8.5,
                "safety_incidents": 0,
                "total_routes": len(routes),
                "active_vehicles": len([v for v in vehicles if v["status"] == "In Transit"])
            }
            
            # Update session state
            st.session_state.real_time_data = {
                "vehicles": vehicles,
                "routes": routes,
                "alerts": alerts,
                "metrics": metrics,
                "last_update": current_time
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch initial data: {e}")
            return False
    
    @staticmethod
    def get_vehicle_updates() -> List[Dict]:
        """Get latest vehicle position updates."""
        return st.session_state.real_time_data.get("vehicles", [])
    
    @staticmethod
    def get_active_alerts() -> List[Dict]:
        """Get active system alerts."""
        alerts = st.session_state.real_time_data.get("alerts", [])
        return [a for a in alerts if a["status"] == "Active"]
    
    @staticmethod
    def add_notification(message: str, type: str = "info"):
        """Add notification to user's notification list."""
        notification = {
            "id": str(uuid.uuid4()),
            "message": message,
            "type": type,
            "timestamp": datetime.now(),
            "read": False
        }
        
        st.session_state.notifications.append(notification)
        
        # Keep only last 50 notifications
        if len(st.session_state.notifications) > 50:
            st.session_state.notifications = st.session_state.notifications[-50:]


class UIComponents:
    """Reusable UI components and widgets."""
    
    @staticmethod
    def render_header():
        """Render application header."""
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.image("https://via.placeholder.com/120x60/1f77b4/white?text=SmartSchoolGo", width=120)
        
        with col2:
            st.markdown(
                "<h1 style='text-align: center; color: #1f77b4; margin: 0;'>SmartSchoolGo</h1>",
                unsafe_allow_html=True
            )
        
        with col3:
            if st.session_state.authenticated:
                st.markdown(f"**Welcome, {st.session_state.user_name}**")
                st.caption(f"Role: {st.session_state.user_role.title()}")
    
    @staticmethod
    def render_navigation():
        """Render navigation menu based on user role."""
        if not st.session_state.authenticated:
            return "login"
        
        role = st.session_state.user_role
        
        # Define menu options based on role
        if role == UserRole.PARENT:
            menu_options = ["Dashboard", "Track Bus", "Route Info", "Notifications", "Profile"]
            icons = ["house", "geo-alt", "map", "bell", "person"]
        elif role == UserRole.ADMIN:
            menu_options = ["Dashboard", "Fleet Management", "Routes", "Analytics", "Alerts", "Settings"]
            icons = ["speedometer2", "truck", "diagram-3", "graph-up", "exclamation-triangle", "gear"]
        elif role == UserRole.PLANNER:
            menu_options = ["Dashboard", "Route Planning", "Optimization", "Performance", "Reports"]
            icons = ["clipboard-data", "map", "cpu", "bar-chart", "file-earmark-text"]
        elif role == UserRole.DRIVER:
            menu_options = ["Dashboard", "My Route", "Schedule", "Maintenance", "Messages"]
            icons = ["speedometer", "geo", "calendar", "tools", "chat"]
        else:
            menu_options = ["Dashboard"]
            icons = ["house"]
        
        # Create navigation menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Navigation",
                options=menu_options,
                icons=icons,
                menu_icon="list",
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "5px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px"},
                    "nav-link-selected": {"background-color": "#1f77b4"}
                }
            )
            
            # Add logout button
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                SessionManager.logout()
        
        return selected.lower().replace(" ", "_")
    
    @staticmethod
    def render_realtime_status():
        """Render real-time connection status."""
        connection = st.session_state.websocket_connection
        
        if connection and connection.get("status") == "connected":
            st.success("🟢 Real-time updates active")
            last_update = st.session_state.real_time_data.get("last_update")
            if last_update:
                time_diff = datetime.now() - last_update
                st.caption(f"Last update: {int(time_diff.total_seconds())}s ago")
        else:
            st.warning("🟡 Real-time updates unavailable")
    
    @staticmethod
    def render_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
        """Render a metric card."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
    
    @staticmethod
    def render_alert_banner():
        """Render active alerts banner."""
        alerts = RealtimeManager.get_active_alerts()
        
        if alerts:
            high_priority = [a for a in alerts if a["severity"] in ["High", "Critical"]]
            
            if high_priority:
                for alert in high_priority:
                    if alert["severity"] == "Critical":
                        st.error(f"🚨 {alert['message']}")
                    else:
                        st.warning(f"⚠️ {alert['message']}")
    
    @staticmethod
    def render_notifications_panel():
        """Render notifications panel."""
        notifications = st.session_state.notifications
        unread = [n for n in notifications if not n["read"]]
        
        if unread:
            with st.expander(f"🔔 Notifications ({len(unread)} unread)", expanded=False):
                for notification in unread[-5:]:  # Show last 5
                    icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(
                        notification["type"], "ℹ️"
                    )
                    
                    st.markdown(f"{icon} {notification['message']}")
                    st.caption(f"{notification['timestamp'].strftime('%H:%M:%S')}")
                    st.markdown("---")
    
    @staticmethod
    def render_vehicle_map(vehicles: List[Dict], height: int = 400):
        """Render interactive map with vehicle positions."""
        try:
            # Create base map centered on Canberra
            center_lat, center_lon = -35.2809, 149.1300
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="OpenStreetMap"
            )
            
            # Add vehicle markers
            colors = {"R1": "red", "R2": "blue", "R3": "green", "R4": "orange", "R5": "purple"}
            
            for vehicle in vehicles:
                lat, lon = vehicle["latitude"], vehicle["longitude"]
                route_id = vehicle["route_id"]
                
                # Create popup content
                popup_content = f"""
                <div style='width: 200px;'>
                    <b>{vehicle['vehicle_id']}</b><br>
                    Route: {route_id}<br>
                    Speed: {vehicle['speed']} km/h<br>
                    Occupancy: {vehicle['occupancy']}/50<br>
                    Status: {vehicle['status']}<br>
                    Updated: {vehicle['last_update'].strftime('%H:%M:%S')}
                </div>
                """
                
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(popup_content, max_width=200),
                    icon=folium.Icon(
                        color=colors.get(route_id, "gray"),
                        icon="bus",
                        prefix="fa"
                    ),
                    tooltip=f"{vehicle['vehicle_id']} - {route_id}"
                ).add_to(m)
            
            # Display map
            map_data = st_folium(m, width=700, height=height)
            
            return map_data
            
        except Exception as e:
            st.error(f"Map rendering failed: {e}")
            return None
    
    @staticmethod
    def render_performance_chart(metrics: Dict[str, float]):
        """Render performance metrics chart."""
        try:
            # Create gauge charts for key metrics
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=("On-Time Performance", "Fleet Utilization", 
                              "Student Satisfaction", "Fuel Efficiency")
            )
            
            # On-time performance
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get("on_time_performance", 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "On-Time %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ), row=1, col=1)
            
            # Fleet utilization
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get("fleet_utilization", 0),
                title={'text': "Utilization %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"}}
            ), row=1, col=2)
            
            # Student satisfaction
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get("student_satisfaction", 0),
                title={'text': "Satisfaction"},
                gauge={'axis': {'range': [0, 5]},
                       'bar': {'color': "darkorange"}}
            ), row=2, col=1)
            
            # Fuel efficiency
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get("fuel_efficiency", 0),
                title={'text': "L/100km"},
                gauge={'axis': {'range': [0, 15]},
                       'bar': {'color': "purple"}}
            ), row=2, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")
    
    @staticmethod
    def render_data_table(data: List[Dict], title: str = "Data"):
        """Render data as an interactive table."""
        try:
            df = pd.DataFrame(data)
            
            if not df.empty:
                st.subheader(title)
                
                # Add filters if multiple columns
                if len(df.columns) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Column filter
                        columns_to_show = st.multiselect(
                            "Select columns to display",
                            options=df.columns.tolist(),
                            default=df.columns.tolist()[:5]  # Show first 5 by default
                        )
                    
                    with col2:
                        # Search filter
                        search_term = st.text_input("Search in data")
                    
                    # Apply filters
                    if columns_to_show:
                        df = df[columns_to_show]
                    
                    if search_term:
                        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                        df = df[mask]
                
                # Display table
                st.dataframe(df, use_container_width=True)
                
                # Export functionality
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv,
                        file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.caption(f"Showing {len(df)} records")
            
        except Exception as e:
            st.error(f"Table rendering failed: {e}")


class LoginPage:
    """Login page implementation."""
    
    @staticmethod
    def render():
        """Render login page."""
        st.markdown(
            "<h2 style='text-align: center; color: #1f77b4;'>Welcome to SmartSchoolGo</h2>",
            unsafe_allow_html=True
        )
        
        # Create centered login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Login to Your Account")
            
            with st.form("login_form"):
                username = st.text_input("Email Address", placeholder="user@example.com")
                password = st.text_input("Password", type="password")
                
                col_login, col_demo = st.columns(2)
                
                with col_login:
                    login_clicked = st.form_submit_button("Login", use_container_width=True)
                
                with col_demo:
                    demo_clicked = st.form_submit_button("Demo Login", use_container_width=True)
                
                if login_clicked and username and password:
                    if SessionManager.login(username, password):
                        st.rerun()
                
                if demo_clicked:
                    # Demo login as parent
                    if SessionManager.login("parent@demo.com", "parent123"):
                        st.rerun()
            
            # Demo credentials
            with st.expander("Demo Credentials"):
                st.markdown("""
                **Parent Account:**
                - Email: parent@demo.com
                - Password: parent123
                
                **Administrator Account:**
                - Email: admin@demo.com
                - Password: admin123
                
                **Planner Account:**
                - Email: planner@demo.com
                - Password: planner123
                """)


class Dashboard:
    """Main dashboard implementation."""
    
    @staticmethod
    def render():
        """Render dashboard based on user role."""
        role = st.session_state.user_role
        
        if role == UserRole.PARENT:
            Dashboard.render_parent_dashboard()
        elif role == UserRole.ADMIN:
            Dashboard.render_admin_dashboard()
        elif role == UserRole.PLANNER:
            Dashboard.render_planner_dashboard()
        elif role == UserRole.DRIVER:
            Dashboard.render_driver_dashboard()
        else:
            st.error("Unknown user role")
    
    @staticmethod
    def render_parent_dashboard():
        """Render parent-specific dashboard."""
        st.title("🏠 Parent Dashboard")
        
        # Real-time status
        UIComponents.render_realtime_status()
        UIComponents.render_alert_banner()
        
        # Key metrics for parents
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("My Children", "2", help="Number of children using transport")
        
        with col2:
            st.metric("Next Pickup", "7:45 AM", "in 25 min", help="Next scheduled pickup time")
        
        with col3:
            st.metric("Bus Status", "On Time", help="Current bus status")
        
        with col4:
            st.metric("Arrival Time", "8:10 AM", "+2 min", help="Expected arrival at school")
        
        st.markdown("---")
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Live Bus Tracking")
            vehicles = RealtimeManager.get_vehicle_updates()
            # Filter to relevant vehicles for this parent
            relevant_vehicles = [v for v in vehicles if v["route_id"] in ["R1", "R2"]]
            UIComponents.render_vehicle_map(relevant_vehicles)
        
        with col2:
            st.subheader("📋 Today's Schedule")
            
            schedule_data = [
                {"Time": "7:45 AM", "Event": "Bus Pickup", "Location": "Corner of Main St"},
                {"Time": "8:10 AM", "Event": "School Arrival", "Location": "Canberra Grammar"},
                {"Time": "3:30 PM", "Event": "School Departure", "Location": "Canberra Grammar"},
                {"Time": "4:05 PM", "Event": "Bus Drop-off", "Location": "Corner of Main St"}
            ]
            
            for item in schedule_data:
                st.markdown(f"**{item['Time']}** - {item['Event']}")
                st.caption(item['Location'])
                st.markdown("---")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        UIComponents.render_notifications_panel()
    
    @staticmethod
    def render_admin_dashboard():
        """Render administrator dashboard."""
        st.title("⚙️ Administrator Dashboard")
        
        UIComponents.render_realtime_status()
        UIComponents.render_alert_banner()
        
        # Admin metrics
        metrics = st.session_state.real_time_data.get("metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Routes", metrics.get("total_routes", 0))
        
        with col2:
            st.metric("Active Vehicles", metrics.get("active_vehicles", 0))
        
        with col3:
            st.metric("On-Time Performance", f"{metrics.get('on_time_performance', 0):.1f}%")
        
        with col4:
            st.metric("Safety Incidents", metrics.get("safety_incidents", 0))
        
        st.markdown("---")
        
        # Performance overview
        st.subheader("📊 Performance Overview")
        UIComponents.render_performance_chart(metrics)
        
        # Fleet status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚌 Fleet Status")
            vehicles = RealtimeManager.get_vehicle_updates()
            UIComponents.render_data_table(vehicles, "Active Vehicles")
        
        with col2:
            st.subheader("🗺️ Live Fleet Map")
            UIComponents.render_vehicle_map(vehicles, height=300)
        
        # Alerts and notifications
        st.subheader("🚨 Active Alerts")
        alerts = RealtimeManager.get_active_alerts()
        if alerts:
            UIComponents.render_data_table(alerts, "System Alerts")
        else:
            st.success("No active alerts")
    
    @staticmethod
    def render_planner_dashboard():
        """Render route planner dashboard."""
        st.title("📋 Route Planner Dashboard")
        
        UIComponents.render_realtime_status()
        UIComponents.render_alert_banner()
        
        # Planner metrics
        metrics = st.session_state.real_time_data.get("metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fleet Efficiency", f"{metrics.get('fleet_utilization', 0):.1f}%")
        
        with col2:
            st.metric("Avg Delay", f"{metrics.get('average_delay', 0):.1f} min")
        
        with col3:
            st.metric("Fuel Efficiency", f"{metrics.get('fuel_efficiency', 0):.1f} L/100km")
        
        with col4:
            st.metric("Student Satisfaction", f"{metrics.get('student_satisfaction', 0):.1f}/5")
        
        st.markdown("---")
        
        # Route optimization
        st.subheader("🎯 Route Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Optimization controls
            st.markdown("#### Optimization Parameters")
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                optimize_for = st.selectbox(
                    "Optimize for:",
                    ["Cost Reduction", "Time Efficiency", "Fuel Savings", "Student Satisfaction"]
                )
                
                max_delay = st.slider("Max Acceptable Delay (minutes)", 0, 30, 5)
            
            with col_opt2:
                route_selection = st.multiselect(
                    "Routes to optimize:",
                    ["R1", "R2", "R3", "R4", "R5"],
                    default=["R1", "R2"]
                )
                
                if st.button("🚀 Run Optimization", use_container_width=True):
                    with st.spinner("Running optimization..."):
                        time.sleep(2)  # Simulate optimization
                        st.success("Optimization complete! Estimated 15% cost reduction.")
        
        with col2:
            st.markdown("#### Quick Actions")
            
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("Report generation started...")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                RealtimeManager.fetch_initial_data()
                st.success("Data refreshed!")
            
            if st.button("⚠️ Emergency Mode", use_container_width=True):
                st.warning("Emergency protocols activated")
        
        # Route performance
        st.subheader("📈 Route Performance")
        routes = st.session_state.real_time_data.get("routes", [])
        UIComponents.render_data_table(routes, "Route Performance")
    
    @staticmethod
    def render_driver_dashboard():
        """Render driver dashboard."""
        st.title("🚌 Driver Dashboard")
        
        UIComponents.render_realtime_status()
        UIComponents.render_alert_banner()
        
        # Driver-specific metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Route", "R1 - North")
        
        with col2:
            st.metric("Next Stop", "Main St & Oak Ave", "in 3 min")
        
        with col3:
            st.metric("Students Onboard", "28/50")
        
        with col4:
            st.metric("Schedule Status", "On Time")
        
        st.markdown("---")
        
        # Driver interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ My Route")
            vehicles = [v for v in RealtimeManager.get_vehicle_updates() if v["vehicle_id"] == "BUS-001"]
            UIComponents.render_vehicle_map(vehicles)
        
        with col2:
            st.subheader("📋 Today's Schedule")
            
            schedule = [
                {"Time": "7:30 AM", "Stop": "Depot Departure"},
                {"Time": "7:45 AM", "Stop": "Main St & Oak Ave", "Students": 5},
                {"Time": "7:52 AM", "Stop": "Elm St & Pine Ave", "Students": 8},
                {"Time": "8:05 AM", "Stop": "School Arrival", "Students": 28}
            ]
            
            for i, stop in enumerate(schedule):
                if i == 1:  # Current stop
                    st.success(f"**{stop['Time']}** - {stop['Stop']} ⭐")
                else:
                    st.markdown(f"**{stop['Time']}** - {stop['Stop']}")
                
                if "Students" in stop:
                    st.caption(f"{stop['Students']} students")
                st.markdown("---")


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #1f77b4 0%, #3498db 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    
    /* Alert styling */
    .alert-banner {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Hide streamlit branding */
    .reportview-container .main footer {visibility: hidden;}
    .reportview-container .main .block-container {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    try:
        # Apply custom styling
        apply_custom_css()
        
        # Initialize session
        SessionManager.initialize_session()
        
        # Render header
        UIComponents.render_header()
        
        # Handle routing
        if not st.session_state.authenticated:
            LoginPage.render()
        else:
            # Get navigation selection
            selected_page = UIComponents.render_navigation()
            
            # Auto-refresh data every 30 seconds
            if st.session_state.authenticated:
                if st.session_state.get("last_refresh"):
                    time_since_refresh = time.time() - st.session_state.last_refresh
                    if time_since_refresh > 30:
                        RealtimeManager.fetch_initial_data()
                        st.session_state.last_refresh = time.time()
                else:
                    st.session_state.last_refresh = time.time()
            
            # Route to appropriate page
            if selected_page == "dashboard":
                Dashboard.render()
            
            elif selected_page == "track_bus":
                st.title("🗺️ Live Bus Tracking")
                UIComponents.render_realtime_status()
                vehicles = RealtimeManager.get_vehicle_updates()
                UIComponents.render_vehicle_map(vehicles, height=500)
            
            elif selected_page == "route_info":
                st.title("🚌 Route Information")
                routes = st.session_state.real_time_data.get("routes", [])
                UIComponents.render_data_table(routes, "Available Routes")
            
            elif selected_page == "notifications":
                st.title("🔔 Notifications")
                UIComponents.render_notifications_panel()
                
                # Add test notification button
                if st.button("Add Test Notification"):
                    RealtimeManager.add_notification("Test notification message", "info")
                    st.rerun()
            
            elif selected_page == "profile":
                st.title("👤 User Profile")
                st.markdown(f"**Name:** {st.session_state.user_name}")
                st.markdown(f"**Email:** {st.session_state.user_id}")
                st.markdown(f"**Role:** {st.session_state.user_role}")
                st.markdown(f"**Login Time:** {st.session_state.login_time}")
            
            elif selected_page == "fleet_management":
                st.title("🚚 Fleet Management")
                UIComponents.render_realtime_status()
                vehicles = RealtimeManager.get_vehicle_updates()
                UIComponents.render_data_table(vehicles, "Fleet Status")
            
            elif selected_page == "analytics":
                st.title("📊 Analytics")
                metrics = st.session_state.real_time_data.get("metrics", {})
                UIComponents.render_performance_chart(metrics)
            
            elif selected_page == "alerts":
                st.title("🚨 System Alerts")
                alerts = st.session_state.real_time_data.get("alerts", [])
                UIComponents.render_data_table(alerts, "All Alerts")
            
            elif selected_page == "route_planning":
                st.title("📋 Route Planning")
                st.info("Route planning interface - coming soon!")
            
            elif selected_page == "optimization":
                st.title("🎯 Route Optimization")
                st.info("Advanced optimization tools - coming soon!")
            
            else:
                st.error(f"Page '{selected_page}' not implemented yet.")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()