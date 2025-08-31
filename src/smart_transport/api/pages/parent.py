"""
Parent Portal Interface for SmartSchoolGo

This module provides a comprehensive parent portal interface with real-time bus tracking,
push notifications, route information, student profile management, communication tools,
and historical travel analytics specifically designed for parents.

Features:
- Interactive map centered on child's route with real-time tracking
- Push notification system for pickup/dropoff alerts
- Comprehensive route information with alternative options
- Student profile management and settings
- Direct communication portal with school transport services
- Historical travel data and punctuality analytics
- Weather integration for travel planning
- Emergency contact management
- Mobile-responsive design optimized for parent use

Author: SmartSchoolGo Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
import uuid

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import folium
    from streamlit_folium import st_folium
    import requests
    from geopy.distance import geodesic
    import streamlit_elements as elements
    PARENT_LIBS_AVAILABLE = True
except ImportError as e:
    st.error(f"Required libraries not available: {e}")
    PARENT_LIBS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationType:
    """Types of notifications for parents."""
    BUS_ARRIVING = "bus_arriving"
    BUS_DELAYED = "bus_delayed"
    ROUTE_CHANGED = "route_changed"
    WEATHER_ALERT = "weather_alert"
    EMERGENCY = "emergency"
    SCHEDULE_UPDATE = "schedule_update"
    MAINTENANCE = "maintenance"


class StudentProfile:
    """Student profile data structure."""
    
    def __init__(self, student_data: Dict[str, Any]):
        self.student_id = student_data.get('student_id', '')
        self.name = student_data.get('name', '')
        self.grade = student_data.get('grade', '')
        self.school = student_data.get('school', '')
        self.home_address = student_data.get('home_address', '')
        self.pickup_location = student_data.get('pickup_location', '')
        self.route_id = student_data.get('route_id', '')
        self.pickup_time = student_data.get('pickup_time', '')
        self.dropoff_time = student_data.get('dropoff_time', '')
        self.emergency_contacts = student_data.get('emergency_contacts', [])
        self.special_needs = student_data.get('special_needs', [])
        self.allergies = student_data.get('allergies', [])


class ParentPortalData:
    """Data management for parent portal."""
    
    @staticmethod
    @st.cache_data(ttl=30)
    def get_student_data(parent_id: str) -> List[StudentProfile]:
        """Get student data for the logged-in parent."""
        # Mock data - replace with actual database queries
        mock_students = [
            {
                'student_id': 'STU001',
                'name': 'Emma Smith',
                'grade': '5',
                'school': 'Canberra Primary School',
                'home_address': '123 Main Street, Canberra ACT 2600',
                'pickup_location': 'Corner of Main St & Oak Ave',
                'route_id': 'R1',
                'pickup_time': '07:45',
                'dropoff_time': '15:30',
                'emergency_contacts': [
                    {'name': 'John Smith', 'phone': '0412345678', 'relationship': 'Father'},
                    {'name': 'Mary Johnson', 'phone': '0423456789', 'relationship': 'Grandmother'}
                ],
                'special_needs': [],
                'allergies': ['Peanuts']
            },
            {
                'student_id': 'STU002',
                'name': 'James Smith',
                'grade': '3',
                'school': 'Canberra Primary School',
                'home_address': '123 Main Street, Canberra ACT 2600',
                'pickup_location': 'Corner of Main St & Oak Ave',
                'route_id': 'R1',
                'pickup_time': '07:45',
                'dropoff_time': '15:30',
                'emergency_contacts': [
                    {'name': 'John Smith', 'phone': '0412345678', 'relationship': 'Father'},
                    {'name': 'Mary Johnson', 'phone': '0423456789', 'relationship': 'Grandmother'}
                ],
                'special_needs': ['Requires assistance boarding'],
                'allergies': []
            }
        ]
        
        return [StudentProfile(student) for student in mock_students]
    
    @staticmethod
    @st.cache_data(ttl=10)
    def get_live_bus_data() -> Dict[str, Any]:
        """Get live bus tracking data."""
        current_time = datetime.now()
        
        return {
            'bus_id': 'BUS-R1-001',
            'route_id': 'R1',
            'current_location': {
                'latitude': -35.2820,
                'longitude': 149.1285,
                'address': 'Northbourne Avenue, Canberra'
            },
            'next_stop': {
                'name': 'Main St & Oak Ave',
                'eta': current_time + timedelta(minutes=8),
                'distance_km': 2.1
            },
            'speed': 35.0,
            'occupancy': 28,
            'capacity': 50,
            'status': 'On Schedule',
            'driver': {
                'name': 'Mike Johnson',
                'phone': '0434567890'
            },
            'last_update': current_time - timedelta(seconds=15)
        }
    
    @staticmethod
    @st.cache_data(ttl=60)
    def get_route_data(route_id: str) -> Dict[str, Any]:
        """Get detailed route information."""
        return {
            'route_id': route_id,
            'route_name': 'Route 1 - Northern Suburbs',
            'school': 'Canberra Primary School',
            'total_stops': 12,
            'total_distance': 18.5,
            'average_duration': 35,
            'stops': [
                {
                    'stop_id': 'STP001',
                    'name': 'Depot',
                    'address': 'Transport Depot, Industrial Ave',
                    'latitude': -35.2750,
                    'longitude': 149.1200,
                    'scheduled_time': '07:30',
                    'type': 'departure'
                },
                {
                    'stop_id': 'STP002', 
                    'name': 'Main St & Oak Ave',
                    'address': 'Corner of Main Street and Oak Avenue',
                    'latitude': -35.2800,
                    'longitude': 149.1250,
                    'scheduled_time': '07:45',
                    'type': 'pickup',
                    'students': ['Emma Smith', 'James Smith', 'Sarah Jones']
                },
                {
                    'stop_id': 'STP003',
                    'name': 'Elm Street',
                    'address': 'Elm Street near shopping center',
                    'latitude': -35.2780,
                    'longitude': 149.1300,
                    'scheduled_time': '07:52',
                    'type': 'pickup',
                    'students': ['Tom Wilson', 'Amy Chen']
                },
                {
                    'stop_id': 'STP004',
                    'name': 'School Arrival',
                    'address': 'Canberra Primary School',
                    'latitude': -35.2850,
                    'longitude': 149.1350,
                    'scheduled_time': '08:10',
                    'type': 'dropoff'
                }
            ],
            'alternative_routes': [
                {
                    'name': 'Public Transport Option',
                    'description': 'Bus Route 3 to City, then Bus Route 7 to school',
                    'duration': '45 minutes',
                    'cost': '$4.50',
                    'walking_distance': '800m total'
                },
                {
                    'name': 'Walking Route',
                    'description': 'Direct walking route via bike path',
                    'duration': '25 minutes',
                    'cost': 'Free',
                    'walking_distance': '2.1km'
                }
            ]
        }
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_travel_history(student_id: str, days: int = 30) -> pd.DataFrame:
        """Get travel history for analytics."""
        # Generate mock travel history
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Filter to weekdays only (school days)
        weekdays = dates[dates.weekday < 5]
        
        history_data = []
        for date in weekdays:
            # Morning trip
            scheduled_pickup = datetime.combine(date.date(), time(7, 45))
            actual_pickup = scheduled_pickup + timedelta(
                minutes=np.random.normal(0, 3)  # Average 0 delay, std 3 minutes
            )
            
            history_data.append({
                'date': date.date(),
                'trip_type': 'Morning',
                'scheduled_time': scheduled_pickup,
                'actual_time': actual_pickup,
                'delay_minutes': (actual_pickup - scheduled_pickup).total_seconds() / 60,
                'bus_id': 'BUS-R1-001',
                'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Windy']),
                'on_time': abs((actual_pickup - scheduled_pickup).total_seconds()) < 300  # 5 minute tolerance
            })
            
            # Afternoon trip
            scheduled_dropoff = datetime.combine(date.date(), time(15, 30))
            actual_dropoff = scheduled_dropoff + timedelta(
                minutes=np.random.normal(2, 5)  # Afternoon usually has more delays
            )
            
            history_data.append({
                'date': date.date(),
                'trip_type': 'Afternoon',
                'scheduled_time': scheduled_dropoff,
                'actual_time': actual_dropoff,
                'delay_minutes': (actual_dropoff - scheduled_dropoff).total_seconds() / 60,
                'bus_id': 'BUS-R1-001',
                'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Windy']),
                'on_time': abs((actual_dropoff - scheduled_dropoff).total_seconds()) < 300
            })
        
        return pd.DataFrame(history_data)
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_weather_data() -> Dict[str, Any]:
        """Get weather data for travel planning."""
        return {
            'current': {
                'temperature': 18,
                'condition': 'Partly Cloudy',
                'humidity': 65,
                'wind_speed': 12,
                'visibility': 10,
                'uv_index': 6
            },
            'forecast': [
                {'time': '06:00', 'temp': 12, 'condition': 'Clear', 'rain_chance': 0},
                {'time': '07:00', 'temp': 14, 'condition': 'Clear', 'rain_chance': 0},
                {'time': '08:00', 'temp': 16, 'condition': 'Partly Cloudy', 'rain_chance': 10},
                {'time': '15:00', 'temp': 22, 'condition': 'Cloudy', 'rain_chance': 30},
                {'time': '16:00', 'temp': 20, 'condition': 'Light Rain', 'rain_chance': 60}
            ],
            'alerts': [
                {
                    'type': 'rain',
                    'message': 'Light rain expected during afternoon pickup (60% chance)',
                    'severity': 'low'
                }
            ]
        }


class ParentNotifications:
    """Notification management for parents."""
    
    @staticmethod
    def initialize_notifications():
        """Initialize notification system."""
        if 'parent_notifications' not in st.session_state:
            st.session_state.parent_notifications = []
            
        if 'notification_preferences' not in st.session_state:
            st.session_state.notification_preferences = {
                'bus_arriving': {'enabled': True, 'advance_minutes': 5},
                'bus_delayed': {'enabled': True, 'threshold_minutes': 5},
                'route_changed': {'enabled': True},
                'weather_alert': {'enabled': True},
                'emergency': {'enabled': True},
                'schedule_update': {'enabled': True}
            }
    
    @staticmethod
    def add_notification(notification_type: str, message: str, 
                        priority: str = 'normal', data: Dict = None):
        """Add a new notification."""
        ParentNotifications.initialize_notifications()
        
        notification = {
            'id': str(uuid.uuid4()),
            'type': notification_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now(),
            'read': False,
            'data': data or {}
        }
        
        st.session_state.parent_notifications.append(notification)
        
        # Keep only last 50 notifications
        if len(st.session_state.parent_notifications) > 50:
            st.session_state.parent_notifications = st.session_state.parent_notifications[-50:]
    
    @staticmethod
    def get_unread_notifications() -> List[Dict]:
        """Get unread notifications."""
        ParentNotifications.initialize_notifications()
        return [n for n in st.session_state.parent_notifications if not n['read']]
    
    @staticmethod
    def mark_notification_read(notification_id: str):
        """Mark notification as read."""
        for notification in st.session_state.parent_notifications:
            if notification['id'] == notification_id:
                notification['read'] = True
                break
    
    @staticmethod
    def check_bus_arrival_alerts():
        """Check and create bus arrival alerts."""
        bus_data = ParentPortalData.get_live_bus_data()
        next_stop = bus_data.get('next_stop', {})
        
        if next_stop:
            eta = next_stop.get('eta')
            if eta:
                time_to_arrival = (eta - datetime.now()).total_seconds() / 60
                advance_minutes = st.session_state.notification_preferences.get(
                    'bus_arriving', {}
                ).get('advance_minutes', 5)
                
                if 0 < time_to_arrival <= advance_minutes:
                    ParentNotifications.add_notification(
                        NotificationType.BUS_ARRIVING,
                        f"Bus arriving at {next_stop['name']} in {int(time_to_arrival)} minutes",
                        'high',
                        {'eta': eta.isoformat(), 'stop': next_stop['name']}
                    )


class ParentPortalUI:
    """User interface components for parent portal."""
    
    @staticmethod
    def render_header():
        """Render parent portal header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.image("https://via.placeholder.com/100x50/1f77b4/white?text=SchoolGo", width=100)
        
        with col2:
            st.markdown(
                "<h2 style='text-align: center; color: #1f77b4; margin: 0;'>Parent Portal</h2>",
                unsafe_allow_html=True
            )
        
        with col3:
            # Notification bell
            unread_count = len(ParentNotifications.get_unread_notifications())
            if unread_count > 0:
                st.markdown(
                    f"<div style='text-align: right;'>"
                    f"<span style='background: red; color: white; border-radius: 50%; "
                    f"padding: 2px 6px; font-size: 12px;'>{unread_count}</span> 🔔"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    @staticmethod
    def render_student_selector() -> StudentProfile:
        """Render student selection widget."""
        students = ParentPortalData.get_student_data(st.session_state.user_id)
        
        if len(students) == 1:
            st.sidebar.markdown(f"**Student:** {students[0].name}")
            return students[0]
        else:
            student_options = {f"{s.name} (Grade {s.grade})": s for s in students}
            selected_name = st.sidebar.selectbox(
                "Select Student",
                options=list(student_options.keys())
            )
            return student_options[selected_name]
    
    @staticmethod
    def render_quick_stats(student: StudentProfile):
        """Render quick statistics cards."""
        bus_data = ParentPortalData.get_live_bus_data()
        next_stop = bus_data.get('next_stop', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if next_stop.get('eta'):
                eta = next_stop['eta']
                minutes_away = int((eta - datetime.now()).total_seconds() / 60)
                st.metric(
                    "Bus Arrives In",
                    f"{minutes_away} min" if minutes_away > 0 else "Arrived",
                    delta="On time" if bus_data.get('status') == 'On Schedule' else "Delayed"
                )
            else:
                st.metric("Bus Status", bus_data.get('status', 'Unknown'))
        
        with col2:
            st.metric(
                "Current Occupancy",
                f"{bus_data.get('occupancy', 0)}/{bus_data.get('capacity', 50)}",
                delta=f"{bus_data.get('capacity', 50) - bus_data.get('occupancy', 0)} seats free"
            )
        
        with col3:
            st.metric(
                "Pickup Time",
                student.pickup_time,
                delta=student.pickup_location
            )
        
        with col4:
            # On-time performance
            history = ParentPortalData.get_travel_history(student.student_id, 7)
            if not history.empty:
                on_time_rate = history['on_time'].mean() * 100
                st.metric(
                    "On-Time Rate (7d)",
                    f"{on_time_rate:.0f}%",
                    delta=f"{'Good' if on_time_rate >= 80 else 'Needs Improvement'}"
                )
    
    @staticmethod
    def render_live_tracking_map(student: StudentProfile):
        """Render live bus tracking map."""
        st.subheader("🗺️ Live Bus Tracking")
        
        try:
            bus_data = ParentPortalData.get_live_bus_data()
            route_data = ParentPortalData.get_route_data(student.route_id)
            
            # Create map centered on bus location
            bus_location = bus_data['current_location']
            map_center = [bus_location['latitude'], bus_location['longitude']]
            
            m = folium.Map(location=map_center, zoom_start=13)
            
            # Add bus marker
            folium.Marker(
                [bus_location['latitude'], bus_location['longitude']],
                popup=f"""
                <div style='width: 200px;'>
                    <h4>🚌 {bus_data['bus_id']}</h4>
                    <p><b>Speed:</b> {bus_data['speed']} km/h</p>
                    <p><b>Occupancy:</b> {bus_data['occupancy']}/{bus_data['capacity']}</p>
                    <p><b>Status:</b> {bus_data['status']}</p>
                    <p><b>Driver:</b> {bus_data['driver']['name']}</p>
                </div>
                """,
                icon=folium.Icon(color='green', icon='bus', prefix='fa'),
                tooltip="Current Bus Location"
            ).add_to(m)
            
            # Add route stops
            colors = {'pickup': 'blue', 'dropoff': 'red', 'departure': 'gray'}
            
            for stop in route_data['stops']:
                color = colors.get(stop['type'], 'blue')
                
                popup_content = f"""
                <div style='width: 180px;'>
                    <h5>{stop['name']}</h5>
                    <p><b>Time:</b> {stop['scheduled_time']}</p>
                    <p><b>Type:</b> {stop['type'].title()}</p>
                """
                
                if 'students' in stop:
                    popup_content += f"<p><b>Students:</b> {', '.join(stop['students'])}</p>"
                
                popup_content += "</div>"
                
                # Highlight student's stop
                if any(student.name in stop.get('students', []) for student in [student]):
                    icon = folium.Icon(color='red', icon='star', prefix='fa')
                    tooltip = f"⭐ {stop['name']} (Your Stop)"
                else:
                    icon = folium.Icon(color=color, icon='map-marker', prefix='fa')
                    tooltip = stop['name']
                
                folium.Marker(
                    [stop['latitude'], stop['longitude']],
                    popup=popup_content,
                    icon=icon,
                    tooltip=tooltip
                ).add_to(m)
            
            # Add route line
            route_coords = [[stop['latitude'], stop['longitude']] for stop in route_data['stops']]
            folium.PolyLine(
                route_coords,
                color='blue',
                weight=3,
                opacity=0.7,
                tooltip="Bus Route"
            ).add_to(m)
            
            # Show next stop ETA circle
            if bus_data.get('next_stop'):
                next_stop = bus_data['next_stop']
                if 'latitude' in next_stop and 'longitude' in next_stop:
                    folium.Circle(
                        [next_stop['latitude'], next_stop['longitude']],
                        radius=200,
                        color='orange',
                        fill=True,
                        opacity=0.3,
                        tooltip="Next Stop"
                    ).add_to(m)
            
            # Display map
            map_data = st_folium(m, width=700, height=400)
            
            # Show ETA information
            if bus_data.get('next_stop'):
                next_stop = bus_data['next_stop']
                eta = next_stop.get('eta')
                if eta:
                    time_to_arrival = eta - datetime.now()
                    minutes = int(time_to_arrival.total_seconds() / 60)
                    
                    if minutes > 0:
                        st.success(f"🚌 Bus arriving at **{next_stop['name']}** in **{minutes} minutes**")
                    else:
                        st.info("🚌 Bus has arrived at the stop")
            
        except Exception as e:
            st.error(f"Unable to load live tracking: {str(e)}")
            logger.error(f"Live tracking error: {e}")
    
    @staticmethod
    def render_schedule_timeline(student: StudentProfile):
        """Render today's schedule timeline."""
        st.subheader("📅 Today's Schedule")
        
        route_data = ParentPortalData.get_route_data(student.route_id)
        current_time = datetime.now()
        
        # Create timeline
        for i, stop in enumerate(route_data['stops']):
            scheduled_time = datetime.strptime(stop['scheduled_time'], '%H:%M').time()
            scheduled_datetime = datetime.combine(current_time.date(), scheduled_time)
            
            # Determine status
            if scheduled_datetime < current_time:
                status = "✅ Completed"
                color = "green"
            elif abs((scheduled_datetime - current_time).total_seconds()) < 600:  # Within 10 minutes
                status = "🚌 Current"
                color = "orange"
            else:
                status = "⏳ Scheduled"
                color = "blue"
            
            # Check if this is student's stop
            is_student_stop = any(student.name in stop.get('students', []))
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**{stop['scheduled_time']}**")
            
            with col2:
                if is_student_stop:
                    st.markdown(f"⭐ **{stop['name']}** ({stop['type'].title()})")
                    st.caption(f"📍 {stop.get('address', '')}")
                else:
                    st.markdown(f"{stop['name']} ({stop['type'].title()})")
                    if 'students' in stop and len(stop['students']) > 0:
                        st.caption(f"👥 {len(stop['students'])} students")
            
            with col3:
                st.markdown(f"<span style='color: {color};'>{status}</span>", unsafe_allow_html=True)
            
            if i < len(route_data['stops']) - 1:
                st.markdown("---")
    
    @staticmethod
    def render_notifications_panel():
        """Render notifications panel."""
        st.subheader("🔔 Notifications")
        
        ParentNotifications.initialize_notifications()
        
        # Notification preferences
        with st.expander("⚙️ Notification Settings"):
            prefs = st.session_state.notification_preferences
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Alert Types**")
                prefs['bus_arriving']['enabled'] = st.checkbox(
                    "Bus Arriving Soon", 
                    value=prefs['bus_arriving']['enabled']
                )
                
                prefs['bus_delayed']['enabled'] = st.checkbox(
                    "Bus Delays", 
                    value=prefs['bus_delayed']['enabled']
                )
                
                prefs['route_changed']['enabled'] = st.checkbox(
                    "Route Changes", 
                    value=prefs['route_changed']['enabled']
                )
            
            with col2:
                st.write("**Settings**")
                if prefs['bus_arriving']['enabled']:
                    prefs['bus_arriving']['advance_minutes'] = st.slider(
                        "Alert me X minutes before arrival",
                        1, 15, prefs['bus_arriving']['advance_minutes']
                    )
                
                if prefs['bus_delayed']['enabled']:
                    prefs['bus_delayed']['threshold_minutes'] = st.slider(
                        "Alert if delayed more than X minutes",
                        1, 20, prefs['bus_delayed']['threshold_minutes']
                    )
        
        # Display notifications
        notifications = sorted(
            st.session_state.parent_notifications, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )
        
        if notifications:
            unread_notifications = [n for n in notifications if not n['read']]
            read_notifications = [n for n in notifications if n['read']]
            
            # Unread notifications
            if unread_notifications:
                st.write("**New Notifications**")
                for notification in unread_notifications[:10]:  # Show last 10 unread
                    ParentPortalUI._render_notification_item(notification, is_new=True)
                
                if st.button("Mark All as Read"):
                    for notification in unread_notifications:
                        notification['read'] = True
                    st.rerun()
            
            # Recent read notifications
            if read_notifications:
                with st.expander(f"Recent Notifications ({len(read_notifications)})"):
                    for notification in read_notifications[:20]:  # Show last 20 read
                        ParentPortalUI._render_notification_item(notification, is_new=False)
        else:
            st.info("No notifications yet.")
        
        # Add test notification button (for demo)
        if st.button("🧪 Add Test Notification"):
            ParentNotifications.add_notification(
                NotificationType.BUS_ARRIVING,
                "Bus will arrive at your stop in 5 minutes",
                'high'
            )
            st.rerun()
    
    @staticmethod
    def _render_notification_item(notification: Dict, is_new: bool = False):
        """Render a single notification item."""
        timestamp = notification['timestamp'].strftime('%m/%d %H:%M')
        
        # Icons for different notification types
        icons = {
            NotificationType.BUS_ARRIVING: "🚌",
            NotificationType.BUS_DELAYED: "⏰",
            NotificationType.ROUTE_CHANGED: "🔄",
            NotificationType.WEATHER_ALERT: "🌧️",
            NotificationType.EMERGENCY: "🚨",
            NotificationType.SCHEDULE_UPDATE: "📅",
            NotificationType.MAINTENANCE: "🔧"
        }
        
        icon = icons.get(notification['type'], "ℹ️")
        
        # Priority styling
        if notification['priority'] == 'high':
            message_style = "background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; border-radius: 4px;"
        elif notification['priority'] == 'critical':
            message_style = "background-color: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; border-radius: 4px;"
        else:
            message_style = "background-color: #d1ecf1; padding: 10px; border-left: 4px solid #17a2b8; border-radius: 4px;"
        
        if is_new:
            message_style += " font-weight: bold;"
        
        st.markdown(
            f"<div style='{message_style}'>"
            f"{icon} {notification['message']}<br>"
            f"<small style='color: #666;'>{timestamp}</small>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("")  # Add spacing
    
    @staticmethod
    def render_student_profile(student: StudentProfile):
        """Render student profile management."""
        st.subheader("👤 Student Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information**")
            st.text_input("Full Name", value=student.name, disabled=True)
            st.text_input("Grade", value=student.grade, disabled=True)
            st.text_input("School", value=student.school, disabled=True)
            st.text_input("Student ID", value=student.student_id, disabled=True)
        
        with col2:
            st.write("**Transport Details**")
            st.text_input("Route", value=student.route_id, disabled=True)
            st.text_input("Pickup Time", value=student.pickup_time, disabled=True)
            st.text_input("Pickup Location", value=student.pickup_location, disabled=True)
            st.text_input("Home Address", value=student.home_address, disabled=True)
        
        # Emergency contacts
        st.write("**Emergency Contacts**")
        for i, contact in enumerate(student.emergency_contacts):
            with st.expander(f"{contact['name']} ({contact['relationship']})"):
                st.text_input("Name", value=contact['name'], key=f"contact_name_{i}")
                st.text_input("Phone", value=contact['phone'], key=f"contact_phone_{i}")
                st.text_input("Relationship", value=contact['relationship'], key=f"contact_rel_{i}")
        
        # Special requirements
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Special Needs**")
            if student.special_needs:
                for need in student.special_needs:
                    st.info(f"• {need}")
            else:
                st.write("None listed")
        
        with col2:
            st.write("**Allergies**")
            if student.allergies:
                for allergy in student.allergies:
                    st.warning(f"⚠️ {allergy}")
            else:
                st.write("None listed")
    
    @staticmethod
    def render_travel_analytics(student: StudentProfile):
        """Render travel history and analytics."""
        st.subheader("📊 Travel Analytics")
        
        # Time period selector
        period = st.selectbox(
            "Analysis Period",
            ["Last 7 days", "Last 30 days", "Last 3 months"],
            index=1
        )
        
        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 3 months": 90}
        days = days_map[period]
        
        # Get travel history
        history = ParentPortalData.get_travel_history(student.student_id, days)
        
        if history.empty:
            st.warning("No travel history available for the selected period.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            on_time_rate = history['on_time'].mean() * 100
            st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
        
        with col2:
            avg_delay = history['delay_minutes'].mean()
            st.metric("Average Delay", f"{avg_delay:.1f} min")
        
        with col3:
            total_trips = len(history)
            st.metric("Total Trips", total_trips)
        
        with col4:
            max_delay = history['delay_minutes'].max()
            st.metric("Max Delay", f"{max_delay:.1f} min")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # On-time performance over time
            daily_performance = history.groupby('date')['on_time'].mean().reset_index()
            daily_performance['on_time_pct'] = daily_performance['on_time'] * 100
            
            fig = px.line(
                daily_performance, 
                x='date', 
                y='on_time_pct',
                title="Daily On-Time Performance",
                labels={'on_time_pct': 'On-Time Rate (%)', 'date': 'Date'}
            )
            fig.update_traces(line=dict(color='#1f77b4', width=3))
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="Target: 80%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delay distribution
            fig = px.histogram(
                history, 
                x='delay_minutes',
                title="Delay Distribution",
                labels={'delay_minutes': 'Delay (minutes)', 'count': 'Number of Trips'},
                nbins=20
            )
            fig.update_traces(marker_color='#ff7f0e')
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekly pattern analysis
        st.subheader("📈 Weekly Patterns")
        
        history['day_of_week'] = pd.to_datetime(history['date']).dt.day_name()
        weekly_stats = history.groupby(['day_of_week', 'trip_type']).agg({
            'delay_minutes': 'mean',
            'on_time': 'mean'
        }).round(2)
        
        # Pivot for better display
        weekly_delay = weekly_stats['delay_minutes'].unstack(fill_value=0)
        weekly_ontime = weekly_stats['on_time'].unstack(fill_value=0) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Average Delay by Day (minutes)**")
            st.dataframe(weekly_delay)
        
        with col2:
            st.write("**On-Time Rate by Day (%)**")
            st.dataframe(weekly_ontime)
        
        # Weather impact analysis
        if 'weather' in history.columns:
            st.subheader("🌤️ Weather Impact")
            
            weather_stats = history.groupby('weather').agg({
                'delay_minutes': 'mean',
                'on_time': 'mean'
            }).round(2)
            weather_stats['on_time_pct'] = weather_stats['on_time'] * 100
            
            fig = px.bar(
                weather_stats.reset_index(),
                x='weather',
                y='delay_minutes',
                title="Average Delay by Weather Condition",
                labels={'delay_minutes': 'Average Delay (minutes)', 'weather': 'Weather'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_communication_center():
        """Render communication center with school transport."""
        st.subheader("💬 Communication Center")
        
        # Quick contact options
        st.write("**Quick Contact**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📞 Call Transport Office"):
                st.info("Calling: +61 2 6123 4567")
        
        with col2:
            if st.button("✉️ Send Email"):
                st.info("Opening email to: transport@canberraschools.edu.au")
        
        with col3:
            if st.button("🚨 Emergency Contact"):
                st.error("Emergency line: +61 2 6123 9999")
        
        st.markdown("---")
        
        # Message center
        st.write("**Send Message**")
        
        message_type = st.selectbox(
            "Message Type",
            ["General Inquiry", "Schedule Change Request", "Route Concern", "Feedback"]
        )
        
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=100)
        
        if st.button("📤 Send Message"):
            if subject and message:
                # In production, this would integrate with actual messaging system
                st.success("Message sent successfully! You will receive a response within 24 hours.")
                
                # Add to notifications as confirmation
                ParentNotifications.add_notification(
                    NotificationType.SCHEDULE_UPDATE,
                    f"Your message '{subject}' has been sent to the transport office",
                    'normal'
                )
            else:
                st.error("Please fill in both subject and message fields.")
        
        # FAQ section
        with st.expander("❓ Frequently Asked Questions"):
            faqs = [
                {
                    "question": "What should I do if the bus is late?",
                    "answer": "Buses can be delayed by traffic or weather. Check the live tracking for real-time updates. If the bus is more than 15 minutes late, contact the transport office."
                },
                {
                    "question": "How do I request a route change?",
                    "answer": "Route change requests must be submitted at least 2 weeks in advance. Use the 'Schedule Change Request' message type above or contact the office directly."
                },
                {
                    "question": "What happens in severe weather?",
                    "answer": "In case of severe weather, routes may be modified or cancelled for safety. You'll receive notifications through this app and the school's communication channels."
                },
                {
                    "question": "My child missed the bus, what should I do?",
                    "answer": "If your child misses the bus, you'll need to arrange alternative transport. The bus cannot wait beyond the scheduled time to maintain the schedule for other students."
                }
            ]
            
            for faq in faqs:
                st.write(f"**Q: {faq['question']}**")
                st.write(f"A: {faq['answer']}")
                st.markdown("---")
    
    @staticmethod
    def render_weather_integration():
        """Render weather information for travel planning."""
        st.subheader("🌤️ Weather & Travel Planning")
        
        weather_data = ParentPortalData.get_weather_data()
        current = weather_data['current']
        forecast = weather_data['forecast']
        alerts = weather_data['alerts']
        
        # Current conditions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Temperature", f"{current['temperature']}°C")
        
        with col2:
            st.metric("Condition", current['condition'])
        
        with col3:
            st.metric("Wind Speed", f"{current['wind_speed']} km/h")
        
        with col4:
            st.metric("Visibility", f"{current['visibility']} km")
        
        # Weather alerts
        if alerts:
            st.write("**Weather Alerts**")
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(f"⚠️ {alert['message']}")
                elif alert['severity'] == 'medium':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        
        # Hourly forecast for travel times
        st.write("**Travel Time Forecast**")
        
        forecast_df = pd.DataFrame(forecast)
        forecast_df['rain_risk'] = forecast_df['rain_chance'].apply(
            lambda x: 'High' if x > 70 else 'Medium' if x > 30 else 'Low'
        )
        
        # Highlight morning and afternoon travel times
        travel_times = ['07:00', '08:00', '15:00', '16:00']
        travel_forecast = forecast_df[forecast_df['time'].isin(travel_times)]
        
        for _, row in travel_forecast.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{row['time']}**")
            
            with col2:
                st.write(f"{row['temp']}°C")
            
            with col3:
                st.write(row['condition'])
            
            with col4:
                rain_chance = row['rain_chance']
                if rain_chance > 50:
                    st.write(f"🌧️ {rain_chance}%")
                else:
                    st.write(f"☀️ {rain_chance}%")


def render_parent_portal():
    """Main function to render the parent portal."""
    try:
        # Initialize notifications
        ParentNotifications.initialize_notifications()
        
        # Check for bus arrival alerts
        ParentNotifications.check_bus_arrival_alerts()
        
        # Render header
        ParentPortalUI.render_header()
        
        # Student selector in sidebar
        selected_student = ParentPortalUI.render_student_selector()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Dashboard", 
            "🗺️ Live Tracking", 
            "🔔 Notifications", 
            "👤 Profile", 
            "📊 Analytics", 
            "💬 Communication"
        ])
        
        with tab1:
            # Dashboard
            ParentPortalUI.render_quick_stats(selected_student)
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ParentPortalUI.render_live_tracking_map(selected_student)
            
            with col2:
                ParentPortalUI.render_schedule_timeline(selected_student)
                
                st.markdown("---")
                ParentPortalUI.render_weather_integration()
        
        with tab2:
            # Live Tracking (focused view)
            ParentPortalUI.render_live_tracking_map(selected_student)
            
            # Route information
            st.markdown("---")
            route_data = ParentPortalData.get_route_data(selected_student.route_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📍 Route Information")
                st.write(f"**Route:** {route_data['route_name']}")
                st.write(f"**Total Stops:** {route_data['total_stops']}")
                st.write(f"**Distance:** {route_data['total_distance']} km")
                st.write(f"**Duration:** ~{route_data['average_duration']} minutes")
            
            with col2:
                st.subheader("🚌 Alternative Options")
                for alt in route_data['alternative_routes']:
                    with st.expander(alt['name']):
                        st.write(f"**Description:** {alt['description']}")
                        st.write(f"**Duration:** {alt['duration']}")
                        st.write(f"**Cost:** {alt['cost']}")
                        st.write(f"**Walking:** {alt['walking_distance']}")
        
        with tab3:
            # Notifications
            ParentPortalUI.render_notifications_panel()
        
        with tab4:
            # Student Profile
            ParentPortalUI.render_student_profile(selected_student)
        
        with tab5:
            # Analytics
            ParentPortalUI.render_travel_analytics(selected_student)
        
        with tab6:
            # Communication
            ParentPortalUI.render_communication_center()
        
        # Auto-refresh every 30 seconds for live data
        if st.session_state.get("auto_refresh", True):
            time.sleep(1)  # Small delay to prevent too frequent updates
            if st.button("🔄 Refresh Live Data", key="refresh_button"):
                st.cache_data.clear()  # Clear cached data
                st.rerun()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Parent portal error: {e}", exc_info=True)


if __name__ == "__main__":
    render_parent_portal()