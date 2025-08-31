"""
Interactive Mapping System for SmartSchoolGo

This module provides comprehensive mapping capabilities using Folium for interactive
visualizations including real-time vehicle tracking, route visualization, safety
heatmaps, catchment area analysis, and demographic data overlays.

Features:
- Multi-layer interactive maps with dynamic controls
- Real-time vehicle tracking with animated markers
- Route visualization with performance metrics
- Safety heatmaps and incident overlays
- Catchment area visualization with demographic data
- Custom marker clusters and popup information panels
- Export capabilities (PNG, HTML, PDF formats)
- Mobile-responsive design with touch controls
- Performance optimization for large datasets

Author: SmartSchoolGo Team
Version: 1.0.0
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import base64

import numpy as np
import pandas as pd
from PIL import Image

try:
    import folium
    from folium import plugins
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString
    import branca.colormap as cm
    import requests
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import plotly.express as px
    import plotly.graph_objects as go
    MAPPING_LIBS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mapping libraries not available: {e}")
    MAPPING_LIBS_AVAILABLE = False

from ..data.models import (
    Coordinate, Route, Stop, Vehicle, School, Student,
    VehiclePosition, SafetyRiskLevel
)


class MapLayer(Enum):
    """Available map layers."""
    VEHICLES = "vehicles"
    ROUTES = "routes"
    STOPS = "stops"
    SCHOOLS = "schools"
    STUDENTS = "students"
    CATCHMENTS = "catchments"
    SAFETY_HEATMAP = "safety_heatmap"
    TRAFFIC = "traffic"
    DEMOGRAPHICS = "demographics"
    INCIDENTS = "incidents"


class MarkerStyle(Enum):
    """Marker style options."""
    DEFAULT = "default"
    CIRCLE = "circle"
    CUSTOM_ICON = "custom"
    CLUSTER = "cluster"


class ExportFormat(Enum):
    """Export format options."""
    HTML = "html"
    PNG = "png"
    PDF = "pdf"
    JPG = "jpg"


@dataclass
class MapConfig:
    """Configuration for interactive maps."""
    center_lat: float = -35.2809
    center_lon: float = 149.1300
    zoom_start: int = 12
    width: str = "100%"
    height: int = 600
    tiles: str = "OpenStreetMap"
    enable_clustering: bool = True
    cluster_max_zoom: int = 15
    max_markers: int = 1000
    update_interval: float = 5.0
    enable_heatmap: bool = True
    show_layer_control: bool = True
    enable_draw_tools: bool = False
    enable_measure_tools: bool = False
    custom_css: Optional[str] = None


@dataclass
class LayerData:
    """Data for a map layer."""
    layer_name: str
    data: List[Dict[str, Any]]
    style: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    clustered: bool = False
    popup_template: Optional[str] = None
    tooltip_template: Optional[str] = None


class InteractiveMap:
    """Advanced interactive mapping system with multiple layers."""
    
    def __init__(self, config: MapConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.InteractiveMap")
        
        # Map components
        self.map = None
        self.layers = {}
        self.layer_groups = {}
        self.markers = {}
        self.heatmaps = {}
        
        # Performance tracking
        self.last_update = None
        self.update_count = 0
        
        # Initialize map
        self._initialize_map()
    
    def _initialize_map(self):
        """Initialize the base map."""
        try:
            if not MAPPING_LIBS_AVAILABLE:
                raise ImportError("Folium not available")
            
            self.map = folium.Map(
                location=[self.config.center_lat, self.config.center_lon],
                zoom_start=self.config.zoom_start,
                width=self.config.width,
                height=self.config.height,
                tiles=self.config.tiles
            )
            
            # Add additional tile layers
            self._add_tile_layers()
            
            # Add measurement tools if enabled
            if self.config.enable_measure_tools:
                self._add_measure_tools()
            
            # Add drawing tools if enabled
            if self.config.enable_draw_tools:
                self._add_draw_tools()
            
            # Apply custom CSS if provided
            if self.config.custom_css:
                self._apply_custom_css()
            
            self.logger.info("Interactive map initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Map initialization failed: {e}")
            raise
    
    def _add_tile_layers(self):
        """Add additional tile layer options."""
        try:
            # Satellite imagery
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(self.map)
            
            # Traffic layer (simplified)
            folium.TileLayer(
                tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='Traffic Data',
                name='Traffic',
                overlay=True,
                control=True
            ).add_to(self.map)
            
            # Dark theme
            folium.TileLayer(
                tiles='https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
                attr='Stadia Maps',
                name='Dark Theme',
                overlay=False,
                control=True
            ).add_to(self.map)
            
        except Exception as e:
            self.logger.warning(f"Failed to add tile layers: {e}")
    
    def _add_measure_tools(self):
        """Add measurement tools to the map."""
        try:
            # Distance measurement
            plugins.MeasureControl(
                primary_length_unit='kilometers',
                secondary_length_unit='miles',
                primary_area_unit='sqkilometers',
                secondary_area_unit='acres'
            ).add_to(self.map)
            
        except Exception as e:
            self.logger.warning(f"Failed to add measure tools: {e}")
    
    def _add_draw_tools(self):
        """Add drawing tools to the map."""
        try:
            # Drawing tools
            plugins.Draw(
                export=True,
                filename='map_drawing.geojson',
                position='topleft'
            ).add_to(self.map)
            
        except Exception as e:
            self.logger.warning(f"Failed to add draw tools: {e}")
    
    def _apply_custom_css(self):
        """Apply custom CSS styling to the map."""
        try:
            css_element = folium.Element(f"""
            <style>
            {self.config.custom_css}
            </style>
            """)
            self.map.get_root().header.add_child(css_element)
            
        except Exception as e:
            self.logger.warning(f"Failed to apply custom CSS: {e}")
    
    def add_layer(self, layer_data: LayerData) -> bool:
        """Add a data layer to the map."""
        try:
            layer_name = layer_data.layer_name
            
            # Create feature group for the layer
            feature_group = folium.FeatureGroup(
                name=layer_name,
                show=layer_data.visible
            )
            
            # Add data points to the layer
            markers = []
            for item in layer_data.data:
                marker = self._create_marker(item, layer_data)
                if marker:
                    if layer_data.clustered and self.config.enable_clustering:
                        markers.append(marker)
                    else:
                        marker.add_to(feature_group)
            
            # Add clustering if enabled for this layer
            if markers and layer_data.clustered:
                marker_cluster = plugins.MarkerCluster(
                    markers,
                    name=f"{layer_name}_cluster",
                    maxClusterRadius=50,
                    disableClusteringAtZoom=self.config.cluster_max_zoom
                ).add_to(feature_group)
            
            # Store layer references
            self.layer_groups[layer_name] = feature_group
            self.layers[layer_name] = layer_data
            
            # Add to map
            feature_group.add_to(self.map)
            
            self.logger.info(f"Layer '{layer_name}' added with {len(layer_data.data)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add layer '{layer_data.layer_name}': {e}")
            return False
    
    def _create_marker(self, item: Dict[str, Any], layer_data: LayerData) -> Optional[folium.Marker]:
        """Create a marker for a data item."""
        try:
            # Extract coordinates
            lat = item.get('latitude', item.get('lat'))
            lon = item.get('longitude', item.get('lon', item.get('lng')))
            
            if lat is None or lon is None:
                return None
            
            # Create popup content
            popup_content = self._generate_popup_content(item, layer_data.popup_template)
            
            # Create tooltip content
            tooltip_content = self._generate_tooltip_content(item, layer_data.tooltip_template)
            
            # Determine marker style
            icon = self._get_marker_icon(item, layer_data.style)
            
            # Create marker
            marker = folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300) if popup_content else None,
                tooltip=tooltip_content if tooltip_content else None,
                icon=icon
            )
            
            return marker
            
        except Exception as e:
            self.logger.error(f"Failed to create marker: {e}")
            return None
    
    def _generate_popup_content(self, item: Dict[str, Any], template: Optional[str]) -> str:
        """Generate popup content for a marker."""
        if template:
            try:
                return template.format(**item)
            except KeyError as e:
                self.logger.warning(f"Template key error: {e}")
        
        # Default popup content
        content_lines = []
        
        # Add title if available
        title = item.get('name', item.get('id', item.get('vehicle_id', 'Item')))
        content_lines.append(f"<h4>{title}</h4>")
        
        # Add key-value pairs
        for key, value in item.items():
            if key not in ['latitude', 'longitude', 'lat', 'lon', 'lng', 'name', 'id']:
                if isinstance(value, (str, int, float)):
                    formatted_key = key.replace('_', ' ').title()
                    content_lines.append(f"<b>{formatted_key}:</b> {value}<br>")
        
        return "".join(content_lines)
    
    def _generate_tooltip_content(self, item: Dict[str, Any], template: Optional[str]) -> str:
        """Generate tooltip content for a marker."""
        if template:
            try:
                return template.format(**item)
            except KeyError:
                pass
        
        # Default tooltip
        name = item.get('name', item.get('id', item.get('vehicle_id', 'Item')))
        status = item.get('status', item.get('state', ''))
        
        if status:
            return f"{name} - {status}"
        else:
            return str(name)
    
    def _get_marker_icon(self, item: Dict[str, Any], style: Dict[str, Any]) -> folium.Icon:
        """Get appropriate icon for a marker."""
        # Default icon properties
        icon_props = {
            'color': 'blue',
            'icon': 'info-sign',
            'prefix': 'glyphicon'
        }
        
        # Update with layer style
        icon_props.update(style)
        
        # Item-specific styling
        if 'color' in item:
            icon_props['color'] = item['color']
        
        if 'icon' in item:
            icon_props['icon'] = item['icon']
        
        # Vehicle-specific icons
        if 'vehicle_id' in item:
            icon_props.update({
                'color': 'green' if item.get('status') == 'Active' else 'red',
                'icon': 'road',
                'prefix': 'fa'
            })
        
        # School-specific icons
        if 'school_name' in item:
            icon_props.update({
                'color': 'darkblue',
                'icon': 'graduation-cap',
                'prefix': 'fa'
            })
        
        # Stop-specific icons
        if 'stop_type' in item:
            icon_props.update({
                'color': 'orange',
                'icon': 'map-marker',
                'prefix': 'fa'
            })
        
        return folium.Icon(**icon_props)
    
    def add_vehicle_layer(self, vehicles: List[Dict[str, Any]]) -> bool:
        """Add vehicle tracking layer with real-time updates."""
        try:
            # Prepare vehicle data
            vehicle_data = []
            for vehicle in vehicles:
                vehicle_item = vehicle.copy()
                
                # Add popup template
                vehicle_item['popup_template'] = """
                <div style='width: 250px; font-family: Arial, sans-serif;'>
                    <h4 style='margin: 0 0 10px 0; color: #2c3e50;'>{vehicle_id}</h4>
                    <table style='width: 100%; font-size: 12px;'>
                        <tr><td><b>Route:</b></td><td>{route_id}</td></tr>
                        <tr><td><b>Speed:</b></td><td>{speed} km/h</td></tr>
                        <tr><td><b>Occupancy:</b></td><td>{occupancy}/50</td></tr>
                        <tr><td><b>Status:</b></td><td>{status}</td></tr>
                        <tr><td><b>Last Update:</b></td><td>{last_update}</td></tr>
                    </table>
                </div>
                """
                
                vehicle_data.append(vehicle_item)
            
            layer_data = LayerData(
                layer_name="Vehicles",
                data=vehicle_data,
                style={
                    'color': 'green',
                    'icon': 'bus',
                    'prefix': 'fa'
                },
                clustered=len(vehicle_data) > 20,
                popup_template="{popup_template}",
                tooltip_template="{vehicle_id} - {route_id}"
            )
            
            return self.add_layer(layer_data)
            
        except Exception as e:
            self.logger.error(f"Failed to add vehicle layer: {e}")
            return False
    
    def add_route_layer(self, routes: List[Dict[str, Any]], 
                       route_geometries: Optional[Dict[str, List[Coordinate]]] = None) -> bool:
        """Add route visualization layer with performance metrics."""
        try:
            # Add route lines if geometries provided
            if route_geometries:
                for route_id, coordinates in route_geometries.items():
                    route_info = next((r for r in routes if r.get('route_id') == route_id), {})
                    
                    # Determine line color based on performance
                    delay = route_info.get('avg_delay', 0)
                    if delay > 10:
                        color = 'red'
                    elif delay > 5:
                        color = 'orange'
                    else:
                        color = 'green'
                    
                    # Create route line
                    coords = [[coord.latitude, coord.longitude] for coord in coordinates]
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=f"Route {route_id}<br>Avg Delay: {delay} min",
                        tooltip=f"Route {route_id}"
                    ).add_to(self.map)
            
            # Add route markers for stops/key points
            route_data = []
            for route in routes:
                route_item = route.copy()
                route_item['popup_template'] = """
                <div style='width: 200px;'>
                    <h4>{route_name}</h4>
                    <p><b>Active Vehicles:</b> {active_vehicles}</p>
                    <p><b>Average Delay:</b> {avg_delay} minutes</p>
                    <p><b>Total Students:</b> {total_students}</p>
                    <p><b>Status:</b> {status}</p>
                </div>
                """
                route_data.append(route_item)
            
            layer_data = LayerData(
                layer_name="Routes",
                data=route_data,
                style={
                    'color': 'blue',
                    'icon': 'route',
                    'prefix': 'fa'
                },
                popup_template="{popup_template}",
                tooltip_template="{route_name}"
            )
            
            return self.add_layer(layer_data)
            
        except Exception as e:
            self.logger.error(f"Failed to add route layer: {e}")
            return False
    
    def add_safety_heatmap(self, safety_data: List[Dict[str, Any]]) -> bool:
        """Add safety risk heatmap layer."""
        try:
            if not self.config.enable_heatmap:
                return False
            
            # Prepare heatmap data
            heat_data = []
            for item in safety_data:
                lat = item.get('latitude', item.get('lat'))
                lon = item.get('longitude', item.get('lon'))
                risk_score = item.get('risk_score', item.get('incidents', 1))
                
                if lat is not None and lon is not None:
                    heat_data.append([lat, lon, risk_score])
            
            if heat_data:
                # Create heatmap
                heatmap = plugins.HeatMap(
                    heat_data,
                    name="Safety Risk Heatmap",
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=15,
                    blur=10,
                    gradient={
                        0.0: 'green',
                        0.3: 'yellow', 
                        0.6: 'orange',
                        1.0: 'red'
                    }
                )
                
                heatmap.add_to(self.map)
                self.heatmaps['safety'] = heatmap
                
                self.logger.info(f"Safety heatmap added with {len(heat_data)} data points")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add safety heatmap: {e}")
            return False
    
    def add_catchment_areas(self, catchment_data: List[Dict[str, Any]]) -> bool:
        """Add school catchment area visualization."""
        try:
            for catchment in catchment_data:
                school_name = catchment.get('school_name', 'Unknown School')
                boundary = catchment.get('boundary_coordinates', [])
                
                if boundary:
                    # Create polygon for catchment area
                    coords = [[coord['lat'], coord['lon']] for coord in boundary]
                    
                    # Color based on school type or capacity
                    color = catchment.get('color', '#3498db')
                    
                    folium.Polygon(
                        coords,
                        color=color,
                        weight=2,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.3,
                        popup=f"""
                        <div>
                            <h4>{school_name} Catchment</h4>
                            <p><b>Capacity:</b> {catchment.get('capacity', 'N/A')}</p>
                            <p><b>Current Enrollment:</b> {catchment.get('enrollment', 'N/A')}</p>
                            <p><b>Transport Required:</b> {catchment.get('transport_students', 'N/A')}</p>
                        </div>
                        """,
                        tooltip=f"{school_name} Catchment"
                    ).add_to(self.map)
            
            self.logger.info(f"Added {len(catchment_data)} catchment areas")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add catchment areas: {e}")
            return False
    
    def add_demographic_overlay(self, demographic_data: Dict[str, Any]) -> bool:
        """Add demographic data overlay."""
        try:
            # This would typically use census or demographic API data
            # Simplified implementation for demonstration
            
            demographic_layer = folium.FeatureGroup(name="Demographics")
            
            # Add demographic markers/polygons based on data
            areas = demographic_data.get('areas', [])
            for area in areas:
                center_lat = area.get('center_lat')
                center_lon = area.get('center_lon')
                
                if center_lat and center_lon:
                    # Create circle marker sized by population
                    population = area.get('population', 0)
                    radius = min(max(population / 100, 5), 50)  # Scale radius
                    
                    folium.CircleMarker(
                        location=[center_lat, center_lon],
                        radius=radius,
                        popup=f"""
                        <div>
                            <h4>{area.get('name', 'Area')}</h4>
                            <p><b>Population:</b> {population:,}</p>
                            <p><b>School Age (5-17):</b> {area.get('school_age', 'N/A'):,}</p>
                            <p><b>Households:</b> {area.get('households', 'N/A'):,}</p>
                            <p><b>Median Income:</b> ${area.get('median_income', 'N/A'):,}</p>
                        </div>
                        """,
                        color='purple',
                        fillColor='purple',
                        fillOpacity=0.6
                    ).add_to(demographic_layer)
            
            demographic_layer.add_to(self.map)
            self.layer_groups['demographics'] = demographic_layer
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add demographic overlay: {e}")
            return False
    
    def update_layer(self, layer_name: str, new_data: List[Dict[str, Any]]) -> bool:
        """Update an existing layer with new data."""
        try:
            if layer_name not in self.layers:
                self.logger.warning(f"Layer '{layer_name}' not found for update")
                return False
            
            # Remove existing layer
            if layer_name in self.layer_groups:
                self.map.keep_in_front(self.layer_groups[layer_name])
                # In Folium, we need to recreate the layer
                old_layer = self.layer_groups[layer_name]
                self.map._children.pop(old_layer._name, None)
            
            # Update layer data
            layer_data = self.layers[layer_name]
            layer_data.data = new_data
            
            # Re-add the layer
            return self.add_layer(layer_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update layer '{layer_name}': {e}")
            return False
    
    def toggle_layer_visibility(self, layer_name: str) -> bool:
        """Toggle visibility of a layer."""
        try:
            if layer_name in self.layers:
                layer = self.layers[layer_name]
                layer.visible = not layer.visible
                
                # Update layer group visibility
                if layer_name in self.layer_groups:
                    layer_group = self.layer_groups[layer_name]
                    if layer.visible:
                        layer_group.add_to(self.map)
                    else:
                        # Remove from map (simplified approach)
                        pass
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to toggle layer visibility: {e}")
            return False
    
    def add_layer_control(self):
        """Add layer control widget to the map."""
        try:
            if self.config.show_layer_control:
                folium.LayerControl(
                    position='topright',
                    collapsed=True,
                    autoZIndex=True
                ).add_to(self.map)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add layer control: {e}")
            return False
    
    def add_search_control(self):
        """Add search functionality to the map."""
        try:
            # Add search plugin (requires folium-plugins)
            search = plugins.Search(
                layer=self.layer_groups.get('vehicles', None),
                search=['vehicle_id', 'route_id'],
                placeholder='Search vehicles...',
                collapsed=True
            )
            search.add_to(self.map)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to add search control: {e}")
            return False
    
    def add_fullscreen_control(self):
        """Add fullscreen control to the map."""
        try:
            plugins.Fullscreen(
                position='topleft',
                title='Enter fullscreen mode',
                title_cancel='Exit fullscreen mode',
                force_separate_button=True
            ).add_to(self.map)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to add fullscreen control: {e}")
            return False
    
    def export_map(self, file_path: str, format: ExportFormat = ExportFormat.HTML) -> bool:
        """Export map to various formats."""
        try:
            if format == ExportFormat.HTML:
                # Save as HTML
                self.map.save(file_path)
                self.logger.info(f"Map exported as HTML to {file_path}")
                return True
            
            elif format in [ExportFormat.PNG, ExportFormat.JPG, ExportFormat.PDF]:
                # Requires selenium for screenshot
                return self._export_as_image(file_path, format)
            
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export map: {e}")
            return False
    
    def _export_as_image(self, file_path: str, format: ExportFormat) -> bool:
        """Export map as image using selenium."""
        try:
            if not MAPPING_LIBS_AVAILABLE:
                self.logger.error("Selenium not available for image export")
                return False
            
            # Save HTML temporarily
            temp_html = "temp_map.html"
            self.map.save(temp_html)
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"--window-size={self.config.width},{self.config.height}")
            
            # Take screenshot
            with webdriver.Chrome(options=chrome_options) as driver:
                driver.get(f"file://{Path(temp_html).absolute()}")
                time.sleep(2)  # Wait for map to load
                screenshot = driver.get_screenshot_as_png()
            
            # Save image
            image = Image.open(BytesIO(screenshot))
            
            if format == ExportFormat.PNG:
                image.save(file_path, 'PNG')
            elif format == ExportFormat.JPG:
                image = image.convert('RGB')
                image.save(file_path, 'JPEG')
            elif format == ExportFormat.PDF:
                image = image.convert('RGB')
                image.save(file_path, 'PDF')
            
            # Cleanup
            Path(temp_html).unlink(missing_ok=True)
            
            self.logger.info(f"Map exported as {format.value.upper()} to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export map as image: {e}")
            return False
    
    def get_map_html(self) -> str:
        """Get map as HTML string."""
        try:
            return self.map._repr_html_()
        except Exception as e:
            self.logger.error(f"Failed to get map HTML: {e}")
            return ""
    
    def get_map_bounds(self) -> Dict[str, float]:
        """Get current map bounds."""
        try:
            # This would typically come from JavaScript map state
            # Simplified implementation
            return {
                'north': self.config.center_lat + 0.1,
                'south': self.config.center_lat - 0.1,
                'east': self.config.center_lon + 0.1,
                'west': self.config.center_lon - 0.1
            }
        except Exception as e:
            self.logger.error(f"Failed to get map bounds: {e}")
            return {}
    
    def center_on_bounds(self, bounds: Dict[str, float]):
        """Center map on specified bounds."""
        try:
            center_lat = (bounds['north'] + bounds['south']) / 2
            center_lon = (bounds['east'] + bounds['west']) / 2
            
            # Update map center
            self.config.center_lat = center_lat
            self.config.center_lon = center_lon
            
            # Re-initialize map with new center
            self._initialize_map()
            
        except Exception as e:
            self.logger.error(f"Failed to center map on bounds: {e}")


class RealtimeVehicleTracker:
    """Real-time vehicle tracking with animated markers."""
    
    def __init__(self, interactive_map: InteractiveMap):
        self.map = interactive_map
        self.logger = logging.getLogger(f"{__name__}.RealtimeVehicleTracker")
        
        # Tracking state
        self.vehicle_positions = {}
        self.vehicle_trails = {}
        self.last_update = None
        
        # Animation settings
        self.trail_length = 10  # Number of positions to keep in trail
        self.update_interval = 5  # Seconds between updates
    
    def update_vehicle_positions(self, vehicles: List[Dict[str, Any]]) -> bool:
        """Update vehicle positions with trail animation."""
        try:
            current_time = datetime.now()
            
            # Process each vehicle
            for vehicle in vehicles:
                vehicle_id = vehicle.get('vehicle_id')
                if not vehicle_id:
                    continue
                
                # Store position with timestamp
                position = {
                    'latitude': vehicle.get('latitude'),
                    'longitude': vehicle.get('longitude'),
                    'timestamp': current_time,
                    'speed': vehicle.get('speed', 0),
                    'bearing': vehicle.get('bearing', 0),
                    'status': vehicle.get('status', 'Unknown')
                }
                
                # Update vehicle trail
                if vehicle_id not in self.vehicle_trails:
                    self.vehicle_trails[vehicle_id] = []
                
                self.vehicle_trails[vehicle_id].append(position)
                
                # Limit trail length
                if len(self.vehicle_trails[vehicle_id]) > self.trail_length:
                    self.vehicle_trails[vehicle_id] = self.vehicle_trails[vehicle_id][-self.trail_length:]
                
                # Store current position
                self.vehicle_positions[vehicle_id] = position
            
            # Update map layer
            return self._update_tracking_layer()
            
        except Exception as e:
            self.logger.error(f"Failed to update vehicle positions: {e}")
            return False
    
    def _update_tracking_layer(self) -> bool:
        """Update the tracking layer on the map."""
        try:
            # Prepare vehicle data for map
            vehicle_data = []
            
            for vehicle_id, position in self.vehicle_positions.items():
                if position['latitude'] and position['longitude']:
                    vehicle_item = {
                        'vehicle_id': vehicle_id,
                        'latitude': position['latitude'],
                        'longitude': position['longitude'],
                        'speed': position['speed'],
                        'bearing': position['bearing'],
                        'status': position['status'],
                        'last_update': position['timestamp'].strftime('%H:%M:%S'),
                        'trail_length': len(self.vehicle_trails.get(vehicle_id, []))
                    }
                    
                    # Add dynamic icon based on movement
                    if position['speed'] > 0:
                        vehicle_item['color'] = 'green'
                        vehicle_item['icon'] = 'play'
                    else:
                        vehicle_item['color'] = 'orange'
                        vehicle_item['icon'] = 'pause'
                    
                    vehicle_data.append(vehicle_item)
            
            # Update vehicle layer
            return self.map.update_layer('vehicles', vehicle_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update tracking layer: {e}")
            return False
    
    def add_vehicle_trails(self):
        """Add trail polylines for vehicle movement history."""
        try:
            for vehicle_id, trail in self.vehicle_trails.items():
                if len(trail) > 1:
                    # Create trail coordinates
                    trail_coords = [
                        [pos['latitude'], pos['longitude']]
                        for pos in trail
                        if pos['latitude'] and pos['longitude']
                    ]
                    
                    if len(trail_coords) > 1:
                        # Create gradient colors for trail (older = more transparent)
                        colors = ['red'] * len(trail_coords)  # Simplified - could use gradient
                        
                        folium.PolyLine(
                            trail_coords,
                            color='blue',
                            weight=2,
                            opacity=0.6,
                            popup=f"{vehicle_id} Trail",
                            tooltip=f"{vehicle_id} Movement History"
                        ).add_to(self.map.map)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vehicle trails: {e}")
            return False
    
    def get_vehicle_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked vehicles."""
        try:
            stats = {
                'total_vehicles': len(self.vehicle_positions),
                'active_vehicles': len([v for v in self.vehicle_positions.values() if v['speed'] > 0]),
                'stopped_vehicles': len([v for v in self.vehicle_positions.values() if v['speed'] == 0]),
                'avg_speed': np.mean([v['speed'] for v in self.vehicle_positions.values()]) if self.vehicle_positions else 0,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get vehicle statistics: {e}")
            return {}


class MapUtils:
    """Utility functions for mapping operations."""
    
    @staticmethod
    def calculate_bounds(coordinates: List[Tuple[float, float]], padding: float = 0.01) -> Dict[str, float]:
        """Calculate bounding box for a set of coordinates."""
        if not coordinates:
            return {}
        
        lats, lons = zip(*coordinates)
        
        return {
            'north': max(lats) + padding,
            'south': min(lats) - padding,
            'east': max(lons) + padding,
            'west': min(lons) - padding
        }
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers."""
        R = 6371  # Earth radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    @staticmethod
    def create_circle_coordinates(center_lat: float, center_lon: float, 
                                radius_km: float, num_points: int = 32) -> List[Tuple[float, float]]:
        """Create coordinates for a circle around a center point."""
        coordinates = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Calculate offset
            lat_offset = (radius_km / 111) * math.cos(angle)  # Approximate: 1 degree ≈ 111 km
            lon_offset = (radius_km / (111 * math.cos(math.radians(center_lat)))) * math.sin(angle)
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            
            coordinates.append((lat, lon))
        
        # Close the circle
        coordinates.append(coordinates[0])
        
        return coordinates
    
    @staticmethod
    def simplify_polyline(coordinates: List[Tuple[float, float]], tolerance: float = 0.001) -> List[Tuple[float, float]]:
        """Simplify polyline using Douglas-Peucker algorithm (simplified version)."""
        if len(coordinates) < 3:
            return coordinates
        
        # Find the point with maximum distance from the line segment
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(coordinates) - 1):
            distance = MapUtils._point_to_line_distance(
                coordinates[i], coordinates[0], coordinates[-1]
            )
            
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Simplify segments
            left_part = MapUtils.simplify_polyline(coordinates[:max_index + 1], tolerance)
            right_part = MapUtils.simplify_polyline(coordinates[max_index:], tolerance)
            
            # Combine results (remove duplicate point)
            return left_part[:-1] + right_part
        else:
            # Return endpoints only
            return [coordinates[0], coordinates[-1]]
    
    @staticmethod
    def _point_to_line_distance(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line segment."""
        # Simplified distance calculation
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate distance using cross product
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        if denominator == 0:
            return MapUtils.haversine_distance(x0, y0, x1, y1)
        
        return numerator / denominator
    
    @staticmethod
    def geocode_address(address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to coordinates (mock implementation)."""
        try:
            # In production, use actual geocoding service
            # Mock Canberra locations
            mock_locations = {
                "canberra": (-35.2809, 149.1300),
                "parliament house": (-35.3075, 149.1244),
                "anu": (-35.2777, 149.1185),
                "airport": (-35.3000, 149.1900),
                "civic": (-35.2809, 149.1300)
            }
            
            address_lower = address.lower()
            for location, coords in mock_locations.items():
                if location in address_lower:
                    return coords
            
            # Default to Canberra center
            return (-35.2809, 149.1300)
            
        except Exception as e:
            logging.error(f"Geocoding failed: {e}")
            return None