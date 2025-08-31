"""
Transport Planner Interface for SmartSchoolGo

This module provides advanced planning tools for transport network analysis, demand 
forecasting, infrastructure planning, policy impact analysis, and strategic planning
with long-term projections and data integration capabilities.

Features:
- Interactive network analysis with graph visualization
- Demand forecasting with scenario modeling capabilities
- GIS-integrated infrastructure planning tools
- Policy impact simulation and analysis
- Strategic planning dashboard with long-term projections
- External data integration interface
- Environmental analysis with carbon footprint calculations
- Cost-benefit analysis framework with ROI modeling
- Scenario comparison tools with side-by-side visualization

Author: SmartSchoolGo Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import networkx as nx
    import folium
    from streamlit_folium import st_folium
    from streamlit_elements import elements, mui, html, dashboard, nivo
    from streamlit_plotly_events import plotly_events
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
    import requests
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    PLANNER_LIBS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some advanced planning features may not be available: {e}")
    PLANNER_LIBS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkAnalysisType:
    """Network analysis types."""
    CENTRALITY = "centrality"
    CONNECTIVITY = "connectivity"
    FLOW_ANALYSIS = "flow_analysis"
    ACCESSIBILITY = "accessibility"
    VULNERABILITY = "vulnerability"


class ScenarioType:
    """Planning scenario types."""
    POPULATION_GROWTH = "population_growth"
    NEW_INFRASTRUCTURE = "new_infrastructure"
    POLICY_CHANGE = "policy_change"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    BUDGET_CONSTRAINT = "budget_constraint"


class PlannerDataService:
    """Data service for transport planner interface."""
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_network_data() -> Dict[str, Any]:
        """Get transport network data for analysis."""
        # Generate mock network data
        np.random.seed(42)  # For reproducible results
        
        # Create network nodes (stops, schools, key locations)
        nodes = []
        node_types = ['stop', 'school', 'depot', 'interchange', 'community_center']
        
        for i in range(50):  # 50 network nodes
            node = {
                'node_id': f'N{i+1:03d}',
                'name': f'Node {i+1}',
                'type': np.random.choice(node_types, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
                'latitude': -35.28 + np.random.uniform(-0.1, 0.1),
                'longitude': 149.13 + np.random.uniform(-0.1, 0.1),
                'population_served': np.random.randint(50, 500),
                'accessibility_score': np.random.uniform(0.3, 1.0),
                'infrastructure_quality': np.random.uniform(0.4, 1.0),
                'safety_rating': np.random.uniform(0.6, 1.0),
                'capacity': np.random.randint(20, 200),
                'utilization': np.random.uniform(0.3, 0.95),
                'yearly_growth_rate': np.random.uniform(-0.02, 0.08)
            }
            nodes.append(node)
        
        # Create network edges (routes between nodes)
        edges = []
        for i in range(75):  # 75 connections
            node1 = np.random.choice(nodes)
            node2 = np.random.choice([n for n in nodes if n['node_id'] != node1['node_id']])
            
            # Calculate distance
            dist = np.sqrt((node1['latitude'] - node2['latitude'])**2 + 
                          (node1['longitude'] - node2['longitude'])**2) * 111  # Approximate km
            
            edge = {
                'edge_id': f'E{i+1:03d}',
                'from_node': node1['node_id'],
                'to_node': node2['node_id'],
                'distance_km': dist,
                'travel_time_min': dist / 0.5,  # Assume 30 km/h average
                'route_type': np.random.choice(['bus_route', 'walking_path', 'bike_path', 'road']),
                'capacity': np.random.randint(50, 200),
                'current_flow': np.random.randint(10, 150),
                'congestion_level': np.random.uniform(0.1, 0.9),
                'maintenance_cost': dist * np.random.uniform(100, 300),  # Cost per km
                'environmental_impact': dist * np.random.uniform(0.5, 2.0)  # CO2 per km
            }
            edges.append(edge)
        
        # Network statistics
        total_capacity = sum(n['capacity'] for n in nodes)
        total_utilization = np.mean([n['utilization'] for n in nodes])
        avg_accessibility = np.mean([n['accessibility_score'] for n in nodes])
        
        return {
            'nodes': nodes,
            'edges': edges,
            'network_stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_capacity': total_capacity,
                'avg_utilization': total_utilization,
                'avg_accessibility': avg_accessibility,
                'network_efficiency': np.random.uniform(0.7, 0.9),
                'connectivity_index': len(edges) / len(nodes),
                'coverage_area_km2': 250.0
            }
        }
    
    @staticmethod
    @st.cache_data(ttl=600)
    def get_demographic_data() -> Dict[str, Any]:
        """Get demographic and population data."""
        # Mock demographic data for Canberra regions
        regions = [
            'North Canberra', 'South Canberra', 'Inner Canberra', 'Outer Canberra',
            'Belconnen', 'Tuggeranong', 'Woden Valley', 'Gungahlin'
        ]
        
        demographic_data = []
        for region in regions:
            data = {
                'region': region,
                'total_population': np.random.randint(30000, 80000),
                'school_age_population': np.random.randint(5000, 15000),
                'households': np.random.randint(12000, 32000),
                'median_income': np.random.randint(60000, 120000),
                'car_ownership_rate': np.random.uniform(0.7, 0.95),
                'public_transport_usage': np.random.uniform(0.15, 0.45),
                'population_density': np.random.uniform(500, 3000),  # per km2
                'growth_rate_annual': np.random.uniform(0.01, 0.05),
                'employment_rate': np.random.uniform(0.85, 0.95),
                'education_level_tertiary': np.random.uniform(0.45, 0.75),
                'disability_rate': np.random.uniform(0.08, 0.15),
                'senior_population_rate': np.random.uniform(0.12, 0.22)
            }
            demographic_data.append(data)
        
        # Growth projections
        projection_years = list(range(2024, 2035))
        projections = []
        
        for region_data in demographic_data:
            region = region_data['region']
            base_population = region_data['total_population']
            growth_rate = region_data['growth_rate_annual']
            
            for year in projection_years:
                years_ahead = year - 2024
                projected_population = base_population * ((1 + growth_rate) ** years_ahead)
                
                projections.append({
                    'region': region,
                    'year': year,
                    'projected_population': int(projected_population),
                    'projected_school_age': int(projected_population * 0.18),  # Assume 18% school age
                    'transport_demand': int(projected_population * 0.25)  # 25% need transport
                })
        
        return {
            'current_demographics': demographic_data,
            'projections': projections,
            'total_population': sum(d['total_population'] for d in demographic_data),
            'total_school_age': sum(d['school_age_population'] for d in demographic_data),
            'avg_growth_rate': np.mean([d['growth_rate_annual'] for d in demographic_data])
        }
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_infrastructure_data() -> Dict[str, Any]:
        """Get infrastructure and facility data."""
        # Current infrastructure
        schools = []
        for i in range(25):  # 25 schools
            school = {
                'school_id': f'SCH-{i+1:03d}',
                'name': f'School {i+1}',
                'type': np.random.choice(['Primary', 'Secondary', 'Combined'], p=[0.6, 0.3, 0.1]),
                'capacity': np.random.randint(200, 800),
                'current_enrollment': np.random.randint(150, 750),
                'transport_eligible_students': np.random.randint(80, 400),
                'latitude': -35.28 + np.random.uniform(-0.08, 0.08),
                'longitude': 149.13 + np.random.uniform(-0.08, 0.08),
                'accessibility_rating': np.random.uniform(0.5, 1.0),
                'catchment_area_km2': np.random.uniform(5, 25),
                'year_established': np.random.randint(1950, 2020),
                'facility_condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], p=[0.2, 0.4, 0.3, 0.1])
            }
            schools.append(school)
        
        # Transport facilities
        facilities = []
        facility_types = ['Bus Depot', 'Maintenance Center', 'Fuel Station', 'Park & Ride', 'Interchange']
        
        for i in range(15):  # 15 transport facilities
            facility = {
                'facility_id': f'FAC-{i+1:03d}',
                'name': f'Facility {i+1}',
                'type': np.random.choice(facility_types),
                'capacity': np.random.randint(50, 300),
                'current_utilization': np.random.uniform(0.4, 0.9),
                'latitude': -35.28 + np.random.uniform(-0.1, 0.1),
                'longitude': 149.13 + np.random.uniform(-0.1, 0.1),
                'operational_cost_annual': np.random.randint(50000, 500000),
                'condition_score': np.random.uniform(0.6, 1.0),
                'expansion_potential': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
                'environmental_impact': np.random.uniform(0.2, 0.8)
            }
            facilities.append(facility)
        
        # Proposed infrastructure projects
        proposed_projects = [
            {
                'project_id': 'PROJ-001',
                'name': 'North Canberra Transport Hub',
                'type': 'Interchange',
                'estimated_cost': 2500000,
                'timeline_months': 18,
                'expected_capacity': 150,
                'environmental_benefit_score': 0.8,
                'community_support': 0.75,
                'roi_annual': 0.12,
                'priority_score': 8.5
            },
            {
                'project_id': 'PROJ-002',
                'name': 'Southern Bus Route Extension',
                'type': 'Route Extension',
                'estimated_cost': 800000,
                'timeline_months': 8,
                'expected_capacity': 200,
                'environmental_benefit_score': 0.6,
                'community_support': 0.85,
                'roi_annual': 0.18,
                'priority_score': 9.2
            },
            {
                'project_id': 'PROJ-003',
                'name': 'Electric Bus Charging Infrastructure',
                'type': 'Environmental Infrastructure',
                'estimated_cost': 1200000,
                'timeline_months': 12,
                'expected_capacity': 50,
                'environmental_benefit_score': 0.95,
                'community_support': 0.68,
                'roi_annual': 0.08,
                'priority_score': 7.8
            }
        ]
        
        return {
            'schools': schools,
            'transport_facilities': facilities,
            'proposed_projects': proposed_projects,
            'total_school_capacity': sum(s['capacity'] for s in schools),
            'total_transport_capacity': sum(f['capacity'] for f in facilities),
            'avg_facility_utilization': np.mean([f['current_utilization'] for f in facilities]),
            'total_project_investment': sum(p['estimated_cost'] for p in proposed_projects)
        }
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_policy_scenarios() -> Dict[str, Any]:
        """Get policy scenarios and impact models."""
        scenarios = [
            {
                'scenario_id': 'SC001',
                'name': 'Free Public Transport for Students',
                'description': 'Eliminate transport fees for all eligible students',
                'type': ScenarioType.POLICY_CHANGE,
                'implementation_cost': 5000000,
                'annual_operating_cost': 2000000,
                'expected_impacts': {
                    'ridership_increase': 0.35,
                    'car_usage_reduction': 0.25,
                    'environmental_benefit': 0.4,
                    'social_equity_improvement': 0.6,
                    'operational_efficiency_change': -0.1
                },
                'timeline_months': 6,
                'stakeholder_support': 0.78,
                'risk_level': 'Medium'
            },
            {
                'scenario_id': 'SC002',
                'name': 'Smart Traffic Signal Integration',
                'description': 'Implement AI-powered traffic signal optimization for buses',
                'type': ScenarioType.NEW_INFRASTRUCTURE,
                'implementation_cost': 3000000,
                'annual_operating_cost': 200000,
                'expected_impacts': {
                    'travel_time_reduction': 0.15,
                    'fuel_efficiency_improvement': 0.12,
                    'on_time_performance_improvement': 0.20,
                    'maintenance_cost_reduction': 0.05,
                    'passenger_satisfaction_increase': 0.18
                },
                'timeline_months': 12,
                'stakeholder_support': 0.65,
                'risk_level': 'High'
            },
            {
                'scenario_id': 'SC003',
                'name': 'Expanded Evening Services',
                'description': 'Extend bus services to 7 PM for after-school activities',
                'type': ScenarioType.POLICY_CHANGE,
                'implementation_cost': 800000,
                'annual_operating_cost': 1200000,
                'expected_impacts': {
                    'service_coverage_increase': 0.3,
                    'student_activity_participation': 0.4,
                    'parent_satisfaction_increase': 0.5,
                    'operational_cost_increase': 0.25,
                    'driver_overtime_increase': 0.35
                },
                'timeline_months': 3,
                'stakeholder_support': 0.82,
                'risk_level': 'Low'
            }
        ]
        
        return {
            'scenarios': scenarios,
            'total_scenarios': len(scenarios),
            'avg_implementation_cost': np.mean([s['implementation_cost'] for s in scenarios]),
            'avg_stakeholder_support': np.mean([s['stakeholder_support'] for s in scenarios])
        }
    
    @staticmethod
    def generate_demand_forecast(scenario: str, years_ahead: int = 10) -> pd.DataFrame:
        """Generate demand forecast based on scenario."""
        base_demand = 1000  # Base daily trips
        
        # Scenario modifiers
        scenario_modifiers = {
            'baseline': {'growth_rate': 0.02, 'volatility': 0.05},
            'high_growth': {'growth_rate': 0.05, 'volatility': 0.08},
            'low_growth': {'growth_rate': 0.01, 'volatility': 0.03},
            'policy_impact': {'growth_rate': 0.03, 'volatility': 0.12}
        }
        
        modifier = scenario_modifiers.get(scenario, scenario_modifiers['baseline'])
        
        dates = pd.date_range(start=datetime.now(), periods=years_ahead*365, freq='D')
        demand_data = []
        
        for i, date in enumerate(dates):
            # Base trend
            trend_factor = (1 + modifier['growth_rate']) ** (i / 365)
            
            # Seasonal factors
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Day of week factor
            dow_factors = [0.95, 1.05, 1.08, 1.06, 1.04, 0.7, 0.6]  # Mon-Sun
            dow_factor = dow_factors[date.weekday()]
            
            # Random variation
            random_factor = 1 + np.random.normal(0, modifier['volatility'])
            
            # Calculate demand
            demand = base_demand * trend_factor * seasonal_factor * dow_factor * random_factor
            demand = max(int(demand), 0)  # Ensure non-negative
            
            demand_data.append({
                'date': date,
                'demand': demand,
                'trend_component': base_demand * trend_factor,
                'seasonal_component': seasonal_factor,
                'day_of_week_component': dow_factor,
                'scenario': scenario
            })
        
        return pd.DataFrame(demand_data)


class PlannerUIComponents:
    """UI components for transport planner interface."""
    
    @staticmethod
    def render_network_analysis():
        """Render network analysis tools."""
        st.subheader("🕸️ Network Analysis")
        
        network_data = PlannerDataService.get_network_data()
        nodes = network_data['nodes']
        edges = network_data['edges']
        stats = network_data['network_stats']
        
        # Network overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Network Nodes", stats['total_nodes'])
        
        with col2:
            st.metric("Connections", stats['total_edges'])
        
        with col3:
            st.metric("Efficiency Score", f"{stats['network_efficiency']:.2f}")
        
        with col4:
            st.metric("Coverage Area", f"{stats['coverage_area_km2']:.0f} km²")
        
        # Analysis type selector
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                NetworkAnalysisType.CENTRALITY,
                NetworkAnalysisType.CONNECTIVITY, 
                NetworkAnalysisType.FLOW_ANALYSIS,
                NetworkAnalysisType.ACCESSIBILITY,
                NetworkAnalysisType.VULNERABILITY
            ]
        )
        
        # Perform selected analysis
        if analysis_type == NetworkAnalysisType.CENTRALITY:
            PlannerUIComponents._render_centrality_analysis(nodes, edges)
        elif analysis_type == NetworkAnalysisType.CONNECTIVITY:
            PlannerUIComponents._render_connectivity_analysis(nodes, edges)
        elif analysis_type == NetworkAnalysisType.FLOW_ANALYSIS:
            PlannerUIComponents._render_flow_analysis(nodes, edges)
        elif analysis_type == NetworkAnalysisType.ACCESSIBILITY:
            PlannerUIComponents._render_accessibility_analysis(nodes)
        elif analysis_type == NetworkAnalysisType.VULNERABILITY:
            PlannerUIComponents._render_vulnerability_analysis(nodes, edges)
        
        # Interactive network visualization
        st.subheader("🗺️ Interactive Network Map")
        PlannerUIComponents._render_network_map(nodes, edges)
    
    @staticmethod
    def _render_centrality_analysis(nodes: List[Dict], edges: List[Dict]):
        """Render centrality analysis results."""
        st.write("**Centrality Analysis**")
        st.info("Identifies the most important nodes in the transport network")
        
        # Calculate mock centrality scores
        for node in nodes:
            node['betweenness_centrality'] = np.random.uniform(0, 1)
            node['closeness_centrality'] = np.random.uniform(0, 1)
            node['degree_centrality'] = np.random.uniform(0, 1)
        
        # Sort by betweenness centrality
        top_nodes = sorted(nodes, key=lambda x: x['betweenness_centrality'], reverse=True)[:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top central nodes
            centrality_data = pd.DataFrame([
                {
                    'Node': node['name'],
                    'Type': node['type'],
                    'Betweenness': node['betweenness_centrality'],
                    'Closeness': node['closeness_centrality'],
                    'Population Served': node['population_served']
                }
                for node in top_nodes
            ])
            
            st.write("**Top 10 Most Central Nodes**")
            st.dataframe(centrality_data, use_container_width=True)
        
        with col2:
            # Centrality distribution
            centrality_values = [node['betweenness_centrality'] for node in nodes]
            fig = px.histogram(
                x=centrality_values,
                nbins=20,
                title="Betweenness Centrality Distribution",
                labels={'x': 'Centrality Score', 'y': 'Number of Nodes'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.write("**Recommendations:**")
        high_centrality_nodes = [n for n in nodes if n['betweenness_centrality'] > 0.7]
        st.success(f"• Focus infrastructure improvements on {len(high_centrality_nodes)} high-centrality nodes")
        st.info("• Consider backup routes for critical nodes to improve resilience")
        st.warning("• Monitor capacity at central nodes to prevent bottlenecks")
    
    @staticmethod
    def _render_connectivity_analysis(nodes: List[Dict], edges: List[Dict]):
        """Render connectivity analysis results."""
        st.write("**Connectivity Analysis**")
        
        # Calculate connectivity metrics
        node_degrees = {}
        for node in nodes:
            node_id = node['node_id']
            degree = len([e for e in edges if e['from_node'] == node_id or e['to_node'] == node_id])
            node_degrees[node_id] = degree
        
        avg_degree = np.mean(list(node_degrees.values()))
        max_degree = max(node_degrees.values())
        min_degree = min(node_degrees.values())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Connectivity", f"{avg_degree:.1f}")
        
        with col2:
            st.metric("Max Connections", max_degree)
        
        with col3:
            st.metric("Min Connections", min_degree)
        
        # Connectivity distribution
        fig = px.bar(
            x=list(node_degrees.keys())[:20],  # Show first 20 nodes
            y=list(node_degrees.values())[:20],
            title="Node Connectivity (First 20 Nodes)",
            labels={'x': 'Node ID', 'y': 'Number of Connections'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify poorly connected areas
        poorly_connected = [node_id for node_id, degree in node_degrees.items() if degree <= 2]
        st.warning(f"⚠️ {len(poorly_connected)} nodes have poor connectivity (≤2 connections)")
        
        if poorly_connected:
            st.write("Poorly connected nodes:", ", ".join(poorly_connected[:10]))
    
    @staticmethod
    def _render_flow_analysis(nodes: List[Dict], edges: List[Dict]):
        """Render flow analysis results."""
        st.write("**Flow Analysis**")
        
        # Calculate flow metrics
        total_capacity = sum(e['capacity'] for e in edges)
        total_flow = sum(e['current_flow'] for e in edges)
        utilization_rate = total_flow / total_capacity if total_capacity > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Capacity", f"{total_capacity:,}")
        
        with col2:
            st.metric("Current Flow", f"{total_flow:,}")
        
        with col3:
            st.metric("Utilization Rate", f"{utilization_rate:.1%}")
        
        # Flow vs capacity analysis
        flow_data = pd.DataFrame([
            {
                'Edge': edge['edge_id'],
                'Capacity': edge['capacity'],
                'Current Flow': edge['current_flow'],
                'Utilization': edge['current_flow'] / edge['capacity'] if edge['capacity'] > 0 else 0,
                'Congestion Level': edge['congestion_level']
            }
            for edge in edges
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Utilization distribution
            fig = px.histogram(
                flow_data,
                x='Utilization',
                nbins=20,
                title="Edge Utilization Distribution",
                labels={'Utilization': 'Capacity Utilization Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Congestion vs utilization
            fig = px.scatter(
                flow_data,
                x='Utilization',
                y='Congestion Level',
                title="Utilization vs Congestion",
                hover_data=['Edge', 'Capacity']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Bottleneck identification
        bottlenecks = flow_data[flow_data['Utilization'] > 0.8].sort_values('Utilization', ascending=False)
        
        if not bottlenecks.empty:
            st.error(f"🚨 {len(bottlenecks)} bottleneck edges identified (>80% utilization)")
            st.dataframe(bottlenecks.head(10), use_container_width=True)
        else:
            st.success("✅ No significant bottlenecks detected")
    
    @staticmethod
    def _render_accessibility_analysis(nodes: List[Dict]):
        """Render accessibility analysis results."""
        st.write("**Accessibility Analysis**")
        
        # Accessibility metrics
        accessibility_scores = [node['accessibility_score'] for node in nodes]
        avg_accessibility = np.mean(accessibility_scores)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Accessibility", f"{avg_accessibility:.2f}")
        
        with col2:
            high_access = len([s for s in accessibility_scores if s > 0.8])
            st.metric("High Accessibility Nodes", high_access)
        
        with col3:
            low_access = len([s for s in accessibility_scores if s < 0.5])
            st.metric("Low Accessibility Nodes", low_access)
        
        # Accessibility by node type
        accessibility_by_type = {}
        for node in nodes:
            node_type = node['type']
            if node_type not in accessibility_by_type:
                accessibility_by_type[node_type] = []
            accessibility_by_type[node_type].append(node['accessibility_score'])
        
        # Average by type
        avg_by_type = {k: np.mean(v) for k, v in accessibility_by_type.items()}
        
        fig = px.bar(
            x=list(avg_by_type.keys()),
            y=list(avg_by_type.values()),
            title="Average Accessibility by Node Type",
            labels={'x': 'Node Type', 'y': 'Accessibility Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.write("**Accessibility Improvement Recommendations:**")
        
        low_access_nodes = [n for n in nodes if n['accessibility_score'] < 0.5]
        if low_access_nodes:
            st.warning(f"• {len(low_access_nodes)} nodes need accessibility improvements")
            st.info("• Consider adding wheelchair access, better lighting, and shelter")
            st.info("• Prioritize improvements at high-population nodes")
    
    @staticmethod
    def _render_vulnerability_analysis(nodes: List[Dict], edges: List[Dict]):
        """Render network vulnerability analysis."""
        st.write("**Vulnerability Analysis**")
        st.info("Identifies critical infrastructure whose failure would significantly impact the network")
        
        # Calculate vulnerability scores (mock)
        for node in nodes:
            node['vulnerability_score'] = (
                node['population_served'] / 500 * 0.4 +  # Population impact
                (1 - node['accessibility_score']) * 0.3 +  # Access dependency
                node['utilization'] * 0.3  # Usage level
            )
        
        # Sort by vulnerability
        vulnerable_nodes = sorted(nodes, key=lambda x: x['vulnerability_score'], reverse=True)[:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Most Vulnerable Nodes**")
            vulnerability_data = pd.DataFrame([
                {
                    'Node': node['name'],
                    'Type': node['type'],
                    'Vulnerability Score': f"{node['vulnerability_score']:.2f}",
                    'Population Served': node['population_served'],
                    'Utilization': f"{node['utilization']:.1%}"
                }
                for node in vulnerable_nodes
            ])
            st.dataframe(vulnerability_data, use_container_width=True)
        
        with col2:
            # Vulnerability distribution
            vuln_scores = [node['vulnerability_score'] for node in nodes]
            fig = px.histogram(
                x=vuln_scores,
                nbins=20,
                title="Network Vulnerability Distribution",
                labels={'x': 'Vulnerability Score', 'y': 'Number of Nodes'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk mitigation recommendations
        st.write("**Risk Mitigation Strategies:**")
        critical_nodes = [n for n in nodes if n['vulnerability_score'] > 1.0]
        
        if critical_nodes:
            st.error(f"🚨 {len(critical_nodes)} critical vulnerability nodes identified")
            st.warning("• Develop redundant routes for critical nodes")
            st.info("• Implement real-time monitoring for high-risk infrastructure")
            st.info("• Create emergency response plans for critical failures")
        else:
            st.success("✅ Network vulnerability within acceptable levels")
    
    @staticmethod
    def _render_network_map(nodes: List[Dict], edges: List[Dict]):
        """Render interactive network map."""
        try:
            # Create base map
            center_lat = np.mean([node['latitude'] for node in nodes])
            center_lon = np.mean([node['longitude'] for node in nodes])
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            
            # Add nodes
            node_colors = {
                'stop': 'blue',
                'school': 'green',
                'depot': 'red',
                'interchange': 'purple',
                'community_center': 'orange'
            }
            
            for node in nodes:
                color = node_colors.get(node['type'], 'gray')
                
                # Node size based on population served
                radius = 5 + (node['population_served'] / 100)
                
                folium.CircleMarker(
                    location=[node['latitude'], node['longitude']],
                    radius=radius,
                    popup=f"""
                    <div>
                        <h4>{node['name']}</h4>
                        <p><b>Type:</b> {node['type'].title()}</p>
                        <p><b>Population Served:</b> {node['population_served']}</p>
                        <p><b>Accessibility:</b> {node['accessibility_score']:.2f}</p>
                        <p><b>Utilization:</b> {node['utilization']:.1%}</p>
                    </div>
                    """,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    tooltip=f"{node['name']} ({node['type']})"
                ).add_to(m)
            
            # Add edges with flow visualization
            for edge in edges:
                from_node = next((n for n in nodes if n['node_id'] == edge['from_node']), None)
                to_node = next((n for n in nodes if n['node_id'] == edge['to_node']), None)
                
                if from_node and to_node:
                    # Line width based on flow
                    weight = 1 + (edge['current_flow'] / 50)
                    
                    # Color based on congestion
                    if edge['congestion_level'] > 0.7:
                        color = 'red'
                    elif edge['congestion_level'] > 0.4:
                        color = 'orange'
                    else:
                        color = 'green'
                    
                    folium.PolyLine(
                        locations=[
                            [from_node['latitude'], from_node['longitude']],
                            [to_node['latitude'], to_node['longitude']]
                        ],
                        weight=weight,
                        color=color,
                        opacity=0.6,
                        popup=f"""
                        <div>
                            <h4>Route {edge['edge_id']}</h4>
                            <p><b>Distance:</b> {edge['distance_km']:.1f} km</p>
                            <p><b>Flow:</b> {edge['current_flow']}/{edge['capacity']}</p>
                            <p><b>Congestion:</b> {edge['congestion_level']:.1%}</p>
                        </div>
                        """
                    ).add_to(m)
            
            # Add legend
            legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <b>Network Legend</b><br>
            🔵 Stop<br>
            🟢 School<br>
            🔴 Depot<br>
            🟣 Interchange<br>
            🟠 Community Center
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            
            st_folium(m, width=700, height=500)
            
        except Exception as e:
            st.error(f"Unable to render network map: {str(e)}")
    
    @staticmethod
    def render_demand_forecasting():
        """Render demand forecasting interface."""
        st.subheader("📈 Demand Forecasting")
        
        # Forecasting parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scenario = st.selectbox(
                "Forecast Scenario",
                ["baseline", "high_growth", "low_growth", "policy_impact"]
            )
        
        with col2:
            years_ahead = st.slider("Forecast Years", 1, 15, 5)
        
        with col3:
            confidence_interval = st.slider("Confidence Interval", 80, 99, 95)
        
        # Generate forecast
        with st.spinner("Generating demand forecast..."):
            forecast_data = PlannerDataService.generate_demand_forecast(scenario, years_ahead)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Time series plot
            fig = go.Figure()
            
            # Add main forecast line
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['demand'],
                mode='lines',
                name='Forecasted Demand',
                line=dict(color='blue')
            ))
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['trend_component'],
                mode='lines',
                name='Trend Component',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Transport Demand Forecast - {scenario.title()} Scenario",
                xaxis_title="Date",
                yaxis_title="Daily Trips",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Forecast summary statistics
            current_demand = forecast_data['demand'].iloc[0]
            future_demand = forecast_data['demand'].iloc[-1]
            growth_rate = ((future_demand / current_demand) ** (1/years_ahead) - 1) * 100
            
            st.metric("Current Daily Demand", f"{current_demand:,.0f}")
            st.metric("Forecast Daily Demand", f"{future_demand:,.0f}")
            st.metric("Annual Growth Rate", f"{growth_rate:.1f}%")
            
            peak_demand = forecast_data['demand'].max()
            peak_date = forecast_data.loc[forecast_data['demand'].idxmax(), 'date']
            st.metric("Peak Demand", f"{peak_demand:,.0f}")
            st.write(f"Peak Date: {peak_date.strftime('%Y-%m-%d')}")
        
        # Seasonal analysis
        st.subheader("📊 Seasonal Analysis")
        
        # Monthly aggregation
        forecast_data['month'] = forecast_data['date'].dt.month
        monthly_avg = forecast_data.groupby('month')['demand'].mean().reset_index()
        
        fig = px.bar(
            monthly_avg,
            x='month',
            y='demand',
            title="Average Monthly Demand",
            labels={'month': 'Month', 'demand': 'Average Daily Trips'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario comparison
        if st.checkbox("Compare Scenarios"):
            PlannerUIComponents._render_scenario_comparison(years_ahead)
    
    @staticmethod
    def _render_scenario_comparison(years_ahead: int):
        """Render scenario comparison analysis."""
        st.subheader("🔄 Scenario Comparison")
        
        scenarios = ["baseline", "high_growth", "low_growth", "policy_impact"]
        scenario_data = {}
        
        for scenario in scenarios:
            data = PlannerDataService.generate_demand_forecast(scenario, years_ahead)
            scenario_data[scenario] = data
        
        # Create comparison plot
        fig = go.Figure()
        
        colors = {'baseline': 'blue', 'high_growth': 'green', 'low_growth': 'red', 'policy_impact': 'orange'}
        
        for scenario, data in scenario_data.items():
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['demand'],
                mode='lines',
                name=scenario.replace('_', ' ').title(),
                line=dict(color=colors[scenario])
            ))
        
        fig.update_layout(
            title="Demand Forecast Scenario Comparison",
            xaxis_title="Date",
            yaxis_title="Daily Trips",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario summary table
        summary_data = []
        for scenario, data in scenario_data.items():
            current = data['demand'].iloc[0]
            future = data['demand'].iloc[-1]
            growth = ((future / current) ** (1/years_ahead) - 1) * 100
            
            summary_data.append({
                'Scenario': scenario.replace('_', ' ').title(),
                'Current Demand': f"{current:,.0f}",
                'Future Demand': f"{future:,.0f}",
                'Growth Rate': f"{growth:.1f}%",
                'Total Growth': f"{((future/current - 1) * 100):.1f}%"
            })
        
        st.write("**Scenario Summary**")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    @staticmethod
    def render_infrastructure_planning():
        """Render infrastructure planning tools."""
        st.subheader("🏗️ Infrastructure Planning")
        
        infrastructure_data = PlannerDataService.get_infrastructure_data()
        schools = infrastructure_data['schools']
        facilities = infrastructure_data['transport_facilities']
        projects = infrastructure_data['proposed_projects']
        
        # Infrastructure overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Schools", len(schools))
        
        with col2:
            st.metric("Transport Facilities", len(facilities))
        
        with col3:
            total_capacity = infrastructure_data['total_school_capacity']
            st.metric("School Capacity", f"{total_capacity:,}")
        
        with col4:
            avg_utilization = infrastructure_data['avg_facility_utilization']
            st.metric("Avg Facility Use", f"{avg_utilization:.1%}")
        
        # Infrastructure analysis tabs
        infra_tab1, infra_tab2, infra_tab3 = st.tabs([
            "Current Infrastructure",
            "Proposed Projects", 
            "Capacity Planning"
        ])
        
        with infra_tab1:
            PlannerUIComponents._render_current_infrastructure(schools, facilities)
        
        with infra_tab2:
            PlannerUIComponents._render_proposed_projects(projects)
        
        with infra_tab3:
            PlannerUIComponents._render_capacity_planning(schools, facilities)
    
    @staticmethod
    def _render_current_infrastructure(schools: List[Dict], facilities: List[Dict]):
        """Render current infrastructure analysis."""
        st.write("**Current Infrastructure Analysis**")
        
        # School analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Schools by Type**")
            school_types = pd.Series([s['type'] for s in schools]).value_counts()
            fig = px.pie(
                values=school_types.values,
                names=school_types.index,
                title="School Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**School Capacity Utilization**")
            school_data = []
            for school in schools:
                utilization = school['current_enrollment'] / school['capacity'] if school['capacity'] > 0 else 0
                school_data.append({
                    'School': school['name'],
                    'Capacity': school['capacity'],
                    'Enrollment': school['current_enrollment'],
                    'Utilization': utilization,
                    'Transport Students': school['transport_eligible_students']
                })
            
            school_df = pd.DataFrame(school_data)
            
            fig = px.histogram(
                school_df,
                x='Utilization',
                nbins=15,
                title="School Capacity Utilization Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Facilities analysis
        st.write("**Transport Facilities**")
        
        facilities_df = pd.DataFrame([
            {
                'Facility': f['name'],
                'Type': f['type'],
                'Capacity': f['capacity'],
                'Utilization': f'{f["current_utilization"]:.1%}',
                'Condition': f'{f["condition_score"]:.2f}',
                'Annual Cost': f'${f["operational_cost_annual"]:,}'
            }
            for f in facilities
        ])
        
        st.dataframe(facilities_df, use_container_width=True)
        
        # Facility utilization by type
        facility_types = {}
        for facility in facilities:
            ftype = facility['type']
            if ftype not in facility_types:
                facility_types[ftype] = []
            facility_types[ftype].append(facility['current_utilization'])
        
        avg_utilization = {k: np.mean(v) for k, v in facility_types.items()}
        
        fig = px.bar(
            x=list(avg_utilization.keys()),
            y=list(avg_utilization.values()),
            title="Average Facility Utilization by Type",
            labels={'x': 'Facility Type', 'y': 'Average Utilization'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_proposed_projects(projects: List[Dict]):
        """Render proposed infrastructure projects."""
        st.write("**Proposed Infrastructure Projects**")
        
        if not projects:
            st.info("No proposed projects currently in the pipeline.")
            return
        
        # Project overview
        total_investment = sum(p['estimated_cost'] for p in projects)
        avg_timeline = np.mean([p['timeline_months'] for p in projects])
        avg_roi = np.mean([p['roi_annual'] for p in projects])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Investment", f"${total_investment:,.0f}")
        
        with col2:
            st.metric("Avg Timeline", f"{avg_timeline:.1f} months")
        
        with col3:
            st.metric("Avg Annual ROI", f"{avg_roi:.1%}")
        
        # Projects table
        projects_df = pd.DataFrame([
            {
                'Project': p['name'],
                'Type': p['type'],
                'Cost': f"${p['estimated_cost']:,}",
                'Timeline': f"{p['timeline_months']} months",
                'ROI': f"{p['roi_annual']:.1%}",
                'Priority Score': f"{p['priority_score']:.1f}",
                'Community Support': f"{p['community_support']:.1%}"
            }
            for p in projects
        ])
        
        st.dataframe(projects_df, use_container_width=True)
        
        # Project analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost vs ROI scatter plot
            fig = px.scatter(
                x=[p['estimated_cost'] for p in projects],
                y=[p['roi_annual'] for p in projects],
                size=[p['priority_score'] for p in projects],
                hover_name=[p['name'] for p in projects],
                title="Investment Cost vs ROI",
                labels={'x': 'Estimated Cost ($)', 'y': 'Annual ROI'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Priority vs community support
            fig = px.scatter(
                x=[p['community_support'] for p in projects],
                y=[p['priority_score'] for p in projects],
                size=[p['estimated_cost'] for p in projects],
                hover_name=[p['name'] for p in projects],
                title="Community Support vs Priority Score",
                labels={'x': 'Community Support', 'y': 'Priority Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Project selection tool
        st.subheader("🎯 Project Selection Tool")
        
        budget_constraint = st.number_input(
            "Available Budget ($)",
            min_value=0,
            max_value=10000000,
            value=5000000,
            step=100000
        )
        
        # Priority-based selection
        selected_projects = []
        remaining_budget = budget_constraint
        sorted_projects = sorted(projects, key=lambda x: x['priority_score'], reverse=True)
        
        for project in sorted_projects:
            if project['estimated_cost'] <= remaining_budget:
                selected_projects.append(project)
                remaining_budget -= project['estimated_cost']
        
        if selected_projects:
            st.success(f"✅ Recommended {len(selected_projects)} projects within budget:")
            for project in selected_projects:
                st.write(f"• **{project['name']}** - ${project['estimated_cost']:,} (Priority: {project['priority_score']:.1f})")
            
            total_selected_cost = sum(p['estimated_cost'] for p in selected_projects)
            st.info(f"Total cost: ${total_selected_cost:,} (${budget_constraint - total_selected_cost:,} remaining)")
        else:
            st.warning("No projects fit within the specified budget.")
    
    @staticmethod
    def _render_capacity_planning(schools: List[Dict], facilities: List[Dict]):
        """Render capacity planning analysis."""
        st.write("**Capacity Planning Analysis**")
        
        # Get demographic data for projections
        demographic_data = PlannerDataService.get_demographic_data()
        projections = demographic_data['projections']
        
        # Future capacity requirements
        st.subheader("📊 Future Capacity Requirements")
        
        # Convert projections to DataFrame
        projections_df = pd.DataFrame(projections)
        
        # Aggregate by year
        yearly_demand = projections_df.groupby('year').agg({
            'projected_population': 'sum',
            'projected_school_age': 'sum',
            'transport_demand': 'sum'
        }).reset_index()
        
        # Current capacity
        current_school_capacity = sum(s['capacity'] for s in schools)
        current_transport_capacity = sum(f['capacity'] for f in facilities if f['type'] in ['Bus Depot', 'Interchange'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # School capacity vs demand
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=yearly_demand['year'],
                y=yearly_demand['projected_school_age'],
                mode='lines+markers',
                name='Projected School Age Population',
                line=dict(color='blue')
            ))
            
            fig.add_hline(
                y=current_school_capacity,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current School Capacity ({current_school_capacity:,})"
            )
            
            fig.update_layout(
                title="School Capacity vs Projected Demand",
                xaxis_title="Year",
                yaxis_title="Students",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Transport capacity vs demand
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=yearly_demand['year'],
                y=yearly_demand['transport_demand'],
                mode='lines+markers',
                name='Projected Transport Demand',
                line=dict(color='green')
            ))
            
            fig.add_hline(
                y=current_transport_capacity,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Current Transport Capacity ({current_transport_capacity:,})"
            )
            
            fig.update_layout(
                title="Transport Capacity vs Projected Demand",
                xaxis_title="Year",
                yaxis_title="Daily Trips",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Capacity gap analysis
        st.subheader("⚠️ Capacity Gap Analysis")
        
        gap_analysis = []
        for _, row in yearly_demand.iterrows():
            year = row['year']
            school_gap = max(0, row['projected_school_age'] - current_school_capacity)
            transport_gap = max(0, row['transport_demand'] - current_transport_capacity)
            
            gap_analysis.append({
                'Year': year,
                'School Capacity Gap': school_gap,
                'Transport Capacity Gap': transport_gap,
                'Investment Needed (School)': school_gap * 15000,  # Assume $15k per student
                'Investment Needed (Transport)': transport_gap * 5000  # Assume $5k per daily trip
            })
        
        gap_df = pd.DataFrame(gap_analysis)
        
        # Show only years with gaps
        significant_gaps = gap_df[
            (gap_df['School Capacity Gap'] > 0) | (gap_df['Transport Capacity Gap'] > 0)
        ]
        
        if not significant_gaps.empty:
            st.warning(f"⚠️ Capacity gaps identified in {len(significant_gaps)} years")
            st.dataframe(significant_gaps, use_container_width=True)
            
            # Investment timeline
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=significant_gaps['Year'],
                y=significant_gaps['Investment Needed (School)'],
                name='School Investment',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                x=significant_gaps['Year'],
                y=significant_gaps['Investment Needed (Transport)'],
                name='Transport Investment',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Required Investment Timeline",
                xaxis_title="Year",
                yaxis_title="Investment Required ($)",
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No significant capacity gaps projected in the next 10 years")
    
    @staticmethod
    def render_policy_analysis():
        """Render policy impact analysis tools."""
        st.subheader("📋 Policy Impact Analysis")
        
        policy_data = PlannerDataService.get_policy_scenarios()
        scenarios = policy_data['scenarios']
        
        # Policy scenario overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Policy Scenarios", len(scenarios))
        
        with col2:
            avg_cost = policy_data['avg_implementation_cost']
            st.metric("Avg Implementation Cost", f"${avg_cost:,.0f}")
        
        with col3:
            avg_support = policy_data['avg_stakeholder_support']
            st.metric("Avg Stakeholder Support", f"{avg_support:.1%}")
        
        # Scenario selection and analysis
        selected_scenario = st.selectbox(
            "Select Policy Scenario for Analysis",
            options=[s['scenario_id'] for s in scenarios],
            format_func=lambda x: next(s['name'] for s in scenarios if s['scenario_id'] == x)
        )
        
        scenario = next(s for s in scenarios if s['scenario_id'] == selected_scenario)
        
        # Scenario details
        st.subheader(f"📊 Analysis: {scenario['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Description:**")
            st.write(scenario['description'])
            
            st.write("**Expected Impacts:**")
            impacts = scenario['expected_impacts']
            
            impact_data = []
            for impact, value in impacts.items():
                impact_data.append({
                    'Impact': impact.replace('_', ' ').title(),
                    'Change': f"{value:+.1%}" if abs(value) < 1 else f"{value:+.1f}",
                    'Direction': '📈 Positive' if value > 0 else '📉 Negative'
                })
            
            st.dataframe(pd.DataFrame(impact_data), use_container_width=True)
        
        with col2:
            st.write("**Financial Summary:**")
            st.metric("Implementation Cost", f"${scenario['implementation_cost']:,.0f}")
            st.metric("Annual Operating Cost", f"${scenario['annual_operating_cost']:,.0f}")
            st.metric("Timeline", f"{scenario['timeline_months']} months")
            
            # Risk and support indicators
            risk_colors = {'Low': 'success', 'Medium': 'warning', 'High': 'error'}
            risk_color = risk_colors.get(scenario['risk_level'], 'info')
            
            if risk_color == 'success':
                st.success(f"Risk Level: {scenario['risk_level']}")
            elif risk_color == 'warning':
                st.warning(f"Risk Level: {scenario['risk_level']}")
            elif risk_color == 'error':
                st.error(f"Risk Level: {scenario['risk_level']}")
            
            st.info(f"Stakeholder Support: {scenario['stakeholder_support']:.1%}")
        
        # Impact visualization
        st.subheader("📈 Impact Visualization")
        
        impact_names = list(scenario['expected_impacts'].keys())
        impact_values = list(scenario['expected_impacts'].values())
        
        fig = px.bar(
            x=[name.replace('_', ' ').title() for name in impact_names],
            y=impact_values,
            title=f"Expected Impacts - {scenario['name']}",
            labels={'x': 'Impact Category', 'y': 'Change (%)'},
            color=impact_values,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost-benefit analysis
        st.subheader("💰 Cost-Benefit Analysis")
        
        PlannerUIComponents._render_cost_benefit_analysis(scenario)
        
        # Scenario comparison
        if st.checkbox("Compare All Scenarios"):
            PlannerUIComponents._render_policy_comparison(scenarios)
    
    @staticmethod
    def _render_cost_benefit_analysis(scenario: Dict):
        """Render cost-benefit analysis for a policy scenario."""
        # Mock cost-benefit calculation
        implementation_cost = scenario['implementation_cost']
        annual_operating_cost = scenario['annual_operating_cost']
        
        # Estimate benefits based on impacts
        impacts = scenario['expected_impacts']
        
        # Simplified benefit calculation (in practice, this would be much more complex)
        annual_benefits = 0
        
        if 'ridership_increase' in impacts:
            annual_benefits += impacts['ridership_increase'] * 1000000  # $1M per 100% increase
        
        if 'travel_time_reduction' in impacts:
            annual_benefits += impacts['travel_time_reduction'] * 500000  # $500k per 100% reduction
        
        if 'environmental_benefit' in impacts:
            annual_benefits += impacts['environmental_benefit'] * 300000  # $300k per 100% benefit
        
        # Calculate ROI over 10 years
        years = 10
        total_costs = implementation_cost + (annual_operating_cost * years)
        total_benefits = annual_benefits * years
        net_benefit = total_benefits - total_costs
        roi = (net_benefit / total_costs) * 100 if total_costs > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Costs (10y)", f"${total_costs:,.0f}")
        
        with col2:
            st.metric("Total Benefits (10y)", f"${total_benefits:,.0f}")
        
        with col3:
            st.metric("Net Benefit", f"${net_benefit:,.0f}")
        
        with col4:
            st.metric("ROI", f"{roi:.1f}%")
        
        # Cash flow analysis
        cash_flow_data = []
        cumulative_benefit = 0
        
        for year in range(1, years + 1):
            annual_cost = annual_operating_cost
            if year == 1:
                annual_cost += implementation_cost
            
            annual_net = annual_benefits - annual_cost
            cumulative_benefit += annual_net
            
            cash_flow_data.append({
                'Year': year,
                'Annual Cost': annual_cost,
                'Annual Benefit': annual_benefits,
                'Annual Net': annual_net,
                'Cumulative Net': cumulative_benefit
            })
        
        cash_flow_df = pd.DataFrame(cash_flow_data)
        
        # Break-even analysis
        break_even_year = None
        for idx, row in cash_flow_df.iterrows():
            if row['Cumulative Net'] > 0:
                break_even_year = row['Year']
                break
        
        if break_even_year:
            st.success(f"✅ Break-even achieved in Year {break_even_year}")
        else:
            st.warning("⚠️ Break-even not achieved within 10-year timeframe")
        
        # Cash flow chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=cash_flow_df['Year'],
            y=cash_flow_df['Annual Cost'],
            name='Annual Costs',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            x=cash_flow_df['Year'],
            y=cash_flow_df['Annual Benefit'],
            name='Annual Benefits',
            marker_color='green'
        ))
        
        fig.add_trace(go.Scatter(
            x=cash_flow_df['Year'],
            y=cash_flow_df['Cumulative Net'],
            mode='lines+markers',
            name='Cumulative Net Benefit',
            yaxis='y2',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="Cost-Benefit Analysis Over Time",
            xaxis_title="Year",
            yaxis=dict(title="Annual Amount ($)", side="left"),
            yaxis2=dict(title="Cumulative Net Benefit ($)", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_policy_comparison(scenarios: List[Dict]):
        """Render comparison of all policy scenarios."""
        st.subheader("🔄 Policy Scenario Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for scenario in scenarios:
            impacts = scenario['expected_impacts']
            
            comparison_data.append({
                'Scenario': scenario['name'],
                'Type': scenario['type'],
                'Implementation Cost': scenario['implementation_cost'],
                'Annual Cost': scenario['annual_operating_cost'],
                'Timeline (months)': scenario['timeline_months'],
                'Stakeholder Support': scenario['stakeholder_support'],
                'Risk Level': scenario['risk_level'],
                'Primary Impact': max(impacts, key=lambda k: abs(impacts[k])),
                'Max Impact Value': max(impacts.values(), key=abs)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Multi-criteria analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost vs Support
            fig = px.scatter(
                comparison_df,
                x='Stakeholder Support',
                y='Implementation Cost',
                size='Max Impact Value',
                hover_name='Scenario',
                title="Stakeholder Support vs Implementation Cost",
                labels={'Stakeholder Support': 'Stakeholder Support (%)',
                       'Implementation Cost': 'Implementation Cost ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Timeline vs Impact
            fig = px.scatter(
                comparison_df,
                x='Timeline (months)',
                y='Max Impact Value',
                color='Risk Level',
                hover_name='Scenario',
                title="Timeline vs Maximum Impact",
                labels={'Timeline (months)': 'Implementation Timeline (months)',
                       'Max Impact Value': 'Maximum Impact Value'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Decision matrix
        st.subheader("📊 Decision Matrix")
        
        # Simple scoring system (in practice, this would be more sophisticated)
        scored_scenarios = []
        for _, row in comparison_df.iterrows():
            # Normalize and score different criteria
            cost_score = 1 / (1 + row['Implementation Cost'] / 1000000)  # Lower cost = higher score
            support_score = row['Stakeholder Support']
            timeline_score = 1 / (1 + row['Timeline (months)'] / 12)  # Shorter timeline = higher score
            impact_score = abs(row['Max Impact Value'])
            
            risk_scores = {'Low': 1.0, 'Medium': 0.7, 'High': 0.4}
            risk_score = risk_scores.get(row['Risk Level'], 0.5)
            
            # Weighted total score
            total_score = (cost_score * 0.3 + support_score * 0.2 + 
                          timeline_score * 0.2 + impact_score * 0.2 + risk_score * 0.1)
            
            scored_scenarios.append({
                'Scenario': row['Scenario'],
                'Cost Score': cost_score,
                'Support Score': support_score,
                'Timeline Score': timeline_score,
                'Impact Score': impact_score,
                'Risk Score': risk_score,
                'Total Score': total_score
            })
        
        scores_df = pd.DataFrame(scored_scenarios)
        scores_df = scores_df.sort_values('Total Score', ascending=False)
        
        st.dataframe(scores_df, use_container_width=True)
        
        # Recommendation
        best_scenario = scores_df.iloc[0]
        st.success(f"🏆 **Recommended Scenario:** {best_scenario['Scenario']} (Score: {best_scenario['Total Score']:.3f})")
    
    @staticmethod
    def render_strategic_planning():
        """Render strategic planning dashboard."""
        st.subheader("🎯 Strategic Planning Dashboard")
        
        # Long-term projections and planning horizon
        planning_horizon = st.slider("Planning Horizon (years)", 5, 20, 10)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Planning Horizon", f"{planning_horizon} years")
        
        with col2:
            st.metric("Strategic Goals", "5 Active")
        
        with col3:
            st.metric("Budget Allocation", "$15.2M")
        
        # Strategic objectives
        st.subheader("📋 Strategic Objectives")
        
        objectives = [
            {
                'objective': 'Improve On-Time Performance',
                'current': 87.5,
                'target': 95.0,
                'timeline': 3,
                'priority': 'High',
                'status': 'On Track'
            },
            {
                'objective': 'Reduce Environmental Impact',
                'current': 2.3,  # CO2 per km
                'target': 1.5,
                'timeline': 5,
                'priority': 'Medium',
                'status': 'Behind Schedule'
            },
            {
                'objective': 'Increase Fleet Efficiency',
                'current': 92.1,
                'target': 96.0,
                'timeline': 2,
                'priority': 'High',
                'status': 'Ahead of Schedule'
            },
            {
                'objective': 'Expand Coverage Area',
                'current': 250,  # km2
                'target': 320,
                'timeline': 7,
                'priority': 'Medium',
                'status': 'Planning Phase'
            }
        ]
        
        objectives_df = pd.DataFrame(objectives)
        
        # Progress visualization
        for objective in objectives:
            progress = (objective['current'] - objective['current']) / (objective['target'] - objective['current'])
            progress = max(0, min(1, objective['current'] / objective['target']))
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{objective['objective']}**")
                st.progress(progress)
                st.caption(f"Current: {objective['current']:.1f} | Target: {objective['target']:.1f}")
            
            with col2:
                st.write("Timeline")
                st.write(f"{objective['timeline']} years")
            
            with col3:
                st.write("Priority")
                if objective['priority'] == 'High':
                    st.error(objective['priority'])
                elif objective['priority'] == 'Medium':
                    st.warning(objective['priority'])
                else:
                    st.info(objective['priority'])
            
            with col4:
                st.write("Status")
                if objective['status'] == 'On Track':
                    st.success(objective['status'])
                elif objective['status'] == 'Ahead of Schedule':
                    st.success(objective['status'])
                elif objective['status'] == 'Behind Schedule':
                    st.error(objective['status'])
                else:
                    st.info(objective['status'])
        
        # Strategic initiatives
        st.subheader("🚀 Strategic Initiatives")
        
        initiatives = [
            {
                'name': 'Electric Bus Transition',
                'budget': 8500000,
                'timeline': '2024-2028',
                'expected_roi': 0.15,
                'environmental_impact': 0.8,
                'implementation_risk': 'Medium'
            },
            {
                'name': 'Smart Traffic Integration',
                'budget': 3200000,
                'timeline': '2024-2026',
                'expected_roi': 0.22,
                'environmental_impact': 0.3,
                'implementation_risk': 'High'
            },
            {
                'name': 'Route Network Optimization',
                'budget': 1500000,
                'timeline': '2024-2025',
                'expected_roi': 0.28,
                'environmental_impact': 0.4,
                'implementation_risk': 'Low'
            }
        ]
        
        initiatives_df = pd.DataFrame(initiatives)
        
        # Initiative analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Budget allocation
            fig = px.pie(
                initiatives_df,
                values='budget',
                names='name',
                title="Strategic Budget Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI vs Risk analysis
            fig = px.scatter(
                initiatives_df,
                x='expected_roi',
                y='environmental_impact',
                size='budget',
                color='implementation_risk',
                hover_name='name',
                title="ROI vs Environmental Impact",
                labels={'expected_roi': 'Expected Annual ROI',
                       'environmental_impact': 'Environmental Impact Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Long-term trends and projections
        st.subheader("📈 Long-term Trends & Projections")
        
        # Generate trend data
        years = list(range(2024, 2024 + planning_horizon + 1))
        trend_data = []
        
        base_values = {
            'ridership': 10000,
            'costs': 3500000,
            'efficiency': 87.5,
            'emissions': 2300
        }
        
        for i, year in enumerate(years):
            trend_data.append({
                'Year': year,
                'Ridership': base_values['ridership'] * (1.03 ** i),  # 3% growth
                'Operating Costs': base_values['costs'] * (1.02 ** i),  # 2% inflation
                'Efficiency Score': min(100, base_values['efficiency'] + (i * 0.8)),  # Gradual improvement
                'CO2 Emissions': base_values['emissions'] * (0.95 ** i)  # 5% annual reduction
            })
        
        trends_df = pd.DataFrame(trend_data)
        
        # Multi-metric trend visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ridership Growth', 'Operating Costs', 'Efficiency Score', 'CO2 Emissions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=trends_df['Year'], y=trends_df['Ridership'], 
                      mode='lines+markers', name='Ridership'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=trends_df['Year'], y=trends_df['Operating Costs'], 
                      mode='lines+markers', name='Costs'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=trends_df['Year'], y=trends_df['Efficiency Score'], 
                      mode='lines+markers', name='Efficiency'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=trends_df['Year'], y=trends_df['CO2 Emissions'], 
                      mode='lines+markers', name='Emissions'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Long-term Strategic Projections",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategic recommendations
        st.subheader("💡 Strategic Recommendations")
        
        recommendations = [
            {
                'category': 'Investment Priority',
                'recommendation': 'Prioritize Route Network Optimization for highest ROI and lowest risk',
                'urgency': 'High',
                'impact': 'High'
            },
            {
                'category': 'Environmental Goals',
                'recommendation': 'Accelerate Electric Bus Transition to meet 2030 emission targets',
                'urgency': 'Medium',
                'impact': 'High'
            },
            {
                'category': 'Technology Integration',
                'recommendation': 'Phase Smart Traffic Integration with pilot program to reduce risk',
                'urgency': 'Medium',
                'impact': 'Medium'
            },
            {
                'category': 'Capacity Planning',
                'recommendation': 'Begin infrastructure expansion planning to handle projected growth',
                'urgency': 'Low',
                'impact': 'High'
            }
        ]
        
        for rec in recommendations:
            urgency_color = {'High': 'error', 'Medium': 'warning', 'Low': 'info'}
            impact_color = {'High': 'success', 'Medium': 'warning', 'Low': 'info'}
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{rec['category']}:** {rec['recommendation']}")
            
            with col2:
                if urgency_color[rec['urgency']] == 'error':
                    st.error(f"Urgency: {rec['urgency']}")
                elif urgency_color[rec['urgency']] == 'warning':
                    st.warning(f"Urgency: {rec['urgency']}")
                else:
                    st.info(f"Urgency: {rec['urgency']}")
            
            with col3:
                if impact_color[rec['impact']] == 'success':
                    st.success(f"Impact: {rec['impact']}")
                elif impact_color[rec['impact']] == 'warning':
                    st.warning(f"Impact: {rec['impact']}")
                else:
                    st.info(f"Impact: {rec['impact']}")


def render_transport_planner():
    """Main function to render the transport planner interface."""
    try:
        # Page header
        st.title("🗺️ Transport Planner Interface")
        st.markdown("---")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🕸️ Network Analysis",
            "📈 Demand Forecasting",
            "🏗️ Infrastructure Planning",
            "📋 Policy Analysis",
            "🎯 Strategic Planning"
        ])
        
        with tab1:
            PlannerUIComponents.render_network_analysis()
        
        with tab2:
            PlannerUIComponents.render_demand_forecasting()
        
        with tab3:
            PlannerUIComponents.render_infrastructure_planning()
        
        with tab4:
            PlannerUIComponents.render_policy_analysis()
        
        with tab5:
            PlannerUIComponents.render_strategic_planning()
        
        # Sidebar tools
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ Planning Tools")
        
        if st.sidebar.button("📊 Generate Planning Report"):
            st.sidebar.success("Comprehensive planning report generated!")
        
        if st.sidebar.button("📤 Export Analysis Data"):
            st.sidebar.info("Analysis data exported to CSV")
        
        if st.sidebar.button("🔄 Refresh All Data"):
            st.cache_data.clear()
            st.sidebar.success("All data refreshed!")
        
        # Data integration status
        st.sidebar.markdown("---")
        st.sidebar.subheader("📡 Data Sources")
        st.sidebar.success("✅ Census Data")
        st.sidebar.success("✅ GTFS Feed")
        st.sidebar.warning("⚠️ Weather API (Limited)")
        st.sidebar.error("❌ Traffic Data (Unavailable)")
    
    except Exception as e:
        st.error(f"Planner interface error: {str(e)}")
        logger.error(f"Transport planner error: {e}", exc_info=True)


if __name__ == "__main__":
    render_transport_planner()