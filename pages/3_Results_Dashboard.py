import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.cost_calculator import CostCalculator

st.set_page_config(page_title="Results Dashboard", layout="wide")

def main():
    st.title("📊 Results Dashboard")
    st.markdown("Comprehensive analysis and visualization of EV recommendations")
    
    # Check if recommendations exist
    if st.session_state.recommendations is None or st.session_state.recommendations.empty:
        st.warning("⚠️ No recommendations available. Please process your fleet data first.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚗 Analyze Individual Vehicle", type="primary", use_container_width=True):
                st.switch_page("pages/1_Individual_Vehicle.py")
        
        with col2:
            if st.button("🚛 Process Fleet Data", type="primary", use_container_width=True):
                st.switch_page("pages/2_Fleet_Analysis.py")
        
        return
    
    recommendations_df = st.session_state.recommendations
    
    # Initialize cost calculator
    if 'cost_calculator' not in st.session_state:
        st.session_state.cost_calculator = CostCalculator()
    
    # Create dashboard sections
    create_executive_summary(recommendations_df)
    
    col1, col2 = st.columns(2)
    with col1:
        create_vehicle_analysis_charts(recommendations_df)
    with col2:
        create_ev_recommendation_charts(recommendations_df)
    
    create_cost_benefit_analysis(recommendations_df)
    create_detailed_metrics(recommendations_df)

def create_executive_summary(recommendations_df):
    """Create executive summary section"""
    st.subheader("📋 Executive Summary")
    
    # Get top recommendations only (one per vehicle)
    top_recs = recommendations_df[recommendations_df['recommendation_rank'] == 1].copy()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Vehicles Analyzed", len(top_recs))
    
    with col2:
        avg_similarity = top_recs['similarity_score'].mean() * 100
        st.metric("Avg Match Quality", f"{avg_similarity:.1f}%")
    
    with col3:
        total_daily_miles = top_recs['daily_mileage'].sum()
        st.metric("Total Daily Miles", f"{total_daily_miles:,.0f}")
    
    with col4:
        avg_ev_efficiency = top_recs['ev_efficiency'].mean()
        st.metric("Avg EV Efficiency", f"{avg_ev_efficiency:.0f} MPGe")
    
    with col5:
        avg_ev_range = top_recs['ev_range'].mean()
        st.metric("Avg EV Range", f"{avg_ev_range:.0f} miles")
    
    # Fleet composition insights
    st.markdown("#### Fleet Transition Readiness")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_match = len(top_recs[top_recs['similarity_score'] >= 0.8])
        st.metric("High Match (≥80%)", f"{high_match} vehicles", delta=f"{high_match/len(top_recs)*100:.0f}% of fleet")
    
    with col2:
        adequate_range = len(top_recs[top_recs['ev_range'] >= top_recs['daily_mileage'] * 3])
        st.metric("Adequate Range", f"{adequate_range} vehicles", delta=f"{adequate_range/len(top_recs)*100:.0f}% of fleet")
    
    with col3:
        if 'current_mpg' in top_recs.columns:
            efficiency_improvement = top_recs['ev_efficiency'].mean() / (top_recs['current_mpg'].mean() * 3) * 100 - 100
            st.metric("Efficiency Gain", f"+{efficiency_improvement:.0f}%", delta="vs current fleet")
        else:
            st.metric("Efficiency Gain", "Excellent", delta="vs ICE vehicles")

def create_vehicle_analysis_charts(recommendations_df):
    """Create charts analyzing the original vehicle fleet"""
    st.subheader("🚗 Original Fleet Analysis")
    
    # Get unique vehicles
    unique_vehicles = recommendations_df.drop_duplicates(subset=['vehicle_id'])
    
    # Vehicle type distribution
    if 'vehicle_type_standardized' in unique_vehicles.columns:
        st.markdown("##### Vehicle Type Distribution")
        type_counts = unique_vehicles['vehicle_type_standardized'].value_counts()
        
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Fleet Composition by Type"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    if 'vehicle_age' in unique_vehicles.columns:
        st.markdown("##### Vehicle Age Distribution")
        
        fig = px.histogram(
            unique_vehicles,
            x='vehicle_age',
            nbins=15,
            title="Fleet Age Distribution",
            labels={'vehicle_age': 'Vehicle Age (years)', 'count': 'Number of Vehicles'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Daily mileage distribution
    st.markdown("##### Daily Mileage Distribution")
    
    fig = px.histogram(
        unique_vehicles,
        x='daily_mileage',
        nbins=20,
        title="Daily Mileage Distribution",
        labels={'daily_mileage': 'Daily Miles', 'count': 'Number of Vehicles'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def create_ev_recommendation_charts(recommendations_df):
    """Create charts analyzing EV recommendations"""
    st.subheader("⚡ EV Recommendations Analysis")
    
    # Get top recommendations
    top_recs = recommendations_df[recommendations_df['recommendation_rank'] == 1].copy()
    
    # Top recommended EV models
    st.markdown("##### Most Recommended EVs")
    ev_counts = top_recs.groupby(['ev_make', 'ev_model']).size().sort_values(ascending=False).head(8)
    ev_labels = [f"{make} {model}" for make, model in ev_counts.index]
    
    fig = px.bar(
        x=ev_counts.values,
        y=ev_labels,
        orientation='h',
        title="Top EV Recommendations",
        labels={'x': 'Number of Vehicles', 'y': 'EV Model'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # EV efficiency vs range scatter plot
    st.markdown("##### EV Efficiency vs Range")
    
    fig = px.scatter(
        top_recs,
        x='ev_efficiency',
        y='ev_range',
        size='daily_mileage',
        hover_data=['ev_make', 'ev_model'],
        title="EV Efficiency vs Range",
        labels={
            'ev_efficiency': 'Efficiency (MPGe)',
            'ev_range': 'Range (miles)',
            'daily_mileage': 'Daily Miles'
        }
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Similarity score distribution
    st.markdown("##### Match Quality Distribution")
    
    fig = px.histogram(
        top_recs,
        x='similarity_score',
        nbins=20,
        title="Similarity Score Distribution",
        labels={'similarity_score': 'Similarity Score', 'count': 'Number of Vehicles'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def create_cost_benefit_analysis(recommendations_df):
    """Create comprehensive cost-benefit analysis"""
    st.subheader("💰 Cost-Benefit Analysis")
    
    # Calculate fleet-wide cost analysis
    cost_summary = st.session_state.cost_calculator.calculate_fleet_cost_summary(recommendations_df)
    
    if not cost_summary:
        st.warning("Cost analysis data not available")
        return
    
    # Financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Investment",
            f"${cost_summary['total_vehicle_investment']:,.0f}",
            help="Total cost to purchase recommended EVs (after tax credits)"
        )
    
    with col2:
        st.metric(
            "5-Year Savings",
            f"${cost_summary['total_operational_savings_5yr']:,.0f}",
            delta="fuel + maintenance",
            help="Combined fuel and maintenance savings over 5 years"
        )
    
    with col3:
        st.metric(
            "Net 5-Year Impact",
            f"${cost_summary['net_fleet_savings_5yr']:,.0f}",
            delta=f"ROI: {cost_summary['roi_percentage']:.1f}%",
            help="Net financial impact after accounting for vehicle costs"
        )
    
    with col4:
        payback_years = cost_summary.get('average_payback_months', 0) / 12
        st.metric(
            "Average Payback",
            f"{payback_years:.1f} years",
            help="Average time to recover initial investment"
        )
    
    # Cost breakdown visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost vs savings waterfall chart
        create_waterfall_chart(cost_summary)
    
    with col2:
        # ROI by vehicle type if available
        create_roi_by_type_chart(recommendations_df)

def create_waterfall_chart(cost_summary):
    """Create waterfall chart showing cost breakdown"""
    categories = ['Investment', 'Fuel Savings', 'Maintenance Savings', 'Net Result']
    values = [
        -cost_summary['total_vehicle_investment'],
        cost_summary['total_fuel_savings_5yr'],
        cost_summary['total_maintenance_savings_5yr'],
        cost_summary['net_fleet_savings_5yr']
    ]
    
    colors = ['red', 'green', 'blue', 'purple']
    
    fig = go.Figure(go.Waterfall(
        name="Fleet Cost Analysis",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=categories,
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "red"}},
        increasing={"marker": {"color": "green"}},
        totals={"marker": {"color": "blue"}}
    ))
    
    fig.update_layout(
        title="5-Year Financial Impact",
        yaxis_title="Amount ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_roi_by_type_chart(recommendations_df):
    """Create ROI analysis by vehicle type"""
    if 'vehicle_type_standardized' not in recommendations_df.columns:
        st.info("Vehicle type data not available for ROI breakdown")
        return
    
    # Get top recommendations by vehicle type
    top_recs = recommendations_df[recommendations_df['recommendation_rank'] == 1].copy()
    
    # Calculate simplified ROI by type
    roi_by_type = []
    
    for vehicle_type in top_recs['vehicle_type_standardized'].unique():
        type_vehicles = top_recs[top_recs['vehicle_type_standardized'] == vehicle_type]
        
        if len(type_vehicles) > 0:
            # Simplified ROI calculation
            avg_daily_miles = type_vehicles['daily_mileage'].mean()
            avg_efficiency = type_vehicles['ev_efficiency'].mean()
            
            # Estimate 5-year savings (simplified)
            annual_miles = avg_daily_miles * 365
            fuel_savings_per_year = annual_miles * (3.50 / 25 - 0.13 * 33.7 / avg_efficiency)  # Rough calculation
            five_year_savings = fuel_savings_per_year * 5
            
            # Estimate vehicle cost based on type
            if vehicle_type in ['SUV', 'Pickup Truck']:
                avg_cost = 60000
            elif vehicle_type in ['Sedan', 'Passenger Car']:
                avg_cost = 40000
            else:
                avg_cost = 50000
            
            roi = (five_year_savings / avg_cost) * 100 if avg_cost > 0 else 0
            
            roi_by_type.append({
                'Vehicle Type': vehicle_type,
                'Count': len(type_vehicles),
                'ROI (%)': roi,
                '5-Year Savings': five_year_savings
            })
    
    if roi_by_type:
        roi_df = pd.DataFrame(roi_by_type)
        
        fig = px.bar(
            roi_df,
            x='Vehicle Type',
            y='ROI (%)',
            color='Count',
            title="ROI by Vehicle Type (5-Year)",
            text='ROI (%)'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)

def create_detailed_metrics(recommendations_df):
    """Create detailed metrics and analysis"""
    st.subheader("📈 Detailed Performance Metrics")
    
    # Get top recommendations
    top_recs = recommendations_df[recommendations_df['recommendation_rank'] == 1].copy()
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Match Quality", "Range Analysis", "Efficiency Comparison"])
    
    with tab1:
        create_match_quality_analysis(top_recs)
    
    with tab2:
        create_range_analysis(top_recs)
    
    with tab3:
        create_efficiency_analysis(top_recs)

def create_match_quality_analysis(top_recs):
    """Analyze match quality metrics"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Match quality distribution
        quality_ranges = pd.cut(
            top_recs['similarity_score'],
            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=['<60%', '60-70%', '70-80%', '80-90%', '90%+']
        )
        
        quality_counts = quality_ranges.value_counts().sort_index()
        
        fig = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title="Match Quality Distribution",
            color_discrete_sequence=px.colors.sequential.RdYlGn_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Size match analysis
        if 'size_match' in top_recs.columns:
            size_match_counts = top_recs['size_match'].value_counts()
            
            fig = px.bar(
                x=size_match_counts.values,
                y=size_match_counts.index,
                orientation='h',
                title="Size Match Quality",
                color=size_match_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

def create_range_analysis(top_recs):
    """Analyze EV range vs requirements"""
    # Calculate range adequacy
    top_recs['range_buffer'] = (top_recs['ev_range'] - top_recs['daily_mileage'] * 3) / top_recs['daily_mileage'] * 100
    top_recs['range_adequacy'] = pd.cut(
        top_recs['range_buffer'],
        bins=[-np.inf, 0, 50, 100, np.inf],
        labels=['Insufficient', 'Adequate', 'Good', 'Excellent']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Range adequacy pie chart
        adequacy_counts = top_recs['range_adequacy'].value_counts()
        
        fig = px.pie(
            values=adequacy_counts.values,
            names=adequacy_counts.index,
            title="Range Adequacy Assessment",
            color_discrete_map={
                'Excellent': 'green',
                'Good': 'lightgreen',
                'Adequate': 'yellow',
                'Insufficient': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Range vs daily mileage scatter
        fig = px.scatter(
            top_recs,
            x='daily_mileage',
            y='ev_range',
            color='range_adequacy',
            title="EV Range vs Daily Mileage",
            labels={'daily_mileage': 'Daily Mileage', 'ev_range': 'EV Range (miles)'}
        )
        
        # Add diagonal lines for reference
        max_daily = top_recs['daily_mileage'].max()
        fig.add_trace(go.Scatter(
            x=[0, max_daily],
            y=[0, max_daily * 3],
            mode='lines',
            name='3x Daily Range',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig, use_container_width=True)

def create_efficiency_analysis(top_recs):
    """Analyze efficiency improvements"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs EV efficiency
        if 'current_mpg' in top_recs.columns:
            efficiency_data = []
            
            for _, row in top_recs.iterrows():
                efficiency_data.extend([
                    {'Type': 'Current ICE', 'Vehicle': f"{row['make']} {row['model']}", 'Efficiency': row['current_mpg']},
                    {'Type': 'Recommended EV', 'Vehicle': f"{row['ev_make']} {row['ev_model']}", 'Efficiency': row['ev_efficiency']}
                ])
            
            efficiency_df = pd.DataFrame(efficiency_data)
            
            fig = px.box(
                efficiency_df,
                x='Type',
                y='Efficiency',
                title="Efficiency Comparison: ICE vs EV",
                labels={'Efficiency': 'Efficiency (MPG/MPGe)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # EV efficiency distribution
        fig = px.histogram(
            top_recs,
            x='ev_efficiency',
            nbins=15,
            title="EV Efficiency Distribution",
            labels={'ev_efficiency': 'EV Efficiency (MPGe)', 'count': 'Number of Vehicles'}
        )
        
        # Add mean line
        mean_efficiency = top_recs['ev_efficiency'].mean()
        fig.add_vline(x=mean_efficiency, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_efficiency:.0f} MPGe")
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
