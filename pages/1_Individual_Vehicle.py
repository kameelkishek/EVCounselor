import streamlit as st
import pandas as pd
from utils.api_client import EPAClient
from utils.recommendation_engine import RecommendationEngine
from utils.cost_calculator import CostCalculator
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Individual Vehicle Analysis", layout="wide")

def main():
    st.title("🚗 Individual Vehicle Analysis")
    st.markdown("Get EV recommendations for a single vehicle")
    
    # Initialize components
    if 'api_client' not in st.session_state:
        st.session_state.api_client = EPAClient()
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
    if 'cost_calculator' not in st.session_state:
        st.session_state.cost_calculator = CostCalculator()
    
    # Input form
    with st.form("vehicle_input_form"):
        st.subheader("Vehicle Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            make = st.text_input("Make", placeholder="e.g., Toyota")
            model = st.text_input("Model", placeholder="e.g., Camry")
        
        with col2:
            year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
            daily_mileage = st.number_input("Daily Mileage", min_value=0, max_value=500, value=40)
        
        with col3:
            vehicle_type = st.selectbox(
                "Vehicle Type",
                ["Passenger Car", "Sedan", "SUV", "Pickup Truck", "Van/Minivan", 
                 "Hatchback", "Coupe", "Convertible", "Wagon"]
            )
            current_mpg = st.number_input("Current MPG (optional)", min_value=0, max_value=100, value=25)
        
        submitted = st.form_submit_button("Get EV Recommendations", type="primary", use_container_width=True)
    
    if submitted and make and model:
        with st.spinner("Generating recommendations..."):
            # Prepare vehicle data
            vehicle_data = {
                'make': make.title(),
                'model': model.title(),
                'year': int(year),
                'daily_mileage': daily_mileage,
                'annual_mileage': daily_mileage * 365,
                'vehicle_type_standardized': vehicle_type,
                'current_mpg': current_mpg if current_mpg > 0 else None,
                'vehicle_age': 2025 - year
            }
            
            # Get recommendations
            try:
                recommendations = st.session_state.recommendation_engine.get_recommendations(
                    vehicle_data, top_n=5
                )
                
                if recommendations:
                    display_recommendations(vehicle_data, recommendations)
                else:
                    st.warning("No recommendations found. Please try different vehicle details.")
            
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    elif submitted:
        st.error("Please fill in at least Make and Model fields.")

def display_recommendations(vehicle_data, recommendations):
    """Display recommendations and analysis"""
    st.success(f"Found {len(recommendations)} EV recommendations!")
    
    # Vehicle summary
    st.subheader("Input Vehicle Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vehicle", f"{vehicle_data['make']} {vehicle_data['model']}")
    with col2:
        st.metric("Year", vehicle_data['year'])
    with col3:
        st.metric("Daily Miles", f"{vehicle_data['daily_mileage']:,.0f}")
    with col4:
        st.metric("Annual Miles", f"{vehicle_data['annual_mileage']:,.0f}")
    
    # Recommendations table
    st.subheader("EV Recommendations")
    
    # Create recommendations DataFrame for display
    rec_df = pd.DataFrame(recommendations)
    
    # Format display columns
    display_df = rec_df.copy()
    display_df['EV Vehicle'] = display_df['ev_make'] + ' ' + display_df['ev_model']
    display_df['Year'] = display_df['ev_year'].astype(int)
    display_df['Efficiency (MPGe)'] = display_df['ev_efficiency'].round(0).astype(int)
    display_df['Range (miles)'] = display_df['ev_range'].round(0).astype(int)
    display_df['Similarity Score'] = (display_df['similarity_score'] * 100).round(1)
    display_df['Size Match'] = display_df['size_match']
    display_df['Performance'] = display_df['performance_match']
    
    # Display table
    st.dataframe(
        display_df[['EV Vehicle', 'Year', 'Efficiency (MPGe)', 'Range (miles)', 
                   'Similarity Score', 'Size Match', 'Performance']],
        use_container_width=True,
        hide_index=True
    )
    
    # Detailed analysis for top recommendation
    st.subheader("Detailed Analysis - Top Recommendation")
    
    top_rec = recommendations[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**{top_rec['ev_make']} {top_rec['ev_model']} ({top_rec['ev_year']})**")
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Efficiency", f"{top_rec['ev_efficiency']:.0f} MPGe")
            st.metric("Range", f"{top_rec['ev_range']:.0f} miles")
        with metrics_col2:
            st.metric("Similarity", f"{top_rec['similarity_score']*100:.1f}%")
            st.metric("Size Match", top_rec['size_match'])
    
    with col2:
        # Create comparison chart
        create_efficiency_comparison_chart(vehicle_data, top_rec)
    
    # Cost-benefit analysis
    st.subheader("Cost-Benefit Analysis")
    
    cost_analysis = st.session_state.cost_calculator.calculate_cost_analysis(
        vehicle_data, top_rec, analysis_years=5
    )
    
    if cost_analysis:
        display_cost_analysis(cost_analysis)
    
    # Range analysis
    st.subheader("Range Analysis")
    display_range_analysis(vehicle_data, top_rec)

def create_efficiency_comparison_chart(vehicle_data, recommendation):
    """Create efficiency comparison chart"""
    current_mpg = vehicle_data.get('current_mpg', 25)
    ev_mpge = recommendation['ev_efficiency']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Current Vehicle', 'Recommended EV'],
            y=[current_mpg, ev_mpge],
            text=[f'{current_mpg} MPG', f'{ev_mpge:.0f} MPGe'],
            textposition='auto',
            marker_color=['lightcoral', 'lightgreen']
        )
    ])
    
    fig.update_layout(
        title="Fuel Efficiency Comparison",
        yaxis_title="Miles per Gallon (Equivalent)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_cost_analysis(cost_analysis):
    """Display comprehensive cost analysis"""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "5-Year Fuel Savings", 
            f"${cost_analysis['fuel_savings']:,.0f}",
            delta=f"vs gasoline"
        )
    
    with col2:
        st.metric(
            "5-Year Maintenance Savings", 
            f"${cost_analysis['maintenance_savings']:,.0f}",
            delta="lower maintenance"
        )
    
    with col3:
        st.metric(
            "Payback Period", 
            f"{cost_analysis['payback_period_years']:.1f} years",
            delta="break-even point"
        )
    
    with col4:
        roi = cost_analysis['roi_percentage']
        st.metric(
            "5-Year ROI", 
            f"{roi:.1f}%",
            delta="return on investment"
        )
    
    # Cost breakdown
    st.markdown("#### Cost Breakdown")
    
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        st.markdown("**Vehicle Costs**")
        vehicle_costs = cost_analysis['vehicle_costs']
        st.write(f"• Estimated Price: ${vehicle_costs['estimated_price']:,}")
        st.write(f"• Federal Tax Credit: -${vehicle_costs['federal_tax_credit']:,}")
        st.write(f"• **Net Price: ${vehicle_costs['net_price']:,}**")
    
    with cost_col2:
        st.markdown("**Operating Costs (5-year)**")
        st.write(f"• Gas Vehicle Fuel: ${cost_analysis['gas_costs']['total']:,.0f}")
        st.write(f"• EV Electricity: ${cost_analysis['electricity_costs']['total']:,.0f}")
        st.write(f"• **Fuel Savings: ${cost_analysis['fuel_savings']:,.0f}**")
    
    # Cost over time chart
    chart_data = st.session_state.cost_calculator.create_cost_comparison_chart_data(cost_analysis)
    
    if chart_data:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=chart_data['years'],
            y=chart_data['gas_costs'],
            mode='lines+markers',
            name='Gas Vehicle Costs',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=chart_data['years'],
            y=chart_data['ev_costs'],
            mode='lines+markers',
            name='EV Electricity Costs',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=chart_data['years'],
            y=chart_data['cumulative_savings'],
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Cost Comparison Over Time",
            xaxis_title="Years",
            yaxis_title="Cumulative Cost ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_range_analysis(vehicle_data, recommendation):
    """Display range analysis and charging requirements"""
    daily_miles = vehicle_data['daily_mileage']
    ev_range = recommendation['ev_range']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_per_charge = ev_range / daily_miles if daily_miles > 0 else float('inf')
        st.metric("Days per Full Charge", f"{days_per_charge:.1f}")
    
    with col2:
        weekly_miles = daily_miles * 7
        charges_per_week = weekly_miles / ev_range if ev_range > 0 else 0
        st.metric("Charges per Week", f"{charges_per_week:.1f}")
    
    with col3:
        range_buffer = (ev_range - daily_miles) / daily_miles * 100 if daily_miles > 0 else 0
        st.metric("Daily Range Buffer", f"{range_buffer:.0f}%")
    
    # Range adequacy assessment
    if daily_miles * 3 <= ev_range:  # 3x daily range
        st.success("✅ Excellent range coverage - suitable for long trips")
    elif daily_miles * 2 <= ev_range:  # 2x daily range
        st.info("ℹ️ Good range coverage - suitable for daily use with occasional charging")
    else:
        st.warning("⚠️ Limited range coverage - may require daily charging")

if __name__ == "__main__":
    main()
