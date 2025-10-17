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
    
    # Similarity Score Breakdown
    st.subheader("Similarity Score Breakdown")
    display_similarity_breakdown(top_rec)
    
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
        st.write(f"• Price Range: {vehicle_costs['price_range']}")
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

def display_similarity_breakdown(recommendation):
    """Display detailed similarity score breakdown"""
    if 'similarity_breakdown' not in recommendation:
        st.info("Similarity breakdown not available for this recommendation")
        return
    
    breakdown = recommendation['similarity_breakdown']
    
    st.markdown("""
    The similarity score is calculated by comparing your vehicle with the recommended EV across five key features.
    Each feature has a weight that determines its importance in the final score:
    """)
    
    # Create breakdown table
    breakdown_data = []
    for feature, values in breakdown.items():
        contribution_val = values['contribution']
        impact = "Helps Match" if contribution_val >= 0 else "Reduces Match"
        breakdown_data.append({
            'Feature': feature,
            'Contribution (pp)': f"{contribution_val:+.2f}",  # pp = percentage points
            'Impact': impact
        })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    # Show the sum
    total_contribution = sum(breakdown[f]['contribution'] for f in breakdown.keys())
    st.caption(f"Sum of contributions: {total_contribution:.2f} percentage points = Overall similarity {recommendation['similarity_score']*100:.1f}%")
    
    # Visual representation
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature match scores
        features = [item['Feature'] for item in breakdown_data]
        scores = [breakdown[f]['similarity'] for f in breakdown.keys()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=features,
                orientation='h',
                text=[f"{s:.1f}%" for s in scores],
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Feature Match Scores",
            xaxis_title="Match Score (%)",
            yaxis_title="Feature",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weighted contributions (can be positive or negative)
        contributions = [breakdown[f]['contribution'] for f in breakdown.keys()]
        
        # Color bars based on positive/negative contribution
        colors = ['green' if c >= 0 else 'red' for c in contributions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=contributions,
                y=features,
                orientation='h',
                text=[f"{c:+.2f} pp" for c in contributions],  # pp = percentage points
                textposition='auto',
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title="Feature Contributions (percentage points)",
            xaxis_title="Contribution to Overall Similarity Score (pp)",
            yaxis_title="Feature",
            height=300
        )
        
        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    **How the Similarity Score is Calculated:**
    
    The **{recommendation['similarity_score']*100:.1f}% similarity score** is computed using cosine similarity, which compares your vehicle with each EV across five normalized features. 
    
    **Feature Contributions (in percentage points):**
    Each feature contributes to or reduces the overall score:
    - ✅ **Positive values (green)**: Features that align well and increase the similarity
    - ❌ **Negative values (red)**: Features that don't match and decrease the similarity
    
    **The math:** The contributions shown above are the per-feature components of the cosine similarity calculation. 
    Their sum equals the overall score ({recommendation['similarity_score']*100:.1f}% = {sum(breakdown[f]['contribution'] for f in breakdown.keys()):.2f} percentage points).
    
    This breakdown shows exactly which aspects of your vehicle match well with this EV and which don't.
    """)

if __name__ == "__main__":
    main()
