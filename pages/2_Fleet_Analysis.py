import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.recommendation_engine import RecommendationEngine
from utils.cost_calculator import CostCalculator
from utils.api_client import EPAClient
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

st.set_page_config(page_title="Fleet Analysis", layout="wide")

def main():
    st.title("🚛 Fleet Batch Analysis")
    st.markdown("Upload and analyze entire vehicle fleets for EV recommendations")
    
    # Initialize components
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
    if 'cost_calculator' not in st.session_state:
        st.session_state.cost_calculator = CostCalculator()
    if 'api_client' not in st.session_state:
        st.session_state.api_client = EPAClient()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "⚙️ Process Fleet", "📊 Results"])
    
    with tab1:
        handle_data_upload()
    
    with tab2:
        handle_fleet_processing()
    
    with tab3:
        display_results()

def handle_data_upload():
    """Handle file upload and validation"""
    st.subheader("Upload Fleet Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Upload your fleet data as CSV or Excel file**
        
        Required columns:
        - `make`: Vehicle manufacturer
        - `model`: Vehicle model
        - `year`: Model year
        - `daily_mileage`: Average daily miles driven
        
        Optional columns:
        - `vehicle_type`: Type of vehicle (Sedan, SUV, etc.)
        - `current_mpg`: Current fuel economy
        - `fuel_type`: Current fuel type
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your fleet vehicle data for analysis"
        )
        
        if uploaded_file:
            # Validate file
            is_valid, message = st.session_state.data_processor.validate_upload_file(uploaded_file)
            
            if is_valid:
                st.success(f"✅ {message}")
                
                # Process file
                with st.spinner("Processing file..."):
                    fleet_df = st.session_state.data_processor.process_upload_file(uploaded_file)
                    
                    if not fleet_df.empty:
                        st.session_state.fleet_data = fleet_df
                        st.success(f"Successfully loaded {len(fleet_df)} vehicles")
                        
                        # Display preview
                        st.subheader("Data Preview")
                        st.dataframe(fleet_df.head(10), use_container_width=True)
                        
                        # Data summary
                        display_data_summary(fleet_df)
            else:
                st.error(f"❌ {message}")
    
    with col2:
        st.subheader("Sample Template")
        st.markdown("Download a sample template to see the expected format:")
        
        # Create sample template
        template_df = st.session_state.data_processor.create_sample_template()
        
        # Convert to CSV
        csv_data = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_data,
            file_name="fleet_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("**Sample Data Preview:**")
        st.dataframe(template_df, use_container_width=True)

def display_data_summary(fleet_df):
    """Display summary statistics of uploaded data"""
    st.subheader("Fleet Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vehicles", len(fleet_df))
    
    with col2:
        avg_mileage = fleet_df['daily_mileage'].mean()
        st.metric("Avg Daily Miles", f"{avg_mileage:.0f}")
    
    with col3:
        total_annual_miles = fleet_df['annual_mileage'].sum() if 'annual_mileage' in fleet_df.columns else 0
        st.metric("Total Annual Miles", f"{total_annual_miles:,.0f}")
    
    with col4:
        if 'current_mpg' in fleet_df.columns:
            avg_mpg = fleet_df['current_mpg'].mean()
            st.metric("Avg Current MPG", f"{avg_mpg:.1f}")
        else:
            st.metric("Avg Current MPG", "N/A")
    
    # Vehicle type distribution
    if 'vehicle_type_standardized' in fleet_df.columns:
        st.subheader("Vehicle Type Distribution")
        
        type_counts = fleet_df['vehicle_type_standardized'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Fleet Composition"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                type_counts.reset_index().rename(columns={
                    'index': 'Vehicle Type', 
                    'vehicle_type_standardized': 'Count'
                }),
                use_container_width=True,
                hide_index=True
            )

def handle_fleet_processing():
    """Handle fleet processing and recommendation generation"""
    st.subheader("Process Fleet for EV Recommendations")
    
    if st.session_state.fleet_data is None:
        st.warning("⚠️ Please upload fleet data first in the 'Upload Data' tab.")
        return
    
    fleet_df = st.session_state.fleet_data
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"Ready to process {len(fleet_df)} vehicles")
        
        # Processing options
        st.markdown("#### Processing Options")
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Vehicle type filter
            if 'vehicle_type_standardized' in fleet_df.columns:
                vehicle_types = ['All'] + list(fleet_df['vehicle_type_standardized'].unique())
                selected_types = st.multiselect(
                    "Filter by Vehicle Type",
                    vehicle_types,
                    default=['All']
                )
                
                if 'All' not in selected_types and selected_types:
                    fleet_df = fleet_df[fleet_df['vehicle_type_standardized'].isin(selected_types)]
        
        with filter_col2:
            # Year range filter
            if 'year' in fleet_df.columns:
                min_year, max_year = int(fleet_df['year'].min()), int(fleet_df['year'].max())
                year_range = st.slider(
                    "Year Range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year)
                )
                
                fleet_df = fleet_df[
                    (fleet_df['year'] >= year_range[0]) & 
                    (fleet_df['year'] <= year_range[1])
                ]
        
        st.info(f"Filtered to {len(fleet_df)} vehicles for processing")
        
        # Process button
        if st.button("🚀 Generate Fleet Recommendations", type="primary", use_container_width=True):
            process_fleet_recommendations(fleet_df)
    
    with col2:
        # Processing status
        st.markdown("#### Processing Status")
        
        if 'processing_status' in st.session_state:
            status = st.session_state.processing_status
            st.metric("Status", status.get('status', 'Ready'))
            
            if 'progress' in status:
                st.progress(status['progress'])
            
            if 'current_vehicle' in status:
                st.text(f"Processing: {status['current_vehicle']}")
        
        # EV Database status
        st.markdown("#### EV Database Status")
        try:
            api_status = st.session_state.api_client.test_connection()
            if api_status:
                st.success("✅ EPA API Connected")
            else:
                st.error("❌ EPA API Unavailable")
                st.info("Will use fallback recommendations")
        except Exception as e:
            st.error(f"❌ API Error: {str(e)}")

def process_fleet_recommendations(fleet_df):
    """Process fleet and generate recommendations"""
    try:
        # Initialize EV database if not already done
        if not hasattr(st.session_state.recommendation_engine, 'ev_database') or \
           st.session_state.recommendation_engine.ev_database is None:
            
            with st.spinner("Initializing EV database..."):
                # Try to get EV data from API
                try:
                    ev_data = st.session_state.api_client.get_electric_vehicles()
                    st.session_state.recommendation_engine.initialize_ev_database(ev_data)
                except Exception as e:
                    st.warning(f"Could not load EV database from API: {str(e)}")
                    st.info("Using fallback recommendation system")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Processing vehicle {current + 1} of {total}")
        
        # Generate recommendations
        with st.spinner("Generating recommendations..."):
            recommendations_df = st.session_state.recommendation_engine.get_batch_recommendations(
                fleet_df, 
                progress_callback=update_progress
            )
        
        progress_bar.empty()
        status_text.empty()
        
        if not recommendations_df.empty:
            st.session_state.recommendations = recommendations_df
            st.success(f"✅ Generated recommendations for {len(recommendations_df)} vehicle-EV matches!")
            
            # Quick preview
            st.subheader("Quick Preview")
            preview_df = recommendations_df.head(10).copy()
            
            # Format for display
            preview_df['Original Vehicle'] = preview_df['make'] + ' ' + preview_df['model'] + ' (' + preview_df['year'].astype(str) + ')'
            preview_df['Recommended EV'] = preview_df['ev_make'] + ' ' + preview_df['ev_model'] + ' (' + preview_df['ev_year'].astype(str) + ')'
            preview_df['Similarity'] = (preview_df['similarity_score'] * 100).round(1).astype(str) + '%'
            preview_df['Daily Miles'] = preview_df['daily_mileage'].round(0).astype(int)
            
            st.dataframe(
                preview_df[['Original Vehicle', 'Recommended EV', 'Similarity', 'Daily Miles']],
                use_container_width=True,
                hide_index=True
            )
            
            st.info("👆 Preview of recommendations. Go to 'Results' tab for complete analysis.")
        
        else:
            st.error("❌ Failed to generate recommendations. Please check your data and try again.")
    
    except Exception as e:
        st.error(f"Error processing fleet: {str(e)}")

def display_results():
    """Display comprehensive results and analysis"""
    st.subheader("Fleet Analysis Results")
    
    if st.session_state.recommendations is None or st.session_state.recommendations.empty:
        st.warning("⚠️ No recommendations available. Please process your fleet data first.")
        return
    
    recommendations_df = st.session_state.recommendations
    
    # Results summary
    st.subheader("Results Summary")
    
    # Get unique vehicles (original fleet)
    unique_vehicles = recommendations_df.drop_duplicates(subset=['vehicle_id'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vehicles Analyzed", len(unique_vehicles))
    
    with col2:
        avg_similarity = recommendations_df['similarity_score'].mean() * 100
        st.metric("Avg Match Score", f"{avg_similarity:.1f}%")
    
    with col3:
        total_recommendations = len(recommendations_df)
        st.metric("Total EV Matches", total_recommendations)
    
    with col4:
        unique_evs = recommendations_df[['ev_make', 'ev_model', 'ev_year']].drop_duplicates()
        st.metric("Unique EVs Recommended", len(unique_evs))
    
    # Cost analysis
    st.subheader("Fleet Cost Analysis")
    
    cost_summary = st.session_state.cost_calculator.calculate_fleet_cost_summary(recommendations_df)
    
    if cost_summary:
        display_fleet_cost_summary(cost_summary)
    
    # Detailed recommendations table
    st.subheader("Detailed Recommendations")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Vehicle type filter
        vehicle_types = ['All'] + list(recommendations_df['vehicle_type_standardized'].unique()) if 'vehicle_type_standardized' in recommendations_df.columns else ['All']
        type_filter = st.selectbox("Filter by Vehicle Type", vehicle_types)
    
    with col2:
        # Recommendation rank filter
        rank_filter = st.selectbox("Show Recommendations", ["Top recommendation only", "Top 2 per vehicle", "All recommendations"])
    
    with col3:
        # Minimum similarity filter
        min_similarity = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.5, 0.1)
    
    # Apply filters
    filtered_df = recommendations_df.copy()
    
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['vehicle_type_standardized'] == type_filter]
    
    if rank_filter == "Top recommendation only":
        filtered_df = filtered_df[filtered_df['recommendation_rank'] == 1]
    elif rank_filter == "Top 2 per vehicle":
        filtered_df = filtered_df[filtered_df['recommendation_rank'] <= 2]
    
    filtered_df = filtered_df[filtered_df['similarity_score'] >= min_similarity]
    
    # Format display table
    display_table = format_recommendations_table(filtered_df)
    
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv_data = st.session_state.data_processor.export_recommendations(filtered_df, 'csv')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"ev_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel export
        excel_data = st.session_state.data_processor.export_recommendations(filtered_df, 'xlsx')
        st.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name=f"ev_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # Summary report
        summary_data = create_summary_report(recommendations_df, cost_summary)
        st.download_button(
            label="📋 Download Summary",
            data=summary_data,
            file_name=f"fleet_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def display_fleet_cost_summary(cost_summary):
    """Display fleet-wide cost summary"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "5-Year Fuel Savings", 
            f"${cost_summary['total_fuel_savings_5yr']:,.0f}",
            help="Total fuel cost savings across entire fleet"
        )
    
    with col2:
        st.metric(
            "5-Year Maintenance Savings", 
            f"${cost_summary['total_maintenance_savings_5yr']:,.0f}",
            help="Total maintenance cost savings across entire fleet"
        )
    
    with col3:
        st.metric(
            "Fleet Investment", 
            f"${cost_summary['total_vehicle_investment']:,.0f}",
            help="Total investment required for EV fleet transition"
        )
    
    with col4:
        st.metric(
            "Net 5-Year Savings", 
            f"${cost_summary['net_fleet_savings_5yr']:,.0f}",
            delta=f"ROI: {cost_summary['roi_percentage']:.1f}%"
        )
    
    # Cost breakdown chart
    create_fleet_cost_chart(cost_summary)

def create_fleet_cost_chart(cost_summary):
    """Create fleet cost analysis chart"""
    categories = ['Fuel Savings', 'Maintenance Savings', 'Vehicle Investment']
    values = [
        cost_summary['total_fuel_savings_5yr'],
        cost_summary['total_maintenance_savings_5yr'],
        -cost_summary['total_vehicle_investment']  # Negative for cost
    ]
    colors = ['green', 'blue', 'red']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Fleet Cost Analysis (5-Year)",
        yaxis_title="Cost/Savings ($)",
        showlegend=False,
        height=400
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    st.plotly_chart(fig, use_container_width=True)

def format_recommendations_table(recommendations_df):
    """Format recommendations for display table"""
    display_df = recommendations_df.copy()
    
    # Create formatted columns
    display_df['Original Vehicle'] = (display_df['make'] + ' ' + 
                                     display_df['model'] + ' (' + 
                                     display_df['year'].astype(str) + ')')
    
    display_df['Recommended EV'] = (display_df['ev_make'] + ' ' + 
                                   display_df['ev_model'] + ' (' + 
                                   display_df['ev_year'].astype(str) + ')')
    
    display_df['Daily Miles'] = display_df['daily_mileage'].round(0).astype(int)
    display_df['Similarity Score'] = (display_df['similarity_score'] * 100).round(1)
    display_df['EV Efficiency (MPGe)'] = display_df['ev_efficiency'].round(0).astype(int)
    display_df['EV Range (miles)'] = display_df['ev_range'].round(0).astype(int)
    display_df['Rank'] = display_df['recommendation_rank']
    
    return display_df[[
        'Original Vehicle', 'Recommended EV', 'Rank', 'Daily Miles', 
        'Similarity Score', 'EV Efficiency (MPGe)', 'EV Range (miles)',
        'size_match', 'performance_match'
    ]].rename(columns={
        'size_match': 'Size Match',
        'performance_match': 'Performance Match'
    })

def create_summary_report(recommendations_df, cost_summary):
    """Create executive summary report"""
    # Get top recommendation for each vehicle
    top_recs = recommendations_df[recommendations_df['recommendation_rank'] == 1].copy()
    
    summary_data = []
    
    # Fleet overview
    summary_data.append(['FLEET ANALYSIS SUMMARY'])
    summary_data.append([''])
    summary_data.append(['Total Vehicles Analyzed', len(top_recs)])
    summary_data.append(['Average Match Score', f"{top_recs['similarity_score'].mean() * 100:.1f}%"])
    summary_data.append([''])
    
    # Cost summary
    if cost_summary:
        summary_data.append(['FINANCIAL ANALYSIS (5-Year)'])
        summary_data.append(['Total Fuel Savings', f"${cost_summary['total_fuel_savings_5yr']:,.0f}"])
        summary_data.append(['Total Maintenance Savings', f"${cost_summary['total_maintenance_savings_5yr']:,.0f}"])
        summary_data.append(['Total Investment Required', f"${cost_summary['total_vehicle_investment']:,.0f}"])
        summary_data.append(['Net Fleet Savings', f"${cost_summary['net_fleet_savings_5yr']:,.0f}"])
        summary_data.append(['ROI Percentage', f"{cost_summary['roi_percentage']:.1f}%"])
        summary_data.append([''])
    
    # Top EV recommendations
    summary_data.append(['TOP EV MODELS RECOMMENDED'])
    ev_counts = top_recs.groupby(['ev_make', 'ev_model']).size().sort_values(ascending=False).head(10)
    for (make, model), count in ev_counts.items():
        summary_data.append([f"{make} {model}", f"{count} vehicles"])
    
    # Convert to CSV string
    summary_df = pd.DataFrame(summary_data)
    return summary_df.to_csv(index=False, header=False)

if __name__ == "__main__":
    main()
