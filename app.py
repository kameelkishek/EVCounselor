import streamlit as st
import pandas as pd
from utils.api_client import EPAClient
from utils.data_processor import DataProcessor
from utils.recommendation_engine import RecommendationEngine

# Configure page
st.set_page_config(
    page_title="EV Fleet Recommendation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = EPAClient()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()
if 'fleet_data' not in st.session_state:
    st.session_state.fleet_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

def main():
    st.title("🚗 EV Fleet Recommendation Engine")
    st.markdown("### Automate Electric Vehicle Recommendations for Energy Consultants")
    
    # Main navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Key statistics if data is loaded
    if st.session_state.fleet_data is not None:
        st.sidebar.success(f"Fleet Data Loaded: {len(st.session_state.fleet_data)} vehicles")
    
    if st.session_state.recommendations is not None:
        st.sidebar.success(f"Recommendations Generated: {len(st.session_state.recommendations)} matches")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Individual Vehicle Analysis**")
        st.markdown("Analyze single vehicles and get EV recommendations")
        if st.button("Analyze Individual Vehicle", type="primary", use_container_width=True):
            st.switch_page("pages/1_Individual_Vehicle.py")
    
    with col2:
        st.info("**Fleet Batch Analysis**")
        st.markdown("Upload CSV/Excel files for bulk fleet analysis")
        if st.button("Batch Fleet Analysis", type="primary", use_container_width=True):
            st.switch_page("pages/2_Fleet_Analysis.py")
    
    with col3:
        st.info("**Results Dashboard**")
        st.markdown("View recommendations, comparisons, and export results")
        if st.button("View Results Dashboard", type="primary", use_container_width=True):
            st.switch_page("pages/3_Results_Dashboard.py")
    
    # Quick overview section
    st.markdown("---")
    st.markdown("### How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1. Input Data**")
        st.markdown("- Vehicle make, model, year")
        st.markdown("- Daily mileage")
        st.markdown("- Vehicle type")
    
    with col2:
        st.markdown("**2. API Integration**")
        st.markdown("- EPA Fuel Economy database")
        st.markdown("- Comprehensive vehicle specs")
        st.markdown("- Real-time data access")
    
    with col3:
        st.markdown("**3. ML Analysis**")
        st.markdown("- Similarity scoring")
        st.markdown("- Performance matching")
        st.markdown("- Size classification")
    
    with col4:
        st.markdown("**4. Recommendations**")
        st.markdown("- Top EV matches")
        st.markdown("- Cost-benefit analysis")
        st.markdown("- Export capabilities")
    
    # Status indicators
    st.markdown("---")
    st.markdown("### System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Test EPA API connection
        try:
            test_result = st.session_state.api_client.test_connection()
            if test_result:
                st.success("✅ EPA API Connection: Active")
            else:
                st.error("❌ EPA API Connection: Failed")
        except Exception as e:
            st.error(f"❌ EPA API Connection: Error - {str(e)}")
    
    with col2:
        # System readiness
        if all([
            st.session_state.api_client,
            st.session_state.data_processor,
            st.session_state.recommendation_engine
        ]):
            st.success("✅ System Status: Ready")
        else:
            st.error("❌ System Status: Initialization Error")

if __name__ == "__main__":
    main()
