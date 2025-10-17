# EV Fleet Recommendation Engine

## Overview

The EV Fleet Recommendation Engine is a Streamlit-based web application designed for energy consultants to automate electric vehicle (EV) recommendations for their clients. The system analyzes both individual vehicles and entire fleets, providing data-driven EV alternatives based on vehicle characteristics, usage patterns, and cost-benefit analysis.

The application integrates with the EPA Fuel Economy API to access real-time vehicle data and uses machine learning algorithms to match conventional vehicles with suitable EV alternatives. It provides comprehensive cost analysis, including fuel savings, maintenance costs, and payback periods.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (October 2025)

### Similarity Score Breakdown Feature
- **Added**: Detailed breakdown of similarity score showing per-feature contributions
- **Display**: Shows each feature's contribution in percentage points (pp) with sign preservation
- **Features Analyzed**: Size Class, Performance, Efficiency, Range, Year
- **Visualization**: 
  - Data table with Feature | Contribution (pp) | Impact columns
  - Horizontal bar chart with color coding (green = positive, red = negative)
  - Zero reference line for clarity
  - Sum verification: "Sum of contributions ≈ Overall similarity score"
- **Calculation Method**: Component-wise products of normalized feature vectors, scaled by 100 to match percentage scale
- **User Benefit**: Energy consultants can explain to clients exactly why a particular EV was recommended

### Federal Tax Credit Removal
- **Removed**: All federal tax credit calculations and displays from cost analysis
- **Rationale**: Tax credits vary by eligibility, vehicle, and time period; removed to avoid misleading clients
- **Changed Elements**:
  - Cost calculator no longer subtracts tax credits from Net Price
  - UI no longer shows "Federal Tax Credit" line item
  - Net Price = Estimated Price (no deduction)
  - Price Range displayed instead of tax credit information

## System Architecture

### Frontend Architecture

**Technology Stack**: Streamlit multi-page application
- **Rationale**: Streamlit provides rapid development for data-driven applications with minimal frontend code, ideal for consultant-facing tools
- **Multi-page Structure**: Separates concerns into distinct workflows (individual analysis, fleet batch processing, results dashboard)
- **Session State Management**: Uses Streamlit's session state to persist user data, API clients, and recommendation results across pages
- **Visualization**: Plotly for interactive charts and graphs, enabling detailed exploration of cost-benefit analysis

**Page Architecture**:
1. `app.py` - Main landing page with navigation
2. `pages/1_Individual_Vehicle.py` - Single vehicle analysis workflow
3. `pages/2_Fleet_Analysis.py` - Batch fleet processing with CSV/Excel upload
4. `pages/3_Results_Dashboard.py` - Comprehensive visualization and reporting

**Design Pattern**: Component-based initialization with lazy loading through session state, ensuring services are only instantiated once per user session.

### Backend Architecture

**Modular Service Layer**: Core business logic separated into utility modules under `utils/`

1. **API Integration (`api_client.py`)**
   - Handles communication with EPA Fuel Economy API
   - XML response parsing for vehicle data
   - Rate limiting and error handling for external service calls
   - **Alternative Considered**: Direct database of vehicle specs, but API ensures current data

2. **Data Processing (`data_processor.py`)**
   - Fleet data validation and cleaning
   - CSV/Excel file parsing
   - Column mapping and standardization
   - **Design Decision**: Flexible column matching to accommodate various fleet data formats

3. **Recommendation Engine (`recommendation_engine.py`)**
   - Machine learning-based vehicle matching
   - Uses scikit-learn for feature scaling and nearest neighbor algorithms
   - Weighted similarity scoring across multiple dimensions (size, performance, efficiency, range)
   - **Algorithm Choice**: K-Nearest Neighbors for interpretable, similarity-based recommendations
   - **Pros**: Fast inference, explainable results for consultants
   - **Cons**: Requires representative EV database for quality matches

4. **Cost Calculator (`cost_calculator.py`)**
   - Financial analysis including TCO (Total Cost of Ownership)
   - Configurable parameters for gas prices, electricity costs, maintenance
   - Multi-year projection calculations
   - Payback period analysis
   - **Design Pattern**: Parameterized calculations allowing regional customization

**Data Flow**:
1. User input (individual or batch) → Data validation
2. Vehicle characteristics → ML recommendation engine
3. Matched EVs → Cost-benefit analysis
4. Results → Visualization and export

### Key Architectural Decisions

**Stateful Session Management**
- **Problem**: Multi-page applications lose state between navigation
- **Solution**: Centralized session state for API clients, processors, and data
- **Benefit**: Avoids redundant API calls and re-initialization

**ML-Powered Matching**
- **Problem**: Simple rule-based matching produces poor EV alternatives
- **Solution**: Feature-weighted similarity using StandardScaler and cosine similarity/KNN
- **Trade-off**: Requires training data (EV database) but produces more relevant matches

**Flexible Data Input**
- **Problem**: Fleet data comes in various formats from different clients
- **Solution**: Fuzzy column matching and optional field support
- **Benefit**: Reduces data preparation burden for consultants

## External Dependencies

### Third-Party APIs

**EPA Fuel Economy API** (`https://www.fueleconomy.gov/ws/rest`)
- **Purpose**: Real-time vehicle specifications, MPG data, and model information
- **Data Format**: XML responses
- **Integration**: Custom `EPAClient` class with requests library
- **Rate Limiting**: Implemented at session level
- **Fallback Strategy**: Error handling with user-facing messages

### Python Libraries

**Web Framework**:
- `streamlit` - Core application framework and UI components

**Data Processing**:
- `pandas` - DataFrame operations and CSV/Excel parsing
- `numpy` - Numerical computations

**Machine Learning**:
- `scikit-learn` - StandardScaler, cosine_similarity, NearestNeighbors for recommendation engine

**Visualization**:
- `plotly` - Interactive charts (express and graph_objects modules)

**HTTP Client**:
- `requests` - EPA API communication

**File I/O**:
- `io.BytesIO` - In-memory file operations for exports

### Data Storage

**Current Implementation**: In-memory session state storage
- **Rationale**: Streamlit applications are stateless by default; session state provides per-user persistence during active sessions
- **Limitations**: Data does not persist between sessions or server restarts
- **Note**: No database currently implemented, but architecture supports future addition of persistent storage (e.g., PostgreSQL with Drizzle ORM) for storing fleet analyses, user preferences, or historical recommendations

### Configuration

**Default Parameters** (in `CostCalculator`):
- Gas price: $3.50/gallon
- Electricity cost: $0.13/kWh
- EV efficiency: 0.3 kWh/mile
- Maintenance savings: $0.05/mile

These values are hardcoded but designed for easy parameterization if regional customization is required.