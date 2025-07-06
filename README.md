# West Reservoir Temperature Tracker

A Streamlit dashboard for tracking and predicting water temperature at West Reservoir, London.

## Features

- Real-time data loading from Google Sheets
- Interactive temperature visualizations
- Statistical analysis and trends
- Advanced temperature prediction system
- CSV data export functionality

## Installation

0. Create env: 
```bash
python -m venv env
```

1. Activate the virtual environment:
```bash
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run west_reservoir_tracker.py
```

## Temperature Prediction System

The application features a sophisticated multi-stage temperature prediction system that provides accurate forecasts by combining weather data with historical water temperature patterns.

### Three-Stage Process

#### 1. Data Imputation 🔧
The system first fills gaps in historical water temperature data using multiple methods:

- **Forward fill** for short gaps (1-2 days) - assumes temperature changes slowly
- **Time interpolation** for medium gaps (up to 7 days) - smooth transitions between known values  
- **Regression-based** imputation for longer gaps - uses air temperature relationships when other methods fail

This ensures we have complete historical data for robust model training.

#### 2. Model Training 🧠
The prediction model uses **9 sophisticated features** that capture both immediate conditions and longer-term patterns:

**Air Temperature Features:**
- `Air_Temp_t0` - Today's air temperature
- `Air_Temp_t1` - Yesterday's air temperature  
- `Air_Temp_t2` - Day before yesterday's air temperature
- `Air_Temp_7day` - 7-day rolling average air temperature
- `Air_Temp_30day` - 30-day rolling average air temperature

**Water Temperature Features:**
- `Water_Temp_t1` - Yesterday's water temperature (thermal inertia)
- `Water_Temp_t2` - Day before yesterday's water temperature
- `Water_Temp_7day` - 7-day rolling average water temperature

**Seasonal Features:**
- `Season_Sin/Cos` - Sine and cosine transformations of day-of-year for seasonal patterns

The model uses **multivariate linear regression** trained on imputed historical data, leveraging the fact that water temperature has strong thermal inertia - yesterday's water temperature is often the best predictor of today's.

#### 3. Future Predictions 🔮
For dates beyond available data, the system:

- **Iteratively predicts** each future day using weather forecasts
- **Uses previous predictions** as features for subsequent predictions (maintaining thermal inertia)
- **Adapts to missing features** by training simplified models when some data is unavailable
- **Extends predictions** up to 7 days into the future when weather data permits

### Visual Display

The temperature chart displays three distinct data types:

- **Blue solid line** - Actual measurements from Google Sheets
- **Green dotted line** - Imputed missing historical data  
- **Orange dashed line** - Future predictions based on weather forecasts

### Prediction Accuracy

The system achieves high accuracy by:
- Capturing **thermal inertia** through yesterday's water temperature
- Incorporating **weather patterns** and **seasonal cycles**
- Using **rolling averages** to smooth out short-term noise
- **Iterative prediction** that maintains realistic temperature transitions

## Debug Mode

Enable debug mode in the sidebar to see detailed information about:
- Data validation and cleaning processes
- Imputation statistics
- Model training details
- Feature engineering diagnostics

## Data Sources

- **Water Temperature**: Google Sheets (manual readings)
- **Weather Data**: Meteostat API (local weather station data)
- **Location**: West Reservoir, London (51.566938, -0.090492)