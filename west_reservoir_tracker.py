"""West Reservoir Temperature Tracker - Streamlit Dashboard"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import requests
from io import StringIO
from typing import Optional
from meteostat import Point, Daily
from sklearn.linear_model import LinearRegression
import numpy as np
from forecast import WaterTemperatureModel

st.set_page_config(
    page_title="West Reservoir Temperature Tracker",
    page_icon="🌡️",
    layout="wide"
)

# Initialize session state for logging
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def add_log_message(message_type: str, message: str) -> None:
    """Add a log message to be displayed at the end."""
    st.session_state.log_messages.append((message_type, message))

def display_log_messages() -> None:
    """Display all accumulated log messages."""
    if st.session_state.log_messages:
        st.subheader("📝 Processing Log")
        for msg_type, msg in st.session_state.log_messages:
            if msg_type == "info":
                st.info(msg)
            elif msg_type == "warning":
                st.warning(msg)
            elif msg_type == "error":
                st.error(msg)
            elif msg_type == "success":
                st.success(msg)
        # Clear messages after displaying
        st.session_state.log_messages = []

SHEET_URL = "https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/export?format=csv&gid=0"
REQUEST_TIMEOUT = 30

# West Reservoir location (London)
RESERVOIR_LOCATION = Point(51.566938, -0.090492)  # West Reservoir coordinates

# Sample data for fallback
SAMPLE_DATA = [
    {"Date": "01/01/2024", "Temperature": 8.5},
    {"Date": "02/01/2024", "Temperature": 7.2},
    {"Date": "03/01/2024", "Temperature": 6.8},
    {"Date": "04/01/2024", "Temperature": 9.1},
    {"Date": "05/01/2024", "Temperature": 10.3},
    {"Date": "06/01/2024", "Temperature": 12.7},
    {"Date": "07/01/2024", "Temperature": 15.2},
    {"Date": "08/01/2024", "Temperature": 17.8},
    {"Date": "09/01/2024", "Temperature": 16.4},
    {"Date": "10/01/2024", "Temperature": 13.9},
    {"Date": "11/01/2024", "Temperature": 11.2},
    {"Date": "12/01/2024", "Temperature": 9.6},
]


def validate_and_clean_data(df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """Validate and clean the raw temperature data.
    
    Args:
        df: Raw DataFrame from data source
        
    Returns:
        pd.DataFrame: Cleaned and validated temperature data
    """
    if df.empty:
        add_log_message("warning", "Data source appears to be empty")
        return pd.DataFrame()
    
    # Ensure we have at least 2 columns
    if len(df.columns) < 2:
        add_log_message("error", "Data source must have at least 2 columns (Date, Temperature)")
        return pd.DataFrame()
    
    # Clean and process the data
    df.columns = ['Date', 'Temperature'] + list(df.columns[2:])  # Preserve extra columns
    df = df[['Date', 'Temperature']]  # Keep only needed columns
    
    # Show debug info if enabled
    if debug_mode and len(df) > 0:
        st.write("🔍 **Debug: Raw data before processing**")
        st.write(df.head(5))
        st.write(f"🔍 **Debug: Raw data shape:** {df.shape}")
        st.write(f"🔍 **Debug: Column names:** {list(df.columns)}")
    
    # Store original data for comparison
    original_df = df.copy() if debug_mode else None
    
    # Convert date and temperature columns
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    
    # Check what's invalid (only show in debug mode)
    if debug_mode:
        invalid_dates = df['Date'].isna().sum()
        invalid_temps = df['Temperature'].isna().sum()
        
        st.write(f"🔍 **Debug: After conversion - Invalid dates:** {invalid_dates}, **Invalid temps:** {invalid_temps}")
        
        if invalid_dates > 0:
            st.warning(f"🔍 Debug: Found {invalid_dates} rows with invalid dates")
            # Show rows with invalid dates (original values)
            invalid_date_mask = df['Date'].isna()
            invalid_date_original = original_df[invalid_date_mask][['Date', 'Temperature']].head(3)
            st.write("🔍 **Original values that failed date parsing:**")
            st.write(invalid_date_original)
        
        if invalid_temps > 0:
            st.warning(f"🔍 Debug: Found {invalid_temps} rows with invalid temperatures")
            # Show rows with invalid temperatures (original values)
            invalid_temp_mask = df['Temperature'].isna()
            invalid_temp_original = original_df[invalid_temp_mask][['Date', 'Temperature']].head(3)
            st.write("🔍 **Original values that failed temperature parsing:**")
            st.write(invalid_temp_original)
        
        if invalid_dates == 0 and invalid_temps == 0:
            st.success("🔍 Debug: All data parsed successfully!")
    
    # Remove rows with invalid data
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    
    if initial_count > final_count:
        if debug_mode:
            add_log_message("info", f"🔍 Debug: Removed {initial_count - final_count} invalid records")
        else:
            add_log_message("info", f"Removed {initial_count - final_count} invalid records")
    
    if df.empty:
        add_log_message("warning", "No valid data found after cleaning")
        return df
    
    return df.sort_values('Date').reset_index(drop=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and process temperature data from Google Sheets.
    
    Returns:
        pd.DataFrame: Cleaned temperature data with Date and Temperature columns
    """
    # Create placeholder for loading message
    loading_placeholder = st.empty()
    
    try:
        loading_placeholder.info("Loading data from Google Sheets...")
        response = requests.get(SHEET_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        result = validate_and_clean_data(df, debug_mode=st.session_state.get('debug_mode', False))
        
        if not result.empty:
            loading_placeholder.empty()  # Clear loading message
            return result
        else:
            loading_placeholder.empty()
            add_log_message("warning", "No valid data in Google Sheets, using sample data")
            return load_sample_data()
        
    except requests.exceptions.Timeout:
        loading_placeholder.empty()
        add_log_message("warning", "Request timed out. Using sample data instead.")
        return load_sample_data()
    except requests.exceptions.RequestException as e:
        loading_placeholder.empty()
        add_log_message("warning", f"Network error: {e}. Using sample data instead.")
        return load_sample_data()
    except Exception as e:
        loading_placeholder.empty()
        add_log_message("warning", f"Error processing data: {e}. Using sample data instead.")
        return load_sample_data()


def load_sample_data() -> pd.DataFrame:
    """Load sample temperature data as fallback.
    
    Returns:
        pd.DataFrame: Sample temperature data
    """
    df = pd.DataFrame(SAMPLE_DATA)
    return validate_and_clean_data(df, debug_mode=st.session_state.get('debug_mode', False))


@st.cache_data
def get_weather_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get local weather data for the specified date range.
    
    Args:
        start_date: Start date for weather data
        end_date: End date for weather data
        
    Returns:
        pd.DataFrame: Weather data with Date and air temperature
    """
    try:
        # Get historical weather data from meteostat
        weather_data = Daily(RESERVOIR_LOCATION, start_date, datetime.now())
        weather_df = weather_data.fetch()
        
        if not weather_df.empty:
            # Reset index to get date as column
            weather_df = weather_df.reset_index()
            weather_df = weather_df[['time', 'tavg']].copy()
            weather_df.columns = ['Date', 'Air_Temperature']
            weather_df = weather_df.dropna()
        else:
            weather_df = pd.DataFrame(columns=['Date', 'Air_Temperature'])
        
        # Add synthetic future weather data for predictions (since meteostat doesn't provide forecasts)
        today = datetime.now().date()
        future_end = end_date.date()
        
        if future_end > today:
            # Create future dates
            future_dates = pd.date_range(start=today + timedelta(days=1), end=future_end, freq='D')
            
            if len(weather_df) > 0:
                # Use recent temperature patterns to create realistic future temps
                recent_temps = weather_df.tail(30)['Air_Temperature']
                mean_temp = recent_temps.mean()
                std_temp = recent_temps.std() if len(recent_temps) > 1 else 3.0
                
                # Add some seasonal variation
                for date in future_dates:
                    day_of_year = date.timetuple().tm_yday
                    seasonal_factor = np.sin(2 * np.pi * day_of_year / 365.25) * 2  # +/- 2°C seasonal variation
                    
                    # Add some random variation
                    temp_variation = np.random.normal(0, std_temp * 0.3)  # Reduced variation for predictions
                    future_temp = mean_temp + seasonal_factor + temp_variation
                    
                    # Add to weather data
                    new_row = pd.DataFrame({
                        'Date': [date],
                        'Air_Temperature': [future_temp]
                    })
                    weather_df = pd.concat([weather_df, new_row], ignore_index=True)
            
            add_log_message("info", f"🌤️ Generated {len(future_dates)} days of synthetic weather data for predictions")
        
        return weather_df.sort_values('Date').reset_index(drop=True)
        
    except Exception as e:
        add_log_message("warning", f"Could not fetch weather data: {e}")
        return pd.DataFrame()


def create_temporal_features(df: pd.DataFrame, air_temp_col: str, water_temp_col: str = None) -> pd.DataFrame:
    """Create temporal features for temperature prediction.
    
    Args:
        df: DataFrame with Date and temperature columns
        air_temp_col: Name of the air temperature column
        water_temp_col: Name of the water temperature column (optional)
        
    Returns:
        pd.DataFrame: DataFrame with temporal features added
    """
    df = df.copy().sort_values('Date')
    
    # Create lagged air temperature features
    df['Air_Temp_t0'] = df[air_temp_col]  # Today
    df['Air_Temp_t1'] = df[air_temp_col].shift(1)  # Yesterday
    df['Air_Temp_t2'] = df[air_temp_col].shift(2)  # Day before yesterday
    
    # Create rolling averages for air temperature
    df['Air_Temp_7day'] = df[air_temp_col].rolling(window=7, min_periods=1).mean()
    df['Air_Temp_30day'] = df[air_temp_col].rolling(window=30, min_periods=1).mean()
    
    # Create water temperature lagged features if available
    if water_temp_col and water_temp_col in df.columns:
        df['Water_Temp_t1'] = df[water_temp_col].shift(1)  # Yesterday's water temp
        df['Water_Temp_t2'] = df[water_temp_col].shift(2)  # Day before yesterday's water temp
        df['Water_Temp_7day'] = df[water_temp_col].rolling(window=7, min_periods=1).mean()
    
    # Create day of year feature (seasonal component)
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Season_Sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365.25)
    df['Season_Cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365.25)
    
    return df


def impute_missing_water_temperatures_physics(reservoir_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing water temperatures using physics-based model.
    
    Args:
        reservoir_df: Existing reservoir temperature data
        weather_df: Weather data for the same period
        
    Returns:
        pd.DataFrame: Extended data with physics-based imputed water temperatures
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df
    
    # Create a complete date range from weather data
    all_dates = weather_df[['Date', 'Air_Temperature']].copy()
    
    # Merge with existing reservoir data
    combined = pd.merge(all_dates, reservoir_df, on='Date', how='left')
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    if len(combined[combined['Temperature'].notna()]) < 10:
        add_log_message("warning", "Not enough existing water temperature data for physics-based imputation")
        return reservoir_df
    
    # Initialize physics model
    model = WaterTemperatureModel()
    
    # Find continuous segments of actual data to train the model
    has_temp = combined['Temperature'].notna()
    actual_segments = []
    
    start_idx = None
    for i, has_data in enumerate(has_temp):
        if has_data and start_idx is None:
            start_idx = i
        elif not has_data and start_idx is not None:
            if i - start_idx >= 10:  # Need at least 10 points to train
                actual_segments.append((start_idx, i))
            start_idx = None
    
    # Check if last segment extends to end
    if start_idx is not None and len(combined) - start_idx >= 10:
        actual_segments.append((start_idx, len(combined)))
    
    if not actual_segments:
        add_log_message("warning", "No sufficiently long segments of actual data found for physics model training")
        return reservoir_df
    
    # Use the longest segment for initial model training
    longest_segment = max(actual_segments, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_segment
    
    training_data = combined.iloc[start_idx:end_idx].copy()
    training_dates = training_data['Date'].dt.to_pydatetime()
    training_air_temps = training_data['Air_Temperature'].values
    training_water_temps = training_data['Temperature'].values
    
    add_log_message("info", f"🧠 Training physics model on {len(training_data)} points for imputation...")
    
    try:
        result = model.fit_parameters(training_air_temps, training_water_temps, training_dates)
        if result and result.success:
            add_log_message("info", f"✅ Physics model trained for imputation (k={model.k:.4f})")
        else:
            add_log_message("warning", "Physics model training failed for imputation, using defaults")
    except Exception as e:
        add_log_message("warning", f"Physics model training error for imputation: {e}")
    
    # Simpler approach: just use statistical imputation for now to avoid issues
    # The complex gap-by-gap physics prediction was causing problems
    add_log_message("info", "🔄 Using statistical fallback for gap filling due to complexity")
    
    imputed_data = combined.copy()
    
    # Method 1: Forward fill for short gaps (1-2 days)
    imputed_data['Temperature'] = imputed_data['Temperature'].ffill(limit=2)
    
    # Method 2: Interpolation for medium gaps
    imputed_data = imputed_data.set_index('Date')
    imputed_data['Temperature'] = imputed_data['Temperature'].interpolate(method='time', limit=7)
    imputed_data = imputed_data.reset_index()
    
    # Method 3: For remaining gaps, use physics model in a simpler way
    still_missing = imputed_data['Temperature'].isna()
    if still_missing.sum() > 0:
        # Get a typical temperature-air relationship from the training data
        available_data = imputed_data[imputed_data['Temperature'].notna()]
        if len(available_data) >= 10:
            # Use the trained physics model to estimate missing values
            for idx in imputed_data.index[still_missing]:
                air_temp = imputed_data.loc[idx, 'Air_Temperature']
                # Simple physics-based estimate: assume equilibrium with seasonal offset
                day_of_year = imputed_data.loc[idx, 'Date'].timetuple().tm_yday
                seasonal_offset = model.seasonal_offset(day_of_year)
                
                # Estimate based on recent average relationship
                recent_data = available_data.tail(30)
                if len(recent_data) > 0:
                    avg_diff = (recent_data['Temperature'] - recent_data['Air_Temperature']).mean()
                    estimated_temp = air_temp + avg_diff + seasonal_offset
                    imputed_data.loc[idx, 'Temperature'] = max(0.1, estimated_temp)
    
    # Add a column to track which data was imputed
    imputed_data['Data_Source'] = 'Actual'
    original_mask = imputed_data['Date'].isin(reservoir_df['Date'])
    imputed_data.loc[~original_mask, 'Data_Source'] = 'Imputed'
    
    physics_imputed = (~original_mask & imputed_data['Temperature'].notna()).sum()
    add_log_message("info", f"🔬 Physics-inspired imputation: {physics_imputed} points")
    
    return imputed_data[['Date', 'Temperature', 'Data_Source']]


def impute_missing_water_temperatures(reservoir_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy imputation function using statistical methods (kept for fallback).
    
    Args:
        reservoir_df: Existing reservoir temperature data
        weather_df: Weather data for the same period
        
    Returns:
        pd.DataFrame: Extended data with imputed water temperatures
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df
    
    # Create a complete date range from weather data
    all_dates = weather_df[['Date', 'Air_Temperature']].copy()
    
    # Merge with existing reservoir data
    combined = pd.merge(all_dates, reservoir_df, on='Date', how='left')
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    if len(combined[combined['Temperature'].notna()]) < 10:
        add_log_message("warning", "Not enough existing water temperature data for imputation")
        return reservoir_df
    
    # Start with simple imputation methods
    imputed_data = combined.copy()
    
    # Method 1: Forward fill for short gaps (1-2 days)
    imputed_data['Temperature'] = imputed_data['Temperature'].ffill(limit=2)
    
    # Method 2: Interpolation for medium gaps
    # Set Date as index for time-weighted interpolation
    imputed_data = imputed_data.set_index('Date')
    imputed_data['Temperature'] = imputed_data['Temperature'].interpolate(method='time', limit=7)
    imputed_data = imputed_data.reset_index()  # Reset index back to normal
    
    # Method 3: For remaining gaps, use regression with air temperature
    still_missing = imputed_data['Temperature'].isna()
    if still_missing.sum() > 0:
        # Train a simple model on available data
        available_data = imputed_data[imputed_data['Temperature'].notna()]
        if len(available_data) >= 5:
            X_simple = available_data[['Air_Temperature']].values
            y_simple = available_data['Temperature'].values
            
            simple_model = LinearRegression()
            simple_model.fit(X_simple, y_simple)
            
            # Predict for missing values
            missing_data = imputed_data[still_missing]
            X_missing = missing_data[['Air_Temperature']].values
            predicted_temps = simple_model.predict(X_missing)
            
            # Fill in the predictions
            imputed_data.loc[still_missing, 'Temperature'] = predicted_temps
    
    # Add a column to track which data was imputed
    imputed_data['Data_Source'] = 'Actual'
    original_mask = imputed_data['Date'].isin(reservoir_df['Date'])
    imputed_data.loc[~original_mask, 'Data_Source'] = 'Imputed'
    
    add_log_message("info", f"🔧 Statistical imputation: {(~original_mask).sum()} missing water temperature readings")
    
    return imputed_data[['Date', 'Temperature', 'Data_Source']]


def create_temperature_predictions_physics(reservoir_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create temperature predictions using physics-based model from forecast.py.
    
    Args:
        reservoir_df: Reservoir temperature data
        weather_df: Weather data
        
    Returns:
        pd.DataFrame: Combined data with predictions
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df
    
    # Separate historical and future weather data
    today = pd.Timestamp(datetime.now().date())
    historical_weather = weather_df[weather_df['Date'] <= today]
    future_weather = weather_df[weather_df['Date'] > today]
    
    if st.session_state.get('debug_mode', False):
        add_log_message("info", f"🔍 Debug: Historical weather: {len(historical_weather)} days")
        add_log_message("info", f"🔍 Debug: Future weather: {len(future_weather)} days")
    
    # Initialize the physics-based model
    model = WaterTemperatureModel()
    
    # First, impute missing water temperatures in historical period 
    # Use the method selected by the user (passed as parameter or default to physics)
    use_physics_imputation = getattr(st.session_state, 'use_physics_imputation', True)
    
    if use_physics_imputation:
        imputed_historical = impute_missing_water_temperatures_physics(reservoir_df, historical_weather)
    else:
        imputed_historical = impute_missing_water_temperatures(reservoir_df, historical_weather)
    
    # Prepare training data - combine imputed reservoir data with historical weather
    merged_historical = pd.merge(imputed_historical, historical_weather, on='Date', how='inner')
    merged_historical = merged_historical.sort_values('Date').reset_index(drop=True)
    
    # Only use actual and imputed data for training (not predictions from other models)
    if 'Data_Source' in merged_historical.columns:
        training_data = merged_historical[merged_historical['Data_Source'].isin(['Actual', 'Imputed'])]
    else:
        training_data = merged_historical
    
    if len(training_data) < len(merged_historical):
        add_log_message("info", f"🔧 Using {len(training_data)} points for training (excluding previous predictions)")
        merged_historical = training_data
    
    if len(merged_historical) < 30:
        add_log_message("warning", "Not enough data for physics model training (need at least 30 days)")
        return reservoir_df
    
    # Train the model on historical data
    training_dates = merged_historical['Date'].dt.to_pydatetime()
    training_air_temps = merged_historical['Air_Temperature'].values
    training_water_temps = merged_historical['Temperature'].values
    
    add_log_message("info", f"🧠 Training physics model on {len(merged_historical)} data points...")
    
    try:
        result = model.fit_parameters(training_air_temps, training_water_temps, training_dates)
        
        if result and result.success:
            add_log_message("success", f"✅ Model training successful!")
            add_log_message("info", f"🔧 Heat transfer coeff: {model.k:.4f}, Seasonal amp: {model.seasonal_amp:.2f}°C")
        else:
            add_log_message("warning", "Model training failed, using default parameters")
    
    except Exception as e:
        add_log_message("warning", f"Model training error: {e}, using default parameters")
    
    # Generate predictions for future dates
    result_parts = []
    
    # Add actual data
    actual_data = reservoir_df.copy()
    actual_data['Type'] = 'Actual'
    result_parts.append(actual_data)
    
    # Add imputed historical data (if any gaps were filled)
    if 'imputed_historical' in locals() and len(imputed_historical) > 0:
        # Only add imputed data points (not the original actual ones)
        imputed_only = imputed_historical[
            (~imputed_historical['Date'].isin(reservoir_df['Date'])) & 
            (imputed_historical['Data_Source'] == 'Imputed')
        ].copy()
        
        if not imputed_only.empty:
            imputed_only['Type'] = 'Imputed'
            result_parts.append(imputed_only[['Date', 'Temperature', 'Type']])
    
    # Create predictions if we have future weather data
    if not future_weather.empty and len(future_weather) > 0:
        # Get the most recent water temperature as starting point (from imputed data if available)
        if 'imputed_historical' in locals() and len(imputed_historical) > 0:
            # Use the most recent imputed temperature (includes actual + filled gaps)
            current_water_temp = imputed_historical.iloc[-1]['Temperature']
            add_log_message("info", f"🎯 Starting forecast from: {current_water_temp:.1f}°C (most recent data point)")
        else:
            # Fallback to original reservoir data
            current_water_temp = reservoir_df.iloc[-1]['Temperature']
            add_log_message("info", f"🎯 Starting forecast from: {current_water_temp:.1f}°C (last actual reading)")
        
        # Prepare forecast data
        forecast_dates = future_weather['Date'].dt.to_pydatetime()
        forecast_air_temps = future_weather['Air_Temperature'].values
        
        # Generate water temperature forecast
        try:
            predicted_water_temps = model.forecast(
                forecast_air_temps, 
                current_water_temp,
                forecast_dates[0],
                len(forecast_dates)
            )
            
            # Create prediction dataframe
            predictions_df = pd.DataFrame({
                'Date': future_weather['Date'],
                'Temperature': predicted_water_temps,
                'Type': 'Predicted'
            })
            
            result_parts.append(predictions_df)
            add_log_message("info", f"🔮 Generated {len(predictions_df)} physics-based predictions")
            
        except Exception as e:
            add_log_message("warning", f"Prediction generation failed: {e}")
    
    # Combine all parts
    final_result = pd.concat(result_parts, ignore_index=True).sort_values('Date').reset_index(drop=True)
    
    return final_result


def create_temperature_predictions(reservoir_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create temperature predictions using multivariate weather and temporal data.
    
    Args:
        reservoir_df: Reservoir temperature data
        weather_df: Weather data
        
    Returns:
        pd.DataFrame: Combined data with predictions and imputed values
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df
    
    # Step 1: Separate historical and future weather data
    today = pd.Timestamp(datetime.now().date())
    historical_weather = weather_df[weather_df['Date'] <= today]
    future_weather_full = weather_df[weather_df['Date'] > today]
    
    if st.session_state.get('debug_mode', False):
        st.write(f"🔍 Debug: Historical weather: {len(historical_weather)} days")
        st.write(f"🔍 Debug: Future weather: {len(future_weather_full)} days")
    
    # Step 2: Impute missing water temperatures in the historical period only
    imputed_df = impute_missing_water_temperatures(reservoir_df, historical_weather)
    
    # Step 3: Create training data from imputed historical data
    merged_df = pd.merge(imputed_df, historical_weather, on='Date', how='inner')
    
    if len(merged_df) < 30:  # Need more data for multivariate model
        add_log_message("warning", "Not enough data for advanced predictions (need at least 30 days)")
        return reservoir_df
    
    # Create temporal features for training data (including water temperature)
    training_df = create_temporal_features(merged_df, 'Air_Temperature', 'Temperature')
    
    # Define feature columns (including water temperature features)
    feature_cols = [
        'Air_Temp_t0', 'Air_Temp_t1', 'Air_Temp_t2',
        'Air_Temp_7day', 'Air_Temp_30day',
        'Water_Temp_t1', 'Water_Temp_t2', 'Water_Temp_7day',
        'Season_Sin', 'Season_Cos'
    ]
    
    # Remove rows with NaN values (from lagged features)
    training_clean = training_df.dropna(subset=feature_cols + ['Temperature'])
    
    if len(training_clean) < 10:
        add_log_message("warning", "Not enough clean training data after creating features")
        return reservoir_df
    
    # Prepare training data
    X_train = training_clean[feature_cols].values
    y_train = training_clean['Temperature'].values
    
    # Train multivariate regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Step 4: Create predictions for future dates using the separated future weather
    # Debug info
    if st.session_state.get('debug_mode', False):
        st.write(f"🔍 Debug: Imputed data date range: {imputed_df['Date'].min().date()} to {imputed_df['Date'].max().date()}")
        st.write(f"🔍 Debug: Future weather data points: {len(future_weather_full)}")
        if not future_weather_full.empty:
            st.write(f"🔍 Debug: Future weather date range: {future_weather_full['Date'].min().date()} to {future_weather_full['Date'].max().date()}")
    
    if future_weather_full.empty:
        # No future predictions needed, just return imputed data with proper types
        result_df = imputed_df.copy()
        result_df['Type'] = result_df['Data_Source'].map({
            'Actual': 'Actual',
            'Imputed': 'Imputed'
        })
        if st.session_state.get('debug_mode', False):
            st.write("🔍 Debug: No future weather data available for predictions")
        return result_df[['Date', 'Temperature', 'Type']]
    
    # Step 5: Create future predictions iteratively
    # Combine imputed historical data with future weather
    extended_data = pd.concat([
        imputed_df.rename(columns={'Data_Source': 'Type'}),
        future_weather_full[['Date', 'Air_Temperature']].assign(Temperature=None, Type='Future')
    ], ignore_index=True).sort_values('Date').reset_index(drop=True)
    
    # Create features for extended data
    extended_features = create_temporal_features(extended_data, 'Air_Temperature', 'Temperature')
    
    # Predict iteratively for future dates
    future_predictions = []
    future_rows = extended_features[extended_features['Type'] == 'Future']
    
    if st.session_state.get('debug_mode', False):
        st.write(f"🔍 Debug: Future rows to predict: {len(future_rows)}")
    
    for idx, row in future_rows.iterrows():
        # Check if we have sufficient features
        row_features = [row[col] for col in feature_cols if not pd.isna(row[col])]
        available_feature_names = [col for col in feature_cols if not pd.isna(row[col])]
        
        if st.session_state.get('debug_mode', False):
            st.write(f"🔍 Debug: Date {row['Date'].date()}: Available features: {len(available_feature_names)}/{len(feature_cols)}")
            st.write(f"🔍 Debug: Missing features: {[col for col in feature_cols if pd.isna(row[col])]}")
        
        if len(row_features) >= 7:  # Need minimum features
            X_pred = np.array([row[col] for col in available_feature_names]).reshape(1, -1)
            
            # Use appropriate model
            if len(available_feature_names) == len(feature_cols):
                predicted_temp = model.predict(X_pred)[0]
                if st.session_state.get('debug_mode', False):
                    st.write(f"🔍 Debug: Used full model, predicted: {predicted_temp:.1f}°C")
            else:
                # Train simplified model with available features
                simple_training = training_clean[available_feature_names + ['Temperature']].dropna()
                if len(simple_training) >= 5:
                    simple_model = LinearRegression()
                    simple_model.fit(simple_training[available_feature_names].values, simple_training['Temperature'].values)
                    predicted_temp = simple_model.predict(X_pred)[0]
                    if st.session_state.get('debug_mode', False):
                        st.write(f"🔍 Debug: Used simplified model, predicted: {predicted_temp:.1f}°C")
                else:
                    if st.session_state.get('debug_mode', False):
                        st.write(f"🔍 Debug: Skipping - not enough training data for simplified model")
                    continue  # Skip this prediction
            
            future_predictions.append({
                'Date': row['Date'],
                'Temperature': predicted_temp,
                'Type': 'Predicted'
            })
            
            # Update extended_features for next iteration
            extended_features.loc[idx, 'Temperature'] = predicted_temp
            extended_features = create_temporal_features(extended_features, 'Air_Temperature', 'Temperature')
        else:
            if st.session_state.get('debug_mode', False):
                st.write(f"🔍 Debug: Skipping - only {len(row_features)} features available (need 7)")
    
    # Combine all data types
    result_parts = []
    
    # Original actual data
    actual_data = reservoir_df.copy()
    actual_data['Type'] = 'Actual'
    result_parts.append(actual_data)
    
    # Imputed data (excluding original actual dates)
    imputed_only = imputed_df[~imputed_df['Date'].isin(reservoir_df['Date'])].copy()
    if not imputed_only.empty:
        imputed_only['Type'] = 'Imputed'
        result_parts.append(imputed_only[['Date', 'Temperature', 'Type']])
    
    # Future predictions
    if future_predictions:
        future_df = pd.DataFrame(future_predictions)
        result_parts.append(future_df)
    
    # Combine all parts
    final_result = pd.concat(result_parts, ignore_index=True).sort_values('Date').reset_index(drop=True)
    
    # Display summary
    n_actual = len(actual_data)
    n_imputed = len(imputed_only) if not imputed_only.empty else 0
    n_predicted = len(future_predictions)
    
    feature_desc = "air temp (t, t-1, t-2), water temp (t-1, t-2), 7-day avgs, seasonal"
    add_log_message("info", f"🧠 Model trained with {len(feature_cols)} features: {feature_desc}")
    add_log_message("info", f"📊 Data: {n_actual} actual, {n_imputed} imputed, {n_predicted} predicted")
    
    return final_result

def display_statistics(df: pd.DataFrame) -> None:
    """Display temperature statistics in the sidebar."""
    st.sidebar.header("📊 Statistics")
    
    latest_temp = df.iloc[-1]['Temperature']
    latest_date = df.iloc[-1]['Date'].strftime('%d/%m/%Y')
    max_temp = df['Temperature'].max()
    min_temp = df['Temperature'].min()
    avg_temp = df['Temperature'].mean()
    
    st.sidebar.metric("Latest Temperature", f"{latest_temp}°C")
    st.sidebar.metric("Latest Reading", latest_date)
    st.sidebar.metric("Maximum", f"{max_temp}°C")
    st.sidebar.metric("Minimum", f"{min_temp}°C")
    st.sidebar.metric("Average", f"{avg_temp:.1f}°C")


def create_line_chart(df: pd.DataFrame) -> None:
    """Create and display the main temperature line chart."""
    st.subheader("📈 Temperature Over Time")
    
    # Check if we have different data types
    if 'Type' in df.columns:
        # Separate different data types
        actual_data = df[df['Type'] == 'Actual']
        imputed_data = df[df['Type'] == 'Imputed']
        predicted_data = df[df['Type'] == 'Predicted']
        
        # Start with actual data as scatter points
        fig = px.scatter(
            actual_data, 
            x='Date', 
            y='Temperature',
            title='West Reservoir Water Temperature (Actual, Imputed & Predicted)',
            labels={'Temperature': 'Temperature (°C)', 'Date': 'Date'}
        )
        
        # Update actual data trace to be points only
        fig.update_traces(
            name='Actual',
            marker=dict(color='#1f77b4', size=8),
            showlegend=True
        )
        
        # Add imputed data as a separate trace
        if not imputed_data.empty:
            fig.add_scatter(
                x=imputed_data['Date'],
                y=imputed_data['Temperature'],
                mode='lines+markers',
                name='Imputed',
                line=dict(color='green', dash='dot'),
                marker=dict(size=4, color='green'),
                showlegend=True
            )
        
        # Add predicted data as a separate trace
        if not predicted_data.empty:
            fig.add_scatter(
                x=predicted_data['Date'],
                y=predicted_data['Temperature'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='orange', dash='dash'),
                marker=dict(size=5, color='orange'),
                showlegend=True
            )
        
    else:
        fig = px.line(
            df, 
            x='Date', 
            y='Temperature',
            title='West Reservoir Water Temperature',
            labels={'Temperature': 'Temperature (°C)', 'Date': 'Date'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=2)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_monthly_analysis(df: pd.DataFrame) -> None:
    """Create monthly analysis charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗓️ Monthly Averages")
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly['Date'].dt.to_period('M')
        monthly_avg = df_monthly.groupby('Month')['Temperature'].mean().reset_index()
        monthly_avg['Month'] = monthly_avg['Month'].astype(str)
        
        if not monthly_avg.empty:
            fig_monthly = px.bar(
                monthly_avg, 
                x='Month', 
                y='Temperature',
                title='Average Temperature by Month',
                labels={'Temperature': 'Avg Temperature (°C)'}
            )
            fig_monthly.update_traces(marker_color='lightblue')
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            add_log_message("info", "Not enough data for monthly analysis")
    
    with col2:
        st.subheader("📊 Temperature Distribution")
        fig_hist = px.histogram(
            df, 
            x='Temperature', 
            nbins=20,
            title='Temperature Distribution',
            labels={'Temperature': 'Temperature (°C)', 'count': 'Frequency'}
        )
        fig_hist.update_traces(marker_color='lightcoral')
        st.plotly_chart(fig_hist, use_container_width=True)


def display_recent_data(df: pd.DataFrame) -> None:
    """Display recent temperature readings table."""
    st.subheader("📋 Recent Readings")
    recent_data = df.tail(10).copy()
    recent_data['Date'] = recent_data['Date'].dt.strftime('%d/%m/%Y')
    st.dataframe(recent_data.iloc[::-1], use_container_width=True, hide_index=True)


def create_download_button(df: pd.DataFrame) -> None:
    """Create CSV download button."""
    csv = df.to_csv(index=False)
    filename = f"west_reservoir_temperature_{datetime.now().strftime('%Y%m%d')}.csv"
    
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def create_forecast_tab(df: pd.DataFrame) -> None:
    """Create the short-term forecast tab content."""
    st.header("🔮 Short-term Temperature Forecast")
    
    # Filter data for the last 7 days and next 14 days
    today = pd.Timestamp(datetime.now().date())
    forecast_start = today - pd.Timedelta(days=7)
    forecast_end = today + pd.Timedelta(days=14)
    
    forecast_df = df[(df['Date'] >= forecast_start) & (df['Date'] <= forecast_end)].copy()
    
    if forecast_df.empty:
        add_log_message("warning", "No forecast data available for the selected period.")
        return
    
    # Show forecast chart
    st.subheader("📈 Recent History & 14-Day Forecast")
    
    if 'Type' in forecast_df.columns:
        # Separate different data types for the forecast period
        actual_data = forecast_df[forecast_df['Type'] == 'Actual']
        imputed_data = forecast_df[forecast_df['Type'] == 'Imputed']
        predicted_data = forecast_df[forecast_df['Type'] == 'Predicted']
        
        # Create forecast chart
        fig = px.scatter(
            actual_data, 
            x='Date', 
            y='Temperature',
            title='Short-term Temperature Forecast (Last 7 Days + Next 14 Days)',
            labels={'Temperature': 'Temperature (°C)', 'Date': 'Date'}
        )
        
        fig.update_traces(
            name='Actual',
            marker=dict(color='#1f77b4', size=8),
            showlegend=True
        )
        
        # Add imputed data
        if not imputed_data.empty:
            fig.add_scatter(
                x=imputed_data['Date'],
                y=imputed_data['Temperature'],
                mode='lines+markers',
                name='Imputed',
                line=dict(color='green', dash='dot'),
                marker=dict(size=4, color='green'),
                showlegend=True
            )
        
        # Add predicted data with emphasis
        if not predicted_data.empty:
            fig.add_scatter(
                x=predicted_data['Date'],
                y=predicted_data['Temperature'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='orange', dash='dash', width=3),
                marker=dict(size=6, color='orange'),
                showlegend=True
            )
            
            # Add vertical line for "today" using shapes instead of add_vline
            fig.add_shape(
                type="line",
                x0=today, x1=today,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="red", dash="dash"),
            )
            fig.add_annotation(
                x=today,
                y=1.02,
                yref="paper",
                text="Today",
                showarrow=False,
                font=dict(color="red", size=12)
            )
    else:
        # Fallback for data without type information
        fig = px.line(
            forecast_df, 
            x='Date', 
            y='Temperature',
            title='Temperature Trend (Last 7 Days)',
            labels={'Temperature': 'Temperature (°C)', 'Date': 'Date'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=2)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast summary
    if 'Type' in forecast_df.columns:
        predicted_data = forecast_df[forecast_df['Type'] == 'Predicted']
        if not predicted_data.empty:
            st.subheader("📋 Forecast Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_pred_temp = predicted_data['Temperature'].min()
                st.metric("Predicted Min", f"{min_pred_temp:.1f}°C")
            
            with col2:
                max_pred_temp = predicted_data['Temperature'].max()
                st.metric("Predicted Max", f"{max_pred_temp:.1f}°C")
            
            with col3:
                avg_pred_temp = predicted_data['Temperature'].mean()
                st.metric("Predicted Avg", f"{avg_pred_temp:.1f}°C")
            
            # Show forecast table
            st.subheader("📅 Detailed Forecast")
            forecast_table = predicted_data[['Date', 'Temperature']].copy()
            forecast_table.loc[:, 'Date'] = forecast_table['Date'].dt.strftime('%d/%m/%Y')
            forecast_table.loc[:, 'Temperature'] = forecast_table['Temperature'].round(1)
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        else:
            add_log_message("info", "No predictions available. Enable predictions in the sidebar to see forecasts.")


def create_historical_tab(df: pd.DataFrame, actual_df: pd.DataFrame) -> None:
    """Create the historical data and statistics tab content."""
    st.header("📊 Historical Data & Statistics")
    
    # Main chart (includes predictions if available)
    create_line_chart(df)
    
    # Monthly analysis (only actual data)
    create_monthly_analysis(actual_df)
    
    # Recent data table (only actual data)
    display_recent_data(actual_df)
    
    # Download data
    create_download_button(df)


def main() -> None:
    """Main application function."""
    st.title("🌡️ West Reservoir Temperature Tracker")
    st.markdown("Tracking water temperature at West Reservoir, London")
    
    # Sidebar options
    st.sidebar.header("🔮 Prediction Options")
    enable_predictions = st.sidebar.checkbox("Enable temperature predictions", value=True)
    
    if enable_predictions:
        prediction_method = st.sidebar.selectbox(
            "Prediction Method",
            ["Physics-based Model", "Statistical Model"],
            index=0,
            help="Physics-based: Uses degree-day accumulation model. Statistical: Uses machine learning with features."
        )
        
        imputation_method = st.sidebar.selectbox(
            "Gap Filling Method",
            ["Physics-based", "Statistical"],
            index=0,
            help="Physics-based: Uses thermal model for gaps. Statistical: Uses interpolation + regression."
        )
        
        # Store in session state for use in prediction functions
        st.session_state.use_physics_imputation = (imputation_method == "Physics-based")
    
    st.sidebar.header("🔧 Debug Options")
    debug_mode = st.sidebar.checkbox("Debug mode", value=False)
    st.session_state.debug_mode = debug_mode
    
    # Load reservoir data
    df = load_data()
    
    if df.empty:
        add_log_message("error", "No data available. Please check the data source.")
        return
    
    # If predictions enabled, get weather data and create predictions
    if enable_predictions:
        with st.spinner("Loading weather data for predictions..."):
            # Get date range for weather data
            start_date = df['Date'].min() - pd.Timedelta(days=30)  # Extra buffer for training
            end_date = datetime.now() + timedelta(days=14)  # Extended future predictions
            
            if debug_mode:
                st.write(f"🔍 Debug: Requesting weather data from {start_date.date()} to {end_date.date()}")
            
            weather_df = get_weather_data(start_date, end_date)
            
            if not weather_df.empty:
                if debug_mode:
                    st.write(f"🔍 Debug: Weather data shape: {weather_df.shape}")
                    st.write(f"🔍 Debug: Weather date range: {weather_df['Date'].min().date()} to {weather_df['Date'].max().date()}")
                    st.write(f"🔍 Debug: Today's date: {datetime.now().date()}")
                    future_weather = weather_df[weather_df['Date'] > datetime.now()]
                    st.write(f"🔍 Debug: Future weather data points: {len(future_weather)}")
                
                if prediction_method == "Physics-based Model":
                    df = create_temperature_predictions_physics(df, weather_df)
                else:
                    df = create_temperature_predictions(df, weather_df)
                prediction_count = len(df[df.get('Type', '') == 'Predicted']) if 'Type' in df.columns else 0
                if prediction_count > 0:
                    add_log_message("info", f"✨ Generated {prediction_count} temperature predictions based on weather data")
                else:
                    add_log_message("warning", "No future predictions generated - may need more recent weather data")
            else:
                add_log_message("warning", "Could not load weather data for predictions")
    
    # Display statistics (only for actual data)
    actual_df = df[df['Type'] == 'Actual'] if 'Type' in df.columns else df
    display_statistics(actual_df)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Short-term Forecast", "📈 Historical Data & Statistics"])
    
    with tab1:
        create_forecast_tab(df)
    
    with tab2:
        create_historical_tab(df, actual_df)
    
    # Display all accumulated log messages at the end
    display_log_messages()

if __name__ == "__main__":
    main()