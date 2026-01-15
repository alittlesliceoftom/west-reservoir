# West Reservoir Temperature Tracker

A simple, transparent Streamlit dashboard for tracking and predicting water temperature at West Reservoir, London.

## Features

- Real-time data loading from Google Sheets
- Interactive temperature visualizations with clear data source labels
- Simple physics-based temperature prediction
- **Always-visible debug panel** showing exact calculations
- No synthetic data - explicit errors when data unavailable

## Quick Start

1. Activate the virtual environment:
```bash
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate     # On Windows
```

2. Install dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Weather Forecast Setup (Optional)

To get real weather forecasts, set your OpenWeatherMap API key:

**Option A: Environment Variable**
```bash
export OPENWEATHER_API_KEY="your_api_key_here"
```

**Option B: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

**Without API key**: The app will show historical data only. Weather forecasts and predictions won't be available.

Get a free API key at: https://openweathermap.org/api

## How It Works

### Simple Physics Model

The prediction system uses a straightforward heat transfer equation:

```
dT/dt = k × (T_air_yesterday - T_water)
```

Where:
- `k` = heat transfer coefficient (optimized during training)
- `T_air_yesterday` = Previous day's air temperature
- `T_water` = Current water temperature

### Prediction Process

1. **Load Data**: Water temps from Google Sheets, air temps from Meteostat
2. **Merge**: Combine into single `temperatures` DataFrame
3. **Train**: Optimize heat transfer coefficient `k` on measured data
4. **Predict**: Apply physics equation iteratively for future days
5. **Display**: Show measured (blue) and predicted (orange) temperatures

### Data Sources

Each temperature reading is labeled with its source:
- **MEASURED**: Actual water temperature from Google Sheets
- **PREDICTED**: Calculated using physics model + weather forecast

### Debug Panel

The debug panel (always visible) shows:
- **Data Overview**: Count of measured vs predicted values
- **Model Parameters**: Heat transfer coefficient and equation
- **Tomorrow's Calculation**: Step-by-step breakdown showing:
  - Current water temperature
  - Yesterday's air temperature
  - Temperature difference
  - Predicted temperature change
  - Final prediction
- **Raw Data Table**: Last 10 rows of the temperatures DataFrame

This makes the prediction process completely transparent and reproducible.

## Architecture

The app consists of 4 simple modules (649 lines total):

- `config.py` (50 lines): Configuration and API keys
- `data.py` (212 lines): Data loading with explicit error handling
- `forecaster.py` (177 lines): Physics-based prediction model
- `app.py` (210 lines): Streamlit UI with debug panel

**Key principle**: Single DataFrame throughout (`temperatures` with columns: date, water_temp, air_temp, source)

## Error Handling

The app raises clear errors instead of silent fallbacks:

- **Google Sheets unavailable**: "Cannot load required data: Failed to fetch data from Google Sheets"
- **No API key**: "Weather forecast unavailable: OpenWeatherMap API key not found"
- **Meteostat down**: "Cannot load required data: Failed to load historical weather data"

## Bug Fixes

This rebuild fixes two critical bugs from the previous version:

1. **Yesterday vs Today confusion**: Now explicitly labels dates and shows warnings when today's data is missing
2. **Temperature floor artifact**: Removed artificial 0.1°C constraint that caused forecasts to increase near freezing

## Data Sources

- **Water Temperature**: Google Sheets (manual readings)
- **Historical Weather**: Meteostat API (London weather station)
- **Weather Forecast**: OpenWeatherMap API (5-day forecast)
- **Location**: West Reservoir, London (51.566938, -0.090492)

## Development

See `REBUILD_PLAN.md` for full design documentation and rationale.
