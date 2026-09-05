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

### Physics Model

The prediction system simulates the reservoir hour by hour, using three
additive heat terms:

```
clearness(t) = 1 - cloud_cover(t) / 100

T_water(t+1h) = T_water(t)
              + k_air   * (T_air(t) - T_water(t))   # conduction/convection
              + k_solar * I(t)                       # shortwave solar heating
              - k_cool  * clearness(t)               # clear-sky radiative cooling
```

Where:
- `k_air` = air/water heat transfer coefficient per hour
- `k_solar` = solar heating coefficient (°C per W/m² per hour)
- `k_cool` = clear-sky cooling rate (°C per hour at 100% clearness)
- `I(t)` = shortwave radiation at hour `t`

All three coefficients are optimised together during training. Water
temperature is measured at 7am, so every simulation period runs 7am to 7am.

When solar and cloud data are unavailable the model degrades gracefully to the
original single-term physics (zero solar, fully overcast).

### Prediction Process

1. **Load Data**: Water temps from Google Sheets, air temps from Meteostat,
   solar radiation and cloud cover from Open-Meteo, forecasts from OpenWeatherMap
2. **Merge**: Combine into single `temperatures` DataFrame, plus an hourly
   weather frame for the model
3. **Train**: Optimise `k_air`, `k_solar` and `k_cool` on measured data
4. **Predict**: Simulate hour by hour, 7am to 7am, for future days
5. **Store**: Save the forecast to MotherDuck so it can later be scored
   against what was actually measured
6. **Display**: Show measured (blue) and predicted (orange) temperatures

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

The app is a small set of focused modules:

- `config.py`: Configuration, API keys, feature flags
- `data.py`: Data loading and frame assembly, with explicit error handling
- `forecaster.py`: Physics-based prediction model
- `forecast_storage.py`: Stores and retrieves forecasts in MotherDuck
- `quotes.py`: Static quotes for the "Heard at the Res" tab
- `app.py`: Streamlit UI with debug panel

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
