# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple, transparent water temperature tracking and prediction system for West Reservoir, London.

**Key principle**: Single DataFrame architecture with explicit error handling.

### File Structure (649 lines total)

```
├── app.py            (210 lines) - Streamlit web dashboard
├── config.py         (50 lines)  - Configuration and API keys
├── data.py           (212 lines) - Data loading functions
├── forecaster.py     (177 lines) - Physics-based prediction model
├── requirements.txt              - Python dependencies
└── REBUILD_PLAN.md               - Full design documentation
```

## Common Development Commands

### Running the Application

```bash
# Activate virtual environment
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows

# Run the Streamlit application
streamlit run app.py

# For development with auto-reload
streamlit run app.py --server.runOnSave=true
```

### Testing

```bash
# Test module imports
source env/bin/activate
python3 -c "import config; import data; import forecaster; print('OK')"

# Test data loading
python3 -c "from data import load_water_temps; print(f'{len(load_water_temps())} readings loaded')"

# Test forecaster
python3 -c "from forecaster import WaterTempForecaster; f = WaterTempForecaster(); print(f.explain_prediction(12.0, 10.0))"
```

## Weather API Setup (Optional)

To enable weather forecasts and predictions:

1. **Get a free API key from OpenWeatherMap:**
   - Visit https://openweathermap.org/api
   - Sign up for a free account
   - Get your API key from the dashboard

2. **Set your API key (choose one method):**

   **Option A: Environment Variable**
   ```bash
   export OPENWEATHER_API_KEY="your_api_key_here"
   ```

   **Option B: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENWEATHER_API_KEY = "your_api_key_here"
   ```

3. **Without API key:** The app will show historical data only. Weather forecasts and predictions won't be available. The app will display a clear warning.

## Architecture

### Core Principle: Single DataFrame

The entire app works with ONE main DataFrame called `temperatures`:

```python
temperatures = pd.DataFrame({
    'date': pd.Timestamp,      # Date
    'water_temp': float,       # Water temperature (°C)
    'air_temp': float,         # Air temperature (°C)
    'source': str              # 'MEASURED' | 'AIR_ONLY' | 'PREDICTED'
})
```

### Data Flow

1. Load water temps from Google Sheets → `water_temp` column
2. Load historical air temps from Meteostat → `air_temp` column
3. Merge into single `temperatures` DataFrame
4. Load future air temps from OpenWeatherMap → extend DataFrame
5. Train forecaster on rows with `source == 'MEASURED'`
6. Predict water temps for rows with `source == 'AIR_ONLY'`
7. Update `source` to `'PREDICTED'` for those rows
8. Display the single `temperatures` DataFrame

### Module Responsibilities

#### `config.py`
- Configuration constants (URLs, coordinates)
- API key retrieval with explicit errors
- No silent fallbacks

#### `data.py`
- `load_water_temps()` - Load from Google Sheets or raise `DataLoadError`
- `load_historical_air_temps()` - Load from Meteostat or raise `DataLoadError`
- `load_forecast_air_temps()` - Load from OpenWeatherMap or raise `DataLoadError`
- All functions raise explicit errors with helpful messages

#### `forecaster.py`
- `WaterTempForecaster` class with simple physics model
- Physics equation: `dT/dt = k × (T_air_yesterday - T_water)`
- `fit()` - Train on measured data, optimize heat transfer coefficient `k`
- `predict()` - Generate predictions iteratively
- `explain_prediction()` - Return calculation breakdown for transparency
- **No temperature constraints** - predicts physical values without artificial floors/ceilings

#### `app.py`
- Streamlit web interface
- Single DataFrame workflow throughout
- Always-visible debug panel showing:
  - Data overview (measured vs predicted counts)
  - Model parameters
  - Tomorrow's calculation breakdown
  - Raw DataFrame view
- Clear date labeling (today vs yesterday vs tomorrow)
- Explicit error messages when data unavailable

### Data Sources

- **Water Temperature**: Google Sheets (manual measurements)
  - URL: `https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/export?format=csv&gid=0`
  - Format: DD/MM/YYYY, Temperature (°C)

- **Historical Weather**: Meteostat API (London weather station data)
  - Location: West Reservoir (51.566938, -0.090492)
  - Provides: Daily average air temperature

- **Weather Forecast**: OpenWeatherMap API (5-day forecast)
  - Requires API key (see setup above)
  - Provides: Daily air temperature predictions

## Development Guidelines

### Error Handling Philosophy

**Do NOT use synthetic/sample data fallbacks.** Always raise explicit errors:

```python
# ✅ Good
if data.empty:
    raise DataLoadError("No data available from Google Sheets")

# ❌ Bad
if data.empty:
    data = load_sample_data()  # Silent fallback
```

### Code Style

- Keep functions simple and focused
- Use explicit variable names
- Avoid creating intermediate DataFrames
- Raise errors with helpful messages
- No silent failures

### UI/UX Guidelines

- **No Emojis**: Keep interface professional and clean
- **Clear Labels**: Always explicit about data sources and dates
- **Transparent**: Show how predictions are calculated
- **Helpful Errors**: Guide users when things break

Examples:
- ✅ Correct: `st.header("Temperature Forecast")`
- ❌ Incorrect: `st.header("🔮 Temperature Forecast")`
- ✅ Correct: "Latest Reading (2026-01-13)"
- ❌ Incorrect: "Latest Reading" (ambiguous date)

## Testing Changes

When making changes, test:

1. **Module imports**: `python3 -c "import app"`
2. **Data loading**: Run each `load_*` function manually
3. **Forecaster**: Test `fit()`, `predict()`, and `explain_prediction()`
4. **Streamlit app**: `streamlit run app.py` and check browser
5. **Error cases**: Test with missing API key, unreachable URLs, etc.

## Common Issues

### "No module named 'meteostat'"
```bash
source env/bin/activate
pip install meteostat scipy
```

### "OpenWeatherMap API key not found"
This is expected if no API key is set. The app will show historical data only.
See "Weather API Setup" above to enable forecasts.

### "Cannot load required data: Failed to fetch data from Google Sheets"
Check internet connection and verify Google Sheets URL is accessible.

## Design Rationale

This is a complete rebuild of the original 3,576-line system. See `REBUILD_PLAN.md` for:
- Detailed design decisions
- Bug fixes (yesterday/today confusion, temperature floor artifact)
- Architecture comparison (old vs new)
- Success metrics and validation

### Key Improvements

| Aspect | Old | New |
|--------|-----|-----|
| Lines of code | 3,576 | 649 (82% reduction) |
| DataFrames | 10+ intermediate | 1 main |
| Forecasting systems | 2 competing | 1 simple |
| Synthetic data | Throughout | None |
| Debug visibility | Toggle/hidden | Always visible |
| Temperature constraints | 0.1°C floor (buggy) | None (physics-based) |

## Future Enhancements

Potential improvements (not currently planned):
- Automated testing suite
- Historical forecast accuracy tracking
- Mobile-responsive layout improvements
- Data export functionality
- Multiple reservoir support

Keep changes aligned with core principle: **Simple, transparent, explicit**.
