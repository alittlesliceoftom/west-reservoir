# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple, transparent water temperature tracking and prediction system for West Reservoir, London.

**Key principle**: Single DataFrame architecture with explicit error handling.

### File Structure

```
├── app.py              - Streamlit web dashboard and tabs
├── config.py           - Configuration, API keys, feature flags
├── data.py             - Data loading and frame assembly
├── forecaster.py       - Physics-based prediction model
├── forecast_storage.py - MotherDuck forecast storage and retrieval
├── accuracy.py         - Forecast accuracy metrics and backtest replay
├── quotes.py           - Static quotes for the "Heard at the Res" tab
├── requirements.txt    - Python dependencies
├── docs/superpowers/   - Design specs and implementation plans
└── REBUILD_PLAN.md     - Original rebuild design documentation
```

Line counts are deliberately not recorded here: they go stale on every commit
and tell you nothing you cannot get from `wc -l`.

## Common Development Commands

### Python Version

**Python 3.14** (verified on 3.14.7). The project ran on 3.10 until September
2026; the whole dependency stack was upgraded together and verified to produce
byte-identical forecasts on both interpreters before switching.

Recreate the environment from scratch:

```bash
python3.14 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned deliberately. Upgrade them together, and re-run both
the suite and the real-data equivalence check before changing a pin — pandas in
particular has changed prediction-relevant defaults across major versions.

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
# Test forecaster (run the whole suite; it needs no network)
python3 -m pytest test_forecaster.py -v
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
2. Load historical air temps from Open-Meteo archive → `air_temp` column
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
- `load_historical_air_temps()` - Load daily air temps from Open-Meteo archive or raise `DataLoadError`
- `load_hourly_air_temps()` - Load hourly air temps from Open-Meteo archive or raise `DataLoadError`
- `load_forecast_air_temps()` - Load from OpenWeatherMap or raise `DataLoadError`
- All functions raise explicit errors with helpful messages

#### `forecaster.py`
- `WaterTempForecaster` class with a three-term hourly physics model
- Per hour:
  ```
  clearness(t) = 1 - cloud_cover(t) / 100
  T_water(t+1h) = T_water(t)
                + k_air   * (T_air(t) - T_water(t))   # conduction/convection
                + k_solar * I(t)                       # shortwave heating
                - k_cool  * clearness(t)               # clear-sky radiative cooling
  ```
- Measurements are taken at 7am, so every training and prediction period runs
  7am to 7am (24 hours)
- `fit()` - Optimise `k_air`, `k_solar` and `k_cool` together against measured data
- `predict_forward(anchor, temp, days_ahead=n | target_dates=[...])` - Forecast
  forward from a known temperature on a known date. One call returns every
  horizon asked for. This is the API to use for new work.
- `fill_predictions()` - Fill `AIR_ONLY` rows in a `temperatures` frame, built
  on `predict_forward`. Used by the dashboard's single-DataFrame flow.
- `explain_prediction()` - Return calculation breakdown for transparency
- Weather is supplied via `set_hourly_weather`, which requires air temp,
  shortwave radiation and cloud cover. Callers fill missing solar/cloud with
  zero solar and 100% cloud, which collapses the model to air conduction alone
- **No temperature constraints** - predicts physical values without artificial floors/ceilings

#### `forecast_storage.py`
- `ForecastStorage` class wrapping MotherDuck (DuckDB in the cloud)
- Stores air-temp forecasts (daily and 3-hourly) and water-temp predictions,
  so today's forecasts can be scored against tomorrow's measurements
- Retrieves stored forecasts to fill any gap between historical data and the
  live OpenWeatherMap forecast
- Gated by `ENABLE_MOTHERDUCK` in `config.py`; the app works without it
- `get_water_predictions_last_run_per_day()` and
  `get_air_forecasts_3hourly_last_run_per_day()` bulk-fetch the final run of
  each creation day for accuracy reporting. They use `rank()`, not
  `row_number()`: one run is many rows sharing a creation timestamp

#### `accuracy.py`
- `compute_metrics()` / `metrics_by_horizon()` - MAE, bias, RMSE, within-0.5C
  hit rate, and N. Bias is `forecast - actual`, so positive means the model
  runs warm. N is reported everywhere because measurement coverage is sparse
- `join_actuals()` - Join forecasts to the measurements they predicted, on
  target date
- `splice_air_history()` - Reconstruct the air series a stored forecast run
  actually had: measured air for the elapsed part of the creation day, forecast
  air after. A run created around 21:00 covers only that day's remainder, so
  simulating from it alone would step through ~10 hours of a 24-hour period
- `replay_current_model()` - Re-run the current model over history to show what
  accuracy would have been. A model-development tool, not a record of real
  performance: solar and cloud forecasts were never stored, so actual
  solar/cloud is used throughout, which biases it optimistically
- Weather reaches the replay through an injected provider, so all I/O stays in
  the caller and the replay is testable without MotherDuck

#### `app.py`
- Streamlit web interface
- Three tabs: Temperature, Forecast Accuracy, Heard at the Res
- The Forecast Accuracy tab is placed **before** the Temperature tab in code.
  The Temperature block calls `st.stop()` on a data error, and Streamlit runs
  tab bodies in code order, so anything after it would silently fail to render
  whenever temperature data is unavailable. It also fits its own model for the
  same reason, rather than borrowing the Temperature tab's forecaster
- The accuracy tab shades the Meteostat outage window (2026-03-20 onward) on
  its time-series charts for stored forecasts. Error in that window measures a
  dead feed, not the model - see issue #33. The replay reads the repaired
  archive, so the band is suppressed there
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

- **Historical Weather**: Open-Meteo archive API
  - Location: West Reservoir (51.566938, -0.090492)
  - Provides: Hourly and daily air temperature, shortwave radiation, cloud cover
  - Replaced Meteostat in Sept 2026: Meteostat moved hosting and left the old
    endpoint frozen since 2026-03-20, silently serving 5-month-old data (#33)

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
- **Debug scripts**: Place debug/test scripts in `debug/` directory (not in project root)

### Agent Workflow

**Use subagents whenever possible** to handle complex, multi-step tasks:

- **When to use subagents:**
  - Multi-step operations (testing, deployment, exploration)
  - Parallel tasks (can run multiple subagents simultaneously)
  - Complex searches or codebase exploration
  - Any task that can be delegated independently

- **Benefits:**
  - Reduces context usage in main conversation
  - Allows parallel execution
  - Specialized agents for specific tasks (Bash, Explore, Plan, etc.)
  - Cleaner conversation flow

- **Example:** Instead of running multiple grep/read commands directly, spawn an Explore subagent to investigate the codebase and return findings.

### Git Workflow

- **Merge PRs with rebase** (`gh pr merge --rebase`) for clean linear history.
- **After every PR merge**, fetch and rebase develop onto main so the next PR only shows new commits:
  ```bash
  git fetch --all && git rebase origin/main
  ```

### Pull Request Format

- **No test plans.** Do not include a "Test plan" checklist section in PR bodies — it will be deleted.
- If testing was actually performed, briefly state what was tested (e.g. "Tested locally: hovered over chart data points and confirmed tooltip format").
- Keep PR bodies concise: a short summary of what changed and why.

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
3. **Forecaster**: Test `fit()`, `predict_forward()`, `fill_predictions()`, and `explain_prediction()`
4. **Streamlit app**: `streamlit run app.py` and check browser
5. **Error cases**: Test with missing API key, unreachable URLs, etc.

## Common Issues

### Data source freshness
A source can go silent while the dashboard still renders - that is exactly how
Meteostat went unnoticed for five months (#33). Run the freshness checks:
```bash
python3 -m pytest test_data_freshness.py -v
```
Skip them (they hit the network) with `pytest -m "not freshness"`.

### "OpenWeatherMap API key not found"
This is expected if no API key is set. The app will show historical data only.
See "Weather API Setup" above to enable forecasts.

### "Cannot load required data: Failed to fetch data from Google Sheets"
Check internet connection and verify Google Sheets URL is accessible.

## Design Rationale

This is a complete rebuild of the original system. See `REBUILD_PLAN.md` for:
- Detailed design decisions
- Bug fixes (yesterday/today confusion, temperature floor artifact)
- Architecture comparison (old vs new)
- Success metrics and validation

### Key Improvements

| Aspect | Old | New |
|--------|-----|-----|
| DataFrames | 10+ intermediate | 1 main |
| Forecasting systems | 2 competing | 1 simple |
| Synthetic data | Throughout | None |
| Debug visibility | Toggle/hidden | Always visible |
| Temperature constraints | 0.1°C floor (buggy) | None (physics-based) |

## Future Enhancements

Potential improvements (not currently planned):
- Mobile-responsive layout improvements
- Data export functionality
- Multiple reservoir support
- Scheduled data-freshness alerting (issue #34)
- Storing solar and cloud forecasts, so the backtest replay stops leaking
  actual solar/cloud into historical runs (issue #29)

Keep changes aligned with core principle: **Simple, transparent, explicit**.
