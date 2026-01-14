# West Reservoir Temperature Tracker - Rebuild Plan

## Current Problems Identified

### 1. **Synthetic/Sample Data Everywhere**
- Sample fallback data in `west_reservoir_tracker.py:249-262`
- Synthetic weather generation in `generate_synthetic_weather_fallback()`
- Synthetic data in `forecast.py:256`
- Falls back silently instead of raising errors

### 2. **Too Many Forecasting Systems**
- Physics-based model (`create_temperature_predictions_physics()`)
- Statistical multi-feature model (`create_temperature_predictions()`)
- Both use different approaches and create confusion
- Both create many intermediate variables and dataframes

### 3. **Critical Bugs**

#### Bug A: Forecast shows yesterday's value if no data for today
**Root cause**: In the forecasting logic, when there's no data for "today", the model starts from yesterday's value as the "current" value. The code doesn't distinguish between "yesterday's reading" and "today's prediction".

**Location**: Lines 1192-1206 in `west_reservoir_tracker.py`
```python
if "imputed_historical" in locals() and len(imputed_historical) > 0:
    current_water_temp = imputed_historical.iloc[-1]["Temperature"]  # Could be yesterday
```

#### Bug B: Forecast increases when approaching 0°C
**Root cause**: Physical constraint `max(0.1, water_temps[i])` in `forecast.py:75`

When the temperature is calculated to be below 0.1°C, it gets clamped to 0.1°C. This creates an artificial floor that prevents the model from following the natural cooling trend, causing the forecast to bounce off the floor and appear to increase.

```python
# Line 75 in forecast.py
water_temps[i] = max(0.1, water_temps[i])  # Above freezing
```

This constraint is physically wrong for a reservoir that won't freeze due to thermal mass and salinity.

### 4. **Too Many Variables and DataFrames**
- Code creates: `reservoir_df`, `weather_df`, `historical_weather`, `future_weather`, `imputed_df`, `merged_df`, `training_df`, `predictions_df`, `result_parts`, `final_result`
- No single source of truth
- Hard to debug what data is being used where

### 5. **Complexity**
- 2,970 lines in main app
- 606 lines in forecast module
- Community submission feature (disabled but still in code)
- Multiple imputation strategies
- Debug modes and logging systems

---

## Rebuild Architecture

### Core Principle: **Single DataFrame, Explicit Operations**

The entire app will work with ONE main DataFrame called `temperatures` that contains:
- `date`: The date
- `water_temp`: Water temperature (actual readings)
- `air_temp`: Air temperature (historical or forecast)
- `source`: Enum of `MEASURED | AIR_ONLY | PREDICTED`

### Data Flow

```
1. Load actual water temps from Google Sheets
   ↓
2. Load historical air temps (Meteostat)
   ↓
3. Merge into main temperatures DataFrame
   ↓
4. Load future air temps (OpenWeatherMap)
   ↓
5. Extend temperatures DataFrame with future dates
   ↓
6. Train forecaster on rows with MEASURED water temps
   ↓
7. Predict water temps for rows with AIR_ONLY
   ↓
8. Update source to PREDICTED for those rows
   ↓
9. Display the single temperatures DataFrame
```

### File Structure

```
west_reservoir/
├── app.py                 # Streamlit app (< 200 lines)
├── data.py               # Data loading only (< 100 lines)
├── forecaster.py         # Single forecasting class (< 150 lines)
├── config.py             # Configuration
└── requirements.txt
```

### Module Responsibilities

#### `config.py`
```python
GOOGLE_SHEETS_URL = "..."
RESERVOIR_LAT = 51.566938
RESERVOIR_LON = -0.090492

# Raise errors if API keys missing
def get_openweather_key() -> str:
    """Get API key or raise ConfigError"""
```

#### `data.py`
```python
class DataLoadError(Exception):
    """Raised when data cannot be loaded"""

def load_water_temps() -> pd.DataFrame:
    """Load from Google Sheets or raise DataLoadError"""

def load_historical_air_temps(start_date, end_date) -> pd.DataFrame:
    """Load from Meteostat or raise DataLoadError"""

def load_forecast_air_temps(days: int) -> pd.DataFrame:
    """Load from OpenWeatherMap or raise DataLoadError"""
```

#### `forecaster.py`
```python
class WaterTempForecaster:
    """Simple physics-based water temperature forecaster"""

    def __init__(self, heat_transfer_coeff: float = 0.05):
        self.k = heat_transfer_coeff

    def fit(self, temperatures: pd.DataFrame) -> None:
        """Train on rows where source == MEASURED"""

    def predict_next_day(self, current_water_temp: float,
                         yesterday_air_temp: float,
                         today_air_temp: float) -> float:
        """
        Predict tomorrow's water temp.

        Makes the calculation explicit and reproducible:
        temp_change = k * (yesterday_air_temp - current_water_temp)
        tomorrow_water_temp = current_water_temp + temp_change

        No constraints, no clamping - just the physics.
        """
        temp_diff = yesterday_air_temp - current_water_temp
        temp_change = self.k * temp_diff
        return current_water_temp + temp_change

    def predict(self, temperatures: pd.DataFrame) -> pd.DataFrame:
        """
        Predict water temps for all rows where source == AIR_ONLY.

        Iterates day by day, using each prediction as input for next day.
        Updates the DataFrame in place.
        """
```

#### `app.py`
```python
def main():
    st.title("West Reservoir Temperature Tracker")

    try:
        # Load data
        water_temps = load_water_temps()

        # Build main DataFrame
        start_date = water_temps['date'].min()
        end_date = datetime.now().date()

        air_temps_hist = load_historical_air_temps(start_date, end_date)

        # Merge
        temperatures = pd.merge(
            water_temps,
            air_temps_hist,
            on='date',
            how='outer'
        ).sort_values('date')

        # Mark sources
        temperatures['source'] = 'MEASURED'
        temperatures.loc[temperatures['water_temp'].isna(), 'source'] = 'AIR_ONLY'

        # Add forecast dates
        try:
            forecast_air = load_forecast_air_temps(days=5)
            forecast_rows = forecast_air.copy()
            forecast_rows['water_temp'] = None
            forecast_rows['source'] = 'AIR_ONLY'
            temperatures = pd.concat([temperatures, forecast_rows])
        except DataLoadError as e:
            st.warning(f"No forecast available: {e}")

        # Train and predict
        forecaster = WaterTempForecaster()
        forecaster.fit(temperatures[temperatures['source'] == 'MEASURED'])
        temperatures = forecaster.predict(temperatures)

        # Display
        display_chart(temperatures)
        display_metrics(temperatures)
        display_debug_info(temperatures)  # Show the actual DataFrame

    except DataLoadError as e:
        st.error(f"Cannot load required data: {e}")
        st.stop()
```

---

## Bug Fixes

### Fix A: Yesterday vs Today Clarity
**Solution**: The forecaster will be explicit about dates and labels:

```python
# In display_chart():
today = datetime.now().date()

# Split by date
historical = temperatures[temperatures['date'] < today]
today_row = temperatures[temperatures['date'] == today]
future = temperatures[temperatures['date'] > today]

# Label clearly
if today_row.empty:
    st.warning("No measurement for today yet. Showing yesterday's reading.")
    latest_reading = historical.iloc[-1]
    st.metric("Latest Reading", f"{latest_reading['water_temp']:.1f}°C",
              label=f"Yesterday ({latest_reading['date']})")
else:
    latest = today_row.iloc[0]
    st.metric("Today's Temperature", f"{latest['water_temp']:.1f}°C")
```

### Fix B: Remove Temperature Floor
**Solution**: Remove the `max(0.1, ...)` constraint entirely from the forecast.

West Reservoir is a large urban reservoir that:
1. Has thermal mass (slow to cool)
2. Won't freeze solid in UK winters
3. Might have salinity/minerals that lower freezing point
4. Is in London (rarely below -5°C ambient)

The physics model should be allowed to predict any temperature. If it predicts below 0°C, that's fine - we can display a warning but shouldn't clamp the value.

```python
def predict_next_day(self, current_water_temp, yesterday_air, today_air):
    temp_diff = yesterday_air - current_water_temp
    temp_change = self.k * temp_diff
    predicted = current_water_temp + temp_change

    # No clamping - return the physics prediction
    return predicted
```

---

## Debuggability: Explicit Tomorrow Calculation

The forecaster will have a method specifically for debugging:

```python
def explain_prediction(self, current_water_temp: float,
                       yesterday_air_temp: float,
                       today_air_temp: float) -> dict:
    """
    Returns a breakdown of tomorrow's prediction for debugging.

    Returns:
        {
            'current_water_temp': float,
            'yesterday_air_temp': float,
            'temperature_difference': float,
            'heat_transfer_coefficient': float,
            'temperature_change': float,
            'predicted_water_temp': float
        }
    """
    temp_diff = yesterday_air_temp - current_water_temp
    temp_change = self.k * temp_diff
    predicted = current_water_temp + temp_change

    return {
        'current_water_temp': current_water_temp,
        'yesterday_air_temp': yesterday_air_temp,
        'temperature_difference': temp_diff,
        'heat_transfer_coefficient': self.k,
        'temperature_change': temp_change,
        'predicted_water_temp': predicted
    }
```

And in the app:

```python
# Debug expander
with st.expander("🔍 How is tomorrow's temperature calculated?"):
    latest = temperatures[temperatures['source'] == 'MEASURED'].iloc[-1]
    tomorrow = temperatures[temperatures['source'] == 'PREDICTED'].iloc[0]

    explanation = forecaster.explain_prediction(
        current_water_temp=latest['water_temp'],
        yesterday_air_temp=latest['air_temp'],
        today_air_temp=tomorrow['air_temp']
    )

    st.write(f"**Current water temp**: {explanation['current_water_temp']:.2f}°C")
    st.write(f"**Yesterday's air temp**: {explanation['yesterday_air_temp']:.2f}°C")
    st.write(f"**Temperature difference**: {explanation['temperature_difference']:.2f}°C")
    st.write(f"**Heat transfer rate (k)**: {explanation['heat_transfer_coefficient']:.4f}")
    st.write(f"**Predicted change**: {explanation['temperature_change']:.2f}°C")
    st.write(f"**→ Tomorrow's prediction**: {explanation['predicted_water_temp']:.2f}°C")
```

---

## Debug Panel (Core Feature)

The debug panel will be **always visible** (no toggle) and provide full transparency:

```python
def display_debug_panel(temperatures: pd.DataFrame, forecaster: WaterTempForecaster):
    """Display comprehensive debug information."""

    with st.expander("🔍 Debug Information", expanded=True):
        # Section 1: Data Overview
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            measured = len(temperatures[temperatures['source'] == 'MEASURED'])
            st.metric("Measured Readings", measured)
        with col2:
            predicted = len(temperatures[temperatures['source'] == 'PREDICTED'])
            st.metric("Predicted Values", predicted)
        with col3:
            st.metric("Total Rows", len(temperatures))

        # Section 2: Model Parameters
        st.subheader("Model Parameters")
        st.write(f"**Heat transfer coefficient (k)**: {forecaster.k:.4f} day⁻¹")
        st.write("**Physics equation**: dT/dt = k × (T_air_yesterday - T_water)")

        # Section 3: Tomorrow's Calculation (if available)
        if any(temperatures['source'] == 'PREDICTED'):
            st.subheader("Tomorrow's Calculation Breakdown")
            latest = temperatures[temperatures['source'] == 'MEASURED'].iloc[-1]
            tomorrow = temperatures[temperatures['source'] == 'PREDICTED'].iloc[0]

            explanation = forecaster.explain_prediction(
                current_water_temp=latest['water_temp'],
                yesterday_air_temp=latest['air_temp'],
                today_air_temp=tomorrow['air_temp']
            )

            st.code(f"""
Current water temperature:    {explanation['current_water_temp']:.2f}°C
Yesterday's air temperature:  {explanation['yesterday_air_temp']:.2f}°C
Temperature difference:       {explanation['temperature_difference']:.2f}°C
Heat transfer rate (k):       {explanation['heat_transfer_coefficient']:.4f}
Temperature change:           {explanation['temperature_change']:.2f}°C
────────────────────────────────────────────
Tomorrow's predicted temp:    {explanation['predicted_water_temp']:.2f}°C
            """)

        # Section 4: Raw DataFrame
        st.subheader("Raw Data (Last 10 Rows)")
        st.dataframe(
            temperatures.tail(10)[['date', 'water_temp', 'air_temp', 'source']],
            use_container_width=True
        )
```

This makes the app **fully transparent** - users can always see:
1. What data is measured vs predicted
2. Exact model parameters
3. Step-by-step calculation for tomorrow
4. Raw DataFrame for verification

---

## What Gets Deleted

### Files to Remove
- `west_reservoir_tracker.py` (replaced by simple `app.py`)
- `forecast.py` (replaced by simple `forecaster.py`)
- `get_weather.py` (logic moved to `data.py`)
- `v2/` directory (incomplete alternative approach)
- `tests/test_data_parsing.py` (will write new tests)

### Code Patterns to Remove
- All sample/synthetic data generation
- All silent fallbacks
- Community submission feature
- Statistical multi-feature model
- Multiple imputation strategies
- Session state logging system
- Data validation beyond "is it valid data?"

### Code Patterns to Keep/Improve
- **Debug panel** - Keep and enhance! Make it always visible (remove the toggle) and show:
  - The temperatures DataFrame structure
  - Calculation breakdown for tomorrow's prediction
  - Data source information
  - Model parameters

---

## Git Workflow

All rebuild work will be done on a new branch starting from main:

```bash
# Switch to main branch first
git checkout main

# Create and checkout new branch from main
git checkout -b rebuild-simplify

# All development happens here
# main branch: preserved as-is
# New branch: rebuild-simplify (all new code)
```

**Branch strategy**:
- `main` - starting point (clean slate)
- `rebuild-simplify` (new) - all rebuild work
- `july-tidy-up` (existing) - preserved for reference if needed
- When complete and tested, can be merged back to main

**Commit strategy**:
- One commit per step (Step 1, Step 2, etc.)
- Clear commit messages describing what was added/removed
- Each commit should be a working state (tests pass)

---

## First Next Steps (Incremental)

### Step 0: Create new branch from main
**Goal**: Set up clean workspace for rebuild starting from main

**Actions**:
```bash
# Switch to main branch
git checkout main

# Create new branch from main
git checkout -b rebuild-simplify
```

### Step 1: Create `config.py` and `data.py`
**Goal**: Load real data or raise errors. No fallbacks.

**Acceptance criteria**:
- `load_water_temps()` returns DataFrame or raises `DataLoadError`
- `load_historical_air_temps()` returns DataFrame or raises `DataLoadError`
- `load_forecast_air_temps()` returns DataFrame or raises `DataLoadError`
- All functions have explicit error messages

### Step 2: Create simple `forecaster.py`
**Goal**: Single physics model, no constraints, fully debuggable

**Acceptance criteria**:
- `WaterTempForecaster` class with `fit()` and `predict()` methods
- `explain_prediction()` method returns calculation breakdown
- No temperature clamping
- Uses i-1 indexing correctly (yesterday's air temp)

### Step 3: Create minimal `app.py`
**Goal**: Single DataFrame workflow, explicit about dates, always-visible debug panel

**Acceptance criteria**:
- Builds one `temperatures` DataFrame
- Clearly shows whether today has data or not
- **Always-visible debug panel** with:
  - Data overview (measured vs predicted counts)
  - Model parameters display
  - Tomorrow's calculation breakdown
  - Raw DataFrame view (last 10 rows)
- < 200 lines of code

### Step 4: Test with missing data scenarios
**Goal**: Verify error handling works

**Test cases**:
- Google Sheets unreachable → Error shown, app stops
- No OpenWeather API key → Forecast section shows error, rest of app works
- Meteostat down → Error shown, app stops

### Step 5: Remove all old code
**Goal**: Clean slate

**Actions**:
- Delete old files
- Update README
- Update requirements.txt
- Create simple tests for the new modules

---

## Success Metrics

### Simplicity
- ✅ Total code < 500 lines (vs 3,576 currently)
- ✅ Single DataFrame throughout
- ✅ No synthetic data
- ✅ One forecasting approach

### Debuggability
- ✅ User can see exact calculation for tomorrow's temp
- ✅ Can inspect the temperatures DataFrame
- ✅ Clear labels for "today" vs "yesterday" vs "tomorrow"
- ✅ Debug panel always visible with full transparency
- ✅ Model parameters and data sources clearly shown

### Correctness
- ✅ Bug A fixed: Always clear about which date each value represents
- ✅ Bug B fixed: No artificial temperature floor
- ✅ Predictions use yesterday's air temp (i-1 indexing correct)

### Reliability
- ✅ Errors are raised, not silently handled
- ✅ User knows when data is missing
- ✅ User knows when forecast is unavailable

---

## Migration Notes

### What Users Lose
- Fallback sample data (intentional)
- Statistical multi-feature predictions (too complex)
- Community submissions (disabled anyway)

### What Users Gain
- **Always-visible debug panel** showing exact calculations and data sources
- Full understanding of how predictions work (no black box)
- Confidence in the data shown (measured vs predicted clearly labeled)
- Faster app performance (simpler code, less computation)
- Easier to contribute/maintain (< 500 lines vs 3,576)
- Clear error messages when things break (no silent fallbacks)

### Backwards Compatibility
None needed - this is a full rebuild. Current functionality is preserved but implementation is completely new.

---

## Timeline

Not providing time estimates per project instructions. Steps are ordered by dependency:
0. Create new branch (rebuild-simplify)
1. Config + Data loading
2. Forecaster
3. App
4. Testing
5. Cleanup

Each step can be completed and tested independently.
