# Plan: Enhanced Physics Model with Solar, Wind, and Radiative Cooling

**Status: STASHED for future implementation**

## Goal
Add physical variables to reduce seasonal prediction bias:
- **Sunshine duration (tsun)** → direct solar heating
- **Wind speed (wspd)** → modifies heat transfer rate
- **Cloud cover at night (coco)** → long-wave radiative cooling

## Current Model
```
T_water(t+1h) = T_water(t) + k × (T_air - T_water)
```
Single parameter `k` optimized on all data. Causes seasonal bias because solar radiation not captured.

## Proposed Model

### Physics Equations

**1. Solar Heating (daytime only)**
```
solar_gain = k_sun × tsun × solar_intensity(hour, day_of_year)
```
- `tsun`: sunshine minutes this hour (0-60)
- `solar_intensity`: varies by hour (peak at noon) and season (higher in summer)
- Only applied during daylight hours (approx 6am-8pm, varies by season)

**2. Wind-Modified Heat Transfer**
```
k_effective = k × (1 + k_wind × wspd)
```
- Higher wind → faster equilibration with air temp
- `wspd`: wind speed in km/h
- This is physically correct: wind increases convective heat transfer

**3. Radiative Cooling (nighttime, clear sky)**
```
radiative_loss = k_rad × (1 - cloud_fraction) × night_factor
```
- Clear nights: water radiates heat to cold sky
- Cloudy nights: clouds act as blanket, reduce heat loss
- `cloud_fraction`: from coco (0-1 scale)
- `night_factor`: 1 at night, 0 during day

### Combined Equation
```
T_water(t+1h) = T_water(t)
               + k_eff × (T_air - T_water)   # wind-modified transfer
               + solar_gain                   # solar heating (day)
               - radiative_loss               # sky cooling (night)
```

## Data Availability

| Variable | Meteostat API | Availability |
|----------|---------------|--------------|
| wspd (wind speed) | Hourly | Good coverage |
| tsun (sunshine mins) | Hourly | Available but may have gaps |
| coco (weather code) | Hourly | Need to map to cloud fraction |

### Weather Code (coco) to Cloud Fraction Mapping
```python
CLOUD_FRACTION = {
    1: 0.0,   # Clear
    2: 0.25,  # Fair
    3: 0.5,   # Cloudy
    4: 0.75,  # Overcast
    5: 0.9,   # Fog
    6: 0.8,   # Freezing Fog
    7: 0.6,   # Light Rain
    8: 0.7,   # Rain
    ...
}
```

## Implementation Steps

### Step 1: Update Data Loading (`data.py`)
- Modify `load_hourly_air_temps()` to include `wspd`, `tsun`, `coco`
- Handle missing values (interpolate or use defaults)
- Add cloud fraction mapping function

### Step 2: Update Forecaster (`forecaster.py`)

**2a. Add new class constants:**
```python
MEASUREMENT_HOUR = 7
MAX_TRAINING_GAP_DAYS = 3
# New coefficients (initial values, will be optimized)
DEFAULT_K_SUN = 0.01    # solar heating coefficient
DEFAULT_K_WIND = 0.05   # wind effect on k
DEFAULT_K_RAD = 0.02    # radiative cooling coefficient
```

**2b. Modify `_simulate_24h()` to accept and use new variables**

**2c. Add helper methods:**
- `_get_solar_intensity(hour, day_of_year)` → returns 0-1 factor
- `_is_night(hour, day_of_year)` → returns True/False
- `_cloud_fraction_from_coco(coco)` → returns 0-1

**2d. Update `fit()` to optimize multiple coefficients**
- Optimize: `k, k_sun, k_wind, k_rad`
- Use scipy.optimize.minimize with bounds

**2e. Update `predict()` and `explain_prediction()`**

### Step 3: Update Training Explorer (`app_training.py`)
- Add visualizations for new variables vs errors
- Show optimized coefficient values
- Add toggles to enable/disable each effect

### Step 4: Update Main App (`app.py`)
- Update debug panel to show new coefficients
- Update model description text

## Files to Modify
1. `data.py` - load additional Meteostat columns
2. `forecaster.py` - enhanced physics model
3. `app_training.py` - visualize new variables
4. `app.py` - display new model info

## Verification
1. Run `app_training.py` and check:
   - Seasonal bias reduced (residuals more balanced)
   - MAE improved
   - New coefficients shown in sidebar
2. Compare error by month before/after
3. Test edge cases (missing tsun data, etc.)

## Questions to Resolve
- Should we optimize all 4 coefficients together or incrementally add them?
- How to handle missing tsun/coco data (common in Meteostat)?
- Should we add these to forecast (OpenWeatherMap) data too?
