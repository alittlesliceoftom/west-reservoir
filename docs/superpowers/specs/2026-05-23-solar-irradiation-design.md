# Solar Irradiation & Radiative Cooling — Design

**Date:** 2026-05-23
**Branch:** `feature-solar-irradiation`
**Status:** Approved

## Problem

The current water-temperature forecaster uses a single hourly air-temperature
term:

```
T_water(t+1h) = T_water(t) + k * (T_air(t) - T_water(t))
```

This misses two physically significant heat fluxes:

1. **Solar shortwave heating** — direct sunlight warms the reservoir during the
   day. Cloud cover and time of day modulate it.
2. **Clear-sky longwave cooling** — at night (and to a lesser extent in the
   day) water radiates infrared to the sky. Clouds re-radiate it back; clear
   skies let it escape, causing meaningful overnight cooling.

Adding these as fitted terms should improve forecast accuracy, especially on
clear days, clear nights, and during weather transitions.

## Goals

- Extend the existing hourly physics model with two additional terms (solar,
  radiative cooling) driven by Open-Meteo data.
- Fit all three coefficients (`k_air`, `k_solar`, `k_cool`) from measured
  water temperatures.
- Keep the existing module boundaries (`data.py`, `forecaster.py`, `app.py`).
- No new external secrets — Open-Meteo is key-less.

## Non-Goals

- Replacing Meteostat / OpenWeatherMap as the air-temperature sources.
- Full Stefan-Boltzmann radiative transfer modelling.
- Wind, humidity, evaporation, or precipitation terms (future work).
- Caching historical Open-Meteo archive data — re-fetched fresh per load to
  match the current Meteostat pattern.

## Model

### Equation

Per hour:

```
clearness(t) = 1 − cloud_cover(t) / 100
ΔT(t) = k_air   * (T_air(t) − T_water(t))      # conduction / convection
      + k_solar *  I(t)                         # shortwave heating
      − k_cool  *  clearness(t)                 # longwave cooling
T_water(t+1h) = T_water(t) + ΔT(t)
```

Where:
- `T_air(t)`     — hourly air temperature (°C), existing input
- `I(t)`         — shortwave irradiance at surface (W/m²), Open-Meteo
- `cloud_cover(t)` — total cloud cover (%), Open-Meteo
- Coefficients fitted via scipy `minimize` on measured water temps

### Coefficient bounds

| Param     | Units                  | Bounds            | Default | Sanity check                            |
|-----------|------------------------|-------------------|---------|-----------------------------------------|
| `k_air`   | per hour               | `(0.001, 0.1)`    | `0.02`  | Unchanged from current model            |
| `k_solar` | °C per (W/m²) per hour | `(1e-5, 1e-3)`    | `5e-4`  | Peak 1000 W/m² × 1e-3 = 1°C/hr max      |
| `k_cool`  | °C per hour            | `(0.0, 0.1)`      | `0.01`  | Fully clear sky → up to 0.1°C/hr loss   |

`k_cool ≥ 0` is enforced so the term can only cool (not invert into heating
under cloud).

## Data Layer (`data.py`)

Two new functions, matching the existing Meteostat function pair.

### `load_historical_solar_cloud(start_date, end_date) -> pd.DataFrame`

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Variables:** `shortwave_radiation`, `cloud_cover` (hourly)
- **Params:** `latitude`, `longitude`, `start_date`, `end_date`, `hourly=<vars>`, `timezone=UTC`
- **Returns:** DataFrame with columns `datetime`, `shortwave_radiation`, `cloud_cover`
- **Errors:** `DataLoadError` for timeout, HTTP failure, empty payload, or
  parse failure (same pattern as `load_hourly_air_temps`)
- **No caching** — re-fetched on every load.

### `load_forecast_solar_cloud(days=5) -> pd.DataFrame`

- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Variables:** `shortwave_radiation`, `cloud_cover` (hourly)
- **Params:** `latitude`, `longitude`, `hourly=<vars>`, `forecast_days=days`, `timezone=UTC`
- **Returns:** DataFrame with columns `datetime`, `shortwave_radiation`, `cloud_cover`
- **Errors:** `DataLoadError` analogous to `load_forecast_air_temps`
- **Caching:** stored in `forecast_storage.py` alongside the OWM forecast,
  using the same hour-level deduplication.

### Assembly

In `app.py`, immediately after the existing hourly-air-temp DataFrame is
built, merge the new solar/cloud DataFrame on `datetime`:

```python
hourly_weather = hourly_air_temps.merge(
    solar_cloud, on="datetime", how="inner"
)
forecaster.set_hourly_weather(hourly_weather)
```

A missing-value strategy is needed for the rare case Open-Meteo lacks an hour
the air-temp source has. **Approach:** inner-join then forward-fill up to 3
hours; drop pairs that still have NaNs from training. This avoids silent
zero-irradiance inputs that would bias `k_solar` low.

## Forecaster Changes (`forecaster.py`)

### Class signature

```python
class WaterTempForecaster:
    def __init__(self,
                 k_air: float = 0.02,
                 k_solar: float = 5e-4,
                 k_cool: float = 0.01):
        self.k_air = k_air
        self.k_solar = k_solar
        self.k_cool = k_cool
        self.hourly_weather: Optional[pd.DataFrame] = None
```

### Methods

| Old                          | New                                            |
|------------------------------|------------------------------------------------|
| `set_hourly_air_temps(df)`   | `set_hourly_weather(df)`                       |
| `_simulate_24h(t0, temps)`   | `_simulate_period(t0, weather_slice)`          |
| `predict_next_day(t0, temps)`| `predict_next_day(t0, weather_slice)`          |
| `explain_prediction(...)`    | Adds per-hour columns: `solar`, `cloud`, contributions `dT_air`, `dT_solar`, `dT_cool` |

`fit()` signature unchanged externally; internally optimises a 3-vector
under the bounds above.

### Inner simulation loop

```python
for row in weather_slice.itertuples():
    clearness = 1.0 - row.cloud_cover / 100.0
    water += (self.k_air   * (row.air_temp - water)
            + self.k_solar *  row.shortwave_radiation
            - self.k_cool  *  clearness)
```

If the inner loop becomes a profile hotspot during fitting, convert each
training pair's slice to three NumPy arrays once and vectorise the per-pair
inner integration as a Python loop over arrays (still sequential — each step
depends on the previous water temp). The 3-coefficient objective is otherwise
identical in shape to today's 1-coefficient version.

## App Integration (`app.py`)

Touch-points (all small):

1. **Data assembly:** call `load_historical_solar_cloud` and
   `load_forecast_solar_cloud` in the same code region that currently builds
   the hourly air-temp frame; merge on `datetime`.
2. **Forecaster wiring:** rename `set_hourly_air_temps` callsite to
   `set_hourly_weather`. No other call-site changes.
3. **Explain panel:** the per-hour breakdown table gains columns for
   `solar`, `cloud %`, and the three contributions (`dT_air`, `dT_solar`,
   `dT_cool`). No emojis (per CLAUDE.md). Use Title Case headers.

No new pages, tabs, or settings.

## Testing

### Unit — data layer
- `load_historical_solar_cloud`: mock Open-Meteo archive JSON; assert
  DataFrame shape, dtypes, sort order.
- `load_forecast_solar_cloud`: mock forecast JSON; same assertions.
- Error paths: timeout → `DataLoadError`; HTTP 4xx/5xx → `DataLoadError`;
  empty `hourly` payload → `DataLoadError`.

### Unit — forecaster
- Known-input simulation: hand-computed expected `ΔT` for a 1-hour step with
  all three terms active.
- Bounds enforcement: `fit()` never returns `k_cool < 0`.
- Coefficient recovery: generate synthetic water-temp series under known
  `(k_air, k_solar, k_cool)`, check `fit()` recovers within tolerance
  (~5% relative).

### No live API calls in CI.

## Migration / Rollback

- New branch `feature-solar-irradiation` off `develop`.
- Single PR. No DB migrations (cache table for solar/cloud forecast is
  created on first write).
- Rollback = revert PR; old single-coefficient model unchanged in git
  history.

## Open Questions

None at design time. If the fit produces a near-zero `k_solar` or `k_cool`
on real data, that's a useful empirical finding — leave the term in for
interpretability and document.
