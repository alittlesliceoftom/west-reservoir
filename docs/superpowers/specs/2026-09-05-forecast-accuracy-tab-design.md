# Forecast Accuracy Reporting Tab — Design

Date: 2026-09-05
Status: Approved for planning

## Purpose

Add a "Forecast Accuracy" tab that answers one question: how good are our
water-temperature forecasts, and how does that degrade as we forecast
further ahead?

Two independent comparisons are built, side by side:

1. **Stored forecasts** — the forecasts we actually made and published,
   scored against subsequent measurements. This is the honest record of
   real-world performance.
2. **Model replay** — the current model re-run over history. This is a
   model-development tool: it shows what accuracy *would have been* had
   today's model been running all along.

The two answer different questions and must never be silently merged.

## Data Landscape (verified 2026-09-05)

MotherDuck `west_reservoir`:

| Table | Useful range | Notes |
|---|---|---|
| `water_temp_predictions` | 2026-02-15 → 2026-09-10 | ~200 creation days, horizons 0–5 |
| `air_temp_forecasts_3hourly` | 2026-02-16 → 2026-09-10 | ~200 creation days, horizons 0–5 |

- Roughly **15 forecast runs per creation day** (hourly cadence).
- **735 distinct `heat_transfer_coeff` values** — the model refits on every run.
- `water_temp_predictions` also contains a large block of rows where
  `target_date < forecast_created_date` (horizons down to −665). These are
  historical backfill written alongside real forecasts. **They must be
  excluded** from accuracy reporting.
- Water measurements: 385 readings, 2024-11-05 → 2026-09-05 (~57% of days).
  Measurements are taken at **7am**; all simulation periods are 7am→7am.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Run selection | **Last run of each creation day** | Matches the forecast a user would have seen at end of day; yields one row per (target_date, horizon). |
| Metrics | MAE, bias, RMSE, within-0.5°C hit rate, **plus N** | N is mandatory — sparse measurement coverage makes some horizon buckets thin. |
| Bias sign | `forecast − actual`; **positive = model runs warm** | Stated once, applied everywhere. |
| Replay fitting | **Current fitted model, applied across all history** | User's explicit choice: "show the accuracy as it would have been". Walk-forward / out-of-sample refitting is deferred to the separate new-model work. |
| Default horizon | **1 day** | As requested. |

## Known Leakage (must be surfaced in the UI)

The replay is only as honest as the inputs available at the time.

1. **Solar & cloud are not stored.** The model uses shortwave radiation and
   cloud cover (commit 9a1ee41), but MotherDuck holds air temperature only.
   The replay therefore uses **historical actual** solar/cloud for every
   date. This is a permanent optimistic bias in the replay until solar/cloud
   forecasts are stored. Out of scope here; listed as follow-up.
2. **Air temp before 2026-02-15.** No stored forecasts exist, so the replay
   falls back to Meteostat actuals. Rows are flagged `air_source='ACTUAL'`
   vs `'FORECAST'` and the UI distinguishes them.

The stored-forecast comparison has **no leakage** — it is a pure record.

## Delivery Order

**PR 1 — `WaterTempForecaster.predict()` refactor.** A prerequisite. The
accuracy work must not be built on the current API (see below). Ships with
tests pinning it behaviour-identical to today's output.

**PR 2 — the accuracy tab.** Everything from "New module `accuracy.py`"
onward, built on the PR 1 API.

## PR 1: Forecaster predict() Refactor

### Why this comes first

The current `predict()` is not a forecasting API:

```python
def predict(self, temperatures: pd.DataFrame) -> pd.DataFrame:
    """Fill in predicted water temps for rows where source == 'AIR_ONLY'."""
```

It is a DataFrame gap-filler coupled to the single-`temperatures`-DataFrame
convention and the `source` string vocabulary. To forecast anything, a caller
must first construct a frame with `AIR_ONLY` rows in the correct order. The
backtest would have to either fabricate such a frame per anchor date (~200
times) or reach past it into the private `_simulate_period`. Both are wrong;
the second is what the first draft of this spec did.

This is also the origin of the negative-horizon backfill: `predict()` fills
*every* `AIR_ONLY` row including historical gaps, and `app.py:942` stores
every `PREDICTED` row it finds.

### Loops: which are inherent

1. **Hour-by-hour physics** — genuinely sequential. Stays in `_simulate_period`.
2. **Target dates from one anchor** — *unnecessary*. An n-day forecast is one
   continuous simulation sampled at each 7am boundary, not n simulations.
   The current per-row re-entry is both slower and where the chaining
   behaviour got buried.
3. **Anchors in a backtest** — inherent (independent trajectories), but should
   be a `map` over anchors, not bespoke logic at the call site.

### New API

```python
def predict(self, start_datetime, start_water_temp, targets) -> pd.DataFrame:
    """
    Forecast water temperature forward from a known starting point.

    Args:
        start_datetime:   anchor time (normalised to MEASUREMENT_HOUR, 7am)
        start_water_temp: measured/known water temp at the anchor
        targets:          int n -> forecast 1..n days ahead
                          or a sequence of dates (supports irregular gaps)

    Returns:
        DataFrame: target_datetime, horizon_days, water_temp
        water_temp is NaN where hourly weather coverage runs out.
    """
```

One simulation pass, checkpointed at each target, targets sorted ascending.
Serves all three consumers: live forecast (`targets=5`), backtest replay
(`targets=5` per anchor), and gap-fill (explicit irregular dates — required,
since the existing filler handles non-consecutive dates).

### Migration

- Existing DataFrame method renamed `fill_predictions(temperatures)` and
  reimplemented as a thin wrapper: for each run of consecutive `AIR_ONLY`
  rows, anchor on the preceding row and predict that run's dates. The
  existing code chains from `prev_row` whether it is `MEASURED` or
  `PREDICTED`, which is exactly equivalent to one continuous simulation from
  the last known anchor — so this is behaviour-preserving.
- `predict_next_day` becomes a one-line alias for `predict(..., targets=1)`.
- Call sites: `app.py:759`, `app.py:934`. Tests in `test_forecaster.py`.

### PR 1 tests

- `predict` with `targets=1` matches `predict_next_day` exactly.
- `predict` with `targets=5` sampled at boundaries equals five chained
  single-day simulations (the continuous-run equivalence).
- `predict` with irregular explicit dates handles gaps correctly.
- NaN — not a fabricated value — where weather coverage runs out.
- **Golden test**: `fill_predictions` output is identical to the pre-refactor
  `predict(temperatures)` on a fixed synthetic frame. This is the guard that
  makes the rename safe.

## PR 2 Architecture

### New module: `accuracy.py`

`app.py` is already 1103 lines against a documented target of 210. All new
logic lands in a new module of pure, testable functions; the tab body in
`app.py` only arranges Streamlit widgets.

```python
BIAS_NOTE = "bias = forecast - actual; positive means the model runs warm"

def compute_metrics(df: pd.DataFrame) -> dict:
    """df has forecast_temp, actual_temp. Returns mae, bias, rmse, hit_rate_0_5, n."""

def metrics_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """One row per horizon_days, columns as above."""

def replay_current_model(forecaster, water_temps, hourly_weather_by_anchor, max_horizon=5) -> pd.DataFrame:
    """Re-run the current model over history. Same output shape as the stored frame."""
```

**Shared output shape** — both comparisons produce identical frames so one
set of charts renders either:

```
target_date       date
horizon_days      int    (stored: 0-5; replay: 1-5 only)
forecast_temp     float
actual_temp       float
error             float  (forecast - actual)
air_source        str    ('FORECAST' | 'ACTUAL')  -- replay only; 'FORECAST' for stored
```

### `forecast_storage.py`: new method

```python
def get_water_predictions_last_run_per_day(self, max_horizon: int = 5) -> pd.DataFrame:
```

One SQL query:

```sql
WITH ranked AS (
  SELECT *,
    row_number() OVER (PARTITION BY forecast_created_date
                       ORDER BY forecast_created_timestamp DESC) AS rn
  FROM water_temp_predictions
  WHERE date_diff('day', forecast_created_date, target_date) BETWEEN 0 AND ?
)
SELECT forecast_created_date, forecast_created_timestamp, target_date,
       water_temp AS forecast_temp,
       date_diff('day', forecast_created_date, target_date) AS horizon_days
FROM ranked WHERE rn = 1
ORDER BY target_date, horizon_days
```

The `BETWEEN 0 AND 5` filter is load-bearing: it drops the negative-horizon
backfill. Timestamp columns get the existing `tz_localize(None)` treatment.

### Joining to actuals

`load_water_temps()`, deduplicated on date (duplicate dates have bitten this
codebase before — commit 164186d), inner-joined on `target_date`. Only days
with a measurement are scored; unmeasured days drop out silently but are
reflected in `N`.

### Replay algorithm

For each measured date `d` in range:

1. Start from the measured water temp at `d` 07:00.
2. Build the hourly weather frame from `d` 07:00 forward 5 days:
   - Air temp: the **last stored 3-hourly run created on `d`**, interpolated
     to hourly via the existing `interpolate_to_hourly`. If absent
     (pre-Feb-2026), fall back to Meteostat hourly actuals and mark
     `air_source='ACTUAL'`.
   - Solar/cloud: historical actuals, via the existing `build_hourly_weather`.
3. Call `forecaster.predict(d_7am, measured_temp, targets=5)` — the PR 1 API.
   One simulation, checkpointed at each 07:00 boundary → up to 5 rows
   (horizons 1–5). No private methods, no per-date loop.
4. Where weather coverage is missing (horizon-5 stored runs are truncated —
   14.5k rows vs ~30k at other horizons), `predict` returns **NaN**. Never
   fabricate.

The replay is therefore a `map` of one `predict` call over measured anchors,
concatenated — the whole point of doing PR 1 first.

The current fitted `k_air / k_solar / k_cool` are used throughout, per the
decision above.

## UI

New tab "Forecast Accuracy" alongside "Temperature" and "Heard at the Res".
No emojis, per project guidelines.

- **Horizon selector** (default 1, options 0–5) at the top; filters everything
  except the by-horizon chart.
- **Source toggle**: "Stored forecasts" / "Model replay (current model)",
  with a plain-language caption explaining what each means and the leakage
  caveat on the replay.
- **Metric tiles** for the selected horizon: MAE, bias, RMSE, hit rate, N.
- **Accuracy by horizon** chart — MAE across horizons 0–5, unfiltered, with
  the selected horizon highlighted. This is the "degradation" view.
- **Forecast vs actual** time series at the selected horizon.
- **Error over time** at the selected horizon, with a zero reference line.
- Replay charts visually distinguish `air_source='ACTUAL'` rows so leaky
  points are never mistaken for honest ones.

Horizon 0 is a same-day nowcast; it is included in the horizon chart but 1 is
the default, as requested. **Horizon 0 exists only for stored forecasts** —
the replay is anchored on the 7am measurement itself, so its first output is
horizon 1. When the replay source is selected, horizon 0 is disabled in the
selector rather than shown as an empty chart.

### Caching

`@st.cache_data` on the storage query. The replay is cached keyed on the
model coefficients so a refit invalidates it.

## Testing

`test_accuracy.py`, synthetic frames only — no MotherDuck required:

- Last-run-per-day selection picks the correct row.
- Negative-horizon rows are excluded.
- Metric values against a hand-computed 4-row case.
- Bias sign: an all-too-warm frame yields positive bias.
- Missing weather coverage yields NaN, not a fabricated number.

## Documented Limitations

### Model version is not recoverable

`water_temp_predictions` stores `heat_transfer_coeff` only. It predates the
three-term model (9a1ee41), so `k_solar` and `k_cool` were never persisted —
verified against the live schema on 2026-09-05. Stored predictions cannot be
attributed to a specific model version.

This does not affect scoring stored forecasts (the forecast value is what it
is), but historical accuracy **cannot be segmented by model version**, and a
step change in the by-horizon chart around the 9a1ee41 deploy should be read
as a model change rather than weather. Persisting the two missing
coefficients is a candidate addition to the storage follow-up (issue #29).

### Effective horizon

"1-day horizon" means one day from `forecast_created_date`, not one day from
the last measurement. If date `d` was not measured, a stored "1-day" forecast
in practice started from an older reading. The table does not record the
simulation start date, so effective horizon cannot be recovered. Documented,
not solved.

## Out of Scope

- Storing solar/cloud forecasts alongside air temp (would make the replay
  genuinely honest — recommended next).
- Walk-forward / out-of-sample refitting (deferred to the new-model work).
- Air-temperature forecast accuracy, though the data supports it.
