# Published corrections for the Meteostat outage

Branch: `feature/forecast-evolution` (PR #70). Build on top of it.

## Goal

From 2026-03-20 to 2026-09-06 the historical air feed was frozen (issue #33),
so the forecasts published in that window are poor. Publish a correction: what
the current model would have forecast for those days, run once and stored in
MotherDuck, so the Forecast Evolution page's Published view can show it in
place of the broken runs.

## Constraints

- **Never write to `water_temp_predictions`.** It is the record of what was
  actually published. The accuracy page picks the last run per creation day
  from it, so replay rows there would silently replace the real forecasts and
  flatter the accuracy numbers.
- **A correction is frozen once written.** The live replay moves every time
  the model is refitted; the stored one must not. Record the coefficients used.
- **Label it honestly.** Before 2026-09-06 no solar/cloud forecasts were kept
  (issue #29), so the replay uses actual solar and cloud for that window. The
  correction is an optimistic bound, made with hindsight weather. The UI says so.
- **The accuracy page does not change.** It keeps scoring published forecasts
  only. Corrections are display-only for now.
- Follow `CLAUDE.md`: explicit errors, no silent fallbacks, no emojis, comments
  describe the code as it is now.

## Steps

### 1. Table and storage methods (`forecast_storage.py`)

Add to `initialize_schema()`:

```sql
CREATE TABLE IF NOT EXISTS water_temp_corrections (
    forecast_created_date DATE NOT NULL,   -- the anchor (measurement) date
    target_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL,
    water_temp DOUBLE NOT NULL,
    k_air DOUBLE NOT NULL,
    k_solar DOUBLE NOT NULL,
    k_cool DOUBLE NOT NULL,
    correction_created_timestamp TIMESTAMP NOT NULL,
    reason VARCHAR NOT NULL,               -- e.g. 'meteostat-outage-#33'
    PRIMARY KEY (forecast_created_date, target_date)
)
```

Add:

- `store_water_corrections(df, coefficients, reason)`: plain INSERT. Raises
  `ForecastStorageError` on a primary-key collision rather than ignoring it,
  because a second run means someone is trying to overwrite a published
  correction.
- `has_water_corrections(reason) -> bool`.
- `get_water_corrections() -> DataFrame` returning the same columns as
  `get_water_predictions_last_run_per_day()`: `forecast_created_date`,
  `target_date`, `forecast_temp`, `horizon_days`, with naive timestamps.

Tests in `test_motherduck.py`, following the existing pattern there. Use a
local in-memory DuckDB if the existing tests do; never the production database.

### 2. Backfill script (`debug/backfill_corrections.py`)

- Loads water temps, fits coefficients exactly as the accuracy page does
  (`cached_fitted_model_coefficients` logic, without the Streamlit cache), and
  runs the same replay as `replay_runs()` in `app.py`, with 5 days ahead.
  Factor the shared pieces out of `app.py` rather than copying them, so the
  stored correction and the live replay cannot drift apart.
- Keeps anchors with `forecast_created_date` in
  [`METEOSTAT_OUTAGE_START`, `METEOSTAT_OUTAGE_END`). Drop the horizon-0 anchor
  rows: they are the measurement, not a forecast.
- `--dry-run` is the default: print the row count, the date range, the
  coefficients and a sample, and write nothing. `--write` writes.
- Refuses to write if `has_water_corrections(reason)` is true.

### 3. Chart (`app.py`)

- `page_evolution()`, Published view: load corrections, drop published runs
  whose `forecast_created_date` is inside the outage window, and pass
  corrections to `create_forecast_evolution_chart` as a separate argument.
- `create_forecast_evolution_chart(runs, water_temps, mark_outage, corrections=None)`:
  draw corrections with the same per-horizon colours but `dash="dot"`, with a
  legend entry "Corrected (hindsight weather)". Hover says "Corrected" rather
  than "Made".
- Relabel the shaded band on this chart only: "Corrected after the fact -
  original forecasts missed live weather data (issue #33)". Keep
  `_shade_meteostat_outage` working unchanged for the accuracy page; add a
  label parameter.
- Update the Published caption to explain the correction and the
  solar/cloud caveat.
- If the corrections table is empty, the page behaves exactly as now.

### 4. Docs

- `CLAUDE.md`: add `water_temp_corrections` under `forecast_storage.py`, and
  explain why it is separate from `water_temp_predictions`.

## Verification

- `python3 -m pytest test_motherduck.py test_app.py test_accuracy.py -v` passes.
- `ruff check .` passes.
- `python debug/backfill_corrections.py` (dry run) prints a plausible
  correction: about 5 rows per measured day in the window, temperatures
  tracking the measurements, coefficients matching the accuracy page's replay.
- AppTest on `page_evolution`: no exceptions in either view, with and without
  corrections present.

## Not in scope

- Running `--write` against production. The owner runs that by hand after
  reviewing the dry run.
- Scoring corrections on the accuracy page.
