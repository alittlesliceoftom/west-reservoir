# Forecast Accuracy Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Forecast Accuracy tab reporting how well our water-temperature forecasts have performed, broken down by how many days ahead they were made — built on a forecaster API that can actually forecast.

**Architecture:** Three PRs. PR 1 replaces `WaterTempForecaster.predict()` — today a DataFrame gap-filler — with `predict_forward`, one simulation checkpointed at each target. PR 2 adds the whole backend (`accuracy.py` plus two MotherDuck queries), unit-tested, touching no UI. PR 3 adds the Streamlit tab. Two independent comparisons share one output schema so the same charts render both: **stored forecasts** (the honest record) and **model replay** (current model re-run over history).

**Tech Stack:** Python 3, pandas, numpy, scipy, duckdb/MotherDuck, Streamlit 1.53.0, Plotly 5.17.0, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-forecast-accuracy-tab-design.md`

## Global Constraints

- **No emojis in UI text.** Project guideline, strictly enforced.
- **No synthetic/sample data fallbacks.** Raise explicit errors; never silently substitute fabricated data.
- **Never fabricate a forecast value.** Missing weather coverage yields `NaN`, always.
- **Bias sign:** `forecast - actual`; positive means the model runs warm. Applied everywhere.
- **Measurements are at 07:00.** All simulation periods run 7am→7am. `MEASUREMENT_HOUR = 7`.
- **Horizon filter `BETWEEN 0 AND 5` is load-bearing** — it excludes the negative-horizon backfill rows in `water_temp_predictions`.
- **Use `rank()`, never `row_number()`, for last-run-per-day selection.** A run is many rows sharing one `forecast_created_timestamp`; `row_number()` would keep only one of them. `rank()` ties them all at 1.
- **MotherDuck returns tz-aware timestamps**; the codebase is tz-naive. Apply `.dt.tz_localize(None)` on every timestamp column read back.
- **Streamlit width:** this codebase uses `width='stretch'`, not the deprecated `use_container_width=True`. Match it (5 existing uses, e.g. `app.py:762`).
- **Date arguments are `pd.Timestamp`, normalised** — matching `app.py:681` / `app.py:814`. Never pass `.date()`.
- **Bulk-fetch, never per-anchor.** The replay runs over ~385 anchors; one query/API call per anchor is not acceptable.
- Run tests with: `env/bin/python -m pytest <file> -v` from the repo root.

---

# PR 1 — Forecaster predict() Refactor

Tasks 1–2. No behaviour change; a pure API improvement with tests proving equivalence.

---

### Task 1: Real forward-simulation API

**Files:**
- Modify: `forecaster.py` (add `predict_forward` after `predict_next_day`)
- Test: `test_forecaster.py`

**Interfaces:**
- Consumes: existing `_simulate_period`, `_get_weather_for_period`, `MEASUREMENT_HOUR`, `set_hourly_weather`.
- Produces:
  ```python
  def predict_forward(self, start_datetime, start_water_temp, targets=1) -> pd.DataFrame
  # columns: target_datetime (Timestamp), horizon_days (int), water_temp (float), has_weather (bool)
  ```
  `targets` is an `int` n (forecast 1..n days ahead) or a sequence of dates.

> **Naming:** `predict_forward` is the final name for the forward-simulation API; `fill_predictions` (Task 2) is the final name for the DataFrame filler. Two different jobs, two different names — a single overloaded `predict` was the original problem.

> **Duplicate targets are NOT de-duplicated.** A repeated target becomes a zero-length leg → empty weather slice → `has_weather=False`. This is exactly what the legacy filler did with duplicate dates, and preserving it is what keeps Task 2's golden test passing. Do not add `set()`.

- [ ] **Step 1: Write the failing tests**

Append to `test_forecaster.py`:

```python
class TestPredictForward:
    """Forward simulation API: one pass, checkpointed at each target."""

    def _fitted(self, n_hours=200, air_temp=15.0):
        f = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        f.set_hourly_weather(
            _make_hourly_weather(datetime(2026, 3, 1, 0), n_hours, air_temp=air_temp)
        )
        return f

    def test_single_target_matches_manual_simulation(self):
        f = self._fitted()
        start = datetime(2026, 3, 1, 7)
        result = f.predict_forward(start, 10.0, targets=1)

        assert list(result.columns) == [
            "target_datetime", "horizon_days", "water_temp", "has_weather"
        ]
        assert len(result) == 1
        assert result.loc[0, "horizon_days"] == 1
        assert result.loc[0, "target_datetime"] == pd.Timestamp(2026, 3, 2, 7)
        assert bool(result.loc[0, "has_weather"]) is True

        expected = f._simulate_period(
            10.0, f._get_weather_for_period(start, datetime(2026, 3, 2, 7))
        )
        assert result.loc[0, "water_temp"] == pytest.approx(expected)

    def test_multi_target_equals_chained_single_days(self):
        """A 3-day forecast is one continuous run, equal to 3 chained 1-day runs."""
        f = self._fitted()
        start = datetime(2026, 3, 1, 7)
        result = f.predict_forward(start, 10.0, targets=3)

        assert list(result["horizon_days"]) == [1, 2, 3]

        water = 10.0
        for day in range(3):
            leg_start = datetime(2026, 3, 1 + day, 7)
            leg_end = datetime(2026, 3, 2 + day, 7)
            water = f._simulate_period(water, f._get_weather_for_period(leg_start, leg_end))
            assert result.loc[day, "water_temp"] == pytest.approx(water)

    def test_start_datetime_normalised_to_measurement_hour(self):
        """A midnight or mid-afternoon anchor is snapped to 7am."""
        f = self._fitted()
        at_seven = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, targets=2)
        at_midnight = f.predict_forward(datetime(2026, 3, 1, 0), 10.0, targets=2)
        pd.testing.assert_frame_equal(at_seven, at_midnight)

    def test_explicit_irregular_dates(self):
        """Targets may be an irregular sequence of dates, not just 1..n."""
        f = self._fitted()
        start = datetime(2026, 3, 1, 7)
        result = f.predict_forward(
            start, 10.0, targets=[datetime(2026, 3, 2), datetime(2026, 3, 5)]
        )
        assert list(result["horizon_days"]) == [1, 4]

        water = 10.0
        water = f._simulate_period(
            water, f._get_weather_for_period(datetime(2026, 3, 1, 7), datetime(2026, 3, 2, 7))
        )
        assert result.loc[0, "water_temp"] == pytest.approx(water)
        water = f._simulate_period(
            water, f._get_weather_for_period(datetime(2026, 3, 2, 7), datetime(2026, 3, 5, 7))
        )
        assert result.loc[1, "water_temp"] == pytest.approx(water)

    def test_targets_sorted_ascending(self):
        f = self._fitted()
        result = f.predict_forward(
            datetime(2026, 3, 1, 7), 10.0,
            targets=[datetime(2026, 3, 4), datetime(2026, 3, 2)],
        )
        assert list(result["horizon_days"]) == [1, 3]

    def test_duplicate_targets_are_kept_as_zero_length_legs(self):
        """One row per target, duplicates included. Preserves legacy filler behaviour."""
        f = self._fitted()
        result = f.predict_forward(
            datetime(2026, 3, 1, 7), 10.0,
            targets=[datetime(2026, 3, 2), datetime(2026, 3, 2)],
        )
        assert len(result) == 2
        assert bool(result.loc[0, "has_weather"]) is True
        assert bool(result.loc[1, "has_weather"]) is False

    def test_nan_when_weather_runs_out(self):
        """No fabrication: legs beyond weather coverage are NaN with has_weather False."""
        f = self._fitted(n_hours=30)  # covers ~1 day past the 7am anchor
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, targets=3)

        assert len(result) == 3
        assert not np.isnan(result.loc[0, "water_temp"])
        assert bool(result.loc[0, "has_weather"]) is True
        assert np.isnan(result.loc[2, "water_temp"])
        assert bool(result.loc[2, "has_weather"]) is False

    def test_nan_propagates_once_coverage_lost(self):
        f = self._fitted(n_hours=30)
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, targets=4)
        tail = result[result["horizon_days"] >= 3]["water_temp"]
        assert tail.isna().all()

    def test_no_weather_set_returns_all_nan(self):
        f = WaterTempForecaster()
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, targets=2)
        assert result["water_temp"].isna().all()
        assert not result["has_weather"].any()

    def test_zero_targets_returns_empty_frame(self):
        f = self._fitted()
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, targets=0)
        assert result.empty
        assert list(result.columns) == [
            "target_datetime", "horizon_days", "water_temp", "has_weather"
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_forecaster.py::TestPredictForward -v`
Expected: FAIL — `AttributeError: 'WaterTempForecaster' object has no attribute 'predict_forward'`

- [ ] **Step 3: Implement `predict_forward`**

Add to `forecaster.py`, immediately after `predict_next_day`:

```python
    def predict_forward(
        self,
        start_datetime,
        start_water_temp: float,
        targets=1,
    ) -> pd.DataFrame:
        """
        Forecast water temperature forward from a known starting point.

        Runs ONE continuous simulation, checkpointed at each target, rather
        than re-simulating per target.

        Args:
            start_datetime: Anchor time. Normalised to MEASUREMENT_HOUR (7am),
                            since water temps are measured at 7am.
            start_water_temp: Known water temperature at the anchor.
            targets: Either an int n (forecast 1..n days ahead), or a sequence
                     of dates/datetimes. Duplicates are kept as zero-length
                     legs rather than de-duplicated, so callers get exactly one
                     row per target they asked for.

        Returns:
            DataFrame with columns:
                target_datetime (Timestamp, at 7am)
                horizon_days    (int, days from the anchor)
                water_temp      (float, NaN where weather coverage is missing)
                has_weather     (bool, whether that leg had any weather data)
        """
        columns = ["target_datetime", "horizon_days", "water_temp", "has_weather"]

        start_dt = pd.Timestamp(start_datetime).normalize() + pd.Timedelta(
            hours=self.MEASUREMENT_HOUR
        )

        if isinstance(targets, (int, np.integer)):
            target_dts = [start_dt + pd.Timedelta(days=i) for i in range(1, int(targets) + 1)]
        else:
            target_dts = sorted(
                pd.Timestamp(t).normalize() + pd.Timedelta(hours=self.MEASUREMENT_HOUR)
                for t in targets
            )

        target_dts = [t for t in target_dts if t >= start_dt]

        if not target_dts:
            return pd.DataFrame({c: [] for c in columns}).astype(
                {"horizon_days": "int64", "water_temp": "float64", "has_weather": "bool"}
            )

        rows = []
        water = float(start_water_temp)
        cursor = start_dt

        for target_dt in target_dts:
            weather_slice = self._get_weather_for_period(cursor, target_dt)
            has_weather = not weather_slice.empty

            if has_weather:
                water = self._simulate_period(water, weather_slice)
            else:
                # No data for this leg: refuse to fabricate, and stay NaN onward.
                water = float("nan")

            rows.append({
                "target_datetime": target_dt,
                "horizon_days": int((target_dt - start_dt) / pd.Timedelta(days=1)),
                "water_temp": water,
                "has_weather": has_weather,
            })
            cursor = target_dt

        return pd.DataFrame(rows, columns=columns)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_forecaster.py::TestPredictForward -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the full forecaster suite for regressions**

Run: `env/bin/python -m pytest test_forecaster.py -v`
Expected: PASS, all pre-existing tests still green.

- [ ] **Step 6: Commit**

```bash
git add forecaster.py test_forecaster.py
git commit -m "Add predict_forward: one-pass forward simulation API

Forecast 1..n days or an irregular set of dates from a known anchor,
checkpointed at each target rather than re-simulating per target.
NaN where weather coverage is missing - never a fabricated value."
```

---

### Task 2: Replace the DataFrame filler with `fill_predictions`

**Files:**
- Modify: `forecaster.py:335-361` (replace `predict`)
- Modify: `app.py:759`, `app.py:934` (call sites)
- Test: `test_forecaster.py`

**Interfaces:**
- Consumes: `predict_forward` from Task 1.
- Produces: `def fill_predictions(self, temperatures: pd.DataFrame) -> pd.DataFrame` — fills rows where `source == "AIR_ONLY"`, setting `water_temp` and flipping `source` to `"PREDICTED"`. Behaviour-identical to the old `predict`, which is **deleted** (no alias — there are exactly two callers and both are updated in this task).

**Why a golden test:** the old method is being replaced, so equivalence must be proven. The test embeds the *old algorithm verbatim* as a reference implementation — no magic numbers to go stale.

- [ ] **Step 1: Write the failing golden test**

Append to `test_forecaster.py`:

```python
def _legacy_fill(forecaster, temperatures):
    """
    Verbatim copy of the pre-refactor WaterTempForecaster.predict().
    Reference implementation for the equivalence test. Do not "improve" it.
    """
    result = temperatures.copy()
    result = result.sort_values("date").reset_index(drop=True)

    for i in range(len(result)):
        if result.loc[i, "source"] != "AIR_ONLY":
            continue
        if i == 0:
            continue

        prev_row = result.iloc[i - 1]
        curr_date = result.loc[i, "date"]
        current_water_temp = prev_row["water_temp"]

        start_dt = pd.Timestamp(prev_row["date"]).replace(hour=forecaster.MEASUREMENT_HOUR)
        end_dt = pd.Timestamp(curr_date).replace(hour=forecaster.MEASUREMENT_HOUR)

        slice_df = forecaster._get_weather_for_period(start_dt, end_dt)
        if not slice_df.empty:
            predicted = forecaster._simulate_period(current_water_temp, slice_df)
            result.loc[i, "water_temp"] = predicted
            result.loc[i, "source"] = "PREDICTED"

    return result


class TestFillPredictions:
    """fill_predictions must be behaviour-identical to the old predict()."""

    def _fitted(self, n_hours=400):
        f = WaterTempForecaster(k_air=0.02, k_solar=1e-4, k_cool=0.005)
        f.set_hourly_weather(
            _make_hourly_weather(
                datetime(2026, 3, 1, 0), n_hours, air_temp=15.0, shortwave=200.0, cloud=40.0
            )
        )
        return f

    def _frame(self, sources, start_day=1, water_start=10.0):
        dates = [datetime(2026, 3, start_day + i) for i in range(len(sources))]
        temps = [water_start if s == "MEASURED" else float("nan") for s in sources]
        return pd.DataFrame({"date": dates, "water_temp": temps, "source": sources})

    def test_matches_legacy_simple_run(self):
        f = self._fitted()
        df = self._frame(["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_with_interleaved_measurements(self):
        """Chain must re-anchor on each MEASURED row."""
        f = self._fitted()
        df = self._frame(
            ["MEASURED", "AIR_ONLY", "MEASURED", "AIR_ONLY", "AIR_ONLY", "MEASURED"]
        )
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_when_air_only_is_first_row(self):
        """Row 0 has no anchor and must be left untouched."""
        f = self._fitted()
        df = self._frame(["AIR_ONLY", "MEASURED", "AIR_ONLY"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_with_date_gaps(self):
        """Non-consecutive dates: a leg may span several days."""
        f = self._fitted()
        df = pd.DataFrame({
            "date": [
                datetime(2026, 3, 1), datetime(2026, 3, 2),
                datetime(2026, 3, 6), datetime(2026, 3, 7),
            ],
            "water_temp": [10.0, float("nan"), float("nan"), float("nan")],
            "source": ["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"],
        })
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_with_duplicate_dates(self):
        """Duplicate dates produce a zero-length leg; the row stays untouched."""
        f = self._fitted()
        df = pd.DataFrame({
            "date": [
                datetime(2026, 3, 1), datetime(2026, 3, 2), datetime(2026, 3, 2),
                datetime(2026, 3, 3),
            ],
            "water_temp": [10.0, float("nan"), float("nan"), float("nan")],
            "source": ["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"],
        })
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_when_weather_runs_out(self):
        f = self._fitted(n_hours=60)
        df = self._frame(["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_unsorted_input(self):
        f = self._fitted()
        df = self._frame(["MEASURED", "AIR_ONLY", "AIR_ONLY"]).iloc[::-1].reset_index(drop=True)
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_all_measured_is_unchanged(self):
        f = self._fitted()
        df = self._frame(["MEASURED", "MEASURED", "MEASURED"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_forecaster.py::TestFillPredictions -v`
Expected: FAIL — `AttributeError: 'WaterTempForecaster' object has no attribute 'fill_predictions'`

- [ ] **Step 3: Replace `predict` with `fill_predictions`**

In `forecaster.py`, delete the existing `predict` method (lines 335-361) entirely and put this in its place:

```python
    def fill_predictions(self, temperatures: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in predicted water temps for rows where source == 'AIR_ONLY'.

        Each maximal run of consecutive AIR_ONLY rows is anchored on the row
        immediately before it and forecast in one pass via predict_forward.
        Equivalent to chaining day by day, because each AIR_ONLY row was
        already chained from the previous row's value.
        """
        result = temperatures.copy()
        result = result.sort_values("date").reset_index(drop=True)

        i = 0
        while i < len(result):
            if result.loc[i, "source"] != "AIR_ONLY" or i == 0:
                i += 1
                continue

            # Collect this maximal run of consecutive AIR_ONLY rows.
            run_start = i
            while i < len(result) and result.loc[i, "source"] == "AIR_ONLY":
                i += 1
            run_end = i  # exclusive

            anchor = result.iloc[run_start - 1]
            predictions = self.predict_forward(
                start_datetime=anchor["date"],
                start_water_temp=anchor["water_temp"],
                targets=list(result.loc[run_start:run_end - 1, "date"]),
            )

            for offset, row_idx in enumerate(range(run_start, run_end)):
                prediction = predictions.iloc[offset]
                # Legs without weather stay untouched, as before.
                if prediction["has_weather"]:
                    result.loc[row_idx, "water_temp"] = prediction["water_temp"]
                    result.loc[row_idx, "source"] = "PREDICTED"

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_forecaster.py::TestFillPredictions -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Update the two call sites**

In `app.py`, at line 759 and line 934, change:

```python
temperatures_deduped = forecaster.predict(temperatures_deduped)
```

to:

```python
temperatures_deduped = forecaster.fill_predictions(temperatures_deduped)
```

Confirm no callers remain:

Run: `grep -rn "forecaster.predict(" app.py`
Expected: no output.

- [ ] **Step 6: Verify the whole suite and imports**

Run: `env/bin/python -m pytest test_forecaster.py test_data.py test_combine_hourly.py -v`
Expected: PASS

Run: `env/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`

- [ ] **Step 7: Commit and open PR 1**

```bash
git add forecaster.py app.py test_forecaster.py
git commit -m "Rename predict to fill_predictions, reimplement on predict_forward

The DataFrame filler and the forecasting API are two different jobs and
now have two different names. fill_predictions is proven equivalent to
the old predict() by a golden test carrying the old algorithm verbatim."

git push -u origin worktree-forecast-accuracy-tab
gh pr create --base main --title "Give the forecaster a real predict API" --body "$(cat <<'PRBODY'
The forecaster had no way to say "forecast N days from this temperature on
this date". `predict()` was a DataFrame gap-filler coupled to the
`temperatures` frame and the `source` string vocabulary, so forecasting
anything meant building a fake frame first - or reaching into the private
`_simulate_period`. The upcoming accuracy backtest needs the former.

- Adds `predict_forward(start_datetime, start_water_temp, targets)`, returning
  `target_datetime / horizon_days / water_temp / has_weather`. `targets` is an
  int n or an irregular sequence of dates. One simulation pass checkpointed at
  each target, not n simulations.
- Renames `predict` to `fill_predictions` and reimplements it on the new API.
- Missing weather coverage yields NaN, never a fabricated value.

Tested: golden test embeds the pre-refactor algorithm verbatim and asserts the
new `fill_predictions` matches it across interleaved measurements, date gaps,
duplicate dates, unsorted input, exhausted weather, and an AIR_ONLY first row.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YL9pqG8tgTqoRdbTGAM3bw
PRBODY
)"
```

---

# PR 2 — Accuracy Backend

Tasks 3–7. All backend, fully unit-tested, **zero `app.py` changes**. Reviews and ships independently of any UI.

Begin after PR 1 is merged.

---

### Task 3: Query stored water-temp forecasts, last run per day

**Files:**
- Modify: `forecast_storage.py`
- Test: `test_accuracy.py` (create)

**Interfaces:**
- Produces: `LAST_WATER_RUN_PER_DAY_SQL` (module constant) and `ForecastStorage.get_water_predictions_last_run_per_day(max_horizon: int = 5) -> pd.DataFrame` with columns `forecast_created_date`, `forecast_created_timestamp`, `target_date`, `forecast_temp`, `horizon_days`.

The SQL is what's under test. Run it against a local in-memory DuckDB with the same schema — no network, identical SQL.

- [ ] **Step 1: Write the failing test**

Create `test_accuracy.py`:

```python
"""Tests for forecast accuracy reporting"""

import duckdb
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from forecast_storage import LAST_WATER_RUN_PER_DAY_SQL


def _local_predictions_db(rows):
    """In-memory DuckDB with the water_temp_predictions schema and given rows."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE water_temp_predictions (
            forecast_created_date DATE NOT NULL,
            forecast_created_timestamp TIMESTAMP NOT NULL,
            target_date DATE NOT NULL,
            water_temp DOUBLE NOT NULL,
            heat_transfer_coeff DOUBLE NOT NULL,
            start_water_temp DOUBLE NOT NULL,
            simulation_hours INTEGER NOT NULL,
            source_air_forecast_timestamp TIMESTAMP NOT NULL
        )
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO water_temp_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r["created_date"], r["created_ts"], r["target_date"], r["water_temp"],
                0.02, 10.0, 24, r["created_ts"],
            ],
        )
    return conn


def _row(created_date, created_hour, target_date, water_temp):
    return {
        "created_date": created_date,
        "created_ts": datetime(
            created_date.year, created_date.month, created_date.day, created_hour
        ),
        "target_date": target_date,
        "water_temp": water_temp,
    }


class TestLastWaterRunPerDay:

    def test_picks_last_run_of_each_creation_day(self):
        rows = [
            _row(datetime(2026, 5, 1).date(), 8, datetime(2026, 5, 2).date(), 11.0),
            _row(datetime(2026, 5, 1).date(), 20, datetime(2026, 5, 2).date(), 12.0),
            _row(datetime(2026, 5, 1).date(), 14, datetime(2026, 5, 2).date(), 99.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert len(result) == 1
        assert result.loc[0, "forecast_temp"] == pytest.approx(12.0)

    def test_keeps_all_horizons_of_the_winning_run(self):
        """Selection is per creation day, not per row - rank(), not row_number()."""
        created = datetime(2026, 5, 10).date()
        rows = [
            _row(created, 20, datetime(2026, 5, 11).date(), 12.0),
            _row(created, 20, datetime(2026, 5, 12).date(), 13.0),
            _row(created, 20, datetime(2026, 5, 13).date(), 14.0),
            _row(created, 8, datetime(2026, 5, 11).date(), 99.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert sorted(result["horizon_days"]) == [1, 2, 3]
        assert 99.0 not in list(result["forecast_temp"])

    def test_excludes_negative_horizon_backfill(self):
        """Rows targeting dates BEFORE the run are historical backfill, not forecasts."""
        rows = [
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 11).date(), 12.0),
            _row(datetime(2026, 5, 10).date(), 12, datetime(2024, 12, 1).date(), 5.0),
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 9).date(), 6.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert set(result["horizon_days"]) == {1}

    def test_excludes_horizons_beyond_max(self):
        rows = [
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 11).date(), 12.0),
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 20).date(), 13.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert list(result["horizon_days"]) == [1]

    def test_horizon_zero_is_included(self):
        rows = [_row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 10).date(), 12.0)]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert list(result["horizon_days"]) == [0]

    def test_separate_creation_days_each_keep_their_own_run(self):
        rows = [
            _row(datetime(2026, 5, 1).date(), 20, datetime(2026, 5, 2).date(), 11.0),
            _row(datetime(2026, 5, 2).date(), 20, datetime(2026, 5, 3).date(), 12.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest test_accuracy.py::TestLastWaterRunPerDay -v`
Expected: FAIL — `ImportError: cannot import name 'LAST_WATER_RUN_PER_DAY_SQL' from 'forecast_storage'`

- [ ] **Step 3: Add the SQL constant and method**

In `forecast_storage.py`, add at module level after the imports:

```python
# Selects the final forecast run of each creation day.
#
# rank(), not row_number(): one run is MANY rows sharing a single
# forecast_created_timestamp (one per target date). row_number() would keep
# only one of them and silently drop the rest of the horizons.
#
# The horizon filter is load-bearing: water_temp_predictions also holds rows
# targeting dates BEFORE the run (historical backfill written alongside real
# forecasts). Those are not forecasts and must never be scored.
LAST_WATER_RUN_PER_DAY_SQL = """
    WITH ranked AS (
        SELECT *,
            rank() OVER (
                PARTITION BY forecast_created_date
                ORDER BY forecast_created_timestamp DESC
            ) AS run_rank
        FROM water_temp_predictions
        WHERE date_diff('day', forecast_created_date, target_date) BETWEEN 0 AND ?
    )
    SELECT
        forecast_created_date,
        forecast_created_timestamp,
        target_date,
        water_temp AS forecast_temp,
        date_diff('day', forecast_created_date, target_date) AS horizon_days
    FROM ranked
    WHERE run_rank = 1
    ORDER BY target_date, horizon_days
"""
```

Then add this method to `ForecastStorage`, after `get_forecasts_for_gap`:

```python
    def get_water_predictions_last_run_per_day(
        self, max_horizon: int = 5
    ) -> pd.DataFrame:
        """
        Retrieve stored water-temp forecasts, one run per creation day.

        Uses the final run of each creation day - the forecast a user would
        have seen at end of day.

        Args:
            max_horizon: Largest days-ahead horizon to return.

        Returns:
            DataFrame: forecast_created_date, forecast_created_timestamp,
                       target_date, forecast_temp, horizon_days
        """
        conn = self._get_connection()
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [max_horizon]).fetchdf()

        if result.empty:
            return result

        # MotherDuck returns tz-aware timestamps; the rest of the app is naive.
        result["forecast_created_timestamp"] = pd.to_datetime(
            result["forecast_created_timestamp"]
        ).dt.tz_localize(None)
        result["forecast_created_date"] = pd.to_datetime(result["forecast_created_date"])
        result["target_date"] = pd.to_datetime(result["target_date"])

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest test_accuracy.py::TestLastWaterRunPerDay -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add forecast_storage.py test_accuracy.py
git commit -m "Add last-run-per-day query for stored water temp forecasts

Uses rank() so every horizon of the winning run survives, and excludes
the negative-horizon backfill rows, which target dates before the run."
```

---

### Task 4: Bulk query for stored 3-hourly air forecasts

**Files:**
- Modify: `forecast_storage.py`
- Test: `test_accuracy.py`

**Interfaces:**
- Produces: `LAST_AIR_RUN_PER_DAY_SQL` and `ForecastStorage.get_air_forecasts_3hourly_last_run_per_day() -> pd.DataFrame` with columns `forecast_created_date`, `target_datetime`, `air_temp`.

**Why this exists:** the replay needs, for each anchor date, the air forecast *created on that date*. The existing `get_forecast_for_date` returns the most recent forecast **covering** a date — for a past anchor that can be a later, better-informed run, which would leak future information into the backtest. It also costs one round trip per anchor (~385). This method fetches everything in one query, correctly.

Note `air_temp_forecasts_3hourly` has no `forecast_created_date` column, so the partition casts the timestamp to a date.

- [ ] **Step 1: Write the failing test**

Append to `test_accuracy.py`:

```python
from forecast_storage import LAST_AIR_RUN_PER_DAY_SQL


def _local_air_db(rows):
    """In-memory DuckDB with the air_temp_forecasts_3hourly schema."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE air_temp_forecasts_3hourly (
            forecast_created_timestamp TIMESTAMP NOT NULL,
            target_datetime TIMESTAMP NOT NULL,
            air_temp DOUBLE NOT NULL,
            source VARCHAR DEFAULT 'OpenWeatherMap'
        )
    """)
    for created_ts, target_dt, air_temp in rows:
        conn.execute(
            "INSERT INTO air_temp_forecasts_3hourly VALUES (?, ?, ?, 'OpenWeatherMap')",
            [created_ts, target_dt, air_temp],
        )
    return conn


class TestLastAirRunPerDay:

    def test_picks_last_run_and_keeps_all_its_rows(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 1, 21), 15.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 14.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 3), 13.0),
            (datetime(2026, 5, 1, 8), datetime(2026, 5, 2, 0), 99.0),
        ]
        conn = _local_air_db(rows)
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        assert len(result) == 3
        assert 99.0 not in list(result["air_temp"])

    def test_groups_by_creation_date_not_timestamp(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 14.0),
            (datetime(2026, 5, 2, 20), datetime(2026, 5, 3, 0), 15.0),
        ]
        conn = _local_air_db(rows)
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        assert len(result) == 2
        assert sorted(pd.to_datetime(result["forecast_created_date"]).dt.day) == [1, 2]

    def test_rows_ordered_by_target_datetime(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 3, 0), 13.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 14.0),
        ]
        conn = _local_air_db(rows)
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        assert list(result["air_temp"]) == [14.0, 13.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest test_accuracy.py::TestLastAirRunPerDay -v`
Expected: FAIL — `ImportError: cannot import name 'LAST_AIR_RUN_PER_DAY_SQL'`

- [ ] **Step 3: Add the SQL constant and method**

In `forecast_storage.py`, after `LAST_WATER_RUN_PER_DAY_SQL`:

```python
# Final 3-hourly air forecast run of each creation day, all rows of that run.
#
# This table has no forecast_created_date column, so the partition casts the
# timestamp. rank() for the same reason as above: a run is many rows sharing
# one creation timestamp.
LAST_AIR_RUN_PER_DAY_SQL = """
    WITH ranked AS (
        SELECT
            CAST(forecast_created_timestamp AS DATE) AS forecast_created_date,
            target_datetime,
            air_temp,
            rank() OVER (
                PARTITION BY CAST(forecast_created_timestamp AS DATE)
                ORDER BY forecast_created_timestamp DESC
            ) AS run_rank
        FROM air_temp_forecasts_3hourly
    )
    SELECT forecast_created_date, target_datetime, air_temp
    FROM ranked
    WHERE run_rank = 1
    ORDER BY forecast_created_date, target_datetime
"""
```

Add this method to `ForecastStorage`:

```python
    def get_air_forecasts_3hourly_last_run_per_day(self) -> pd.DataFrame:
        """
        Retrieve every stored 3-hourly air forecast, one run per creation day.

        Fetched in bulk for backtesting: the replay needs the forecast created
        ON each anchor date. Do not use get_forecast_for_date for this - it
        returns the most recent forecast covering a date, which for a past
        anchor may be a later, better-informed run (future leakage).

        Returns:
            DataFrame: forecast_created_date, target_datetime, air_temp
        """
        conn = self._get_connection()
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        if result.empty:
            return result

        result["forecast_created_date"] = pd.to_datetime(result["forecast_created_date"])
        result["target_datetime"] = pd.to_datetime(
            result["target_datetime"]
        ).dt.tz_localize(None)

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest test_accuracy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add forecast_storage.py test_accuracy.py
git commit -m "Add bulk query for stored 3-hourly air forecasts by creation day

The backtest needs the forecast created ON each anchor date.
get_forecast_for_date returns the most recent forecast covering a date,
which for a past anchor leaks later information - and costs one round
trip per anchor."
```

---

### Task 5: Accuracy metrics

**Files:**
- Create: `accuracy.py`
- Test: `test_accuracy.py`

**Interfaces:**
- Produces:
  ```python
  BIAS_NOTE: str
  METRIC_COLUMNS: list
  def compute_metrics(df: pd.DataFrame) -> dict   # keys: mae, bias, rmse, hit_rate_0_5, n
  def metrics_by_horizon(df: pd.DataFrame) -> pd.DataFrame
  ```

- [ ] **Step 1: Write the failing tests**

Append to `test_accuracy.py`:

```python
from accuracy import BIAS_NOTE, compute_metrics, metrics_by_horizon


def _scored(pairs, horizons=None):
    """pairs: list of (forecast, actual)."""
    df = pd.DataFrame(
        {"forecast_temp": [p[0] for p in pairs], "actual_temp": [p[1] for p in pairs]}
    )
    if horizons is not None:
        df["horizon_days"] = horizons
    return df


class TestComputeMetrics:

    def test_hand_computed_case(self):
        # errors: +1.0, -1.0, +2.0, 0.0
        # mae = 4.0/4 = 1.0 ; bias = 2.0/4 = 0.5
        # rmse = sqrt((1+1+4+0)/4) = sqrt(1.5)
        # within 0.5: only the 0.0 error -> 25%
        m = compute_metrics(_scored([(11.0, 10.0), (9.0, 10.0), (14.0, 12.0), (8.0, 8.0)]))

        assert m["mae"] == pytest.approx(1.0)
        assert m["bias"] == pytest.approx(0.5)
        assert m["rmse"] == pytest.approx(np.sqrt(1.5))
        assert m["hit_rate_0_5"] == pytest.approx(25.0)
        assert m["n"] == 4

    def test_bias_positive_means_model_runs_warm(self):
        m = compute_metrics(_scored([(12.0, 10.0), (13.0, 10.0)]))
        assert m["bias"] > 0
        assert "warm" in BIAS_NOTE

    def test_bias_negative_means_model_runs_cold(self):
        m = compute_metrics(_scored([(8.0, 10.0), (7.0, 10.0)]))
        assert m["bias"] < 0

    def test_hit_rate_boundary_is_inclusive(self):
        """Exactly 0.5 C off counts as a hit."""
        m = compute_metrics(_scored([(10.5, 10.0), (10.51, 10.0)]))
        assert m["hit_rate_0_5"] == pytest.approx(50.0)

    def test_nan_rows_are_excluded_not_counted(self):
        m = compute_metrics(_scored([(11.0, 10.0), (np.nan, 10.0), (9.0, 10.0)]))
        assert m["n"] == 2
        assert m["mae"] == pytest.approx(1.0)

    def test_empty_frame_returns_nan_metrics_and_zero_n(self):
        m = compute_metrics(_scored([]))
        assert m["n"] == 0
        assert np.isnan(m["mae"])
        assert np.isnan(m["bias"])
        assert np.isnan(m["rmse"])
        assert np.isnan(m["hit_rate_0_5"])

    def test_all_nan_frame_returns_zero_n(self):
        m = compute_metrics(_scored([(np.nan, 10.0)]))
        assert m["n"] == 0


class TestMetricsByHorizon:

    def test_one_row_per_horizon_sorted(self):
        df = _scored([(11.0, 10.0), (13.0, 10.0), (10.0, 10.0)], horizons=[2, 1, 1])
        result = metrics_by_horizon(df)

        assert list(result["horizon_days"]) == [1, 2]
        assert list(result.columns) == [
            "horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"
        ]

    def test_metrics_computed_within_horizon(self):
        df = _scored([(11.0, 10.0), (12.0, 10.0)], horizons=[1, 2])
        result = metrics_by_horizon(df).set_index("horizon_days")

        assert result.loc[1, "mae"] == pytest.approx(1.0)
        assert result.loc[2, "mae"] == pytest.approx(2.0)
        assert result.loc[1, "n"] == 1

    def test_horizon_with_only_nan_reports_zero_n(self):
        df = _scored([(np.nan, 10.0), (12.0, 10.0)], horizons=[1, 2])
        result = metrics_by_horizon(df).set_index("horizon_days")

        assert result.loc[1, "n"] == 0
        assert np.isnan(result.loc[1, "mae"])

    def test_empty_frame_returns_empty_with_columns(self):
        result = metrics_by_horizon(_scored([], horizons=[]))
        assert result.empty
        assert list(result.columns) == [
            "horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_accuracy.py::TestComputeMetrics -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'accuracy'`

- [ ] **Step 3: Create `accuracy.py`**

```python
"""Forecast accuracy metrics and backtest replay"""

import numpy as np
import pandas as pd


BIAS_NOTE = "bias = forecast - actual; positive means the model runs warm"

HIT_THRESHOLD_C = 0.5

METRIC_COLUMNS = ["horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"]

SCORED_COLUMNS = [
    "target_date", "horizon_days", "forecast_temp",
    "actual_temp", "error", "air_source",
]

REPLAY_COLUMNS = ["target_date", "horizon_days", "forecast_temp", "air_source"]


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute accuracy metrics for scored forecasts.

    Args:
        df: DataFrame with 'forecast_temp' and 'actual_temp' columns.
            Rows with a missing value in either are excluded.

    Returns:
        dict with mae, bias, rmse, hit_rate_0_5 (percent), and n.
        Metrics are NaN when n == 0; n is always an int.
    """
    if df.empty:
        return {"mae": np.nan, "bias": np.nan, "rmse": np.nan,
                "hit_rate_0_5": np.nan, "n": 0}

    valid = df.dropna(subset=["forecast_temp", "actual_temp"])
    n = len(valid)

    if n == 0:
        return {"mae": np.nan, "bias": np.nan, "rmse": np.nan,
                "hit_rate_0_5": np.nan, "n": 0}

    error = valid["forecast_temp"] - valid["actual_temp"]

    return {
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "hit_rate_0_5": float((error.abs() <= HIT_THRESHOLD_C).mean() * 100),
        "n": int(n),
    }


def metrics_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics separately for each forecast horizon.

    Args:
        df: DataFrame with 'forecast_temp', 'actual_temp', 'horizon_days'.

    Returns:
        DataFrame with one row per horizon, sorted ascending.
    """
    if df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    rows = []
    for horizon, group in df.groupby("horizon_days", sort=True):
        metrics = compute_metrics(group)
        metrics["horizon_days"] = int(horizon)
        rows.append(metrics)

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_accuracy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add accuracy.py test_accuracy.py
git commit -m "Add forecast accuracy metrics

MAE, bias, RMSE, within-0.5C hit rate, and N. N is reported everywhere
because measurement coverage is sparse and some horizon buckets are thin."
```

---

### Task 6: Join forecasts to actual measurements

**Files:**
- Modify: `accuracy.py`
- Test: `test_accuracy.py`

**Interfaces:**
- Produces: `def join_actuals(forecasts: pd.DataFrame, water_temps: pd.DataFrame) -> pd.DataFrame` returning `SCORED_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

Append to `test_accuracy.py`:

```python
from accuracy import join_actuals


class TestJoinActuals:

    def _forecasts(self, rows):
        """rows: list of (target_date, horizon_days, forecast_temp)."""
        return pd.DataFrame({
            "target_date": [pd.Timestamp(r[0]) for r in rows],
            "horizon_days": [r[1] for r in rows],
            "forecast_temp": [r[2] for r in rows],
        })

    def _measurements(self, rows):
        """rows: list of (date, water_temp)."""
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
        })

    def test_inner_join_keeps_only_measured_days(self):
        forecasts = self._forecasts([
            (datetime(2026, 5, 1), 1, 11.0),
            (datetime(2026, 5, 2), 1, 12.0),
        ])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)

        assert len(result) == 1
        assert result.loc[0, "actual_temp"] == pytest.approx(10.0)

    def test_error_is_forecast_minus_actual(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.5)])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert result.loc[0, "error"] == pytest.approx(1.5)

    def test_duplicate_measurement_dates_do_not_multiply_rows(self):
        """Duplicate dates have broken this codebase before - keep the last."""
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 1), 10.4),
        ])

        result = join_actuals(forecasts, measurements)

        assert len(result) == 1
        assert result.loc[0, "actual_temp"] == pytest.approx(10.4)

    def test_time_component_on_dates_does_not_break_join(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1, 7, 0), 1, 11.0)])
        measurements = self._measurements([(datetime(2026, 5, 1, 0, 0), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert len(result) == 1

    def test_air_source_preserved_when_present(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        forecasts["air_source"] = ["ACTUAL"]
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert result.loc[0, "air_source"] == "ACTUAL"

    def test_air_source_defaults_to_forecast_when_absent(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert result.loc[0, "air_source"] == "FORECAST"

    def test_nan_forecasts_are_kept_for_metrics_to_exclude(self):
        """Coverage gaps stay visible in the frame; compute_metrics drops them."""
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, np.nan)])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert len(result) == 1
        assert np.isnan(result.loc[0, "forecast_temp"])

    def test_missing_measurement_values_are_dropped(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        measurements = self._measurements([(datetime(2026, 5, 1), np.nan)])

        result = join_actuals(forecasts, measurements)
        assert result.empty

    def test_empty_forecasts_returns_empty_with_schema(self):
        result = join_actuals(self._forecasts([]), self._measurements([]))
        assert result.empty
        assert list(result.columns) == [
            "target_date", "horizon_days", "forecast_temp",
            "actual_temp", "error", "air_source",
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_accuracy.py::TestJoinActuals -v`
Expected: FAIL — `ImportError: cannot import name 'join_actuals' from 'accuracy'`

- [ ] **Step 3: Implement `join_actuals`**

Add to `accuracy.py` after `metrics_by_horizon`:

```python
def join_actuals(forecasts: pd.DataFrame, water_temps: pd.DataFrame) -> pd.DataFrame:
    """
    Join forecasts to the measurements they were predicting.

    Only days with a measurement can be scored; unmeasured days drop out and
    are reflected in the reported N. Forecast NaNs are kept so coverage gaps
    stay visible - compute_metrics excludes them.

    Args:
        forecasts: target_date, horizon_days, forecast_temp, optional air_source
        water_temps: load_water_temps() frame with date, water_temp

    Returns:
        DataFrame with SCORED_COLUMNS. air_source defaults to 'FORECAST'.
    """
    if forecasts.empty or water_temps.empty:
        return pd.DataFrame(columns=SCORED_COLUMNS)

    actuals = water_temps.dropna(subset=["water_temp"]).copy()
    actuals["target_date"] = pd.to_datetime(actuals["date"]).dt.normalize()
    # Duplicate measurement dates have broken prediction chains before; keep last.
    actuals = actuals.drop_duplicates(subset=["target_date"], keep="last")
    actuals = actuals[["target_date", "water_temp"]].rename(
        columns={"water_temp": "actual_temp"}
    )

    scored = forecasts.copy()
    scored["target_date"] = pd.to_datetime(scored["target_date"]).dt.normalize()
    if "air_source" not in scored.columns:
        scored["air_source"] = "FORECAST"

    scored = scored.merge(actuals, on="target_date", how="inner")
    scored["error"] = scored["forecast_temp"] - scored["actual_temp"]

    return scored[SCORED_COLUMNS].sort_values(
        ["target_date", "horizon_days"]
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_accuracy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add accuracy.py test_accuracy.py
git commit -m "Join forecasts to measurements on target date

Deduplicates measurement dates, keeping the last - duplicate dates have
broken prediction chains in this codebase before."
```

---

### Task 7: Backtest replay of the current model

**Files:**
- Modify: `accuracy.py`
- Test: `test_accuracy.py`

**Interfaces:**
- Consumes: `WaterTempForecaster.predict_forward` (Task 1).
- Produces:
  ```python
  def replay_current_model(forecaster, water_temps, weather_provider, max_horizon=5) -> pd.DataFrame
  ```
  `weather_provider` is `Callable[[pd.Timestamp], tuple[pd.DataFrame | None, str]]` returning the hourly weather to simulate from that anchor plus an `air_source` label. Injecting it keeps the replay testable without MotherDuck and keeps all I/O in the caller, where it can be bulk-fetched. Returns `REPLAY_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

Append to `test_accuracy.py`:

```python
from accuracy import replay_current_model
from forecaster import WaterTempForecaster


def _weather_frame(start, n_hours, air_temp=15.0):
    return pd.DataFrame({
        "datetime": [pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(n_hours)],
        "air_temp": [air_temp] * n_hours,
        "shortwave_radiation": [0.0] * n_hours,
        "cloud_cover": [100.0] * n_hours,
    })


class TestReplayCurrentModel:

    def _measurements(self, dates_temps):
        return pd.DataFrame({
            "date": [pd.Timestamp(d) for d, _ in dates_temps],
            "water_temp": [t for _, t in dates_temps],
        })

    def _provider(self, label="FORECAST", n_hours=200, air_temp=15.0):
        def provider(anchor_date):
            return _weather_frame(anchor_date, n_hours, air_temp), label
        return provider

    def test_one_row_per_anchor_and_horizon(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 2), 10.5),
        ])

        result = replay_current_model(
            forecaster, measurements, self._provider(), max_horizon=3
        )

        assert set(result["horizon_days"]) == {1, 2, 3}
        assert len(result) == 6

    def test_horizon_zero_is_never_produced(self):
        """The replay is anchored on the measurement itself; it starts at day 1."""
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = replay_current_model(
            forecaster, measurements, self._provider(), max_horizon=3
        )
        assert 0 not in set(result["horizon_days"])

    def test_air_source_label_from_provider(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = replay_current_model(
            forecaster, measurements, self._provider(label="ACTUAL"), max_horizon=2
        )
        assert set(result["air_source"]) == {"ACTUAL"}

    def test_per_anchor_air_source_is_preserved(self):
        """Anchors with stored forecasts and anchors falling back must be distinguishable."""
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 2), 10.5),
        ])

        def provider(anchor_date):
            label = "FORECAST" if anchor_date.day == 1 else "ACTUAL"
            return _weather_frame(anchor_date, 200), label

        result = replay_current_model(forecaster, measurements, provider, max_horizon=1)

        by_date = result.set_index("target_date")["air_source"]
        assert by_date[pd.Timestamp(2026, 5, 2)] == "FORECAST"
        assert by_date[pd.Timestamp(2026, 5, 3)] == "ACTUAL"

    def test_warming_air_produces_warming_water(self):
        forecaster = WaterTempForecaster(k_air=0.05, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = replay_current_model(
            forecaster, measurements, self._provider(air_temp=25.0), max_horizon=3
        ).sort_values("horizon_days").reset_index(drop=True)

        assert result.loc[0, "forecast_temp"] > 10.0
        assert result["forecast_temp"].is_monotonic_increasing

    def test_anchor_with_no_weather_yields_nan_not_fabrication(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        def provider(anchor_date):
            return None, "ACTUAL"

        result = replay_current_model(forecaster, measurements, provider, max_horizon=2)

        assert len(result) == 2
        assert result["forecast_temp"].isna().all()
        assert set(result["air_source"]) == {"ACTUAL"}

    def test_measurements_with_nan_are_not_used_as_anchors(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([
            (datetime(2026, 5, 1), np.nan),
            (datetime(2026, 5, 2), 10.0),
        ])

        result = replay_current_model(
            forecaster, measurements, self._provider(), max_horizon=1
        )

        assert len(result) == 1
        assert result.loc[0, "target_date"] == pd.Timestamp(2026, 5, 3)

    def test_forecaster_weather_state_is_restored(self):
        """The replay must not leave the forecaster pointing at replay weather."""
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        forecaster.set_hourly_weather(_weather_frame(datetime(2026, 1, 1), 48))
        before = forecaster.hourly_weather.copy()

        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])
        replay_current_model(forecaster, measurements, self._provider(), max_horizon=2)

        pd.testing.assert_frame_equal(forecaster.hourly_weather, before)

    def test_empty_measurements_returns_empty_with_schema(self):
        forecaster = WaterTempForecaster()
        result = replay_current_model(
            forecaster, self._measurements([]), self._provider(), max_horizon=2
        )
        assert result.empty
        assert list(result.columns) == [
            "target_date", "horizon_days", "forecast_temp", "air_source"
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_accuracy.py::TestReplayCurrentModel -v`
Expected: FAIL — `ImportError: cannot import name 'replay_current_model' from 'accuracy'`

- [ ] **Step 3: Implement `replay_current_model`**

Add to `accuracy.py`:

```python
def replay_current_model(
    forecaster,
    water_temps: pd.DataFrame,
    weather_provider,
    max_horizon: int = 5,
) -> pd.DataFrame:
    """
    Re-run the current model over history: what accuracy would have been.

    For each measured day, anchor on that measurement and forecast forward
    max_horizon days. Each anchor is an independent trajectory.

    This is a model-development tool, not a record of real performance. It is
    optimistically biased: solar and cloud forecasts were never stored, so
    actual solar/cloud is used for every date. See the design spec.

    Args:
        forecaster: A fitted WaterTempForecaster. Its coefficients are used
                    as-is; nothing is refitted.
        water_temps: load_water_temps() frame with date, water_temp.
        weather_provider: Callable taking an anchor date (pd.Timestamp) and
                          returning (hourly_weather_df_or_None, air_source).
                          All I/O belongs in the caller so it can be bulk-fetched.
        max_horizon: Days ahead to forecast from each anchor.

    Returns:
        DataFrame with REPLAY_COLUMNS. forecast_temp is NaN where weather
        coverage was unavailable. Horizons start at 1 - the anchor is the
        measurement itself, so there is no horizon 0.
    """
    if water_temps.empty:
        return pd.DataFrame(columns=REPLAY_COLUMNS)

    anchors = water_temps.dropna(subset=["water_temp"]).copy()
    anchors["date"] = pd.to_datetime(anchors["date"]).dt.normalize()
    anchors = anchors.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    if anchors.empty:
        return pd.DataFrame(columns=REPLAY_COLUMNS)

    # The replay swaps weather in per anchor; put back whatever was there.
    saved_weather = forecaster.hourly_weather

    frames = []
    try:
        for _, anchor in anchors.iterrows():
            anchor_date = anchor["date"]
            hourly_weather, air_source = weather_provider(anchor_date)

            if hourly_weather is None or hourly_weather.empty:
                frames.append(pd.DataFrame({
                    "target_date": [
                        anchor_date + pd.Timedelta(days=h)
                        for h in range(1, max_horizon + 1)
                    ],
                    "horizon_days": list(range(1, max_horizon + 1)),
                    "forecast_temp": [np.nan] * max_horizon,
                    "air_source": [air_source] * max_horizon,
                }))
                continue

            forecaster.set_hourly_weather(hourly_weather)
            predictions = forecaster.predict_forward(
                start_datetime=anchor_date,
                start_water_temp=float(anchor["water_temp"]),
                targets=max_horizon,
            )

            frames.append(pd.DataFrame({
                "target_date": predictions["target_datetime"].dt.normalize(),
                "horizon_days": predictions["horizon_days"],
                "forecast_temp": predictions["water_temp"],
                "air_source": air_source,
            }))
    finally:
        forecaster.hourly_weather = saved_weather

    result = pd.concat(frames, ignore_index=True)
    return result[REPLAY_COLUMNS].sort_values(
        ["target_date", "horizon_days"]
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_accuracy.py -v`
Expected: PASS (all classes)

- [ ] **Step 5: Commit and open PR 2**

```bash
git add accuracy.py test_accuracy.py
git commit -m "Add backtest replay of the current model over history

One predict_forward call per measured anchor. Weather is injected via a
provider so the replay is testable without MotherDuck, all I/O stays in
the caller for bulk fetching, and the forecaster's weather state is
restored afterwards."

git push
gh pr create --base main --title "Add forecast accuracy backend" --body "$(cat <<'PRBODY'
Backend for the forecast accuracy tab. No UI changes - all of this is unit
tested and ships on its own.

- `accuracy.py`: `compute_metrics` (MAE, bias, RMSE, within-0.5C hit rate, N),
  `metrics_by_horizon`, `join_actuals`, and `replay_current_model`.
- `forecast_storage.py`: two bulk queries selecting the final run of each
  creation day, for stored water-temp forecasts and stored 3-hourly air
  forecasts.

Two details worth review attention:

- The queries use `rank()`, not `row_number()`. A run is many rows sharing one
  `forecast_created_timestamp`; `row_number()` would keep only one and silently
  drop the other horizons.
- `water_temp_predictions` holds rows targeting dates *before* their run -
  historical backfill, not forecasts. The horizon filter excludes them.

The new air-forecast query replaces `get_forecast_for_date` for backtesting:
that method returns the most recent forecast *covering* a date, which for a
past anchor can be a later, better-informed run - future leakage - and costs
one round trip per anchor.

Bias is `forecast - actual`; positive means the model runs warm. N is reported
everywhere because measurement coverage is sparse.

Tested: last-run selection including all horizons of the winning run, backfill
exclusion, metrics against a hand-computed case, bias sign, measurement-date
dedup, per-anchor air_source labelling, NaN handling where coverage is missing,
and restoration of the forecaster's weather state.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YL9pqG8tgTqoRdbTGAM3bw
PRBODY
)"
```

---

# PR 3 — Forecast Accuracy Tab

Tasks 8–10. UI only, on top of the PR 2 backend. Verified by running the app; the logic underneath is already unit-tested.

Begin after PR 2 is merged.

---

### Task 8: Chart builders

**Files:**
- Modify: `app.py` (add three functions near `create_temperature_chart`)

**Interfaces:**
- Produces: `create_horizon_accuracy_chart(horizon_metrics, selected_horizon) -> go.Figure`, `create_forecast_vs_actual_chart(scored, horizon) -> go.Figure`, `create_error_over_time_chart(scored) -> go.Figure`. All pure — frames in, figures out.

- [ ] **Step 1: Add the three chart builders**

Add to `app.py`, after `create_temperature_chart`:

```python
def create_horizon_accuracy_chart(
    horizon_metrics: pd.DataFrame, selected_horizon: int
) -> go.Figure:
    """MAE by forecast horizon, with the selected horizon highlighted."""
    colors = [
        "#1f77b4" if h == selected_horizon else "#c6dbef"
        for h in horizon_metrics["horizon_days"]
    ]

    fig = go.Figure(go.Bar(
        x=horizon_metrics["horizon_days"],
        y=horizon_metrics["mae"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in horizon_metrics["mae"]],
        textposition="outside",
        customdata=horizon_metrics["n"],
        hovertemplate="MAE %{y:.2f} C<br>%{customdata} forecasts scored<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Days ahead",
        yaxis_title="Mean absolute error (C)",
        height=340,
        showlegend=False,
        margin=dict(t=30),
    )
    fig.update_xaxes(dtick=1)
    return fig


def create_forecast_vs_actual_chart(scored: pd.DataFrame, horizon: int) -> go.Figure:
    """Forecast and measured water temp over time at one horizon."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scored["target_date"], y=scored["actual_temp"],
        name="Measured", mode="lines+markers",
        line=dict(color="#2ca02c", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=scored["target_date"], y=scored["forecast_temp"],
        name=f"Forecast ({horizon} day ahead)", mode="lines+markers",
        line=dict(color="#1f77b4", width=2, dash="dot"), marker=dict(size=5),
    ))

    leaky = scored[scored["air_source"] == "ACTUAL"]
    if not leaky.empty:
        fig.add_trace(go.Scatter(
            x=leaky["target_date"], y=leaky["forecast_temp"],
            name="Used actual air temp (optimistic)", mode="markers",
            marker=dict(size=9, color="#d62728", symbol="x"),
        ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Water temperature (C)",
        height=400, hovermode="x unified", margin=dict(t=30),
    )
    return fig


def create_error_over_time_chart(scored: pd.DataFrame) -> go.Figure:
    """Signed forecast error over time, with a zero reference line."""
    fig = go.Figure(go.Scatter(
        x=scored["target_date"], y=scored["error"],
        mode="lines+markers", name="Error",
        line=dict(color="#ff7f0e", width=1.5), marker=dict(size=4),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Forecast - actual (C)",
        height=320, showlegend=False, margin=dict(t=30),
    )
    return fig
```

- [ ] **Step 2: Verify the module still imports**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Add chart builders for forecast accuracy tab"
```

---

### Task 9: Cached loaders and the bulk weather provider

**Files:**
- Modify: `app.py` (imports, plus loaders after the existing `cached_load_*` block near line 77)

**Interfaces:**
- Consumes: `accuracy.replay_current_model`, `ForecastStorage.get_water_predictions_last_run_per_day`, `ForecastStorage.get_air_forecasts_3hourly_last_run_per_day`, and existing `interpolate_to_hourly` / `build_hourly_weather` / `cached_load_hourly_air_temps` / `cached_load_historical_solar_cloud`.
- Produces: `cached_load_stored_forecasts(max_horizon)` and `cached_replay(_forecaster, water_temps, coefficients, max_horizon)`.

**Bulk-fetch discipline:** the replay covers ~385 anchors. Everything it needs is fetched **once** — one MotherDuck query for stored air runs, one Meteostat call, one Open-Meteo call — then sliced per anchor in memory. Never one call per anchor.

- [ ] **Step 1: Add the accuracy imports**

In `app.py`, after `from quotes import QUOTES` (line 26):

```python
from accuracy import (
    BIAS_NOTE,
    compute_metrics,
    join_actuals,
    metrics_by_horizon,
    replay_current_model,
)
```

- [ ] **Step 2: Add the cached loaders**

Add after the existing `cached_load_*` functions (near line 77):

```python
@st.cache_data(ttl=3600)
def cached_load_stored_forecasts(max_horizon: int = 5):
    """Stored water-temp forecasts, one run per creation day."""
    storage = ForecastStorage()
    return storage.get_water_predictions_last_run_per_day(max_horizon=max_horizon)


@st.cache_data(ttl=3600)
def cached_replay(_forecaster, water_temps, coefficients, max_horizon: int = 5):
    """
    Backtest replay over history.

    Cached on `coefficients` so refitting the model invalidates the result;
    the argument is otherwise unused. `_forecaster` is underscore-prefixed so
    Streamlit does not try to hash it.

    All weather is fetched in bulk and sliced per anchor. The replay covers
    hundreds of anchors, so per-anchor fetching is not an option.
    """
    storage = ForecastStorage()

    stored_air = storage.get_air_forecasts_3hourly_last_run_per_day()
    runs_by_date = {
        date: group for date, group in stored_air.groupby("forecast_created_date")
    } if not stored_air.empty else {}

    start_date = pd.Timestamp(water_temps["date"].min()).normalize()
    end_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=max_horizon)

    # Fallback air temps for anchors with no stored forecast (pre-Feb-2026).
    try:
        actual_hourly = cached_load_hourly_air_temps(start_date, end_date)
    except DataLoadError:
        actual_hourly = pd.DataFrame(columns=["datetime", "air_temp"])

    # Solar/cloud was never stored, so actuals are all we have - for every date.
    try:
        solar_hist = cached_load_historical_solar_cloud(start_date, end_date)
    except DataLoadError:
        solar_hist = None

    def weather_provider(anchor_date):
        window_end = anchor_date + pd.Timedelta(days=max_horizon + 1)

        run = runs_by_date.get(anchor_date)
        if run is not None and not run.empty:
            hourly_air = interpolate_to_hourly(
                run[["target_datetime", "air_temp"]].rename(
                    columns={"target_datetime": "datetime"}
                )
            )
            air_source = "FORECAST"
        else:
            hourly_air = actual_hourly[
                (actual_hourly["datetime"] >= anchor_date)
                & (actual_hourly["datetime"] < window_end)
            ]
            air_source = "ACTUAL"

        if hourly_air.empty:
            return None, air_source

        return build_hourly_weather(hourly_air, solar_hist, None), air_source

    return replay_current_model(
        _forecaster, water_temps, weather_provider, max_horizon=max_horizon
    )
```

- [ ] **Step 3: Verify the module still imports**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add cached loaders for stored forecasts and backtest replay

All replay weather is fetched in bulk and sliced per anchor - the replay
covers hundreds of anchors. Anchors with no stored air forecast fall back
to Meteostat actuals, flagged ACTUAL."
```

---

### Task 10: The tab body

**Files:**
- Modify: `app.py:774` (tab list), and add the tab body after the `with tab_temp:` block
- Test: manual, via `streamlit run app.py`

**Scope note — `forecaster` is defined inside `with tab_temp:` (line 929).** Python scoping makes it visible afterwards, but only if that block ran far enough. So: initialise `forecaster = None` before `st.tabs(...)`, place the accuracy tab **after** `with tab_temp:`, and gate only the replay branch on it. Stored-forecasts mode needs no model at all.

- [ ] **Step 1: Initialise `forecaster` and add the tab**

In `app.py`, change line 774 from:

```python
    tab_temp, tab_quotes = st.tabs(["Temperature", "Heard at the Res"])
```

to:

```python
    # Defined inside the Temperature tab; the accuracy tab checks for it before
    # using it, so a failure there cannot NameError here.
    forecaster = None

    tab_temp, tab_accuracy, tab_quotes = st.tabs(
        ["Temperature", "Forecast Accuracy", "Heard at the Res"]
    )
```

- [ ] **Step 2: Add the tab body**

Add immediately **after** the entire `with tab_temp:` block ends (find the end of that block; it is the last top-level statement inside `main()` before the trailing helper calls):

```python
    with tab_accuracy:
        st.header("Forecast Accuracy")

        if not ENABLE_MOTHERDUCK:
            st.warning(
                "Forecast accuracy requires MotherDuck, which is not configured. "
                "Set MOTHERDUCK_TOKEN to enable this tab."
            )
        else:
            source_label = st.radio(
                "Comparison",
                ["Stored forecasts", "Model replay (current model)"],
                horizontal=True,
                help=(
                    "Stored forecasts: the forecasts we actually published, scored "
                    "against later measurements. Model replay: today's model re-run "
                    "over history, for comparing model versions."
                ),
            )
            is_replay = source_label.startswith("Model replay")

            if is_replay:
                st.caption(
                    "Optimistic. Solar and cloud forecasts were never stored, so "
                    "actual solar and cloud are used for every date. Points marked "
                    "with a cross also used actual air temperature."
                )
                horizon_options = [1, 2, 3, 4, 5]
            else:
                st.caption(
                    "The honest record: what we published, scored against what was "
                    "then measured."
                )
                horizon_options = [0, 1, 2, 3, 4, 5]

            selected_horizon = st.selectbox(
                "Days ahead",
                horizon_options,
                index=horizon_options.index(1),
                help="0 is a same-day nowcast, available for stored forecasts only.",
            )

            if is_replay and forecaster is None:
                st.warning(
                    "The model is not available because the Temperature tab could "
                    "not load its data. Stored forecasts still work."
                )
            else:
                try:
                    water_temps = cached_load_water_temps()

                    if is_replay:
                        raw = cached_replay(
                            forecaster,
                            water_temps,
                            (forecaster.k_air, forecaster.k_solar, forecaster.k_cool),
                            5,
                        )
                    else:
                        raw = cached_load_stored_forecasts(max_horizon=5)

                    scored = join_actuals(raw, water_temps)

                    if scored.empty:
                        st.warning(
                            "No forecasts could be matched to measurements yet. "
                            "Accuracy needs stored forecasts whose target dates "
                            "have since been measured."
                        )
                    else:
                        at_horizon = scored[
                            scored["horizon_days"] == selected_horizon
                        ].sort_values("target_date")
                        metrics = compute_metrics(at_horizon)

                        st.subheader(f"{selected_horizon}-day-ahead accuracy")
                        if metrics["n"] == 0:
                            st.warning(
                                f"No scored forecasts at {selected_horizon} days ahead."
                            )
                        else:
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("Mean absolute error", f"{metrics['mae']:.2f} C")
                            c2.metric("Bias", f"{metrics['bias']:+.2f} C")
                            c3.metric("RMSE", f"{metrics['rmse']:.2f} C")
                            c4.metric("Within 0.5 C", f"{metrics['hit_rate_0_5']:.0f}%")
                            c5.metric("Forecasts scored", f"{metrics['n']}")
                            st.caption(BIAS_NOTE)
                            st.caption(
                                f"Covering {at_horizon['target_date'].min().date()} "
                                f"to {at_horizon['target_date'].max().date()}"
                            )

                        st.subheader("Accuracy by forecast horizon")
                        st.caption(
                            "All horizons, unfiltered. Shows how forecasts degrade "
                            "further ahead."
                        )
                        st.plotly_chart(
                            create_horizon_accuracy_chart(
                                metrics_by_horizon(scored), selected_horizon
                            ),
                            width='stretch',
                        )

                        if not at_horizon.empty:
                            st.subheader(
                                f"Forecast vs measured ({selected_horizon} days ahead)"
                            )
                            st.plotly_chart(
                                create_forecast_vs_actual_chart(
                                    at_horizon, selected_horizon
                                ),
                                width='stretch',
                            )

                            st.subheader(
                                f"Error over time ({selected_horizon} days ahead)"
                            )
                            st.plotly_chart(
                                create_error_over_time_chart(at_horizon),
                                width='stretch',
                            )

                            with st.expander("Scored forecasts"):
                                st.dataframe(at_horizon, width='stretch')

                except ForecastStorageError as e:
                    st.error(f"Could not load stored forecasts: {e}")
                except DataLoadError as e:
                    st.error(f"Could not load measurements: {e}")
```

- [ ] **Step 3: Verify imports and the test suite**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`

Run: `env/bin/python -m pytest test_accuracy.py test_forecaster.py test_data.py test_combine_hourly.py -v`
Expected: PASS

- [ ] **Step 4: Manual verification**

Run: `env/bin/streamlit run app.py`

Check each:
1. The "Forecast Accuracy" tab appears and opens without error.
2. Default horizon is 1; tiles show MAE, bias, RMSE, within-0.5C, and N.
3. The by-horizon chart shows bars 0–5 and highlights the selected horizon.
4. Changing horizon updates tiles, both time-series charts, and the highlight.
5. "Model replay" removes horizon 0 and shows the leakage caption.
6. The replay completes in reasonable time — if it hangs, per-anchor fetching has crept back in.
7. Replay points before Feb 2026 are marked with a red cross (ACTUAL air temp).
8. No emojis anywhere in the tab.
9. No Streamlit deprecation warnings in the console.
10. The Temperature tab still renders correctly.

- [ ] **Step 5: Commit and open PR 3**

```bash
git add app.py
git commit -m "Add Forecast Accuracy tab

Stored forecasts and model replay side by side, filtered by horizon with
a 1-day default, plus an unfiltered by-horizon chart showing how accuracy
degrades further ahead. Replay leakage is labelled in the UI."

git push
gh pr create --base main --title "Add forecast accuracy reporting tab" --body "$(cat <<'PRBODY'
The UI for the accuracy backend added in the previous PR.

A "Forecast Accuracy" tab with a horizon filter (default 1 day) driving metric
tiles, a forecast-vs-measured series, and an error-over-time series, plus an
unfiltered by-horizon chart showing how accuracy degrades further ahead.

Two comparisons, switchable:

- **Stored forecasts** - what we actually published, scored against later
  measurements. No leakage.
- **Model replay** - today's model re-run over history, for comparing model
  versions. Labelled optimistic in the UI: solar/cloud forecasts were never
  stored (see #29), so actual solar/cloud is used throughout, and anchors
  before Feb 2026 fall back to actual air temperature, marked with a cross.

All replay weather is fetched in bulk and sliced per anchor - the replay covers
hundreds of anchors.

Tested: checked in the browser across both comparisons and every horizon, and
confirmed the Temperature tab is unaffected.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YL9pqG8tgTqoRdbTGAM3bw
PRBODY
)"
```

---

## Self-Review

**Spec coverage, checked honestly against each spec section:**

| Spec requirement | Task |
|---|---|
| PR 1 `predict()` refactor, migration, PR 1 tests | 1, 2 |
| Last-run-per-day selection, backfill exclusion | 3 |
| Stored air forecasts for replay input | 4 |
| MAE / bias / RMSE / hit rate / N | 5 |
| Join to actuals, measurement dedup | 6 |
| Replay, `air_source` flagging, NaN on missing coverage | 7 |
| **ACTUAL fallback to Meteostat pre-Feb-2026** | **9** (provider) + 7 (labelling) |
| Tab, horizon selector default 1, source toggle, leakage caption | 10 |
| By-horizon chart, forecast-vs-actual, error-over-time | 8, 10 |
| Caching keyed on coefficients | 9 |
| Documented limitations | spec only, no task |

**Corrections made after the first draft:**

1. **The ACTUAL fallback was missing.** The first draft's provider returned `None` for anchors without a stored run, which `replay_current_model` turns into all-NaN — so the replay would have silently covered only Feb–Sep 2026, and the "less accurate where we're using actuals" case the user asked for would never occur. Now implemented in Task 9 via bulk Meteostat, with Task 7 covering per-anchor labelling. The first self-review wrongly marked this covered.
2. **`row_number()` → `rank()`.** A run is many rows sharing one timestamp; `row_number()` keeps one and drops the rest of the horizons. The spec's SQL sketch has the same bug and Task 3's own test would have caught it.
3. **N+1 fetching.** Draft called `get_forecast_for_date` and `cached_load_historical_solar_cloud` per anchor — ~385 MotherDuck round trips plus ~385 Open-Meteo calls. Replaced with bulk fetch (Task 4 query, Task 9 provider).
4. **`get_forecast_for_date` leakage.** It returns the most recent forecast *covering* a date, which for a past anchor may be a later run. Draft shipped this as a documented caveat; it is now fixed properly by Task 4 instead.
5. **`forecaster` scope.** Draft placed the tab before `with tab_temp:`, where `forecaster` does not yet exist — a guaranteed `NameError`. The line-756 definition is inside an embed branch ending in `st.stop()` and never runs in the full view. Fixed by initialising to `None`, placing the tab after `tab_temp`, and gating only the replay branch.
6. **`set()` on targets** broke legacy equivalence for duplicate dates and would have raised `IndexError` in `fill_predictions`. Removed; a golden test now covers duplicates.
7. **`use_container_width=True` → `width='stretch'`**, matching the 5 existing uses.
8. **Deprecated `predict` alias deleted** — it had zero callers after Task 2.
9. **Two PRs → three**, splitting backend from UI.

**Known behaviour, not a defect:** `has_weather` is boolean, so a leg with partial coverage (say 6 of 24 hours) still simulates and reports `True`. Stored forecasts behaved the same way, so the replay is faithful — but it explains any oddity in the horizon-5 bar, where stored runs are truncated.
