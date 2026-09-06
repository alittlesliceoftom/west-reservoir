# Forecast Accuracy Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Forecast Accuracy tab reporting how well our water-temperature forecasts have performed, broken down by how many days ahead they were made — built on a forecaster API that can actually forecast.

**Architecture:** Five PRs. PR 1.5 was inserted mid-flight after a routine check found the historical air-temperature feed had been dead for five months (issue #33), and covers that repair plus the dependency upgrade it exposed. PR 0 is housekeeping: correct the docs, clean dependencies, stop storing backfilled predictions as forecasts, and lift the duplicated data pipeline into `data.py`. PR 1 replaces `WaterTempForecaster.predict()` — today a DataFrame gap-filler — with `predict_forward`, one simulation checkpointed at each target. PR 2 adds the whole backend (`accuracy.py` plus two MotherDuck queries), unit-tested, touching no UI. PR 3 adds the Streamlit tab. Two independent comparisons share one output schema so the same charts render both: **stored forecasts** (the honest record) and **model replay** (current model re-run over history).

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

# PR 0 — Housekeeping Refactor

Tasks 0.1–0.5. Clears the ground before the accuracy work. No behaviour change; the pure data-assembly logic moves to `data.py`, where it is testable without mocking Streamlit.

**Why first:** the data pipeline currently exists twice (`app.py:676-770` and `app.py:809-935`), identical in logic and differing only in side effects (MotherDuck writes, `st.warning` calls). That duplication is why PR 1 has to change `forecaster.predict` in two places, and why PR 3 would otherwise add a third partial copy.

---

### Task 0.1: Correct the documentation

**Files:**
- Modify: `CLAUDE.md`, `README.md`

CLAUDE.md claims "649 lines total" and "app.py (210 lines)". Actual: 3,186 and 1,103. It also omits `forecast_storage.py`, MotherDuck, the solar/cloud model, and the quotes tab. Every agent that reads it starts with a false map.

- [ ] **Step 1: Replace the File Structure section in CLAUDE.md**

Remove all line counts — they rot on every commit and add nothing. Replace the structure block with:

```
├── app.py              - Streamlit web dashboard and tabs
├── config.py           - Configuration, API keys, feature flags
├── data.py             - Data loading and frame assembly
├── forecaster.py       - Physics-based prediction model
├── forecast_storage.py - MotherDuck forecast storage and retrieval
├── quotes.py           - Static quotes for the "Heard at the Res" tab
├── requirements.txt    - Python dependencies
└── docs/superpowers/   - Design specs and implementation plans
```

- [ ] **Step 2: Correct the model description in CLAUDE.md**

The "Module Responsibilities" section describes a single-term model:

> Physics equation: `dT/dt = k × (T_air_yesterday - T_water)`

Replace with the actual three-term hourly model (see `forecaster.py`):

```
Per hour:
  clearness(t) = 1 - cloud_cover(t) / 100
  T_water(t+1h) = T_water(t)
                + k_air   * (T_air(t) - T_water(t))   # conduction/convection
                + k_solar * I(t)                       # shortwave heating
                - k_cool  * clearness(t)               # clear-sky radiative cooling

Measurements are at 7am, so every training and prediction period runs 7am to 7am.
`fit()` optimises k_air, k_solar and k_cool together against measured data.
```

Add `forecast_storage.py` to Module Responsibilities: stores and retrieves air-temp forecasts and water-temp predictions in MotherDuck; gated by `ENABLE_MOTHERDUCK` in `config.py`.

- [ ] **Step 3: Remove the line-count comparison table**

Delete the "Key Improvements" table rows citing line counts (3,576 → 649) and the "Lines of code" row. Keep the qualitative rows.

- [ ] **Step 4: Update README.md the same way**

Remove any line counts; make sure the described structure and data sources match reality (MotherDuck, Open-Meteo solar/cloud).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Correct project docs; drop line counts

Documented structure was substantially out of date: no forecast_storage,
no MotherDuck, and a single-term model description for what is now a
three-term hourly model. Line counts removed - they rot on every commit."
```

---

### Task 0.2: Clean up dependencies

**Files:**
- Modify: `requirements.txt`

`scikit-learn`, `seaborn` and `matplotlib` have **zero imports** anywhere in the codebase. `pytest` is missing despite five test files.

- [ ] **Step 1: Verify the three are genuinely unused**

Run: `grep -rn "sklearn\|seaborn\|matplotlib" *.py`
Expected: no output.

- [ ] **Step 2: Edit `requirements.txt`**

Remove `scikit-learn==1.3.2`, `seaborn==0.12.2`, `matplotlib==3.7.4`. Add `pytest==7.4.4`.

- [ ] **Step 3: Verify the app and tests still work**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Run: `env/bin/python -m pytest test_forecaster.py test_data.py test_combine_hourly.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Drop unused deps, add pytest

scikit-learn, seaborn and matplotlib have no imports anywhere. pytest was
missing despite five test files, so a fresh clone could not run them."
```

---

### Task 0.3: Stop storing backfilled predictions

**Files:**
- Modify: `app.py` (the prediction-storage block, around line 942)
- Test: `test_accuracy.py` is not created until PR 2, so add this test to `test_data.py`

`app.py:942` stores every row with `source == "PREDICTED"`, with no date filter. That includes historical gap-fills, which is why `water_temp_predictions` holds rows at horizons down to −665. Those are not forecasts and pollute any accuracy analysis.

- [ ] **Step 1: Write the failing test**

Append to `test_data.py`:

```python
from data import select_storable_predictions


class TestSelectStorablePredictions:
    """Only genuine forward-looking forecasts should be stored."""

    def _frame(self, rows):
        """rows: list of (date, water_temp, source)."""
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
            "source": [r[2] for r in rows],
        })

    def test_keeps_predictions_from_today_onward(self):
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2026, 5, 10), 12.0, "PREDICTED"),
            (datetime(2026, 5, 11), 12.5, "PREDICTED"),
        ])
        result = select_storable_predictions(df, today)
        assert len(result) == 2

    def test_drops_predictions_for_past_dates(self):
        """Backfilled gap-fills target dates before the run - not forecasts."""
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2024, 12, 1), 5.0, "PREDICTED"),
            (datetime(2026, 5, 9), 11.0, "PREDICTED"),
            (datetime(2026, 5, 11), 12.5, "PREDICTED"),
        ])
        result = select_storable_predictions(df, today)
        assert list(result["date"]) == [pd.Timestamp(2026, 5, 11)]

    def test_drops_non_predicted_rows(self):
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2026, 5, 11), 12.5, "PREDICTED"),
            (datetime(2026, 5, 11), 12.5, "MEASURED"),
            (datetime(2026, 5, 12), 13.0, "AIR_ONLY"),
        ])
        result = select_storable_predictions(df, today)
        assert list(result["source"]) == ["PREDICTED"]

    def test_drops_rows_with_missing_water_temp(self):
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2026, 5, 11), float("nan"), "PREDICTED"),
            (datetime(2026, 5, 12), 13.0, "PREDICTED"),
        ])
        result = select_storable_predictions(df, today)
        assert len(result) == 1

    def test_empty_frame_returns_empty(self):
        result = select_storable_predictions(self._frame([]), pd.Timestamp(2026, 5, 10))
        assert result.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest test_data.py::TestSelectStorablePredictions -v`
Expected: FAIL — `ImportError: cannot import name 'select_storable_predictions' from 'data'`

- [ ] **Step 3: Implement it in `data.py`**

```python
def select_storable_predictions(
    temperatures: pd.DataFrame, run_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Select the predictions worth storing as forecasts.

    predict()/fill_predictions() fills every AIR_ONLY row, including historical
    gaps. Those backfilled rows target dates BEFORE the run and are not
    forecasts - storing them pollutes accuracy analysis with rows at negative
    horizons.

    Args:
        temperatures: Frame with date, water_temp, source.
        run_date: The date this forecast run is being made.

    Returns:
        Rows where source == 'PREDICTED', water_temp is present, and the
        target date is not in the past.
    """
    if temperatures.empty:
        return temperatures

    predictions = temperatures[temperatures["source"] == "PREDICTED"].copy()
    predictions = predictions.dropna(subset=["water_temp"])
    return predictions[
        pd.to_datetime(predictions["date"]).dt.normalize()
        >= pd.Timestamp(run_date).normalize()
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest test_data.py::TestSelectStorablePredictions -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Use it at the storage call site**

In `app.py`, add `select_storable_predictions` to the `from data import (...)` block. Then replace these two lines around line 942:

```python
                        predictions_df = temperatures_deduped[temperatures_deduped["source"] == "PREDICTED"].copy()
                        # Filter out any rows with NULL water_temp (shouldn't happen but safety check)
                        predictions_df = predictions_df.dropna(subset=["water_temp"])
```

with:

```python
                        predictions_df = select_storable_predictions(
                            temperatures_deduped, pd.Timestamp.now()
                        )
```

- [ ] **Step 6: Verify**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Run: `env/bin/python -m pytest test_data.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add data.py app.py test_data.py
git commit -m "Stop storing backfilled predictions as forecasts

fill_predictions fills every AIR_ONLY row including historical gaps, and
all of them were being stored - which is why water_temp_predictions holds
rows at horizons down to -665. Only forward-looking rows are forecasts."
```

---

### Task 0.4: Move the pure hourly-weather helpers to `data.py`

**Files:**
- Modify: `data.py` (receive `combine_hourly_temps`, `build_hourly_weather`)
- Modify: `app.py` (remove them, import instead)
- Modify: `test_combine_hourly.py`, `test_data.py` (import from `data`)

Both functions are pure frame-in/frame-out with no Streamlit dependency. The tests already import them from `app` and have to **mock out `sys.modules['streamlit']`** to do it — the placement is the problem, and moving them removes the mock.

- [ ] **Step 1: Move both functions verbatim**

Cut `combine_hourly_temps` (`app.py:154`, including its nested `normalize_datetime_col` helper) and `build_hourly_weather` (`app.py:78`) from `app.py` and paste into `data.py`. Change nothing inside them.

`build_hourly_weather` uses `Optional` — confirm `data.py` imports it from `typing`, adding it if not.

- [ ] **Step 2: Import them in `app.py`**

Add `build_hourly_weather` and `combine_hourly_temps` to the existing `from data import (...)` block.

- [ ] **Step 3: Update the tests to import from `data`**

In `test_combine_hourly.py`, delete the streamlit-mocking block:

```python
# We need to mock streamlit before importing app
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

from app import combine_hourly_temps
```

and replace with:

```python
from data import combine_hourly_temps
```

The `sys.path.insert(0, '.')` lines can go too.

In `test_data.py`, change `from app import combine_hourly_temps` to import it from `data` alongside the existing imports.

- [ ] **Step 4: Verify**

Run: `env/bin/python -m pytest test_combine_hourly.py test_data.py -v`
Expected: PASS, with no streamlit mocking

Run: `env/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`

- [ ] **Step 5: Commit**

```bash
git add app.py data.py test_combine_hourly.py test_data.py
git commit -m "Move pure hourly-weather helpers to data.py

combine_hourly_temps and build_hourly_weather are frame-in/frame-out with
no Streamlit dependency. Their tests had to mock sys.modules['streamlit']
to import them from app - that mock is now gone."
```

---

### Task 0.5: Extract the duplicated pipeline into `data.py`

**Files:**
- Modify: `data.py` (three new functions)
- Modify: `app.py` (both pipeline copies call them)
- Test: `test_data.py`

The two pipelines are identical in data logic. Extract the three shared blocks so both call sites shrink to orchestration plus their own side effects.

**Interfaces:**
```python
def fill_daily_from_hourly(air_temps_daily, hourly_air_temps) -> pd.DataFrame
def build_temperatures_frame(water_temps, air_temps_hist) -> pd.DataFrame
def deduplicate_temperatures(temperatures) -> pd.DataFrame
```

- [ ] **Step 1: Write the failing tests**

Append to `test_data.py`:

```python
from data import (
    build_temperatures_frame,
    deduplicate_temperatures,
    fill_daily_from_hourly,
)


class TestFillDailyFromHourly:

    def test_fills_missing_daily_values_from_hourly(self):
        """The daily API lags ~2 days; hourly is more current."""
        daily = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)],
            "air_temp": [10.0], "air_temp_min": [8.0], "air_temp_max": [12.0],
        })
        hourly = pd.DataFrame({
            "datetime": [
                pd.Timestamp(2026, 5, 2) + pd.Timedelta(hours=h) for h in range(24)
            ],
            "air_temp": [float(h) for h in range(24)],
        })

        result = fill_daily_from_hourly(daily, hourly).set_index("date")

        assert pd.Timestamp(2026, 5, 2) in result.index
        assert result.loc[pd.Timestamp(2026, 5, 2), "air_temp"] == pytest.approx(11.5)
        assert result.loc[pd.Timestamp(2026, 5, 2), "air_temp_min"] == pytest.approx(0.0)
        assert result.loc[pd.Timestamp(2026, 5, 2), "air_temp_max"] == pytest.approx(23.0)

    def test_existing_daily_values_take_precedence(self):
        daily = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)],
            "air_temp": [10.0], "air_temp_min": [8.0], "air_temp_max": [12.0],
        })
        hourly = pd.DataFrame({
            "datetime": [
                pd.Timestamp(2026, 5, 1) + pd.Timedelta(hours=h) for h in range(24)
            ],
            "air_temp": [99.0] * 24,
        })

        result = fill_daily_from_hourly(daily, hourly).set_index("date")
        assert result.loc[pd.Timestamp(2026, 5, 1), "air_temp"] == pytest.approx(10.0)

    def test_rows_without_air_temp_are_dropped(self):
        daily = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)],
            "air_temp": [float("nan")],
            "air_temp_min": [float("nan")], "air_temp_max": [float("nan")],
        })
        hourly = pd.DataFrame(columns=["datetime", "air_temp"])

        result = fill_daily_from_hourly(daily, hourly)
        assert result.empty

    def test_returns_expected_columns(self):
        daily = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)],
            "air_temp": [10.0], "air_temp_min": [8.0], "air_temp_max": [12.0],
        })
        hourly = pd.DataFrame(columns=["datetime", "air_temp"])

        result = fill_daily_from_hourly(daily, hourly)
        assert list(result.columns) == ["date", "air_temp", "air_temp_min", "air_temp_max"]


class TestBuildTemperaturesFrame:

    def test_measured_where_water_temp_present(self):
        water = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)], "water_temp": [10.0]
        })
        air = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)],
            "air_temp": [15.0], "air_temp_min": [12.0], "air_temp_max": [18.0],
        })

        result = build_temperatures_frame(water, air)
        assert list(result["source"]) == ["MEASURED"]

    def test_air_only_where_water_temp_absent(self):
        water = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)], "water_temp": [10.0]
        })
        air = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1), pd.Timestamp(2026, 5, 2)],
            "air_temp": [15.0, 16.0],
            "air_temp_min": [12.0, 13.0], "air_temp_max": [18.0, 19.0],
        })

        result = build_temperatures_frame(water, air).set_index("date")
        assert result.loc[pd.Timestamp(2026, 5, 2), "source"] == "AIR_ONLY"

    def test_sorted_by_date(self):
        water = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 3), pd.Timestamp(2026, 5, 1)],
            "water_temp": [11.0, 10.0],
        })
        air = pd.DataFrame({
            "date": [pd.Timestamp(2026, 5, 1)],
            "air_temp": [15.0], "air_temp_min": [12.0], "air_temp_max": [18.0],
        })

        result = build_temperatures_frame(water, air)
        assert result["date"].is_monotonic_increasing


class TestDeduplicateTemperatures:

    def _frame(self, rows):
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
            "source": [r[2] for r in rows],
        })

    def test_measured_beats_air_only_on_the_same_date(self):
        df = self._frame([
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert len(result) == 1
        assert result.loc[0, "source"] == "MEASURED"

    def test_air_only_beats_predicted_on_the_same_date(self):
        df = self._frame([
            (datetime(2026, 5, 1), 9.0, "PREDICTED"),
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
        ])
        result = deduplicate_temperatures(df)

        assert len(result) == 1
        assert result.loc[0, "source"] == "AIR_ONLY"

    def test_one_row_per_date(self):
        df = self._frame([
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
            (datetime(2026, 5, 2), 11.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert len(result) == 2
        assert not result["date"].duplicated().any()

    def test_helper_column_is_not_left_behind(self):
        df = self._frame([(datetime(2026, 5, 1), 10.0, "MEASURED")])
        result = deduplicate_temperatures(df)

        assert "_sort_priority" not in result.columns

    def test_result_is_sorted_and_reindexed(self):
        df = self._frame([
            (datetime(2026, 5, 2), 11.0, "MEASURED"),
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert result["date"].is_monotonic_increasing
        assert list(result.index) == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'fill_daily_from_hourly' from 'data'`

- [ ] **Step 3: Implement the three functions in `data.py`**

Lift the logic verbatim from `app.py:809-900`; only the wrapping changes.

```python
SOURCE_PRIORITY = {"MEASURED": 0, "AIR_ONLY": 1, "PREDICTED": 2}


def fill_daily_from_hourly(
    air_temps_daily: pd.DataFrame, hourly_air_temps: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill gaps in daily air temps using hourly data.

    The daily Meteostat feed lags roughly two days; the hourly feed is more
    current. Daily values win wherever they exist.

    Returns:
        DataFrame: date, air_temp, air_temp_min, air_temp_max
    """
    columns = ["date", "air_temp", "air_temp_min", "air_temp_max"]

    if hourly_air_temps is None or hourly_air_temps.empty:
        return air_temps_daily[columns].dropna(subset=["air_temp"])

    hourly_daily_stats = (
        hourly_air_temps.assign(date=hourly_air_temps["datetime"].dt.normalize())
        .groupby("date")["air_temp"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    hourly_daily_stats.columns = ["date", "air_temp_h", "air_temp_min_h", "air_temp_max_h"]
    hourly_daily_stats["date"] = pd.to_datetime(hourly_daily_stats["date"])

    merged = pd.merge(air_temps_daily, hourly_daily_stats, on="date", how="outer")
    merged["air_temp"] = merged["air_temp"].fillna(merged["air_temp_h"])
    merged["air_temp_min"] = merged["air_temp_min"].fillna(merged["air_temp_min_h"])
    merged["air_temp_max"] = merged["air_temp_max"].fillna(merged["air_temp_max_h"])

    return merged[columns].dropna(subset=["air_temp"])


def build_temperatures_frame(
    water_temps: pd.DataFrame, air_temps_hist: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge water and air temperatures into the main temperatures frame.

    Rows with a water measurement are MEASURED; the rest are AIR_ONLY and are
    candidates for prediction.
    """
    temperatures = pd.merge(water_temps, air_temps_hist, on="date", how="outer")
    temperatures = temperatures.sort_values("date").reset_index(drop=True)
    temperatures["source"] = "MEASURED"
    temperatures.loc[temperatures["water_temp"].isna(), "source"] = "AIR_ONLY"
    return temperatures


def deduplicate_temperatures(temperatures: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per date, preferring MEASURED over AIR_ONLY over PREDICTED.

    Duplicate dates break the prediction chain, which walks row to row.
    """
    result = temperatures.copy()
    result["_sort_priority"] = result["source"].map(SOURCE_PRIORITY)
    result = result.sort_values(["date", "_sort_priority"]).reset_index(drop=True)
    result = result.drop(columns=["_sort_priority"])
    return result.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_data.py -v`
Expected: PASS

- [ ] **Step 5: Use them in both pipelines**

Add the three names to `app.py`'s `from data import (...)` block.

In **both** the embed pipeline (`app.py:676-770`) and the full pipeline (`app.py:809-935`), replace:

- the `hourly_daily_stats` / merge / fillna block → `air_temps_hist = fill_daily_from_hourly(air_temps_hist, hourly_air_temps)`
- the merge / sort / `source` marking block → `temperatures = build_temperatures_frame(water_temps, air_temps_hist)`
- the `_sort_priority` sort / drop / `drop_duplicates` block → `temperatures_deduped = deduplicate_temperatures(temperatures)`

Leave each pipeline's own side effects untouched: the embed view stays silent on `DataLoadError`, the full view keeps its `st.warning` calls and MotherDuck writes.

- [ ] **Step 6: Verify nothing changed behaviourally**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Run: `env/bin/python -m pytest -v`
Expected: PASS

Run: `env/bin/streamlit run app.py`

Check by hand:
1. The Temperature tab renders with the same chart as before the refactor.
2. The forecast still extends into the future.
3. The debug panel shows the same measured/predicted counts.
4. No new warnings in the console.

- [ ] **Step 7: Commit and open PR 0**

```bash
git add app.py data.py test_data.py
git commit -m "Extract duplicated pipeline logic into data.py

The data pipeline existed twice, identical in logic and differing only in
side effects. The shared parts now live in data.py as tested functions."

git push -u origin worktree-forecast-accuracy-tab
gh pr create --base main --title "Housekeeping: correct docs, clean deps, de-duplicate the data pipeline" --body "$(cat <<'PRBODY'
Groundwork before the forecast-accuracy work. No behaviour change.

- **Docs corrected.** CLAUDE.md described a 649-line project with a 210-line
  `app.py` (actually 3,186 and 1,103), omitted `forecast_storage.py`,
  MotherDuck and the solar/cloud model, and documented a single-term model
  that is now three-term. Line counts removed entirely - they rot on every
  commit.
- **Dependencies.** `scikit-learn`, `seaborn` and `matplotlib` had zero
  imports anywhere; `pytest` was missing despite five test files.
- **Backfilled predictions are no longer stored as forecasts.** Every
  `PREDICTED` row was being written, including historical gap-fills - which
  is why `water_temp_predictions` holds rows at horizons down to -665.
- **Pure data logic moved to `data.py`.** `combine_hourly_temps` and
  `build_hourly_weather` had no Streamlit dependency, yet their tests had to
  mock `sys.modules['streamlit']` to import them from `app`. That mock is gone.
- **The duplicated pipeline is gone.** `app.py:676-770` and `app.py:809-935`
  were identical in data logic, differing only in side effects. The shared
  parts are now three tested functions in `data.py`.

Tested: new unit tests cover the extracted functions and the storage filter;
existing tests pass without the streamlit mock; the app was run and the
Temperature tab checked against its pre-refactor output.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YL9pqG8tgTqoRdbTGAM3bw
PRBODY
)"
```

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


# PR 1.5 — Data Source Repair and Dependency Upgrade

Inserted mid-flight after a routine verification found the historical air
temperature feed had been dead for five months. Split into two shipped PRs:

**PR 1.5a — Open-Meteo migration and freshness tests.** DONE.

**PR 1.5b — Python and dependency upgrade.** Next.

## Why this exists

Meteostat moved its data hosting to `data.meteostat.net` and stopped rebuilding
`bulk.meteostat.net/v2/` on **2026-03-20**. The old host still returns HTTP 200,
so `meteostat` 1.6.5 kept reading a frozen snapshot — globally, not just London
(issue #33).

Nobody noticed for five months because `combine_hourly_temps` linearly
interpolated across the resulting 161-day hole. The model was handed a smooth
ramp and trained on it without complaint: the whole first week of July 2026
varied by **0.64 C**, where real London air swings roughly 10 C a day. **106 of
383 training pairs** were fitted against that line.

This directly threatened the accuracy work. The replay's design falls back to
"actual" air temperature where no stored forecast exists — which would have
meant scoring the model against a straight line and reporting the result as
accuracy. The stored-forecast comparison was never affected, since it needs
only water measurements.

## PR 1.5a — completed

- Historical daily and hourly air temperature now come from the Open-Meteo
  archive, already in use for solar and cloud. Verified current to today, with
  **18.5 C** of real variation across that same July week.
- Interpolation capped at `MAX_INTERPOLATION_HOURS` (6). Bridging 3-hourly
  forecast data is legitimate; papering over an outage is not. Real gaps stay
  gaps so `fit()` skips them.
- `test_data_freshness.py` checks every source is current **and that hourly air
  actually varies** — flatness catches interpolated fiction that a staleness
  check alone would pass. Confirmed it would have caught the outage: the old
  interpolated series had a median daily swing of 0.089 C against a 2.0 C
  threshold.
- `meteostat` dependency dropped. Recoverable from git history if ever wanted;
  issue #33 records exactly how to rebuild it against the new host in ~30 lines.
- Fixed `predict_forward` treating a zero-length leg (duplicate target date) as
  a missing-weather gap, which poisoned the rest of the chain with NaN.

Issue #34 tracks running the freshness checks on a schedule and opening a
GitHub issue automatically when a source goes stale.

## PR 1.5b — Python and dependency upgrade

The Meteostat investigation surfaced a second problem: the project runs
**Python 3.10.4**, and `meteostat` 2.x requires `>=3.11`. That specific upgrade
is no longer needed since Meteostat is gone, but it revealed how far behind the
runtime and pins have drifted — several are two or more years old.

### Task 1.5b.1: Upgrade Python and dependency pins

**Files:**
- Modify: `requirements.txt`
- Modify: `CLAUDE.md` (record the required Python version)
- Possibly: `.devcontainer/`, if it pins a Python version

- [ ] **Step 1: Establish the target Python version**

The only interpreter installed besides 3.10.4 is **3.14.7**. Before committing
to it, verify the whole stack installs and the suite passes on it in a
throwaway venv — 3.14 is recent enough that some scientific wheels may lag.

If any dependency has no 3.14 wheel and will not build, fall back to installing
3.12 or 3.13 via Homebrew rather than forcing it. Record whichever version is
chosen, and why, in CLAUDE.md.

- [ ] **Step 2: Install the stack in a throwaway venv and run the suite**

```bash
python3.14 -m venv /tmp/testenv && /tmp/testenv/bin/pip install -q --upgrade pip
/tmp/testenv/bin/pip install streamlit pandas plotly requests scipy duckdb pytest
/tmp/testenv/bin/python -m pytest -q
```

Expected: all tests pass. Investigate every failure individually — a genuine
incompatibility must not be papered over by loosening a pin.

Watch specifically for:
- **pandas 3.x behaviour changes.** `pandas==2.1.4` is pinned today. Copy-on-write
  became the default in pandas 3.0, and `deduplicate_temperatures` /
  `fill_predictions` mutate frames in place after copying. The golden tests and
  the real-data equivalence script are the guard here.
- **`resample(...).interpolate(limit=..., limit_area=...)`**, used by the new
  interpolation cap.
- **Streamlit API drift**, particularly `width='stretch'` versus the older
  `use_container_width`.

- [ ] **Step 3: Pin the resolved versions**

Record the exact versions that passed, not floating ranges — this project pins
deliberately (see `duckdb==1.4.4`, pinned for MotherDuck compatibility, commit
ab5e3f9). Keep that pin unless MotherDuck is verified against a newer one.

- [ ] **Step 4: Verify equivalence on real data**

Run the golden equivalence check (legacy `predict()` versus `fill_predictions`
over the real production frame) on the new interpreter. A pandas upgrade
changing prediction values would be a silent, serious regression, and this is
the check that catches it.

- [ ] **Step 5: Verify the app renders**

Launch Streamlit on the new interpreter and confirm the Temperature tab
matches: measured, today's and tomorrow's forecasts, weekly extremes, chart
series and reading count. Compare against a capture taken on 3.10 first.

- [ ] **Step 6: Recreate the project venv**

Only after the throwaway venv is green. `env/` is the user's working
environment, shared with their own shell, so **confirm before replacing it**.

- [ ] **Step 7: Document and commit**

CLAUDE.md should state the required Python version and how to recreate the
venv. Commit with the tested version numbers in the message.

---
# PR 2 — Accuracy Backend

Tasks 3–8. All backend, fully unit-tested, **zero `app.py` changes**. Reviews and ships independently of any UI.

Begin after PR 1 is merged.

---

### Task 3: Query stored water-temp forecasts, last run per day

**Files:**
- Modify: `forecast_storage.py`
- Test: `test_accuracy.py` (create)

**Interfaces:**
- Produces: `LAST_WATER_RUN_PER_DAY_SQL` (module constant) and `ForecastStorage.get_water_predictions_last_run_per_day(max_horizon: int = 5) -> pd.DataFrame` with columns `forecast_created_date`, `forecast_created_timestamp`, `target_date`, `forecast_temp`, `horizon_days`.

The SQL is what's under test. Run it against a local in-memory DuckDB with the same schema — no network, identical SQL.

- [ ] **Step 0: Rebase onto merged PR 1**

All three PRs share one branch, so without this PR 2 would show PR 1's commits again. Per CLAUDE.md, after every merge:

```bash
git fetch --all && git rebase origin/main
```

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

### Task 7: Splice actual air history onto a stored forecast run

**Files:**
- Modify: `accuracy.py`
- Test: `test_accuracy.py`

**Interfaces:**
- Produces: `def splice_air_history(actual_hourly, stored_run_hourly, anchor_datetime) -> pd.DataFrame` returning `datetime`, `air_temp`.

**Why this exists — this is the difference between a meaningful replay and a meaningless one.**

The last run of day `d` is created around 21:00–23:00, and an OpenWeatherMap fetch only covers slots *after* the fetch time. The stored data confirms it: horizon 0 has **16,014 rows against ~30,400** at horizons 1–4 — a run covers only the remainder of its creation day, and a late run covers almost none of it.

So for anchor `d`, feeding the raw stored run to the forecaster gives hourly air starting ~21:00. `_get_weather_for_period(d 07:00, d+1 07:00)` then returns ~10 rows, `has_weather` is `True`, and `_simulate_period` takes 10 steps instead of 24. The horizon-1 replay would not be "optimistic" — it would be a **different computation from anything the model ever runs**, and its MAE would be meaningless.

What the live forecast actually had at 21:00 on `d` was: measured air for 07:00→21:00 (already elapsed) spliced with the OWM forecast after. This function reconstructs that.

> **Not reusable:** `combine_hourly_temps` (`app.py:154`) gives *historical precedence on overlap*. For a past anchor, Meteostat actuals cover the whole window, so it would override the entire stored forecast and the replay would silently become 100% actuals. This needs its own function.

- [ ] **Step 1: Write the failing tests**

Append to `test_accuracy.py`:

```python
from accuracy import splice_air_history


def _air_frame(start, n_hours, air_temp, step_hours=1):
    return pd.DataFrame({
        "datetime": [
            pd.Timestamp(start) + pd.Timedelta(hours=i * step_hours)
            for i in range(n_hours)
        ],
        "air_temp": [air_temp] * n_hours,
    })


class TestSpliceAirHistory:

    def test_actuals_fill_the_head_before_the_run_starts(self):
        """The live forecast had measured air for the elapsed part of the day."""
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)

        assert result["datetime"].min() == anchor
        before = result[result["datetime"] < pd.Timestamp(2026, 5, 1, 21)]
        after = result[result["datetime"] >= pd.Timestamp(2026, 5, 1, 21)]
        assert set(before["air_temp"]) == {10.0}
        assert set(after["air_temp"]) == {20.0}

    def test_no_hours_are_missing_across_the_join(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        gaps = result["datetime"].diff().dropna().unique()

        assert list(gaps) == [pd.Timedelta(hours=1)]

    def test_no_duplicate_datetimes(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        assert not result["datetime"].duplicated().any()

    def test_stored_wins_where_both_have_the_same_hour(self):
        """Past the forecast's creation time, the forecast is what was used."""
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        at_23 = result[result["datetime"] == pd.Timestamp(2026, 5, 1, 23)]

        assert at_23["air_temp"].iloc[0] == pytest.approx(20.0)

    def test_rows_before_the_anchor_are_dropped(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 4, 30, 0), 72, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        assert result["datetime"].min() == anchor

    def test_empty_stored_run_returns_actuals_from_anchor(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)

        result = splice_air_history(
            actuals, pd.DataFrame(columns=["datetime", "air_temp"]), anchor
        )

        assert result["datetime"].min() == anchor
        assert set(result["air_temp"]) == {10.0}

    def test_empty_actuals_returns_stored_run_from_anchor(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(
            pd.DataFrame(columns=["datetime", "air_temp"]), stored, anchor
        )

        assert set(result["air_temp"]) == {20.0}
        assert len(result) == 24

    def test_both_empty_returns_empty_with_schema(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        empty = pd.DataFrame(columns=["datetime", "air_temp"])

        result = splice_air_history(empty, empty, anchor)

        assert result.empty
        assert list(result.columns) == ["datetime", "air_temp"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/bin/python -m pytest test_accuracy.py::TestSpliceAirHistory -v`
Expected: FAIL — `ImportError: cannot import name 'splice_air_history' from 'accuracy'`

- [ ] **Step 3: Implement `splice_air_history`**

Add to `accuracy.py`:

```python
AIR_COLUMNS = ["datetime", "air_temp"]


def splice_air_history(
    actual_hourly: pd.DataFrame,
    stored_run_hourly: pd.DataFrame,
    anchor_datetime,
) -> pd.DataFrame:
    """
    Reconstruct the air-temperature series the live forecast actually had.

    A stored forecast run created late on day d only covers the remainder of
    that day - roughly 16k rows at horizon 0 against 30k at horizon 1. Feeding
    the raw run to the forecaster would simulate ~10 hours of a 24-hour period
    and produce a number the model never computes.

    What the live run had was measured air for the elapsed part of the day,
    then forecast air from its creation time onward. This splices that.

    Args:
        actual_hourly: Measured hourly air temps (datetime, air_temp).
        stored_run_hourly: The stored forecast run, interpolated to hourly.
        anchor_datetime: Start of the window (the 7am measurement time).

    Returns:
        DataFrame (datetime, air_temp) from anchor_datetime onward, with the
        stored run taking precedence wherever it has data.
    """
    anchor = pd.Timestamp(anchor_datetime)

    def _prepared(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=AIR_COLUMNS)
        out = df[AIR_COLUMNS].copy()
        out["datetime"] = pd.to_datetime(out["datetime"])
        return out[out["datetime"] >= anchor]

    actuals = _prepared(actual_hourly)
    stored = _prepared(stored_run_hourly)

    if stored.empty:
        return actuals.sort_values("datetime").reset_index(drop=True)

    # Actuals only cover the head, up to where the forecast run begins.
    head = actuals[actuals["datetime"] < stored["datetime"].min()]

    combined = pd.concat([head, stored], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"], keep="last")

    return combined.sort_values("datetime").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/bin/python -m pytest test_accuracy.py::TestSpliceAirHistory -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add accuracy.py test_accuracy.py
git commit -m "Splice measured air history onto stored forecast runs

A run created late in the day only covers that day's remainder, so
simulating from it alone would cover ~10 hours of a 24-hour period. The
live forecast had measured air for the elapsed part and forecast air
after; this reconstructs that."
```

---

### Task 8: Backtest replay of the current model

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

Tasks 9–11. UI only, on top of the PR 2 backend. Verified by running the app; the logic underneath is already unit-tested.

Begin after PR 2 is merged.

---

### Task 9: Chart builders

**Files:**
- Modify: `app.py` (add three functions near `create_temperature_chart`)

**Interfaces:**
- Produces: `create_horizon_accuracy_chart(horizon_metrics, selected_horizon) -> go.Figure`, `create_forecast_vs_actual_chart(scored, horizon) -> go.Figure`, `create_error_over_time_chart(scored) -> go.Figure`. All pure — frames in, figures out.

- [ ] **Step 0: Rebase onto merged PR 2**

```bash
git fetch --all && git rebase origin/main
```

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
        # A horizon with n == 0 has NaN MAE; label it blank, not "nan".
        text=[
            "" if pd.isna(v) else f"{v:.2f}" for v in horizon_metrics["mae"]
        ],
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

### Task 10: Cached loaders, fitted model, and the bulk weather provider

**Files:**
- Modify: `app.py` (imports, plus loaders after the existing `cached_load_*` block near line 77)

**Interfaces:**
- Consumes: `accuracy.replay_current_model`, `accuracy.splice_air_history`, both new `ForecastStorage` queries, and existing `interpolate_to_hourly` / `build_hourly_weather` / `cached_load_hourly_air_temps` / `cached_load_historical_solar_cloud`.
- Produces: `cached_load_stored_forecasts(max_horizon)`, `cached_fitted_model_coefficients(water_temps, start_date, end_date)`, `cached_replay(water_temps, coefficients, max_horizon)`.

**Two design points:**

1. **The accuracy tab fits its own model.** It must not borrow the `forecaster` local from the Temperature tab: that block ends in `st.stop()` on a data error (`app.py:1099`), which halts the whole script. Fitting needs only *historical* weather plus measured temps — no forecast data — so the helper is small. It returns the three coefficients as a **tuple**, which is hashable, so `st.cache_data` works without underscore-prefix hacks.
2. **Bulk-fetch discipline.** The replay covers ~385 anchors. Everything is fetched **once** — one MotherDuck query, one Meteostat call, one Open-Meteo call — then sliced per anchor in memory. Never one call per anchor.

- [ ] **Step 1: Add the accuracy imports**

In `app.py`, after `from quotes import QUOTES` (line 26):

```python
from accuracy import (
    BIAS_NOTE,
    compute_metrics,
    join_actuals,
    metrics_by_horizon,
    replay_current_model,
    splice_air_history,
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
def cached_fitted_model_coefficients(water_temps, start_date, end_date):
    """
    Fit the model on historical weather and return its coefficients.

    The accuracy tab fits its own model rather than reusing the Temperature
    tab's: that block calls st.stop() on a data error, which halts the script.
    Fitting needs only historical weather, so this is cheap and self-contained.

    Returns:
        (k_air, k_solar, k_cool) - a tuple, so it is hashable as a cache key.
    """
    hourly_air = cached_load_hourly_air_temps(start_date, end_date)

    try:
        solar_hist = cached_load_historical_solar_cloud(start_date, end_date)
    except DataLoadError:
        solar_hist = None

    weather = build_hourly_weather(hourly_air, solar_hist, None)

    forecaster = WaterTempForecaster()
    forecaster.set_hourly_weather(weather)
    forecaster.fit(water_temps.assign(source="MEASURED"))

    return (forecaster.k_air, forecaster.k_solar, forecaster.k_cool)


@st.cache_data(ttl=3600)
def cached_replay(water_temps, coefficients, max_horizon: int = 5):
    """
    Backtest replay over history, using the given model coefficients.

    Keyed on `coefficients`, so refitting invalidates the cached result.

    All weather is fetched in bulk and sliced per anchor - the replay covers
    hundreds of anchors, so per-anchor fetching is not an option.
    """
    k_air, k_solar, k_cool = coefficients
    forecaster = WaterTempForecaster(k_air=k_air, k_solar=k_solar, k_cool=k_cool)

    storage = ForecastStorage()
    stored_air = storage.get_air_forecasts_3hourly_last_run_per_day()
    runs_by_date = (
        {date: group for date, group in stored_air.groupby("forecast_created_date")}
        if not stored_air.empty
        else {}
    )

    start_date = pd.Timestamp(water_temps["date"].min()).normalize()
    end_date = pd.Timestamp.now().normalize()

    # Measured air: the head of each window before its forecast was made, and
    # the whole window for anchors with no stored forecast at all.
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
        anchor_dt = anchor_date + pd.Timedelta(hours=WaterTempForecaster.MEASUREMENT_HOUR)
        window_end = anchor_dt + pd.Timedelta(days=max_horizon)

        window_actuals = actual_hourly[
            (actual_hourly["datetime"] >= anchor_dt)
            & (actual_hourly["datetime"] < window_end)
        ]

        run = runs_by_date.get(anchor_date)
        if run is not None and not run.empty:
            stored_hourly = interpolate_to_hourly(
                run[["target_datetime", "air_temp"]].rename(
                    columns={"target_datetime": "datetime"}
                )
            )
            # A run made at ~21:00 covers only the day's remainder; splice the
            # measured air the live forecast already had for the elapsed hours.
            hourly_air = splice_air_history(window_actuals, stored_hourly, anchor_dt)
            air_source = "FORECAST"
        else:
            hourly_air = window_actuals
            air_source = "ACTUAL"

        if hourly_air.empty:
            return None, air_source

        return build_hourly_weather(hourly_air, solar_hist, None), air_source

    return replay_current_model(
        forecaster, water_temps, weather_provider, max_horizon=max_horizon
    )
```

- [ ] **Step 3: Verify the module still imports**

Run: `env/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add cached loaders for stored forecasts and backtest replay

The accuracy tab fits its own model rather than reusing the Temperature
tab's, which calls st.stop() on a data error. All replay weather is
fetched in bulk and sliced per anchor, with measured air spliced onto the
head of each stored run."
```

---

### Task 11: The tab body

**Files:**
- Modify: `app.py:774` (tab list), and add the tab body between the `with tab_quotes:` and `with tab_temp:` blocks
- Test: manual, via `streamlit run app.py`

**Placement is load-bearing.** `app.py:1099` calls `st.stop()` inside an `except DataLoadError` within `with tab_temp:`. Streamlit executes tab bodies in *code* order, so anything written after that block never runs when the Temperature tab hits a data error. The accuracy tab therefore goes **before** `with tab_temp:`, and depends on no variable defined inside it.

- [ ] **Step 1: Add the tab to the tab list**

In `app.py`, change line 774 from:

```python
    tab_temp, tab_quotes = st.tabs(["Temperature", "Heard at the Res"])
```

to:

```python
    tab_temp, tab_accuracy, tab_quotes = st.tabs(
        ["Temperature", "Forecast Accuracy", "Heard at the Res"]
    )
```

- [ ] **Step 2: Add the tab body**

Insert immediately **after** the `with tab_quotes:` block ends and **before** `with tab_temp:` (currently line 791):

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
                    "Today's model re-run over history, using the air temperature "
                    "each forecast actually had available: measured up to the time "
                    "the forecast was made, forecast after. Optimistic, because "
                    "solar and cloud forecasts were never stored, so actual solar "
                    "and cloud are used throughout. Points marked with a cross had "
                    "no stored air forecast either and used measured air for the "
                    "whole window."
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

            try:
                water_temps = cached_load_water_temps()

                if is_replay:
                    coefficients = cached_fitted_model_coefficients(
                        water_temps,
                        pd.Timestamp(water_temps["date"].min()).normalize(),
                        pd.Timestamp.now().normalize(),
                    )
                    raw = cached_replay(water_temps, coefficients, 5)
                else:
                    raw = cached_load_stored_forecasts(max_horizon=5)

                scored = join_actuals(raw, water_temps)

                if scored.empty:
                    st.warning(
                        "No forecasts could be matched to measurements yet. "
                        "Accuracy needs stored forecasts whose target dates have "
                        "since been measured."
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
7. Replay points before Feb 2026 are marked with a red cross (no stored air forecast).
8. **Sanity-check the splice:** replay horizon-1 MAE should be broadly comparable to stored-forecast horizon-1 MAE. If the replay is dramatically better, the splice is likely not applying and the simulation is running on partial days.
9. No emojis anywhere in the tab.
10. No Streamlit deprecation warnings in the console.
11. The Temperature tab still renders correctly.

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
  stored (see #29), so actual solar/cloud is used throughout, and anchors with
  no stored air forecast fall back to measured air, marked with a cross.

Two placement details worth knowing:

- The tab sits before the Temperature tab in code order and fits its own model,
  because the Temperature tab calls `st.stop()` on a data error, which would
  otherwise stop the accuracy tab rendering at all.
- Replay weather is fetched in bulk and sliced per anchor; the replay covers
  hundreds of anchors.

Tested: checked in the browser across both comparisons and every horizon,
confirmed replay and stored horizon-1 MAE are comparable (a large gap would
indicate the air-history splice is not applying), and confirmed the Temperature
tab is unaffected.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YL9pqG8tgTqoRdbTGAM3bw
PRBODY
)"
```

---

## Self-Review

**Spec coverage, checked against each spec section:**

| Spec requirement | Task |
|---|---|
| PR 1 `predict()` refactor, migration, PR 1 tests | 1, 2 |
| Last-run-per-day selection, backfill exclusion | 3 |
| Stored air forecasts as replay input | 4 |
| MAE / bias / RMSE / hit rate / N | 5 |
| Join to actuals, measurement dedup | 6 |
| Faithful air input for each anchor | 7 |
| Replay, `air_source` flagging, NaN on missing coverage | 8 |
| ACTUAL fallback where no stored run exists | 8 (labelling) + 10 (provider) |
| By-horizon chart, forecast-vs-actual, error-over-time | 9, 11 |
| Caching keyed on coefficients | 10 |
| Tab, horizon selector default 1, source toggle, leakage caption | 11 |
| Documented limitations | spec only, no task |

**Corrections made across three drafts.** Recording these because several were
caught only by re-reading the code rather than the spec:

1. **Stored runs do not cover their own creation day** (draft 3). A run created
   ~21:00 covers only the day's remainder — horizon 0 holds 16,014 rows against
   ~30,400 at horizons 1–4. Feeding it raw to the forecaster would simulate ~10
   hours of a 24-hour period: not "optimistic", simply a different computation
   from anything the model runs, with a meaningless MAE. Task 7 splices the
   measured air the live forecast already had. `combine_hourly_temps` cannot be
   reused — it gives historical precedence on overlap, which for a past anchor
   would override the whole forecast and make the replay 100% actuals.
2. **`st.stop()` at `app.py:1099`** sits inside `except DataLoadError` within
   `with tab_temp:`. Streamlit runs tab bodies in code order, so a tab placed
   after it never renders on a Temperature-tab data failure. The accuracy tab
   now sits *before* `tab_temp` and fits its own model (Task 10), depending on
   no variable from that block. The earlier `forecaster = None` guard would not
   have helped.
3. **The ACTUAL fallback was missing** (draft 2). The provider returned `None`
   for anchors without a stored run, which becomes all-NaN — so the replay would
   have silently covered only Feb–Sep 2026, and the "less accurate where we're
   using actuals" case would never have occurred. The first self-review wrongly
   marked this covered.
4. **`row_number()` → `rank()`.** A run is many rows sharing one timestamp;
   `row_number()` keeps one and drops the remaining horizons. The spec's SQL
   sketch had the same bug; both are now fixed.
5. **N+1 fetching.** Draft called `get_forecast_for_date` and the solar loader
   per anchor — ~385 MotherDuck round trips plus ~385 Open-Meteo calls.
   Replaced with bulk fetch (Task 4 query, Task 10 provider).
6. **`get_forecast_for_date` leakage.** It returns the most recent forecast
   *covering* a date, which for a past anchor may be a later run. Draft shipped
   this as a documented caveat; Task 4 fixes it properly instead.
7. **`set()` on targets** broke legacy equivalence for duplicate dates and would
   have raised `IndexError` in `fill_predictions`. Removed; a golden test covers
   duplicates.
8. **`use_container_width=True` → `width='stretch'`**, matching the 5 existing
   uses. **Deprecated `predict` alias deleted** — zero callers after Task 2.
   **NaN bar labels** guarded so an empty horizon renders blank, not "nan".
9. **Two PRs → three**, splitting backend from UI, with a rebase step between.

**Minor, known, not defects:**

- `has_weather` is boolean, so a leg with partial coverage still simulates and
  reports `True`. Stored forecasts behaved the same way, so the replay is
  faithful — but it explains any oddity in the horizon-5 bar, where stored runs
  are truncated.
- `_legacy_fill` uses `.replace(hour=7)`, which preserves any minutes on the
  input date; `predict_forward` uses `.normalize() + 7h`, which zeroes them.
  These agree for midnight-normalised dates, which is what production carries.
  The golden tests use midnight dates accordingly.
- Task 11 step 4 check 8 is the practical guard on the splice: if replay
  horizon-1 MAE comes out dramatically better than stored horizon-1 MAE, the
  splice is probably not applying.
