"""Tests for data loading and processing functions"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from data import (
    build_hourly_weather,
    combine_hourly_temps,
    interpolate_to_hourly,
    DataLoadError,
)
from forecaster import WaterTempForecaster


class TestInterpolateToHourly:
    """Tests for interpolate_to_hourly function"""

    def test_empty_dataframe(self):
        """Empty input returns empty output"""
        df = pd.DataFrame(columns=["datetime", "air_temp"])
        result = interpolate_to_hourly(df)
        assert result.empty

    def test_single_row(self):
        """Single row returns single row (no interpolation possible)"""
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        result = interpolate_to_hourly(df)
        assert len(result) == 1
        assert result.iloc[0]["air_temp"] == 10.0

    def test_3hourly_to_hourly(self):
        """3-hourly data interpolates to hourly"""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 15:00"),
                pd.Timestamp("2024-01-15 18:00"),
            ],
            "air_temp": [10.0, 13.0, 16.0]
        })
        result = interpolate_to_hourly(df)

        # Should have 7 rows: 12, 13, 14, 15, 16, 17, 18
        assert len(result) == 7

        # Check interpolated values
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 13:00")]["air_temp"].iloc[0] == 11.0
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 14:00")]["air_temp"].iloc[0] == 12.0
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 16:00")]["air_temp"].iloc[0] == 14.0

    def test_preserves_original_values(self):
        """Original values are preserved exactly"""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 15:00"),
            ],
            "air_temp": [10.5, 13.5]
        })
        result = interpolate_to_hourly(df)

        # Original timestamps should have exact values
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 12:00")]["air_temp"].iloc[0] == 10.5
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 15:00")]["air_temp"].iloc[0] == 13.5


class TestCombineHourlyTemps:
    """Tests for combine_hourly_temps function"""

    def test_empty_both(self):
        """Both empty returns empty"""
        hist = pd.DataFrame(columns=["datetime", "air_temp"])
        fore = pd.DataFrame(columns=["datetime", "air_temp"])
        result = combine_hourly_temps(hist, fore)
        assert result.empty

    def test_empty_historical(self):
        """Empty historical returns forecast only"""
        hist = pd.DataFrame(columns=["datetime", "air_temp"])
        fore = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        result = combine_hourly_temps(hist, fore)
        assert len(result) == 1

    def test_empty_forecast(self):
        """Empty forecast returns historical only"""
        hist = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        fore = pd.DataFrame(columns=["datetime", "air_temp"])
        result = combine_hourly_temps(hist, fore)
        assert len(result) == 1

    def test_historical_takes_precedence(self):
        """Historical data overrides forecast for overlapping times"""
        hist = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 13:00"),
            ],
            "air_temp": [10.0, 11.0]
        })
        fore = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 13:00"),  # Overlaps
                pd.Timestamp("2024-01-15 14:00"),
            ],
            "air_temp": [99.0, 12.0]  # 99.0 should be ignored
        })
        result = combine_hourly_temps(hist, fore)

        # Should have 3 rows
        assert len(result) == 3

        # Historical value should be used for 13:00
        val_13 = result[result["datetime"] == pd.Timestamp("2024-01-15 13:00")]["air_temp"].iloc[0]
        assert val_13 == 11.0

    def test_no_overlap(self):
        """Non-overlapping data combines correctly"""
        hist = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 10:00"),
                pd.Timestamp("2024-01-15 11:00"),
            ],
            "air_temp": [8.0, 9.0]
        })
        fore = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 14:00"),
                pd.Timestamp("2024-01-15 15:00"),
            ],
            "air_temp": [12.0, 13.0]
        })
        result = combine_hourly_temps(hist, fore)

        # Should have gap interpolated: 10, 11, 12, 13, 14, 15 = 6 rows
        assert len(result) == 6

    def test_sorted_output(self):
        """Output is sorted by datetime"""
        hist = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        fore = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 15:00")],
            "air_temp": [13.0]
        })
        result = combine_hourly_temps(hist, fore)

        # Verify sorted
        assert result["datetime"].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


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

    def test_run_date_with_time_component_still_keeps_today(self):
        """A run made at 9pm must still store today's prediction."""
        df = self._frame([(datetime(2026, 5, 10), 12.0, "PREDICTED")])
        result = select_storable_predictions(df, pd.Timestamp(2026, 5, 10, 21, 30))
        assert len(result) == 1

    def test_empty_frame_returns_empty(self):
        result = select_storable_predictions(self._frame([]), pd.Timestamp(2026, 5, 10))
        assert result.empty


from data import (
    build_temperatures_frame,
    deduplicate_temperatures,
)


class TestBuildTemperaturesFrame:

    def _water(self, rows):
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
        })

    def _air(self, rows):
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "air_temp": [r[1] for r in rows],
            "air_temp_min": [r[1] - 2 for r in rows],
            "air_temp_max": [r[1] + 2 for r in rows],
        })

    def test_measured_where_water_temp_present(self):
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 1), 10.0)]),
            self._air([(datetime(2026, 5, 1), 15.0)]),
        )
        assert list(result["source"]) == ["MEASURED"]

    def test_air_only_where_water_temp_absent(self):
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 1), 10.0)]),
            self._air([(datetime(2026, 5, 1), 15.0), (datetime(2026, 5, 2), 16.0)]),
        ).set_index("date")

        assert result.loc[pd.Timestamp(2026, 5, 2), "source"] == "AIR_ONLY"

    def test_sorted_by_date(self):
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 3), 11.0), (datetime(2026, 5, 1), 10.0)]),
            self._air([(datetime(2026, 5, 1), 15.0)]),
        )
        assert result["date"].is_monotonic_increasing

    def test_water_only_dates_are_kept(self):
        """An outer join - a measurement with no air data must survive."""
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 9), 12.0)]),
            self._air([(datetime(2026, 5, 1), 15.0)]),
        )
        assert pd.Timestamp(2026, 5, 9) in list(result["date"])


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
        assert "_sort_priority" not in deduplicate_temperatures(df).columns

    def test_result_is_sorted_and_reindexed(self):
        df = self._frame([
            (datetime(2026, 5, 2), 11.0, "MEASURED"),
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert result["date"].is_monotonic_increasing
        assert list(result.index) == [0, 1]

    def test_input_frame_is_not_mutated(self):
        df = self._frame([
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
        ])
        before = df.copy()
        deduplicate_temperatures(df)

        pd.testing.assert_frame_equal(df, before)


from data import MAX_INTERPOLATION_HOURS


class TestInterpolationCap:
    """
    combine_hourly_temps must not invent weather across long gaps.

    This is the direct test of the guarantee. When Meteostat silently died
    (issue #33), the uncapped version drew a straight line across 161 days and
    the model trained on it. The gap is bridged by OUR code, so assert on our
    code rather than inferring it from how flat the resulting data looks.
    """

    def _series(self, start, n_hours, temp):
        return pd.DataFrame({
            "datetime": [pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(n_hours)],
            "air_temp": [float(temp)] * n_hours,
        })

    def test_short_gap_is_bridged(self):
        """A 3-hourly forecast leaves 2-hour holes; those must still be filled."""
        hist = self._series("2026-05-01 00:00", 6, 10.0)          # 00:00-05:00
        fore = self._series("2026-05-01 08:00", 6, 20.0)          # 08:00-13:00
        # gap is 06:00 and 07:00 -> 2 hours, well within the cap

        result = combine_hourly_temps(hist, fore).set_index("datetime")

        assert pd.Timestamp("2026-05-01 06:00") in result.index
        assert pd.Timestamp("2026-05-01 07:00") in result.index

    def test_long_gap_is_not_bridged(self):
        """The #33 failure mode: a gap far beyond the cap must stay a gap."""
        hist = self._series("2026-05-01 00:00", 6, 10.0)           # ends 05:00
        fore = self._series("2026-05-10 00:00", 6, 20.0)           # 9 days later

        result = combine_hourly_temps(hist, fore).set_index("datetime")

        # Real data on both sides survives...
        assert pd.Timestamp("2026-05-01 00:00") in result.index
        assert pd.Timestamp("2026-05-10 00:00") in result.index
        # ...but the middle is absent, not invented.
        assert pd.Timestamp("2026-05-05 12:00") not in result.index

    def test_gap_exactly_at_the_cap_is_bridged(self):
        hist = self._series("2026-05-01 00:00", 1, 10.0)                     # 00:00
        gap_end = 1 + MAX_INTERPOLATION_HOURS                                # 07:00
        fore = self._series(f"2026-05-01 0{gap_end}:00", 3, 20.0)

        result = combine_hourly_temps(hist, fore).set_index("datetime")

        missing = [
            pd.Timestamp("2026-05-01 00:00") + pd.Timedelta(hours=h)
            for h in range(1, MAX_INTERPOLATION_HOURS + 1)
        ]
        assert all(ts in result.index for ts in missing), (
            f"a gap of exactly MAX_INTERPOLATION_HOURS ({MAX_INTERPOLATION_HOURS}) "
            f"should still be bridged"
        )

    def test_gap_one_hour_past_the_cap_is_not_bridged(self):
        hist = self._series("2026-05-01 00:00", 1, 10.0)                     # 00:00
        start = 1 + MAX_INTERPOLATION_HOURS + 1                              # 08:00
        fore = self._series(f"2026-05-01 0{start}:00", 3, 20.0)

        result = combine_hourly_temps(hist, fore).set_index("datetime")

        assert pd.Timestamp("2026-05-01 04:00") not in result.index

    def test_no_fabricated_values_across_a_months_long_outage(self):
        """
        Regression test for issue #33, at the real scale.

        Historical stops in March, forecast starts in September. Every hour in
        between must be absent - previously they were filled with a smooth ramp
        and 106 of 383 training pairs were fitted against it.
        """
        hist = self._series("2026-03-29 00:00", 24, 8.0)
        fore = self._series("2026-09-06 00:00", 24, 23.0)

        result = combine_hourly_temps(hist, fore)

        gap_rows = result[
            (result["datetime"] > pd.Timestamp("2026-03-30 00:00"))
            & (result["datetime"] < pd.Timestamp("2026-09-06 00:00"))
        ]
        assert gap_rows.empty, (
            f"{len(gap_rows)} hours of weather were invented across the outage"
        )

    def test_surviving_rows_keep_their_real_values(self):
        """Capping must not corrupt the data either side of the gap."""
        hist = self._series("2026-05-01 00:00", 6, 10.0)
        fore = self._series("2026-05-10 00:00", 6, 20.0)

        result = combine_hourly_temps(hist, fore).set_index("datetime")

        assert result.loc[pd.Timestamp("2026-05-01 00:00"), "air_temp"] == pytest.approx(10.0)
        assert result.loc[pd.Timestamp("2026-05-10 00:00"), "air_temp"] == pytest.approx(20.0)


class TestCombineHourlyTempsTimezones:
    """
    Timezone and dtype normalisation in combine_hourly_temps.

    Disjoint from TestCombineHourlyTemps above, which covers precedence,
    empties and sort order. This class only asks whether inputs of mixed
    tz-awareness, resolution and dtype are normalised onto one naive
    datetime64[s] axis without shifting any instant.
    """

    def _hours(self, start, n, temp, tz=None):
        return pd.DataFrame({
            "datetime": pd.date_range(start, periods=n, freq="h", tz=tz),
            "air_temp": [float(temp)] * n,
        })

    def test_all_naive_datetimes(self):
        result = combine_hourly_temps(
            self._hours("2024-01-01 00:00", 24, 10.0),
            self._hours("2024-01-02 00:00", 24, 12.0),
        )

        assert len(result) == 48
        assert result["datetime"].dt.tz is None

    def test_historical_tz_aware_forecast_naive(self):
        result = combine_hourly_temps(
            self._hours("2024-01-01 00:00", 24, 10.0, tz="UTC"),
            self._hours("2024-01-02 00:00", 24, 12.0),
        )

        assert len(result) == 48
        assert result["datetime"].dt.tz is None

    def test_forecast_tz_aware_historical_naive(self):
        result = combine_hourly_temps(
            self._hours("2024-01-01 00:00", 24, 10.0),
            self._hours("2024-01-02 00:00", 24, 12.0, tz="UTC"),
        )

        assert len(result) == 48
        assert result["datetime"].dt.tz is None

    def test_gap_fill_tz_aware(self):
        """
        The production bug: a tz-aware gap-fill frame among naive ones.

        Asserting only "not empty and naive" would pass with the gap-fill
        hours shifted or missing entirely, so assert the hours themselves and
        the values they carry.
        """
        hist = self._hours("2024-01-01 00:00", 12, 10.0)                # 00:00-11:00
        gap_fill = self._hours("2024-01-01 12:00", 6, 11.0, tz="UTC")   # 12:00-17:00
        fore = self._hours("2024-01-01 18:00", 12, 12.0)                # 18:00-05:00

        result = combine_hourly_temps(
            hist, fore, gap_fill=gap_fill
        ).set_index("datetime")

        assert result.index.tz is None
        for hour in range(12, 18):
            stamp = pd.Timestamp(2024, 1, 1, hour)
            assert stamp in result.index, f"gap hour {stamp} was dropped"
            assert result.loc[stamp, "air_temp"] == pytest.approx(11.0)
        # The neighbours either side are untouched, so the block did not slide.
        assert result.loc[pd.Timestamp("2024-01-01 11:00"), "air_temp"] == pytest.approx(10.0)
        assert result.loc[pd.Timestamp("2024-01-01 18:00"), "air_temp"] == pytest.approx(12.0)

    def test_all_tz_aware_different_timezones(self):
        """
        Three zones must be CONVERTED to UTC, not stripped of their offset.

        tz_localize(None) would also produce a naive column, so a naive-ness
        assertion cannot tell the two apart. New York is the discriminator:
        18:00 EST is 23:00 UTC. Under a strip it would land at 18:00 and the
        frame would end five hours earlier.
        """
        hist = self._hours("2024-01-01 00:00", 12, 10.0, tz="UTC")
        gap_fill = self._hours("2024-01-01 12:00", 6, 11.0, tz="Europe/London")
        fore = self._hours("2024-01-01 18:00", 12, 12.0, tz="America/New_York")

        result = combine_hourly_temps(
            hist, fore, gap_fill=gap_fill
        ).set_index("datetime")

        assert result.index.tz is None
        # 18:00 New York == 23:00 UTC: the forecast block starts there.
        assert result.loc[pd.Timestamp("2024-01-01 23:00"), "air_temp"] == pytest.approx(12.0)
        # ...and runs to 05:00 New York == 10:00 UTC, the last row.
        assert result.index.max() == pd.Timestamp("2024-01-02 10:00")
        # London in January is UTC, so its block does not move.
        assert result.loc[pd.Timestamp("2024-01-01 12:00"), "air_temp"] == pytest.approx(11.0)
        # 18:00 falls inside the 5-hour hole the conversion opens, so it is
        # bridged - it must NOT be the forecast's flat 12.0.
        assert result.loc[pd.Timestamp("2024-01-01 18:00"), "air_temp"] < 12.0

    def test_empty_gap_fill_no_crash(self):
        result = combine_hourly_temps(
            self._hours("2024-01-01 00:00", 24, 10.0),
            self._hours("2024-01-02 00:00", 24, 12.0),
            gap_fill=pd.DataFrame(columns=["datetime", "air_temp"]),
        )

        assert len(result) == 48

    def test_none_gap_fill(self):
        result = combine_hourly_temps(
            self._hours("2024-01-01 00:00", 24, 10.0),
            self._hours("2024-01-02 00:00", 24, 12.0),
            gap_fill=None,
        )

        assert len(result) == 48

    def test_mixed_resolutions_normalise_to_datetime64_s(self):
        """
        datetime64[ns] and datetime64[us] inputs must land on one resolution.

        The datetime64[s] assertion is deliberate, not incidental: the
        normaliser casts explicitly so concat does not raise on mismatched
        units. Do not relax it to "is some kind of datetime".
        """
        hist = self._hours("2024-01-01 00:00", 12, 10.0)
        fore = self._hours("2024-01-01 18:00", 12, 12.0)
        fore["datetime"] = fore["datetime"].astype("datetime64[us]")

        result = combine_hourly_temps(hist, fore)

        assert result["datetime"].dtype == "datetime64[s]"
        assert len(result) == 30
        assert result["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00")

    def test_object_dtype_datetime(self):
        """
        A datetime column arriving as object dtype must still parse and align.

        Asserting only non-emptiness would pass with the forecast block
        misplaced, so pin the length and the values at both ends.
        """
        hist = self._hours("2024-01-01 00:00", 12, 10.0)          # 00:00-11:00
        fore = self._hours("2024-01-01 18:00", 12, 12.0)          # 18:00-05:00
        fore["datetime"] = fore["datetime"].astype(object)

        result = combine_hourly_temps(hist, fore)

        assert result["datetime"].dtype == "datetime64[s]"
        # 12 historical + 6 bridged (exactly at the cap) + 12 forecast.
        assert len(result) == 30
        indexed = result.set_index("datetime")
        assert indexed.loc[pd.Timestamp("2024-01-01 00:00"), "air_temp"] == pytest.approx(10.0)
        assert indexed.loc[pd.Timestamp("2024-01-01 18:00"), "air_temp"] == pytest.approx(12.0)
        # Bridged hours sit strictly between the two levels, not at either.
        bridged = indexed.loc[pd.Timestamp("2024-01-01 14:00"), "air_temp"]
        assert 10.0 < bridged < 12.0


class TestStoredSolarCloudFeedsTheModel:
    """
    Issue #29's acceptance test: a stored run can be read back and fed straight
    into build_hourly_weather in place of the actuals.

    Moved here from test_accuracy.py, which is where it was written but not
    where it belongs: build_hourly_weather lives in data.py and this class is
    its only coverage.
    """

    def _air(self, start, hours):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(hours)
            ],
            "air_temp": [15.0] * hours,
        })

    def _solar(self, start, hours, radiation, cloud):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(hours)
            ],
            "shortwave_radiation": [radiation] * hours,
            "cloud_cover": [cloud] * hours,
        })

    def test_stored_forecast_beats_actuals_on_overlap(self):
        """The replay must use what the forecast had, not what happened."""
        air = self._air("2026-05-01 07:00", 12)
        actual = self._solar("2026-05-01 07:00", 12, radiation=900.0, cloud=0.0)
        stored = self._solar("2026-05-01 07:00", 12, radiation=100.0, cloud=95.0)

        weather = build_hourly_weather(air, actual, stored)

        assert set(weather["shortwave_radiation"]) == {100.0}
        assert set(weather["cloud_cover"]) == {95.0}

    def test_actuals_fill_hours_the_stored_run_misses(self):
        air = self._air("2026-05-01 07:00", 12)
        actual = self._solar("2026-05-01 07:00", 12, radiation=900.0, cloud=0.0)
        stored = self._solar("2026-05-01 13:00", 6, radiation=100.0, cloud=95.0)

        weather = build_hourly_weather(air, actual, stored).set_index("datetime")

        assert weather.loc[pd.Timestamp("2026-05-01 08:00"), "cloud_cover"] == 0.0
        assert weather.loc[pd.Timestamp("2026-05-01 14:00"), "cloud_cover"] == 95.0

    def test_model_output_differs_with_stored_versus_actual_solar(self):
        """If swapping the source changed nothing, storing it would be pointless."""
        air = self._air("2026-05-01 07:00", 25)
        sunny = self._solar("2026-05-01 07:00", 25, radiation=900.0, cloud=0.0)
        overcast = self._solar("2026-05-01 07:00", 25, radiation=50.0, cloud=100.0)

        forecaster = WaterTempForecaster(k_air=0.01, k_solar=5e-4, k_cool=0.01)

        forecaster.set_hourly_weather(build_hourly_weather(air, sunny, None))
        with_sun = forecaster.predict_forward(
            pd.Timestamp("2026-05-01"), 12.0, days_ahead=1
        ).loc[0, "water_temp"]

        forecaster.set_hourly_weather(build_hourly_weather(air, overcast, None))
        with_cloud = forecaster.predict_forward(
            pd.Timestamp("2026-05-01"), 12.0, days_ahead=1
        ).loc[0, "water_temp"]

        assert with_sun > with_cloud

    def test_no_stored_run_falls_back_to_actuals(self):
        air = self._air("2026-05-01 07:00", 12)
        actual = self._solar("2026-05-01 07:00", 12, radiation=900.0, cloud=0.0)

        weather = build_hourly_weather(air, actual, None)

        assert set(weather["shortwave_radiation"]) == {900.0}

    def test_no_solar_at_all_degrades_to_air_conduction_only(self):
        """
        Both solar frames absent: the documented graceful degradation.

        Zero solar and 100% cloud collapse the three-term model to its air
        term, which is the whole point of the fallback - the alternative is a
        KeyError inside set_hourly_weather. This branch had no test at all.
        """
        air = self._air("2026-05-01 07:00", 12)

        weather = build_hourly_weather(air, None, None)

        assert list(weather.columns) == [
            "datetime", "air_temp", "shortwave_radiation", "cloud_cover"
        ]
        assert len(weather) == 12
        assert (weather["shortwave_radiation"] == 0.0).all()
        assert (weather["cloud_cover"] == 100.0).all()

        # The neutral values must actually neutralise the two optional terms:
        # a forecaster with large k_solar and k_cool must land where one with
        # zeroed coefficients does.
        three_term = WaterTempForecaster(k_air=0.02, k_solar=1e-3, k_cool=0.5)
        air_only = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        slice_df = weather[["air_temp", "shortwave_radiation", "cloud_cover"]]

        assert three_term._simulate_period(10.0, slice_df) == pytest.approx(
            air_only._simulate_period(10.0, slice_df)
        )

    def test_empty_solar_frames_are_treated_as_absent(self):
        """An empty frame is not a source; it must take the fallback too."""
        air = self._air("2026-05-01 07:00", 6)
        empty = pd.DataFrame(
            columns=["datetime", "shortwave_radiation", "cloud_cover"]
        )

        weather = build_hourly_weather(air, empty, empty)

        assert len(weather) == 6
        assert (weather["shortwave_radiation"] == 0.0).all()
        assert (weather["cloud_cover"] == 100.0).all()
