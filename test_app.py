"""
Tests for app.py's pure functions.

Everything here is a plain function or a chart builder: import
app headless, call the builder, and assert on fig.layout / fig.data. Nothing
in this file starts Streamlit or touches the network.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app import (
    METEOSTAT_OUTAGE_START,
    _horizon_label,
    _one_sided,
    _shade_meteostat_outage,
    create_error_over_time_chart,
    create_forecast_vs_actual_chart,
    create_horizon_accuracy_chart,
    filter_to_period,
)


def _scored(dates, forecasts, actuals, air_source="FORECAST", horizon=1):
    """A scored-forecast frame of the shape the accuracy charts consume."""
    dates = [pd.Timestamp(d) for d in dates]
    sources = (
        air_source if isinstance(air_source, list) else [air_source] * len(dates)
    )
    return pd.DataFrame({
        "target_date": dates,
        "horizon_days": [horizon] * len(dates),
        "forecast_temp": [float(f) for f in forecasts],
        "actual_temp": [float(a) for a in actuals],
        "error": [float(f) - float(a) for f, a in zip(forecasts, actuals)],
        "air_source": sources,
    })


class TestShadeMeteostatOutage:
    """
    The outage band. Its left edge is clamped to the first plotted date
    because plotly widens an axis to fit its shapes.
    """

    def _fig(self):
        return go.Figure(go.Scatter(x=[pd.Timestamp("2026-08-01")], y=[1.0]))

    def test_outage_band_left_edge_never_precedes_first_plotted_date(self):
        """
        This bug already shipped once.

        The band was anchored at METEOSTAT_OUTAGE_START (2026-03-20). Plotly
        stretches the x-axis to contain every shape, so on a last-30-days view
        the axis snapped back five months and the period filter was undone -
        the data was windowed correctly, the shape was not.

        Deleting the max() clamp in _shade_meteostat_outage must fail this.
        """
        first_date = pd.Timestamp("2026-08-01")
        fig = self._fig()

        _shade_meteostat_outage(fig, first_date, pd.Timestamp("2026-09-01"))

        assert len(fig.layout.shapes) == 1
        assert first_date > METEOSTAT_OUTAGE_START, "test setup no longer discriminates"
        assert pd.Timestamp(fig.layout.shapes[0].x0) >= first_date

    def test_band_starts_at_the_outage_when_the_chart_reaches_back_that_far(self):
        """The clamp must not move the edge forward when it need not."""
        fig = self._fig()

        _shade_meteostat_outage(
            fig, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-01")
        )

        assert pd.Timestamp(fig.layout.shapes[0].x0) == METEOSTAT_OUTAGE_START

    def test_outage_band_absent_when_chart_ends_before_the_outage(self):
        """Nothing to warn about: shading here would assert a problem that
        did not exist yet on those dates."""
        fig = self._fig()

        _shade_meteostat_outage(
            fig, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")
        )

        assert len(fig.layout.shapes) == 0
        assert len(fig.layout.annotations) == 0

    @pytest.mark.parametrize("last_date", [None, pd.NaT], ids=["none", "nat"])
    def test_outage_band_is_a_noop_on_missing_last_date(self, last_date):
        """An empty scored frame yields NaT here; it must not raise."""
        fig = self._fig()

        _shade_meteostat_outage(fig, pd.Timestamp("2026-08-01"), last_date)

        assert len(fig.layout.shapes) == 0

    def test_missing_first_date_still_shades_from_the_outage(self):
        fig = self._fig()

        _shade_meteostat_outage(fig, None, pd.Timestamp("2026-09-01"))

        assert pd.Timestamp(fig.layout.shapes[0].x0) == METEOSTAT_OUTAGE_START

    def test_outage_annotation_sits_at_top_right_so_it_is_visible_in_the_default_window(self):
        """
        Left-anchored, the label sits at 2026-03-20 - outside the default
        last-30-days view - and is invisible until you scrub back.
        """
        last_date = pd.Timestamp("2026-09-01")
        fig = self._fig()

        _shade_meteostat_outage(fig, pd.Timestamp("2026-08-01"), last_date)

        annotation = fig.layout.annotations[0]
        assert annotation.xanchor == "right"
        assert annotation.yanchor == "top"
        assert pd.Timestamp(annotation.x) == last_date
        assert "issue #33" in annotation.text


class TestFilterToPeriod:
    """Chart windowing. The window is the data, so the y-axis always fits."""

    def _frame(self, end, days):
        return pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp(end), periods=days, freq="D"),
            "value": range(days),
        })

    def test_filter_to_period_measures_back_from_the_data_not_today(self):
        """
        A source that stopped updating must show its final weeks, not an
        empty chart. Measuring back from pd.Timestamp.now() would return
        nothing here, and the bug would be invisible in any test written
        against fresh data.
        """
        df = self._frame(end="2026-01-01", days=90)

        result = filter_to_period(df, "date", days=30)

        assert not result.empty
        assert result["date"].max() == pd.Timestamp("2026-01-01")
        assert result["date"].min() == pd.Timestamp("2025-12-02")

    def test_filter_to_period_none_returns_the_whole_frame(self):
        df = self._frame(end="2026-01-01", days=90)

        result = filter_to_period(df, "date", days=None)

        assert len(result) == 90

    def test_filter_to_period_empty_frame_unchanged(self):
        empty = pd.DataFrame(columns=["date", "value"])

        result = filter_to_period(empty, "date", days=30)

        assert result.empty
        assert list(result.columns) == ["date", "value"]

    def test_window_shorter_than_the_data_actually_drops_rows(self):
        df = self._frame(end="2026-01-01", days=90)

        assert len(filter_to_period(df, "date", days=30)) < 90


class TestOneSided:
    """Whether every error in the window falls on one side of zero."""

    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param([-1.0, -0.5, -2.0], True, id="all_negative"),
            pytest.param([1.0, 0.5, 2.0], True, id="all_positive"),
            pytest.param([-1.0, 0.5], False, id="straddles_zero"),
            pytest.param([-1.0, np.nan, -0.5], True, id="ignores_nan"),
            pytest.param([np.nan, np.nan], False, id="all_nan"),
            pytest.param([], False, id="empty"),
        ],
    )
    def test_one_sided(self, values, expected):
        assert _one_sided(values) is expected


class TestErrorOverTimeChart:
    """Bars grow from zero, so zero has to stay in frame."""

    def test_error_chart_pins_zero_in_frame_when_every_error_is_one_sided(self):
        """
        Without rangemode="tozero" plotly frames the bars themselves, and a
        run of all-negative errors renders as bars hanging from an axis that
        is not zero - which reads as far smaller error than it is.
        """
        scored = _scored(
            ["2026-08-01", "2026-08-02"], [9.0, 9.5], [10.0, 10.0]
        )

        fig = create_error_over_time_chart(scored)

        assert (scored["error"] < 0).all(), "test setup no longer discriminates"
        assert fig.layout.yaxis.rangemode == "tozero"

    def test_error_chart_uses_normal_range_when_errors_straddle_zero(self):
        scored = _scored(
            ["2026-08-01", "2026-08-02"], [11.0, 9.0], [10.0, 10.0]
        )

        fig = create_error_over_time_chart(scored)

        assert fig.layout.yaxis.rangemode == "normal"

    def test_error_chart_draws_bars_not_a_connecting_line(self):
        """Measurements skip days; a line asserts a trend nobody observed."""
        scored = _scored(["2026-08-01", "2026-08-09"], [11.0, 9.0], [10.0, 10.0])

        fig = create_error_over_time_chart(scored)

        assert fig.data[0].type == "bar"


class TestHorizonLabel:

    @pytest.mark.parametrize(
        "horizon, expected",
        [
            pytest.param(0, "0 days ahead", id="zero_is_plural"),
            pytest.param(1, "1 day ahead", id="one_is_singular"),
            pytest.param(2, "2 days ahead", id="two_is_plural"),
        ],
    )
    def test_horizon_label_is_singular_only_at_one_day(self, horizon, expected):
        assert _horizon_label(horizon) == expected


class TestHorizonAccuracyChart:

    def _metrics(self, horizons, maes, ns):
        return pd.DataFrame({"horizon_days": horizons, "mae": maes, "n": ns})

    def test_horizon_chart_labels_a_zero_n_horizon_blank_not_nan(self):
        """
        A horizon nothing was scored at has NaN MAE. f"{nan:.2f}" renders the
        string "nan" on the bar, which users read as a value.
        """
        metrics = self._metrics([0, 1], [np.nan, 0.5], [0, 3])

        fig = create_horizon_accuracy_chart(metrics, selected_horizon=1)

        assert fig.data[0].text[0] == ""
        assert fig.data[0].text[1] == "0.50"

    def test_horizon_chart_highlights_exactly_the_selected_bar(self):
        metrics = self._metrics([1, 2, 3], [0.5, 0.7, 0.9], [3, 3, 3])

        fig = create_horizon_accuracy_chart(metrics, selected_horizon=2)

        colors = list(fig.data[0].marker.color)
        assert len(set(colors)) == 2
        assert colors[1] != colors[0]
        assert colors[0] == colors[2]

    def test_horizon_chart_carries_the_sample_size_into_the_hover(self):
        """MAE without n invites reading a one-sample horizon as a result."""
        metrics = self._metrics([1, 2], [0.5, 0.7], [3, 17])

        fig = create_horizon_accuracy_chart(metrics, selected_horizon=1)

        assert list(fig.data[0].customdata) == [3, 17]


class TestForecastVsActualChart:

    def _water_temps(self, start, days):
        return pd.DataFrame({
            "date": pd.date_range(start, periods=days, freq="D"),
            "water_temp": [10.0 + i * 0.1 for i in range(days)],
        })

    def test_forecast_vs_actual_measured_line_spans_the_full_record_when_water_temps_given(self):
        """
        Stored forecasts start 2026-02-16. Without the full record the
        measured line is a narrow window with no context either side of the
        outage, which is the whole point of the chart.
        """
        scored = _scored(["2026-08-01", "2026-08-02"], [10.0, 10.5], [10.2, 10.4])
        water_temps = self._water_temps("2026-01-01", 300)

        fig = create_forecast_vs_actual_chart(
            scored, horizon=1, water_temps=water_temps
        )

        measured = fig.data[0]
        assert measured.name == "Measured"
        assert len(measured.x) == 300
        assert pd.Timestamp(measured.x[0]) == pd.Timestamp("2026-01-01")

    def test_measured_line_falls_back_to_the_scored_dates_without_water_temps(self):
        scored = _scored(["2026-08-01", "2026-08-02"], [10.0, 10.5], [10.2, 10.4])

        fig = create_forecast_vs_actual_chart(scored, horizon=1, water_temps=None)

        assert len(fig.data[0].x) == 2

    def test_forecast_vs_actual_marks_actual_air_rows_as_optimistic(self):
        """
        A replay row built from actual air temperature is leakage: it knows
        the weather. Those points get cross-marks so the chart cannot be read
        as honest forecast performance.
        """
        scored = _scored(
            ["2026-08-01", "2026-08-02", "2026-08-03"],
            [10.0, 10.5, 11.0],
            [10.2, 10.4, 11.1],
            air_source=["FORECAST", "ACTUAL", "ACTUAL"],
        )

        fig = create_forecast_vs_actual_chart(scored, horizon=1)

        leak = [t for t in fig.data if "optimistic" in (t.name or "")]
        assert len(leak) == 1
        assert leak[0].mode == "markers"
        assert leak[0].marker.symbol == "x"
        assert [pd.Timestamp(d) for d in leak[0].x] == [
            pd.Timestamp("2026-08-02"), pd.Timestamp("2026-08-03")
        ]

    def test_no_leakage_marks_when_every_row_used_a_real_forecast(self):
        scored = _scored(["2026-08-01"], [10.0], [10.2], air_source="FORECAST")

        fig = create_forecast_vs_actual_chart(scored, horizon=1)

        assert not [t for t in fig.data if "optimistic" in (t.name or "")]

    def test_outage_shading_is_opt_in(self):
        """The replay reads the repaired archive, so shading its errors would
        blame the outage for something it did not cause."""
        scored = _scored(["2026-08-01", "2026-09-01"], [10.0, 11.0], [10.2, 11.1])

        unshaded = create_forecast_vs_actual_chart(scored, horizon=1)
        shaded = create_forecast_vs_actual_chart(scored, horizon=1, mark_outage=True)

        assert len(unshaded.layout.shapes) == 0
        assert len(shaded.layout.shapes) == 1
