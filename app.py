"""West Reservoir Temperature Tracker - Streamlit App"""

import warnings
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Suppress known pandas/plotly compatibility warning (harmless)
warnings.filterwarnings("ignore", message=".*DatetimeProperties.to_pydatetime.*")

from data import (
    load_water_temps,
    load_historical_air_temps,
    load_hourly_air_temps,
    load_forecast_air_temps,
    load_forecast_air_temps_3hourly,
    load_historical_solar_cloud,
    load_forecast_solar_cloud,
    interpolate_to_hourly,
    select_storable_predictions,
    build_hourly_weather,
    combine_hourly_temps,
    build_temperatures_frame,
    deduplicate_temperatures,
    DataLoadError,
)
from forecaster import WaterTempForecaster
from config import ENABLE_MOTHERDUCK
from quotes import QUOTES
from accuracy import (
    BIAS_NOTE,
    compute_metrics,
    join_actuals,
    metrics_by_horizon,
    replay_current_model,
    splice_air_history,
)

if ENABLE_MOTHERDUCK:
    from forecast_storage import ForecastStorage, ForecastStorageError

CACHE_TTL = timedelta(hours=6)


@st.cache_data(ttl=CACHE_TTL)
def cached_load_water_temps():
    """Load water temps with 6-hour cache."""
    return load_water_temps()


@st.cache_data(ttl=CACHE_TTL)
def cached_load_historical_air_temps(start_date, end_date):
    """Load historical air temps with 6-hour cache."""
    return load_historical_air_temps(start_date, end_date)


@st.cache_data(ttl=CACHE_TTL)
def cached_load_hourly_air_temps(start_date, end_date):
    """Load hourly air temps with 6-hour cache."""
    return load_hourly_air_temps(start_date, end_date)


@st.cache_data(ttl=CACHE_TTL)
def cached_load_forecast_air_temps(days):
    """Load forecast air temps with 6-hour cache."""
    return load_forecast_air_temps(days=days)


@st.cache_data(ttl=CACHE_TTL)
def cached_load_forecast_air_temps_3hourly(days):
    """Load 3-hourly forecast air temps with 6-hour cache."""
    return load_forecast_air_temps_3hourly(days=days)


@st.cache_data(ttl=CACHE_TTL)
def cached_load_historical_solar_cloud(start_date, end_date):
    """Load historical solar/cloud from Open-Meteo with 6-hour cache."""
    return load_historical_solar_cloud(start_date, end_date)


@st.cache_data(ttl=CACHE_TTL)
def cached_load_forecast_solar_cloud(days):
    """Load forecast solar/cloud from Open-Meteo with 6-hour cache."""
    return load_forecast_solar_cloud(days=days)


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



def retrieve_gap_fill_forecasts(
    hist_end: datetime,
    fore_start: datetime
) -> pd.DataFrame:
    """
    Retrieve stored 3-hourly forecasts from MotherDuck to fill the gap
    between the Open-Meteo archive and the live OWM forecast.

    Args:
        hist_end: Last timestamp from the Open-Meteo hourly archive
        fore_start: First timestamp from OWM 3-hourly forecast

    Returns:
        DataFrame with 'datetime' and 'air_temp' columns (interpolated to hourly),
        or empty DataFrame if no data available
    """
    if not ENABLE_MOTHERDUCK:
        return pd.DataFrame(columns=["datetime", "air_temp"])

    try:
        storage = ForecastStorage()
        gap_data = storage.get_forecasts_for_gap(hist_end, fore_start)

        if gap_data is None or gap_data.empty:
            return pd.DataFrame(columns=["datetime", "air_temp"])

        gap_hourly = interpolate_to_hourly(gap_data)
        return gap_hourly

    except ForecastStorageError:
        return pd.DataFrame(columns=["datetime", "air_temp"])
    except Exception:
        return pd.DataFrame(columns=["datetime", "air_temp"])



st.set_page_config(
    page_title="West Reservoir Water Temperature Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def display_debug_panel(
    temperatures: pd.DataFrame,
    forecaster: WaterTempForecaster,
    hourly_air_temps: pd.DataFrame,
    forecast_3hourly: pd.DataFrame = None,
    gap_fill_hourly: pd.DataFrame = None,
):
    """Display comprehensive debug information."""
    with st.expander("Details for nerds", expanded=False):
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            measured = len(temperatures[temperatures["source"] == "MEASURED"])
            st.metric("Measured Readings", measured)
        with col2:
            predicted = len(temperatures[temperatures["source"] == "PREDICTED"])
            st.metric("Predicted Values", predicted)
        with col3:
            st.metric("Hourly Air Temps", len(hourly_air_temps))

        st.subheader("Model Parameters")
        st.write(f"**k_air (conduction)**: {forecaster.k_air:.4f} per hour")
        st.write(f"**k_solar (shortwave heating)**: {forecaster.k_solar:.6f} °C per (W/m²) per hour")
        st.write(f"**k_cool (clear-sky cooling)**: {forecaster.k_cool:.4f} °C per hour at fully clear sky")
        daily_response = 1 - (1 - forecaster.k_air) ** 24
        st.write(f"**Daily air response**: {daily_response:.1%} of air/water temperature difference")
        st.write(
            "**Physics**: ΔT = k_air·(T_air − T_water) + k_solar·I − k_cool·(1 − cloud/100)"
        )

        has_predictions = any(temperatures["source"] == "PREDICTED")
        if has_predictions:
            st.subheader("Tomorrow's Prediction (24h Simulation)")

            measured_data = temperatures[temperatures["source"] == "MEASURED"]

            if not measured_data.empty:
                latest = measured_data.iloc[-1]
                latest_date = pd.Timestamp(latest["date"])

                start_dt = latest_date.replace(hour=forecaster.MEASUREMENT_HOUR)
                end_dt = start_dt + timedelta(hours=24)

                weather_slice = forecaster._get_weather_for_period(start_dt, end_dt)

                if not weather_slice.empty:
                    explanation = forecaster.explain_prediction(
                        current_water_temp=latest["water_temp"],
                        weather_slice=weather_slice,
                    )

                    st.code(
                        f"""
Current water temp (7am):  {explanation['current_water_temp']:.2f} C
Hours simulated:           {explanation['hours_simulated']}
Air temp range:            {explanation['air_temp_min']:.1f} C to {explanation['air_temp_max']:.1f} C
Air temp average:          {explanation['air_temp_avg']:.1f} C
Solar avg / peak:          {explanation['solar_avg']:.0f} / {explanation['solar_max']:.0f} W/m²
Cloud cover average:       {explanation['cloud_avg']:.0f}%
k_air:                     {explanation['k_air']:.4f} per hour
k_solar:                   {explanation['k_solar']:.6f} °C per W/m² per hour
k_cool:                    {explanation['k_cool']:.4f} °C per hour
Total temp change:         {explanation['total_temp_change']:.2f} C
--------------------------------------------
Tomorrow's predicted temp: {explanation['predicted_water_temp']:.2f} C
                        """
                    )

                    with st.expander("Hourly Simulation Detail"):
                        breakdown_df = pd.DataFrame(explanation["hourly_breakdown"])
                        breakdown_df["hour"] = breakdown_df["hour"].apply(
                            lambda h: f"{h:02d}:00"
                        )
                        breakdown_df = breakdown_df.rename(
                            columns={
                                "hour": "Time",
                                "air_temp": "Air (C)",
                                "shortwave_radiation": "Solar (W/m²)",
                                "cloud_cover": "Cloud (%)",
                                "water_temp_before": "Water Before (C)",
                                "dT_air": "ΔT Air",
                                "dT_solar": "ΔT Solar",
                                "dT_cool": "ΔT Cool",
                                "temp_change": "Change (C)",
                                "water_temp_after": "Water After (C)",
                            }
                        )
                        st.dataframe(
                            breakdown_df.style.format(
                                {
                                    "Air (C)": "{:.1f}",
                                    "Solar (W/m²)": "{:.0f}",
                                    "Cloud (%)": "{:.0f}",
                                    "Water Before (C)": "{:.2f}",
                                    "ΔT Air": "{:.3f}",
                                    "ΔT Solar": "{:.3f}",
                                    "ΔT Cool": "{:.3f}",
                                    "Change (C)": "{:.3f}",
                                    "Water After (C)": "{:.2f}",
                                }
                            ),
                            width='stretch',
                        )
                else:
                    st.info("Waiting for hourly air temperature data")
            else:
                st.info("No measured data available")

        st.subheader("Air Temperature: Last 48h + Next 48h")
        if not hourly_air_temps.empty:
            now = datetime.now()
            cutoff_past = now - timedelta(hours=48)
            cutoff_future = now + timedelta(hours=48)

            past_hourly = hourly_air_temps[hourly_air_temps["datetime"] >= cutoff_past]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=past_hourly["datetime"],
                    y=past_hourly["air_temp"],
                    mode="lines",
                    name="Historical (hourly archive)",
                    line=dict(color="red", width=1),
                )
            )

            if gap_fill_hourly is not None and not gap_fill_hourly.empty:
                gap_window = gap_fill_hourly[
                    (gap_fill_hourly["datetime"] >= cutoff_past) &
                    (gap_fill_hourly["datetime"] <= cutoff_future)
                ]
                if not gap_window.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=gap_window["datetime"],
                            y=gap_window["air_temp"],
                            mode="lines+markers",
                            name="Gap fill (stored forecast)",
                            line=dict(color="blue", width=1, dash="dot"),
                            marker=dict(color="blue", size=6),
                        )
                    )

            if forecast_3hourly is not None and not forecast_3hourly.empty:
                forecast_window = forecast_3hourly[
                    forecast_3hourly["datetime"] <= cutoff_future
                ]

                if not forecast_window.empty:
                    forecast_interpolated = interpolate_to_hourly(forecast_window)

                    fig.add_trace(
                        go.Scatter(
                            x=forecast_interpolated["datetime"],
                            y=forecast_interpolated["air_temp"],
                            mode="lines",
                            name="Forecast (OWM interpolated)",
                            line=dict(color="orange", width=1, dash="dash"),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=forecast_window["datetime"],
                            y=forecast_window["air_temp"],
                            mode="markers",
                            name="Forecast (OWM 3-hourly)",
                            marker=dict(color="orange", size=8),
                        )
                    )

            fig.add_vline(
                x=now.timestamp() * 1000,
                line=dict(color="gray", width=1, dash="dot"),
                annotation_text="Now",
            )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Temperature (C)",
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, width='stretch')
            st.caption("Chart shows raw data points. Non-hourly data (e.g. 3-hourly forecasts) is resampled to hourly before feeding into the prediction model.")

        st.subheader("Raw Data (Last 10 Rows)")
        display_df = temperatures.tail(10)[
            ["date", "water_temp", "air_temp", "source"]
        ].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, width='stretch')

        st.subheader("Forecast Storage (MotherDuck)")

        if ENABLE_MOTHERDUCK:
            try:
                storage = ForecastStorage()
                conn = storage._get_connection()

                result = conn.execute("""
                    SELECT
                        MAX(forecast_created_timestamp) as last_stored,
                        COUNT(DISTINCT DATE_TRUNC('hour', forecast_created_timestamp)) as forecast_runs,
                        COUNT(*) as total_forecasts
                    FROM air_temp_forecasts_3hourly
                """).fetchone()

                if result and result[0]:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Last Stored", result[0].strftime("%Y-%m-%d %H:%M"))
                    with col2:
                        st.metric("Forecast Runs", result[1])
                    with col3:
                        st.metric("Total Forecasts", result[2])
                else:
                    st.info("No forecasts stored yet. Will store on next forecast fetch.")

            except ForecastStorageError as e:
                st.warning(f"Storage not configured: {e}")
                st.info("Set MOTHERDUCK_TOKEN to enable forecast storage")
            except Exception as e:
                st.warning(f"Could not retrieve storage status: {e}")
        else:
            st.info("MotherDuck storage is currently disabled. Set ENABLE_MOTHERDUCK = True in config.py to enable.")


def _ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _fmt_hover_dates(dates) -> list:
    result = []
    for d in pd.to_datetime(dates):
        result.append(d.strftime(f"%a {d.day}{_ordinal(d.day)} %b"))
    return result


def create_temperature_chart(temperatures: pd.DataFrame) -> go.Figure:
    """Create temperature chart with last 5 days + forecast."""
    fig = go.Figure()

    today = datetime.now().date()
    cutoff_date = today - timedelta(days=5)

    filtered = temperatures[
        (temperatures["date"].dt.date >= cutoff_date)
    ].copy()

    measured = filtered[filtered["source"] == "MEASURED"]
    predicted = filtered[filtered["source"] == "PREDICTED"]

    # Add vertical line for today (FIRST so it renders behind data)
    fig.add_shape(
        type="line",
        x0=str(today),
        x1=str(today),
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=15),
        opacity = 0.3,
        layer="below"  # Explicitly render below traces
    )
    fig.add_annotation(
        x=str(today),
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        yshift=10,
    )

    # For past dates prefer the archive (historical), for today prefer forecast (the archive is partial day only)
    air_data = filtered[filtered["air_temp"].notna()].copy()
    past_air = air_data[air_data["date"].dt.date < today].drop_duplicates(subset=["date"], keep="first")
    today_air = air_data[air_data["date"].dt.date == today].drop_duplicates(subset=["date"], keep="last")
    future_air = air_data[air_data["date"].dt.date > today].drop_duplicates(subset=["date"], keep="first")
    all_with_air = pd.concat([past_air, today_air, future_air]).sort_values("date").reset_index(drop=True)
    if not all_with_air.empty:
        has_minmax = all_with_air["air_temp_min"].notna().any()
        if has_minmax:
            all_with_air["error_plus"] = all_with_air["air_temp_max"] - all_with_air["air_temp"]
            all_with_air["error_minus"] = all_with_air["air_temp"] - all_with_air["air_temp_min"]

            fig.add_trace(
                go.Scatter(
                    x=all_with_air["date"],
                    y=all_with_air["air_temp"],
                    mode="markers",
                    name="Air temperature range",
                    marker=dict(size=0),  # Hide markers, only show error bars
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=all_with_air["error_plus"],
                        arrayminus=all_with_air["error_minus"],
                        color="rgba(255, 100, 100, 0.4)",
                        thickness=7,
                        width=0,
                    ),
                    legendgroup="air",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=all_with_air["date"],
                    y=all_with_air["air_temp"],
                    mode="markers",
                    name="Air temperature (avg)",
                    marker=dict(color="black", size=6, symbol="line-ew", line=dict(width=2)),
                    legendgroup="air",
                    showlegend=False,
                    customdata=list(zip(all_with_air["air_temp_min"], all_with_air["air_temp_max"], _fmt_hover_dates(all_with_air["date"]))),
                    hovertemplate="Air: %{y:.1f}C (Low: %{customdata[0]:.1f}, High: %{customdata[1]:.1f})<br>%{customdata[2]}<extra></extra>",
                )
            )

    if not measured.empty:
        fig.add_trace(
            go.Scatter(
                x=measured["date"],
                y=measured["water_temp"],
                mode="lines+markers+text",
                name="Water Temp",
                line=dict(color="#095988", width=2),
                marker=dict(size=6),
                text=[f"{v:.1f}" for v in measured["water_temp"]],
                textposition="bottom center",
                textfont=dict(size=14, color="#095988"),
                customdata=_fmt_hover_dates(measured["date"]),
                legendgroup="water",
                hovertemplate="Water: %{y:.1f}C<br>%{customdata}<extra></extra>",
            )
        )

    # Split predicted into gap-fills (past) and future forecasts
    if not predicted.empty and not measured.empty:
        last_measured_date = measured["date"].max()

        past_gaps = predicted[predicted["date"] < last_measured_date].sort_values("date")
        future_forecast = predicted[predicted["date"] >= last_measured_date].sort_values("date")

        if not past_gaps.empty:
            fig.add_trace(
                go.Scatter(
                    x=past_gaps["date"],
                    y=past_gaps["water_temp"],
                    mode="markers+text",
                    name="Water Temp (Gap-fill)",
                    marker=dict(size=6, color="#095988", symbol="circle-open"),
                    text=[f"{v:.1f}" for v in past_gaps["water_temp"]],
                    textposition="bottom center",
                    textfont=dict(size=14, color="#095988"),
                    customdata=_fmt_hover_dates(past_gaps["date"]),
                    legendgroup="water",
                    showlegend=False,
                    hovertemplate="Water (gap-fill): %{y:.1f}C<br>%{customdata}<extra></extra>",
                )
            )

        if not future_forecast.empty:
            last_measured = measured[measured["date"] == last_measured_date].iloc[[-1]]
            predicted_with_connection = pd.concat([last_measured, future_forecast]).sort_values("date")

            fig.add_trace(
                go.Scatter(
                    x=predicted_with_connection["date"],
                    y=predicted_with_connection["water_temp"],
                    mode="lines+markers+text",
                    name="Water Temp (Forecast)",
                    line=dict(color="#095988", width=2, dash="dash"),
                    marker=dict(size=6),
                    text=[f"{v:.1f}" for v in predicted_with_connection["water_temp"]],
                    textposition="bottom center",
                    textfont=dict(size=14, color="#095988"),
                    customdata=_fmt_hover_dates(predicted_with_connection["date"]),
                    legendgroup="water",
                    showlegend=False,
                    hovertemplate="Water (forecast): %{y:.1f}C<br>%{customdata}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Water Temperature (Last 5 Days + Forecast)",
        xaxis_title="Date",
        yaxis_title="Temperature (C)",
        hovermode="closest",
        height=500,
    )

    return fig


def _horizon_label(horizon: int) -> str:
    """'1 day ahead', '2 days ahead', '0 days ahead' (same-day nowcast)."""
    return f"{horizon} day{'' if horizon == 1 else 's'} ahead"


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
        name=f"Forecast ({_horizon_label(horizon)})", mode="lines+markers",
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


def main():
    """Main application."""
    view = st.query_params.get("view")

    if view == "forecast_graph":
        try:
            water_temps = cached_load_water_temps()
            start_date = pd.Timestamp(water_temps["date"].min()).normalize()
            end_date = pd.Timestamp.now().normalize()
            air_temps_hist = cached_load_historical_air_temps(start_date, end_date)
            hourly_air_temps = cached_load_hourly_air_temps(start_date, end_date)

            temperatures = build_temperatures_frame(water_temps, air_temps_hist)

            combined_hourly = hourly_air_temps
            try:
                forecast_3hourly = cached_load_forecast_air_temps_3hourly(days=5)
                forecast_hourly = interpolate_to_hourly(forecast_3hourly)

                gap_fill_hourly = None
                if ENABLE_MOTHERDUCK and not hourly_air_temps.empty and not forecast_3hourly.empty:
                    hist_end = hourly_air_temps["datetime"].max()
                    fore_start = forecast_3hourly["datetime"].min()
                    gap_hours = (fore_start - hist_end).total_seconds() / 3600
                    if gap_hours > 1:
                        gap_fill_hourly = retrieve_gap_fill_forecasts(hist_end, fore_start)

                combined_hourly = combine_hourly_temps(
                    hourly_air_temps, forecast_hourly, gap_fill_hourly
                )

                forecast = cached_load_forecast_air_temps(days=5)
                forecast["source"] = "AIR_ONLY"
                temperatures = pd.concat([temperatures, forecast], ignore_index=True)
                # Create deduplicated version for prediction chain
                temperatures_deduped = deduplicate_temperatures(temperatures)
            except DataLoadError:
                temperatures_deduped = temperatures.copy()

            solar_hist = None
            solar_fore = None
            try:
                solar_hist = cached_load_historical_solar_cloud(start_date, end_date)
            except DataLoadError:
                pass
            try:
                solar_fore = cached_load_forecast_solar_cloud(days=5)
            except DataLoadError:
                pass

            hourly_weather = build_hourly_weather(combined_hourly, solar_hist, solar_fore)

            forecaster = WaterTempForecaster()
            forecaster.set_hourly_weather(hourly_weather)
            forecaster.fit(temperatures_deduped[temperatures_deduped["source"] == "MEASURED"])
            temperatures_deduped = forecaster.fill_predictions(temperatures_deduped)

            chart = create_temperature_chart(temperatures_deduped)
            st.plotly_chart(chart, width='stretch')
            st.stop()

        except DataLoadError as e:
            st.error(f"Cannot load required data: {e}")
            st.stop()

    st.title("West Reservoir Temperature Tracker + Forecaster")
    st.markdown("Tracking and forecasting water temperature at West Reservoir, London.")

    tab_temp, tab_accuracy, tab_quotes = st.tabs(
        ["Temperature", "Forecast Accuracy", "Heard at the Res"]
    )

    with tab_quotes:
        st.header("Heard at the Res")
        st.markdown("Funny snippets overheard at West Reservoir. Got one? Let me know!")
        if QUOTES:
            import random
            shuffled = random.sample(QUOTES, len(QUOTES))
            for q in shuffled:
                st.markdown(f"> *\"{q['quote']}\"*")
                caption_parts = [p for p in [q.get("context"), str(q["year"]) if q.get("year") else None] if p]
                if caption_parts:
                    st.caption(" | ".join(caption_parts))
                st.divider()
        else:
            st.info("No quotes yet - check back soon!")

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
                            f"Forecast vs measured ({_horizon_label(selected_horizon)})"
                        )
                        st.plotly_chart(
                            create_forecast_vs_actual_chart(
                                at_horizon, selected_horizon
                            ),
                            width='stretch',
                        )

                        st.subheader(
                            f"Error over time ({_horizon_label(selected_horizon)})"
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

    with tab_temp:
        col_info, col_image = st.columns([1, 1])
        with col_info:
            st.info(
                "Water temperatures are taken each morning around 7am. "
                "The water will often be warmer by the time you get in!\n\n "
                "The forecast simulates hourly heat transfer using air temperature, "
                "shortwave solar radiation, and cloud cover (clear-sky overnight "
                "cooling). It does not yet account for wind or evaporation.\n\n"
                "Additionally, temperature varies throughout the reservoir "
                "by both position and depth - this is just a snapshot of conditions."
            )
        with col_image:
            st.image("image.png",)

        try:
            water_temps = cached_load_water_temps()

            # Normalize dates to day-level for consistent caching
            start_date = pd.Timestamp(water_temps["date"].min()).normalize()
            end_date = pd.Timestamp.now().normalize()
            air_temps_hist = cached_load_historical_air_temps(start_date, end_date)

            hourly_air_temps = cached_load_hourly_air_temps(start_date, end_date)

            temperatures = build_temperatures_frame(water_temps, air_temps_hist)

            forecast_3hourly = None
            gap_fill_hourly = None
            try:
                forecast_3hourly = cached_load_forecast_air_temps_3hourly(days=5)

                # Store 3-hourly in MotherDuck (only once per day)
                if ENABLE_MOTHERDUCK:
                    if 'last_forecast_fetch_date' not in st.session_state or \
                       st.session_state['last_forecast_fetch_date'] != datetime.now().date():
                        try:
                            storage = ForecastStorage()
                            storage.initialize_schema()
                            forecast_timestamp = datetime.now()
                            storage.store_air_forecast_3hourly(forecast_3hourly, forecast_timestamp)
                            st.session_state['last_forecast_fetch_date'] = datetime.now().date()
                            st.session_state['last_forecast_timestamp'] = forecast_timestamp
                        except ForecastStorageError as e:
                            st.warning(f"Could not store forecast: {e}")
                        except Exception as e:
                            st.warning(f"Forecast storage error: {e}")

                forecast_hourly = interpolate_to_hourly(forecast_3hourly)

                # Gap is between: last archive timestamp -> first OWM timestamp
                if ENABLE_MOTHERDUCK and not hourly_air_temps.empty and not forecast_3hourly.empty:
                    hist_end = hourly_air_temps["datetime"].max()
                    fore_start = forecast_3hourly["datetime"].min()

                    gap_hours = (fore_start - hist_end).total_seconds() / 3600
                    if gap_hours > 1:
                        gap_fill_hourly = retrieve_gap_fill_forecasts(hist_end, fore_start)

                combined_hourly = combine_hourly_temps(
                    hourly_air_temps, forecast_hourly, gap_fill_hourly
                )

                # Also load daily forecast for the temperatures DataFrame (for chart display)
                forecast = cached_load_forecast_air_temps(days=5)
                forecast["source"] = "AIR_ONLY"
                temperatures = pd.concat([temperatures, forecast], ignore_index=True)
                temperatures = temperatures.sort_values("date").reset_index(drop=True)

                # Deduplicated version for the prediction chain (MEASURED > AIR_ONLY)
                temperatures_deduped = deduplicate_temperatures(temperatures)

            except DataLoadError as e:
                st.warning(f"Weather forecast unavailable: {e}")
                st.info("Showing historical data only")
                combined_hourly = hourly_air_temps
                temperatures_deduped = temperatures.copy()  # No duplicates without forecast

            solar_hist = None
            solar_fore = None
            try:
                solar_hist = cached_load_historical_solar_cloud(start_date, end_date)
            except DataLoadError as e:
                st.warning(f"Open-Meteo historical solar/cloud unavailable: {e}")
            try:
                solar_fore = cached_load_forecast_solar_cloud(days=5)
            except DataLoadError as e:
                st.warning(f"Open-Meteo forecast solar/cloud unavailable: {e}")

            hourly_weather = build_hourly_weather(combined_hourly, solar_hist, solar_fore)

            forecaster = WaterTempForecaster()
            forecaster.set_hourly_weather(hourly_weather)
            forecaster.fit(temperatures_deduped[temperatures_deduped["source"] == "MEASURED"])

            temperatures_deduped = forecaster.fill_predictions(temperatures_deduped)

            # Store water predictions in MotherDuck (only once per day)
            if ENABLE_MOTHERDUCK:
                if 'last_prediction_store_date' not in st.session_state or \
                   st.session_state['last_prediction_store_date'] != datetime.now().date():
                    try:
                        storage = ForecastStorage()
                        # Only forward-looking rows are forecasts. Backfilled
                        # gap-fills target dates before the run and would land
                        # in storage at negative horizons.
                        predictions_df = select_storable_predictions(
                            temperatures_deduped, pd.Timestamp.now()
                        )
                        measured_temps = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]

                        if not predictions_df.empty and not measured_temps.empty:
                            forecast_timestamp = st.session_state.get(
                                'last_forecast_timestamp',
                                datetime.now()
                            )
                            storage.store_water_predictions(
                                predictions_df=predictions_df,
                                forecast_created_timestamp=forecast_timestamp,
                                heat_transfer_coeff=forecaster.k_air,
                                start_water_temp=measured_temps.iloc[-1]["water_temp"]
                            )
                            st.session_state['last_prediction_store_date'] = datetime.now().date()
                    except ForecastStorageError as e:
                        st.warning(f"Could not store predictions: {e}")
                    except Exception as e:
                        st.warning(f"Prediction storage error: {e}")

            with col_info:
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                tomorrow = today + timedelta(days=1)

                st.header("Current Temperature")

                measured_data = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]
                today_data = temperatures_deduped[temperatures_deduped["date"].dt.date == today]
                has_today_measurement = any(today_data["source"] == "MEASURED")

                if has_today_measurement:
                    today_measured = today_data[today_data["source"] == "MEASURED"].iloc[0]
                    st.metric("Today's Measured", f"{today_measured['water_temp']:.1f}C")
                else:
                    if not measured_data.empty:
                        latest = measured_data.iloc[-1]
                        latest_date = latest["date"].strftime("%Y-%m-%d")
                        st.warning(
                            f"No measurement for today yet. Last measured: {latest_date}\n\n"
                            f"Please contribute the temperature to the spreadsheet [here](https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/edit?usp=sharing)"
                        )

                # Compute forecasts independently (always from yesterday's measurement)
                st.subheader("Forecasts")

                yesterday_data = measured_data[measured_data["date"].dt.date == yesterday]
                today_forecast_temp = None
                tomorrow_forecast_temp = None

                if not yesterday_data.empty:
                    yesterday_temp = yesterday_data.iloc[-1]["water_temp"]
                    yesterday_dt = pd.Timestamp(yesterday).replace(hour=forecaster.MEASUREMENT_HOUR)
                    today_dt = pd.Timestamp(today).replace(hour=forecaster.MEASUREMENT_HOUR)
                    weather_slice = forecaster._get_weather_for_period(yesterday_dt, today_dt)
                    if not weather_slice.empty:
                        today_forecast_temp = forecaster._simulate_period(
                            yesterday_temp, weather_slice
                        )

                tomorrow_data = temperatures_deduped[temperatures_deduped["date"].dt.date == tomorrow]
                if not tomorrow_data.empty and tomorrow_data.iloc[0]["source"] == "PREDICTED":
                    tomorrow_forecast_temp = tomorrow_data.iloc[0]["water_temp"]

                col_today_fc, col_tomorrow_fc, col_hottest, col_coldest = st.columns(4)
                with col_today_fc:
                    if today_forecast_temp is not None and pd.notna(today_forecast_temp):
                        st.metric("Today's Forecast (excludes today's measurement)", f"{today_forecast_temp:.1f}C")
                    else:
                        st.metric("Today's Forecast (excludes today's measurement)", "N/A")
                with col_tomorrow_fc:
                    if tomorrow_forecast_temp is not None and pd.notna(tomorrow_forecast_temp):
                        st.metric("Tomorrow's Forecast", f"{tomorrow_forecast_temp:.1f}C")
                    else:
                        st.metric("Tomorrow's Forecast", "N/A")
                with col_hottest:
                    week_ahead = today + timedelta(days=7)
                    upcoming = temperatures_deduped[
                        (temperatures_deduped["date"].dt.date >= today) &
                        (temperatures_deduped["date"].dt.date <= week_ahead) &
                        (temperatures_deduped["source"].isin(["PREDICTED", "MEASURED"]))
                    ]
                    if not upcoming.empty:
                        hottest_temp = upcoming["water_temp"].max()
                        hottest_date = upcoming.loc[upcoming["water_temp"].idxmax(), "date"].strftime("%a %d %b")
                        st.metric("Hottest This Week", f"{hottest_temp:.1f}C", delta=hottest_date, delta_color="off")
                    else:
                        st.metric("Hottest This Week", "N/A")
                with col_coldest:
                    if not upcoming.empty:
                        coldest_temp = upcoming["water_temp"].min()
                        coldest_date = upcoming.loc[upcoming["water_temp"].idxmin(), "date"].strftime("%a %d %b")
                        st.metric("Coldest This Week", f"{coldest_temp:.1f}C", delta=coldest_date, delta_color="off")
                    else:
                        st.metric("Coldest This Week", "N/A")

                if st.button("Data looks old? Press to refresh weather forecast and water temperature data", icon = '🔄' ):
                    st.cache_data.clear()
                    st.rerun()

            st.header("Temperature History and Forecast")
            st.text("""The chart shows the temperature history and forecast for the last 5 days, and next 5 days.
            Red bar shows the air temp range each day, with the black line being the average. The blue line is the water tempterature. It is dotted for forecast days.""")

            chart = create_temperature_chart(temperatures_deduped)
            st.plotly_chart(chart, width='stretch')

            st.header("Summary Statistics")
            measured = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lowest Recorded at West Reservoir! ❄️", f"{measured['water_temp'].min():.1f}C")
            with col2:
                st.metric("Hottest Recorded at West Reservoir! 🥵", f"{measured['water_temp'].max():.1f}C")
            with col3:
                st.metric("Total Readings Taken", len(measured))

            display_debug_panel(temperatures_deduped, forecaster, hourly_air_temps, forecast_3hourly, gap_fill_hourly)

            st.divider()
            st.subheader("About the Project")
            st.markdown(
                """
    Hi! I'm Tom, a local and regular swimmer at the reservoir for over a year.

    I enjoy tracking the temperatures so I decided to make this app. I've recorded
    most of the temperatures since November 2024, and used that data to train a
    simple physics model to predict future temperatures.

    The model simulates hour-by-hour heat transfer between air and water. Forecast
    weather data from OpenWeatherMap and historic data from Open-Meteo inform the
    predictions.
                """
            )

        except DataLoadError as e:
            st.error(f"Cannot load required data: {e}")
            st.info(
                "Please check:\n"
                "- Internet connection is working\n"
                "- Google Sheets is accessible\n"
                "- Open-Meteo service is available"
            )
            st.stop()


if __name__ == "__main__":
    main()
