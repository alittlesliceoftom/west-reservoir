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
    load_forecast_weather,
    daily_from_hourly_forecast,
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

# Meteostat stopped rebuilding its bulk endpoint on this date and served a frozen
# snapshot for five months (issue #33). Forecasts published in that window were
# simulated from interpolated air temperature, and trained on it too, so their
# error is an artefact of the dead feed rather than a property of the model.
# Open-Meteo replaced it in the code on 2026-09-06; the window closes for real
# once that ships to the deployed app.
METEOSTAT_OUTAGE_START = pd.Timestamp("2026-03-20")

# Days of forecast to fetch. Open-Meteo serves 16 free where OpenWeatherMap
# capped us at 5. Held at 5 for now: water-temp error grows with horizon
# (0.28 C at one day, 1.31 C at five), so a 16-day water forecast would be
# mostly drift. Raise this to show more of the air forecast.
FORECAST_DAYS = 5


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


@st.cache_data(ttl=CACHE_TTL)
def cached_load_forecast_weather(days):
    """Load the hourly forecast for every model input in one call, 6-hour cache."""
    return load_forecast_weather(days=days)


def get_storage():
    """
    One MotherDuck connection per browser session, schema checked once.

    Connecting costs about 4 seconds; a query over the largest table costs
    0.05. Six call sites each building their own ForecastStorage meant paying
    that several times over on a single page load, which is what actually made
    the accuracy tab slow (issue #43).

    Kept in session_state rather than st.cache_resource because a DuckDB
    connection is not safe for concurrent use and cache_resource is shared by
    every session; within one session, reruns are sequential.

    A cached connection can die between reruns (idle timeout, network blip), so
    it is pinged before being handed out. The ping is a local round trip at
    about 1ms against the 4 seconds a reconnect costs, and without it a dropped
    connection would fail every later call until the user reloaded the page.
    """
    storage = st.session_state.get("_forecast_storage")
    if storage is not None:
        try:
            storage._get_connection().execute("SELECT 1")
        except Exception:
            storage = None

    if storage is None:
        storage = ForecastStorage()
        storage.initialize_schema()
        st.session_state["_forecast_storage"] = storage
    return storage


@st.cache_data(ttl=3600)
def cached_load_stored_forecasts(max_horizon: int = 5):
    """Stored water-temp forecasts, one run per creation day."""
    return get_storage().get_water_predictions_last_run_per_day(
        max_horizon=max_horizon
    )


@st.cache_data(ttl=3600)
def cached_fitted_model_coefficients(water_temps, start_date, end_date):
    """
    Fit the model on historical weather and return its coefficients.

    The accuracy tab fits its own model rather than reusing the Temperature
    tab's, so the two tabs share no state and neither can block the other.
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


def solar_cloud_runs_by_date(stored_weather):
    """
    Group stored weather runs by creation date, keeping only solar/cloud rows.

    The storage reader returns every source's last run, and a source publishes
    only what it publishes. This provider supplies solar and cloud, so a row
    carrying neither must not reach it: the model would receive NULL where it
    expects radiation, and degrade silently rather than fail.
    """
    if stored_weather.empty:
        return {}

    usable = stored_weather.dropna(
        subset=["shortwave_radiation", "cloud_cover"], how="all"
    )
    if usable.empty:
        return {}

    return {date: group for date, group in usable.groupby("forecast_created_date")}


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

    storage = get_storage()
    stored_air = storage.get_air_forecasts_3hourly_last_run_per_day()
    runs_by_date = (
        {date: group for date, group in stored_air.groupby("forecast_created_date")}
        if not stored_air.empty
        else {}
    )

    # Stored weather forecasts, kept from 2026-09-06 onward (issue #29).
    # Anchors from before then have none and fall back to actual solar/cloud -
    # the remaining leak, and why replay numbers are an optimistic bound until
    # this table has been accumulating for a while.
    weather_runs_by_date = solar_cloud_runs_by_date(
        storage.get_weather_forecasts_last_run_per_day()
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

        # Stored solar/cloud is passed as the forecast argument, so it wins on
        # overlap and the actuals only fill hours it does not cover - the same
        # precedence the live forecast had.
        # Grouped by creation date alone, which is safe only while the stored
        # measures are disjoint by source: the dropna below keeps solar rows
        # and discards air-only ones. When air temperature joins this table
        # (#39), extract it with its own dropna or group by (date, source).
        run = weather_runs_by_date.get(anchor_date)
        solar_forecast = None
        if run is not None and not run.empty:
            solar_forecast = run[
                ["target_datetime", "shortwave_radiation", "cloud_cover"]
            ].rename(columns={"target_datetime": "datetime"}).dropna(
                subset=["shortwave_radiation", "cloud_cover"]
            )
            # A source that publishes only air temperature leaves these NULL.
            if solar_forecast.empty:
                solar_forecast = None

        weather = build_hourly_weather(hourly_air, solar_hist, solar_forecast)
        return weather, air_source

    return replay_current_model(
        forecaster, water_temps, weather_provider, max_horizon=max_horizon
    )



def retrieve_gap_fill_forecasts(
    hist_end: datetime,
    fore_start: datetime
) -> pd.DataFrame:
    """
    Retrieve stored 3-hourly forecasts from MotherDuck to fill the gap
    between the Open-Meteo archive and the live forecast.

    The stored rows are the OpenWeatherMap history kept from before the
    Open-Meteo switch (#39), which is why they are 3-hourly and get
    interpolated on the way out.

    Args:
        hist_end: Last timestamp from the Open-Meteo hourly archive
        fore_start: First timestamp from the live hourly forecast

    Returns:
        DataFrame with 'datetime' and 'air_temp' columns (interpolated to hourly),
        or empty DataFrame if no data available
    """
    if not ENABLE_MOTHERDUCK:
        return pd.DataFrame(columns=["datetime", "air_temp"])

    try:
        gap_data = get_storage().get_forecasts_for_gap(hist_end, fore_start)

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
    forecast_hourly: pd.DataFrame = None,
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

            if forecast_hourly is not None and not forecast_hourly.empty:
                forecast_window = forecast_hourly[
                    forecast_hourly["datetime"] <= cutoff_future
                ]

                if not forecast_window.empty:
                    # One trace, no interpolation: the forecast arrives hourly
                    # now, so there is nothing to bridge and no separate
                    # 3-hourly points to mark (#39).
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_window["datetime"],
                            y=forecast_window["air_temp"],
                            mode="lines+markers",
                            name="Forecast (Open-Meteo hourly)",
                            line=dict(color="orange", width=1, dash="dash"),
                            marker=dict(color="orange", size=5),
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
            st.caption(
                "Chart shows raw data points. The live forecast is hourly, so "
                "nothing is resampled; stored 3-hourly forecasts kept from "
                "before the Open-Meteo switch still are."
            )

        st.subheader("Raw Data (Last 10 Rows)")
        display_df = temperatures.tail(10)[
            ["date", "water_temp", "air_temp", "source"]
        ].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, width='stretch')

        st.subheader("Forecast Storage (MotherDuck)")

        if ENABLE_MOTHERDUCK:
            try:
                conn = get_storage()._get_connection()

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


# Chart windows offered above the time-series charts. A plotly range slider
# looks like the obvious control here, but plotly does not rescale the y-axis
# when the slider moves, so scrubbing back to a colder month leaves the trace
# pinned off-screen. Filtering the data in Streamlit instead means the window
# IS the data, so the y-axis always fits what is on screen.
CHART_PERIODS = {
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last year": 365,
    "All": None,
}


def filter_to_period(df: pd.DataFrame, date_column: str, days) -> pd.DataFrame:
    """
    Keep the last `days` of a frame, measured back from its own last date.

    Measured back from the data, not from today: a source that stops updating
    should show its final weeks rather than an empty chart.

    Args:
        df: Frame to filter.
        date_column: Column holding the dates.
        days: Window length, or None for everything.

    Returns:
        The filtered frame. Unchanged when days is None or the frame is empty.
    """
    if days is None or df.empty:
        return df

    dates = pd.to_datetime(df[date_column])
    cutoff = dates.max() - pd.Timedelta(days=days)
    return df[dates >= cutoff]

def _shade_meteostat_outage(fig: go.Figure, first_date, last_date) -> None:
    """
    Shade the window in which historical air temperature was interpolated.

    Only meaningful for stored forecasts: those were published while the feed
    was dead. The replay reads the repaired archive, so its errors in this
    window are not caused by the outage and shading them would mislead.

    The band starts at the later of the outage date and the first plotted
    date. Plotly widens an axis to fit its shapes, so a band anchored at
    2026-03-20 stretched the x-axis back five months and undid the chart
    period filter - the data was windowed correctly, the shape was not.

    Args:
        fig: Figure to shade.
        first_date: First date plotted, used to clamp the band's left edge.
        last_date: Right edge of the band - the last date on the chart, which
                   may extend past the last scored forecast.
    """
    if last_date is None or pd.isna(last_date):
        return

    last_date = pd.Timestamp(last_date)
    if last_date < METEOSTAT_OUTAGE_START:
        return

    start = METEOSTAT_OUTAGE_START
    if first_date is not None and not pd.isna(first_date):
        start = max(start, pd.Timestamp(first_date))

    fig.add_vrect(
        x0=start,
        x1=last_date,
        fillcolor="#d62728",
        opacity=0.10,
        line_width=0,
        layer="below",
        annotation_text=(
            "Missing up-to-date weather data - accuracy affected (issue #33)"
        ),
        # Right edge, not left: the band starts 2026-03-20, which is outside
        # the default last-30-days view, so a left-anchored label is invisible
        # until you scrub back.
        annotation_position="top right",
        annotation=dict(font_size=11, font_color="#d62728"),
    )


def create_forecast_vs_actual_chart(
    scored: pd.DataFrame,
    horizon: int,
    mark_outage: bool = False,
    water_temps: pd.DataFrame = None,
) -> go.Figure:
    """
    Forecast and measured water temp over time at one horizon.

    Args:
        scored: Scored forecasts at one horizon.
        horizon: Days ahead, for the legend.
        mark_outage: Shade the Meteostat outage window.
        water_temps: Full measurement record. When given, the measured line
                     spans all of it rather than only the dates a forecast
                     exists for - stored forecasts start 2026-02-16, so
                     without this the chart is a narrow window with no
                     context either side of the outage.
    """
    fig = go.Figure()

    if water_temps is not None and not water_temps.empty:
        measured = water_temps.dropna(subset=["water_temp"]).sort_values("date")
        measured_x, measured_y = measured["date"], measured["water_temp"]
    else:
        measured_x, measured_y = scored["target_date"], scored["actual_temp"]

    fig.add_trace(go.Scatter(
        x=measured_x, y=measured_y,
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

    if mark_outage:
        _shade_meteostat_outage(
            fig,
            min(measured_x.min(), scored["target_date"].min()),
            max(measured_x.max(), scored["target_date"].max()),
        )

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Water temperature (C)",
        height=440, hovermode="x unified", margin=dict(t=30),
    )
    return fig


def _one_sided(values) -> bool:
    """True when every value sits on the same side of zero."""
    v = pd.Series(values).dropna()
    return not v.empty and (bool((v >= 0).all()) or bool((v <= 0).all()))


def create_error_over_time_chart(
    scored: pd.DataFrame, mark_outage: bool = False
) -> go.Figure:
    """
    Signed forecast error over time, as bars from zero.

    Bars, not a line: measurements are manual and skip days, and a connecting
    line draws a slope across those gaps that asserts a trend nobody observed.
    A bar stands only where a forecast was actually scored.
    """
    fig = go.Figure(go.Bar(
        x=scored["target_date"], y=scored["error"],
        name="Error", marker_color="#ff7f0e",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f} C<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    if mark_outage:
        _shade_meteostat_outage(
            fig, scored["target_date"].min(), scored["target_date"].max()
        )

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Forecast - actual (C)",
        height=360, showlegend=False, margin=dict(t=30),
    )
    # Bars grow from zero, so zero must stay in frame even when every error
    # in the window falls on one side of it.
    fig.update_yaxes(rangemode="tozero" if _one_sided(scored["error"]) else "normal")
    return fig


def store_forecast_and_predictions(forecast_weather, temperatures_deduped, forecaster):
    """
    Write today's forecast and predictions to MotherDuck, once per day.

    Both are guarded by a session_state date so a rerun does not rewrite them.
    Failures warn rather than raise: storage is a record of what was forecast,
    and losing a day of it must not take the dashboard down with it.
    """
    if forecast_weather is not None:
        if 'last_forecast_fetch_date' not in st.session_state or \
           st.session_state['last_forecast_fetch_date'] != datetime.now().date():
            try:
                storage = get_storage()
                forecast_timestamp = datetime.now()
                storage.store_weather_forecast(
                    forecast_weather, forecast_timestamp, source="Open-Meteo"
                )
                st.session_state['last_forecast_fetch_date'] = datetime.now().date()
                st.session_state['last_forecast_timestamp'] = forecast_timestamp
            except ForecastStorageError as e:
                st.warning(f"Could not store forecast: {e}")
            except Exception as e:
                st.warning(f"Forecast storage error: {e}")

    if 'last_prediction_store_date' not in st.session_state or \
       st.session_state['last_prediction_store_date'] != datetime.now().date():
        try:
            storage = get_storage()
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


def page_temperature():
    """The dashboard: measurements, forecast and the model debug panel."""
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

        forecast_weather = None
        gap_fill_hourly = None
        try:
            forecast_weather = cached_load_forecast_weather(days=FORECAST_DAYS)

            # Gap is between: last archive timestamp -> first forecast timestamp
            if ENABLE_MOTHERDUCK and not hourly_air_temps.empty and not forecast_weather.empty:
                hist_end = hourly_air_temps["datetime"].max()
                fore_start = forecast_weather["datetime"].min()

                gap_hours = (fore_start - hist_end).total_seconds() / 3600
                if gap_hours > 1:
                    gap_fill_hourly = retrieve_gap_fill_forecasts(hist_end, fore_start)

            combined_hourly = combine_hourly_temps(
                hourly_air_temps,
                forecast_weather[["datetime", "air_temp"]],
                gap_fill_hourly,
            )

            # Daily min/mean/max for the chart, derived from the same hourly
            # series the model runs on, so the two cannot disagree.
            forecast = daily_from_hourly_forecast(forecast_weather)
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
        try:
            solar_hist = cached_load_historical_solar_cloud(start_date, end_date)
        except DataLoadError as e:
            st.warning(f"Open-Meteo historical solar/cloud unavailable: {e}")

        hourly_weather = build_hourly_weather(
            combined_hourly, solar_hist, forecast_weather
        )

        forecaster = WaterTempForecaster()
        forecaster.set_hourly_weather(hourly_weather)
        forecaster.fit(temperatures_deduped[temperatures_deduped["source"] == "MEASURED"])

        temperatures_deduped = forecaster.fill_predictions(temperatures_deduped)

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

        # Storage writes go here, below the chart, not beside the fetch that
        # produced the data. Streamlit streams output as it is produced, so
        # anything above the chart delays first paint - and the first write of
        # a server's life opens the MotherDuck connection, which is slow.
        # Late enough not to block the dashboard, early enough to still run
        # before the page finishes.
        if ENABLE_MOTHERDUCK:
            store_forecast_and_predictions(
                forecast_weather, temperatures_deduped, forecaster
            )

        st.header("Summary Statistics")
        measured = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lowest Recorded at West Reservoir! ❄️", f"{measured['water_temp'].min():.1f}C")
        with col2:
            st.metric("Hottest Recorded at West Reservoir! 🥵", f"{measured['water_temp'].max():.1f}C")
        with col3:
            st.metric("Total Readings Taken", len(measured))

        display_debug_panel(
            temperatures_deduped, forecaster, hourly_air_temps,
            forecast_weather, gap_fill_hourly,
        )

        st.divider()
        st.subheader("About the Project")
        st.markdown(
            """
Hi! I'm Tom, a local and regular swimmer at the reservoir for over a year.

I enjoy tracking the temperatures so I decided to make this app. I've recorded
most of the temperatures since November 2024, and used that data to train a
simple physics model to predict future temperatures.

The model simulates hour-by-hour heat transfer between air and water. Both
the forecast and the historic weather come from Open-Meteo, hourly.
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
        # Deliberately no st.stop() here. It halts the entire script, so
        # every tab defined below this one would silently fail to render
        # whenever temperature data is unavailable (issue #43).


def page_accuracy():
    """
    Forecast accuracy against measurements.

    A separate page rather than a tab because it opens a MotherDuck
    connection, which costs several seconds. st.navigation runs only the
    selected page, so that cost is paid by whoever opens this page and
    never by the dashboard.
    """
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
                "the forecast was made, forecast after. Solar and cloud use "
                "the stored forecast where we have one - we only started "
                "keeping those on 2026-09-06 (issue #29), and dates before "
                "that fall back to actual solar and cloud, which the forecast "
                "never had. Read those as an optimistic bound. Points marked "
                "with a cross had no stored air forecast either and used "
                "measured air for the whole window."
            )
            horizon_options = [1, 2, 3, 4, 5]
        else:
            st.caption(
                "The honest record: what we published, scored against what "
                "was then measured."
            )
            st.caption(
                "Shaded period: from 2026-03-20 we were missing up-to-date "
                "weather data, and forecast accuracy was affected. "
                "One-day-ahead error went from 0.21 C before to 0.61 C after, "
                "and got worse the longer it went on (April 0.42 C, June "
                "0.59 C, August 1.35 C). The weather data has since been "
                "fixed. See "
                "[issue #33](https://github.com/alittlesliceoftom/"
                "west-reservoir/issues/33) on GitHub for more information."
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

                st.subheader("Error by forecast horizon")
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
                    period_label = st.radio(
                        "Chart period",
                        list(CHART_PERIODS),
                        horizontal=True,
                        help=(
                            "Applies to the two charts below. The metrics "
                            "above cover the whole record."
                        ),
                    )
                    days = CHART_PERIODS[period_label]
                    windowed = filter_to_period(at_horizon, "target_date", days)
                    measured_window = filter_to_period(
                        water_temps, "date", days
                    )

                    st.subheader(
                        f"Forecast vs measured ({_horizon_label(selected_horizon)})"
                    )
                    st.plotly_chart(
                        create_forecast_vs_actual_chart(
                            windowed, selected_horizon,
                            mark_outage=not is_replay,
                            water_temps=measured_window,
                        ),
                        width='stretch',
                    )

                    st.subheader(
                        f"Error over time ({_horizon_label(selected_horizon)})"
                    )
                    st.plotly_chart(
                        create_error_over_time_chart(
                            windowed, mark_outage=not is_replay
                        ),
                        width='stretch',
                    )

                    with st.expander("Scored forecasts"):
                        st.dataframe(at_horizon, width='stretch')

        except ForecastStorageError as e:
            st.error(f"Could not load stored forecasts: {e}")
        except DataLoadError as e:
            st.error(f"Could not load measurements: {e}")


def page_quotes():
    """Static quotes."""
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
            forecast_weather = None
            try:
                forecast_weather = cached_load_forecast_weather(days=FORECAST_DAYS)

                gap_fill_hourly = None
                if ENABLE_MOTHERDUCK and not hourly_air_temps.empty and not forecast_weather.empty:
                    hist_end = hourly_air_temps["datetime"].max()
                    fore_start = forecast_weather["datetime"].min()
                    gap_hours = (fore_start - hist_end).total_seconds() / 3600
                    if gap_hours > 1:
                        gap_fill_hourly = retrieve_gap_fill_forecasts(hist_end, fore_start)

                combined_hourly = combine_hourly_temps(
                    hourly_air_temps,
                    forecast_weather[["datetime", "air_temp"]],
                    gap_fill_hourly,
                )

                forecast = daily_from_hourly_forecast(forecast_weather)
                forecast["source"] = "AIR_ONLY"
                temperatures = pd.concat([temperatures, forecast], ignore_index=True)
                # Create deduplicated version for prediction chain
                temperatures_deduped = deduplicate_temperatures(temperatures)
            except DataLoadError:
                temperatures_deduped = temperatures.copy()

            solar_hist = None
            try:
                solar_hist = cached_load_historical_solar_cloud(start_date, end_date)
            except DataLoadError:
                pass

            hourly_weather = build_hourly_weather(
                combined_hourly, solar_hist, forecast_weather
            )

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

    # st.navigation runs ONLY the selected page. Tabs did the opposite:
    # every tab body executed on every rerun, so the accuracy page's
    # MotherDuck connection was billed to anyone opening the dashboard.
    page = st.navigation(
        [
            st.Page(page_temperature, title="Temperature", default=True),
            st.Page(page_accuracy, title="Forecast Accuracy"),
            st.Page(page_quotes, title="Heard at the Res"),
        ],
        # Top rather than the sidebar default, so the three views stay
        # side by side where the tabs were.
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
