"""West Reservoir Temperature Tracker - Streamlit App"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data import (
    load_water_temps,
    load_historical_air_temps,
    load_hourly_air_temps,
    load_forecast_air_temps,
    load_forecast_air_temps_3hourly,
    interpolate_to_hourly,
    DataLoadError,
)
from forecaster import WaterTempForecaster
from config import ENABLE_MOTHERDUCK
from quotes import QUOTES

# Conditional MotherDuck import
if ENABLE_MOTHERDUCK:
    from forecast_storage import ForecastStorage, ForecastStorageError

# Cache TTL for data (6 hours)
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


def retrieve_gap_fill_forecasts(
    hist_end: datetime,
    fore_start: datetime
) -> pd.DataFrame:
    """
    Retrieve stored 3-hourly forecasts from MotherDuck to fill the gap
    between Meteostat historical data and live OWM forecast.

    Args:
        hist_end: Last timestamp from Meteostat hourly data
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

        # Interpolate 3-hourly to hourly
        gap_hourly = interpolate_to_hourly(gap_data)
        return gap_hourly

    except ForecastStorageError:
        return pd.DataFrame(columns=["datetime", "air_temp"])
    except Exception:
        return pd.DataFrame(columns=["datetime", "air_temp"])


def combine_hourly_temps(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    gap_fill: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Combine historical hourly temps (Meteostat) with forecast hourly temps (OWM interpolated).

    Historical data takes precedence for overlapping times.
    Gap-fill data (from stored MotherDuck forecasts) fills the gap between historical and forecast.
    If no gap-fill data, falls back to linear interpolation.

    Priority order:
    1. Historical (Meteostat) - trusted measured data
    2. Gap-fill (stored forecasts from MotherDuck) - yesterday's forecast for today
    3. Forecast (live OWM) - current forecast for future

    Args:
        historical: DataFrame with 'datetime' and 'air_temp' columns (Meteostat)
        forecast: DataFrame with 'datetime' and 'air_temp' columns (interpolated OWM)
        gap_fill: Optional DataFrame with 'datetime' and 'air_temp' columns (stored forecasts)

    Returns:
        Combined DataFrame with 'datetime' and 'air_temp' columns
    """
    if historical.empty and forecast.empty:
        return pd.DataFrame(columns=["datetime", "air_temp"])

    if historical.empty:
        return forecast.copy()

    if forecast.empty:
        return historical.copy()

    def normalize_datetime_col(dt_series: pd.Series) -> pd.Series:
        """Ensure datetime series is timezone-naive datetime64[s]."""
        dt = pd.to_datetime(dt_series)
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        return dt.astype("datetime64[s]")

    # Normalize column names and ensure timezone-naive datetimes
    hist = historical[["datetime", "air_temp"]].copy()
    fore = forecast[["datetime", "air_temp"]].copy()
    hist["datetime"] = normalize_datetime_col(hist["datetime"])
    fore["datetime"] = normalize_datetime_col(fore["datetime"])

    # Find where historical ends and forecast begins
    hist_end = hist["datetime"].max()
    fore_start = fore["datetime"].min()

    # Only use forecast data after historical ends
    fore_future = fore[fore["datetime"] > hist_end].copy()

    # Process gap-fill data if available
    gap_data = None
    if gap_fill is not None and not gap_fill.empty:
        gap = gap_fill[["datetime", "air_temp"]].copy()
        gap["datetime"] = normalize_datetime_col(gap["datetime"])
        # Only use gap data that's after historical and before forecast
        filtered = gap[(gap["datetime"] > hist_end) & (gap["datetime"] < fore_start)]
        if not filtered.empty:
            gap_data = filtered.copy()

    # Combine all sources: historical + gap_fill + forecast (only non-empty)
    to_concat = [hist, fore_future]
    if gap_data is not None:
        to_concat.insert(1, gap_data)  # Insert between hist and fore
    combined = pd.concat(to_concat, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    # If there's still a gap (gap_fill didn't fully cover), interpolate it
    if not combined.empty:
        # Set index for resampling
        combined = combined.set_index("datetime").sort_index()
        # Resample to hourly and interpolate any remaining gaps
        combined = combined.resample("h").interpolate(method="linear")
        combined = combined.reset_index()

    return combined


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
        # Section 1: Data Overview
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

        # Section 2: Model Parameters
        st.subheader("Model Parameters")
        st.write(f"**Heat transfer coefficient (k)**: {forecaster.k:.4f} per hour")
        daily_response = 1 - (1 - forecaster.k) ** 24
        st.write(f"**Daily response**: {daily_response:.1%} of temperature difference")
        st.write("**Physics**: T_water(t+1h) = T_water(t) + k * (T_air(t) - T_water(t))")

        # Section 3: Tomorrow's Calculation (if available)
        has_predictions = any(temperatures["source"] == "PREDICTED")
        if has_predictions:
            st.subheader("Tomorrow's Prediction (24h Simulation)")

            # Find latest measured row
            measured_data = temperatures[temperatures["source"] == "MEASURED"]

            if not measured_data.empty:
                latest = measured_data.iloc[-1]
                latest_date = pd.Timestamp(latest["date"])

                # Get the 24 hours from this measurement to next
                start_dt = latest_date.replace(hour=forecaster.MEASUREMENT_HOUR)
                end_dt = start_dt + timedelta(hours=24)

                hourly_temps = forecaster._get_hourly_temps_for_period(start_dt, end_dt)

                if hourly_temps:
                    explanation = forecaster.explain_prediction(
                        current_water_temp=latest["water_temp"],
                        hourly_air_temps=hourly_temps,
                    )

                    st.code(
                        f"""
Current water temp (7am):  {explanation['current_water_temp']:.2f} C
Hours simulated:           {explanation['hours_simulated']}
Air temp range:            {explanation['air_temp_min']:.1f} C to {explanation['air_temp_max']:.1f} C
Air temp average:          {explanation['air_temp_avg']:.1f} C
Heat transfer rate (k):    {explanation['heat_transfer_coefficient']:.4f} per hour
Total temp change:         {explanation['total_temp_change']:.2f} C
--------------------------------------------
Tomorrow's predicted temp: {explanation['predicted_water_temp']:.2f} C
                        """
                    )

                    # Show hourly breakdown in sub-expander
                    with st.expander("Hourly Simulation Detail"):
                        breakdown_df = pd.DataFrame(explanation["hourly_breakdown"])
                        breakdown_df["hour"] = breakdown_df["hour"].apply(
                            lambda h: f"{h:02d}:00"
                        )
                        breakdown_df = breakdown_df.rename(
                            columns={
                                "hour": "Time",
                                "air_temp": "Air (C)",
                                "water_temp_before": "Water Before (C)",
                                "temp_change": "Change (C)",
                                "water_temp_after": "Water After (C)",
                            }
                        )
                        st.dataframe(
                            breakdown_df.style.format(
                                {
                                    "Air (C)": "{:.1f}",
                                    "Water Before (C)": "{:.2f}",
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

        # Section 4: Hourly Air Temperature Chart
        st.subheader("Air Temperature: Last 48h + Next 48h")
        if not hourly_air_temps.empty:
            now = datetime.now()
            cutoff_past = now - timedelta(hours=48)
            cutoff_future = now + timedelta(hours=48)

            # Historical: last 48h of hourly Meteostat data
            past_hourly = hourly_air_temps[hourly_air_temps["datetime"] >= cutoff_past]

            fig = go.Figure()

            # Meteostat hourly (solid red)
            fig.add_trace(
                go.Scatter(
                    x=past_hourly["datetime"],
                    y=past_hourly["air_temp"],
                    mode="lines",
                    name="Historical (Meteostat hourly)",
                    line=dict(color="red", width=1),
                )
            )

            # Gap fill from MotherDuck stored forecasts
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

            # OWM 3-hourly raw points + interpolated line
            if forecast_3hourly is not None and not forecast_3hourly.empty:
                # Filter to next 48h
                forecast_window = forecast_3hourly[
                    forecast_3hourly["datetime"] <= cutoff_future
                ]

                if not forecast_window.empty:
                    # Interpolate to hourly for the line
                    forecast_interpolated = interpolate_to_hourly(forecast_window)

                    # Interpolated line (orange dashed)
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_interpolated["datetime"],
                            y=forecast_interpolated["air_temp"],
                            mode="lines",
                            name="Forecast (OWM interpolated)",
                            line=dict(color="orange", width=1, dash="dash"),
                        )
                    )

                    # Raw 3-hourly points (orange markers)
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

        # Section 5: Raw DataFrame
        st.subheader("Raw Data (Last 10 Rows)")
        display_df = temperatures.tail(10)[
            ["date", "water_temp", "air_temp", "source"]
        ].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, width='stretch')

        # Section 6: Forecast Storage Status
        st.subheader("Forecast Storage (MotherDuck)")

        if ENABLE_MOTHERDUCK:
            try:
                storage = ForecastStorage()
                conn = storage._get_connection()

                result = conn.execute("""
                    SELECT
                        MAX(forecast_created_timestamp) as last_stored,
                        COUNT(DISTINCT DATE(forecast_created_timestamp)) as forecast_runs,
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

    # Filter to last 5 days + any future dates
    filtered = temperatures[
        (temperatures["date"].dt.date >= cutoff_date)
    ].copy()

    # Split data by source
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

    # Daily air temperature as whisker plot (min-avg-max)
    # For past dates prefer Meteostat (historical), for today prefer forecast (Meteostat is partial day only)
    air_data = filtered[filtered["air_temp"].notna()].copy()
    past_air = air_data[air_data["date"].dt.date < today].drop_duplicates(subset=["date"], keep="first")
    today_air = air_data[air_data["date"].dt.date == today].drop_duplicates(subset=["date"], keep="last")
    future_air = air_data[air_data["date"].dt.date > today].drop_duplicates(subset=["date"], keep="first")
    all_with_air = pd.concat([past_air, today_air, future_air]).sort_values("date").reset_index(drop=True)
    if not all_with_air.empty:
        has_minmax = all_with_air["air_temp_min"].notna().any()
        if has_minmax:
            # Calculate error bar distances
            all_with_air["error_plus"] = all_with_air["air_temp_max"] - all_with_air["air_temp"]
            all_with_air["error_minus"] = all_with_air["air_temp"] - all_with_air["air_temp_min"]

            # Trace 1: Red error bars (range)
            fig.add_trace(
                go.Scatter(
                    x=all_with_air["date"],
                    y=all_with_air["air_temp"],
                    mode="markers",
                    name="Air temperature range",
                    marker=dict(size=0),  # Hide markers, only show error bars
                    # line=dict(color="rgba(255, 100, 100, 0.4)", width=10),
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

            # Trace 2: Black center markers (average)
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

    # Water temperature - solid line for measured data
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

        # Past gap-fills: isolated markers only (no connecting line)
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

        # Future forecasts: dashed line connected from last measured point
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
        title="Water Temperature (Last 10 Days + Forecast)",
        xaxis_title="Date",
        yaxis_title="Temperature (C)",
        hovermode="closest",
        height=500,
    )

    return fig


def main():
    """Main application."""
    # Check for forecast_graph view - early return for graph-only view
    view = st.query_params.get("view")

    if view == "forecast_graph":
        # Graph-only view: load data, create chart, display, and exit
        try:
            water_temps = cached_load_water_temps()
            start_date = pd.Timestamp(water_temps["date"].min()).normalize()
            end_date = pd.Timestamp.now().normalize()
            air_temps_hist = cached_load_historical_air_temps(start_date, end_date)
            hourly_air_temps = cached_load_hourly_air_temps(start_date, end_date)

            # Fill missing daily temps from hourly data
            hourly_daily_stats = (
                hourly_air_temps.assign(date=hourly_air_temps["datetime"].dt.normalize())
                .groupby("date")["air_temp"]
                .agg(["mean", "min", "max"])
                .reset_index()
            )
            hourly_daily_stats.columns = ["date", "air_temp_h", "air_temp_min_h", "air_temp_max_h"]
            hourly_daily_stats["date"] = pd.to_datetime(hourly_daily_stats["date"])

            air_temps_hist = pd.merge(air_temps_hist, hourly_daily_stats, on="date", how="outer")
            air_temps_hist["air_temp"] = air_temps_hist["air_temp"].fillna(air_temps_hist["air_temp_h"])
            air_temps_hist["air_temp_min"] = air_temps_hist["air_temp_min"].fillna(air_temps_hist["air_temp_min_h"])
            air_temps_hist["air_temp_max"] = air_temps_hist["air_temp_max"].fillna(air_temps_hist["air_temp_max_h"])
            air_temps_hist = air_temps_hist[["date", "air_temp", "air_temp_min", "air_temp_max"]].dropna(subset=["air_temp"])

            temperatures = pd.merge(water_temps, air_temps_hist, on="date", how="outer")
            temperatures = temperatures.sort_values("date").reset_index(drop=True)
            temperatures["source"] = "MEASURED"
            temperatures.loc[temperatures["water_temp"].isna(), "source"] = "AIR_ONLY"

            # Add forecast with 3-hourly data
            combined_hourly = hourly_air_temps
            try:
                # Load 3-hourly and combine
                forecast_3hourly = cached_load_forecast_air_temps_3hourly(days=5)
                forecast_hourly = interpolate_to_hourly(forecast_3hourly)

                # Retrieve gap-fill data from MotherDuck if enabled
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

                # Also load daily for temperatures DataFrame
                forecast = cached_load_forecast_air_temps(days=5)
                forecast["source"] = "AIR_ONLY"
                temperatures = pd.concat([temperatures, forecast], ignore_index=True)
                # Create deduplicated version for prediction chain
                temperatures["_sort_priority"] = temperatures["source"].map(
                    {"MEASURED": 0, "AIR_ONLY": 1, "PREDICTED": 2}
                )
                temperatures = temperatures.sort_values(["date", "_sort_priority"]).reset_index(drop=True)
                temperatures = temperatures.drop(columns=["_sort_priority"])
                temperatures_deduped = temperatures.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
            except DataLoadError:
                temperatures_deduped = temperatures.copy()

            # Train and predict
            forecaster = WaterTempForecaster()
            forecaster.set_hourly_air_temps(combined_hourly)
            forecaster.fit(temperatures_deduped[temperatures_deduped["source"] == "MEASURED"])
            temperatures_deduped = forecaster.predict(temperatures_deduped)

            # Show only the chart
            chart = create_temperature_chart(temperatures_deduped)
            st.plotly_chart(chart, width='stretch')
            st.stop()

        except DataLoadError as e:
            st.error(f"Cannot load required data: {e}")
            st.stop()

    # Full view continues below (no conditionals needed)
    st.title("West Reservoir Temperature Tracker + Forecaster")
    st.markdown("Tracking and forecasting water temperature at West Reservoir, London.")

    tab_temp, tab_quotes = st.tabs(["Temperature", "Heard at the Res"])

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

    with tab_temp:

        # Header with info and image
        col_info, col_image = st.columns([1, 1])
        with col_info:
            st.info(
                "Water temperatures are taken each morning around 7am. "
                "The water will often be warmer by the time you get in!\n\n "
                "The forecast is based on the weather forecast and the water temperature history. "
                "Currently the forecast is only based on temperature exchange between air and water. "
                "It does not take into account other factors such as wind, cloud cover, or solar radiation.\n\n"
                "Additionally, temperature varies throughout the reservoir "
                "by both position and depth - this is just a snapshot of conditions."
            )
        with col_image:
            st.image("image.png",)

        try:
            # Step 1: Load water temperature measurements
            water_temps = cached_load_water_temps()

            # Step 2: Load historical air temperatures (daily for chart)
            # Normalize dates to day-level for consistent caching
            start_date = pd.Timestamp(water_temps["date"].min()).normalize()
            end_date = pd.Timestamp.now().normalize()
            air_temps_hist = cached_load_historical_air_temps(start_date, end_date)

            # Step 3: Load hourly air temperatures (for model)
            hourly_air_temps = cached_load_hourly_air_temps(start_date, end_date)

            # Step 3b: Fill missing daily temps from hourly data
            # (Daily API has ~2 day lag, but hourly is more current)
            hourly_daily_stats = (
                hourly_air_temps.assign(date=hourly_air_temps["datetime"].dt.normalize())
                .groupby("date")["air_temp"]
                .agg(["mean", "min", "max"])
                .reset_index()
            )
            hourly_daily_stats.columns = ["date", "air_temp_h", "air_temp_min_h", "air_temp_max_h"]
            hourly_daily_stats["date"] = pd.to_datetime(hourly_daily_stats["date"])

            # Merge hourly stats into daily data where missing
            air_temps_hist = pd.merge(
                air_temps_hist, hourly_daily_stats, on="date", how="outer"
            )
            # Fill gaps with hourly values
            air_temps_hist["air_temp"] = air_temps_hist["air_temp"].fillna(air_temps_hist["air_temp_h"])
            air_temps_hist["air_temp_min"] = air_temps_hist["air_temp_min"].fillna(air_temps_hist["air_temp_min_h"])
            air_temps_hist["air_temp_max"] = air_temps_hist["air_temp_max"].fillna(air_temps_hist["air_temp_max_h"])
            air_temps_hist = air_temps_hist[["date", "air_temp", "air_temp_min", "air_temp_max"]].dropna(subset=["air_temp"])

            # Step 4: Merge daily data into main temperatures DataFrame
            temperatures = pd.merge(water_temps, air_temps_hist, on="date", how="outer")
            temperatures = temperatures.sort_values("date").reset_index(drop=True)

            # Step 5: Mark sources
            temperatures["source"] = "MEASURED"
            temperatures.loc[temperatures["water_temp"].isna(), "source"] = "AIR_ONLY"

            # Step 6: Load 3-hourly forecast and combine with historical hourly
            forecast_3hourly = None
            gap_fill_hourly = None
            try:
                # Load raw 3-hourly forecast
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

                # Interpolate 3-hourly to hourly
                forecast_hourly = interpolate_to_hourly(forecast_3hourly)

                # Step 6b: Retrieve stored forecasts from MotherDuck to fill the gap
                # Gap is between: last Meteostat timestamp -> first OWM timestamp
                if ENABLE_MOTHERDUCK and not hourly_air_temps.empty and not forecast_3hourly.empty:
                    hist_end = hourly_air_temps["datetime"].max()
                    fore_start = forecast_3hourly["datetime"].min()

                    # Only try to fill if there's actually a gap (more than 1 hour)
                    gap_hours = (fore_start - hist_end).total_seconds() / 3600
                    if gap_hours > 1:
                        gap_fill_hourly = retrieve_gap_fill_forecasts(hist_end, fore_start)

                # Combine historical hourly + gap fill + forecast hourly
                combined_hourly = combine_hourly_temps(
                    hourly_air_temps, forecast_hourly, gap_fill_hourly
                )

                # Also load daily forecast for the temperatures DataFrame (for chart display)
                forecast = cached_load_forecast_air_temps(days=5)
                forecast["source"] = "AIR_ONLY"
                temperatures = pd.concat([temperatures, forecast], ignore_index=True)
                temperatures = temperatures.sort_values("date").reset_index(drop=True)

                # Keep full temperatures for forecast accuracy analysis
                # Create deduplicated version for prediction chain (MEASURED > AIR_ONLY)
                temperatures["_sort_priority"] = temperatures["source"].map(
                    {"MEASURED": 0, "AIR_ONLY": 1, "PREDICTED": 2}
                )
                temperatures = temperatures.sort_values(["date", "_sort_priority"]).reset_index(drop=True)
                temperatures = temperatures.drop(columns=["_sort_priority"])

                temperatures_deduped = temperatures.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

            except DataLoadError as e:
                st.warning(f"Weather forecast unavailable: {e}")
                st.info("Showing historical data only")
                combined_hourly = hourly_air_temps
                temperatures_deduped = temperatures.copy()  # No duplicates without forecast

            # Step 7: Train forecaster with combined hourly data
            forecaster = WaterTempForecaster()
            forecaster.set_hourly_air_temps(combined_hourly)
            forecaster.fit(temperatures_deduped[temperatures_deduped["source"] == "MEASURED"])

            # Step 8: Generate predictions (use deduped for proper chaining)
            temperatures_deduped = forecaster.predict(temperatures_deduped)

            # Store water predictions in MotherDuck (only once per day)
            if ENABLE_MOTHERDUCK:
                if 'last_prediction_store_date' not in st.session_state or \
                   st.session_state['last_prediction_store_date'] != datetime.now().date():
                    try:
                        storage = ForecastStorage()
                        predictions_df = temperatures_deduped[temperatures_deduped["source"] == "PREDICTED"].copy()
                        # Filter out any rows with NULL water_temp (shouldn't happen but safety check)
                        predictions_df = predictions_df.dropna(subset=["water_temp"])
                        measured_temps = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]

                        if not predictions_df.empty and not measured_temps.empty:
                            forecast_timestamp = st.session_state.get(
                                'last_forecast_timestamp',
                                datetime.now()
                            )
                            storage.store_water_predictions(
                                predictions_df=predictions_df,
                                forecast_created_timestamp=forecast_timestamp,
                                heat_transfer_coeff=forecaster.k,
                                start_water_temp=measured_temps.iloc[-1]["water_temp"]
                            )
                            st.session_state['last_prediction_store_date'] = datetime.now().date()
                    except ForecastStorageError as e:
                        st.warning(f"Could not store predictions: {e}")
                    except Exception as e:
                        st.warning(f"Prediction storage error: {e}")

            with col_info:
                # Display: Current temperature with clear date labeling
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                tomorrow = today + timedelta(days=1)

                st.header("Current Temperature")

                # Get measured data (use deduped for display)
                measured_data = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]
                today_data = temperatures_deduped[temperatures_deduped["date"].dt.date == today]
                has_today_measurement = any(today_data["source"] == "MEASURED")

                # Show measured temperature status
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

                # Find yesterday's measurement for today's forecast
                yesterday_data = measured_data[measured_data["date"].dt.date == yesterday]
                today_forecast_temp = None
                tomorrow_forecast_temp = None

                if not yesterday_data.empty:
                    # Compute today's forecast from yesterday's measurement
                    yesterday_temp = yesterday_data.iloc[-1]["water_temp"]
                    yesterday_dt = pd.Timestamp(yesterday).replace(hour=forecaster.MEASUREMENT_HOUR)
                    today_dt = pd.Timestamp(today).replace(hour=forecaster.MEASUREMENT_HOUR)
                    hourly_temps = forecaster._get_hourly_temps_for_period(yesterday_dt, today_dt)
                    if hourly_temps:
                        today_forecast_temp = forecaster._simulate_24h(
                            yesterday_temp, hourly_temps
                        )

                # Get tomorrow's forecast from the predictions DataFrame
                tomorrow_data = temperatures_deduped[temperatures_deduped["date"].dt.date == tomorrow]
                if not tomorrow_data.empty and tomorrow_data.iloc[0]["source"] == "PREDICTED":
                    tomorrow_forecast_temp = tomorrow_data.iloc[0]["water_temp"]

                # Display forecasts
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

                # Refresh button to clear cache
                if st.button("Data looks old? Press to refresh weather forecast and water temperature data", icon = '🔄' ):
                    st.cache_data.clear()
                    st.rerun()

            # Display: Temperature chart
            st.header("Temperature History and Forecast")
            st.text("""The chart shows the temperature history and forecast for the last 5 days, and next 5 days.
            Red bar shows the air temp range each day, with the black line being the average. The blue line is the water tempterature. It is dotted for forecast days.""")

            chart = create_temperature_chart(temperatures_deduped)
            st.plotly_chart(chart, width='stretch')

            # Display: Summary statistics
            st.header("Summary Statistics")
            measured = temperatures_deduped[temperatures_deduped["source"] == "MEASURED"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lowest Recorded at West Reservoir! ❄️", f"{measured['water_temp'].min():.1f}C")
            with col2:
                st.metric("Hottest Recorded at West Reservoir! 🥵", f"{measured['water_temp'].max():.1f}C")
            with col3:
                st.metric("Total Readings Taken", len(measured))

            # Display: Debug panel (always visible)
            display_debug_panel(temperatures_deduped, forecaster, hourly_air_temps, forecast_3hourly, gap_fill_hourly)

            # About section
            st.divider()
            st.subheader("About the Project")
            st.markdown(
                """
    Hi! I'm Tom, a local and regular swimmer at the reservoir for over a year.

    I enjoy tracking the temperatures so I decided to make this app. I've recorded
    most of the temperatures since November 2024, and used that data to train a
    simple physics model to predict future temperatures.

    The model simulates hour-by-hour heat transfer between air and water. Forecast
    weather data from OpenWeatherMap and historic data from Meteostat inform the
    predictions.
                """
            )

        except DataLoadError as e:
            st.error(f"Cannot load required data: {e}")
            st.info(
                "Please check:\n"
                "- Internet connection is working\n"
                "- Google Sheets is accessible\n"
                "- Meteostat service is available"
            )
            st.stop()


if __name__ == "__main__":
    main()
