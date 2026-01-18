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
    DataLoadError,
)
from forecaster import WaterTempForecaster

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


st.set_page_config(
    page_title="West Reservoir Water Temperature Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def display_debug_panel(
    temperatures: pd.DataFrame,
    forecaster: WaterTempForecaster,
    hourly_air_temps: pd.DataFrame,
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
                            use_container_width=True,
                        )
                else:
                    st.info("Waiting for hourly air temperature data")
            else:
                st.info("No measured data available")

        # Section 4: Hourly Air Temperature Chart
        st.subheader("Hourly Air Temperature (Last 72h)")
        if not hourly_air_temps.empty:
            recent_hourly = hourly_air_temps.tail(72)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=recent_hourly["datetime"],
                    y=recent_hourly["air_temp"],
                    mode="lines",
                    name="Air Temp",
                    line=dict(color="red", width=1),
                )
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Temperature (C)",
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Section 5: Raw DataFrame
        st.subheader("Raw Data (Last 10 Rows)")
        display_df = temperatures.tail(10)[
            ["date", "water_temp", "air_temp", "source"]
        ].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True)


def create_temperature_chart(temperatures: pd.DataFrame) -> go.Figure:
    """Create temperature chart with last 10 days + forecast."""
    fig = go.Figure()

    today = datetime.now().date()
    cutoff_date = today - timedelta(days=10)

    # Filter to last 10 days + any future dates
    filtered = temperatures[
        (temperatures["date"].dt.date >= cutoff_date)
    ].copy()

    # Split data by source
    measured = filtered[filtered["source"] == "MEASURED"]
    predicted = filtered[filtered["source"] == "PREDICTED"]

    # Daily air temperature as whisker plot (min-avg-max)
    all_with_air = filtered[filtered["air_temp"].notna()].copy()
    if not all_with_air.empty:
        has_minmax = all_with_air["air_temp_min"].notna().any()
        if has_minmax:
            # Calculate error bar distances
            all_with_air["error_plus"] = all_with_air["air_temp_max"] - all_with_air["air_temp"]
            all_with_air["error_minus"] = all_with_air["air_temp"] - all_with_air["air_temp_min"]

            fig.add_trace(
                go.Scatter(
                    x=all_with_air["date"],
                    y=all_with_air["air_temp"],
                    mode="markers",
                    name="Air Temp",
                    marker=dict(color="red", size=6, symbol="line-ew", line=dict(width=2)),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=all_with_air["error_plus"],
                        arrayminus=all_with_air["error_minus"],
                        color="rgba(255, 100, 100, 0.6)",
                        thickness=2,
                        width=4,
                    ),
                    legendgroup="air",
                    customdata=list(zip(all_with_air["air_temp_min"], all_with_air["air_temp_max"])),
                    hovertemplate="Air: %{y:.1f}C (Low: %{customdata[0]:.1f}, High: %{customdata[1]:.1f})<extra></extra>",
                )
            )

    # Water temperature - solid line for measured data
    if not measured.empty:
        fig.add_trace(
            go.Scatter(
                x=measured["date"],
                y=measured["water_temp"],
                mode="lines+markers",
                name="Water Temp",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
                legendgroup="water",
                hovertemplate="Water: %{y:.1f}C<extra></extra>",
            )
        )

    # Connect last measured to predictions with dashed line
    if not predicted.empty and not measured.empty:
        # Include last measured point to connect the lines
        last_measured = measured.iloc[[-1]]
        predicted_with_connection = pd.concat([last_measured, predicted])

        fig.add_trace(
            go.Scatter(
                x=predicted_with_connection["date"],
                y=predicted_with_connection["water_temp"],
                mode="lines+markers",
                name="Water Temp (Forecast)",
                line=dict(color="blue", width=2, dash="dash"),
                marker=dict(size=6),
                legendgroup="water",
                showlegend=False,
                hovertemplate="Water (forecast): %{y:.1f}C<extra></extra>",
            )
        )

    # Add vertical line for today
    fig.add_shape(
        type="line",
        x0=str(today),
        x1=str(today),
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=15),
        opacity = 0.3
    )
    fig.add_annotation(
        x=str(today),
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        yshift=10,
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
    st.title("West Reservoir Temperature Tracker + Forecaster")
    st.markdown("Tracking and forecasting water temperature at West Reservoir, London.")

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

        # Step 6: Add forecast dates
        try:
            forecast = cached_load_forecast_air_temps(days=5)
            forecast["source"] = "AIR_ONLY"
            temperatures = pd.concat([temperatures, forecast], ignore_index=True)
            temperatures = temperatures.sort_values("date").reset_index(drop=True)
        except DataLoadError as e:
            st.warning(f"Weather forecast unavailable: {e}")
            st.info("Showing historical data only")

        # Step 7: Train forecaster with hourly data
        forecaster = WaterTempForecaster()
        forecaster.set_hourly_air_temps(hourly_air_temps)
        forecaster.fit(temperatures[temperatures["source"] == "MEASURED"])

        # Step 8: Generate predictions
        temperatures = forecaster.predict(temperatures)


        with col_info:
            # Display: Current temperature with clear date labeling
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            tomorrow = today + timedelta(days=1)

            st.header("Current Temperature")

            # Get measured data
            measured_data = temperatures[temperatures["source"] == "MEASURED"]
            today_data = temperatures[temperatures["date"].dt.date == today]
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
            tomorrow_data = temperatures[temperatures["date"].dt.date == tomorrow]
            if not tomorrow_data.empty and tomorrow_data.iloc[0]["source"] == "PREDICTED":
                tomorrow_forecast_temp = tomorrow_data.iloc[0]["water_temp"]

            # Display forecasts
            col_today_fc, col_tomorrow_fc = st.columns(2)
            with col_today_fc:
                if today_forecast_temp is not None:
                    st.metric("Today's Forecast (excludes today's measurement)", f"{today_forecast_temp:.1f}C")
                else:
                    st.metric("Today's Forecast (excludes today's measurement)", "N/A")
            with col_tomorrow_fc:
                if tomorrow_forecast_temp is not None:
                    st.metric("Tomorrow's Forecast", f"{tomorrow_forecast_temp:.1f}C")
                else:
                    st.metric("Tomorrow's Forecast", "N/A")

                        # Refresh button to clear cache
            if st.button("Data looks old? Press to refresh weather forecast and water temperature data", icon = '🔄' ):
                st.cache_data.clear()
                st.rerun()


        # Display: Temperature chart
        st.header("Temperature History and Forecast")
        st.text("""The chart shows the temperature history and forecast for the last 10 days, and next 5 days.
        Red bar shows the air temp range each day, with the black line being the average. The blue line is the water tempterature. It is dotted for forecast days.""")
        chart = create_temperature_chart(temperatures)
        st.plotly_chart(chart, use_container_width=True)

        # Display: Summary statistics
        st.header("Summary Statistics")
        measured = temperatures[temperatures["source"] == "MEASURED"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lowest Recorded at West Reservoir! ❄️", f"{measured['water_temp'].min():.1f}C")
        with col2:
            st.metric("Hottest Recorded at West Reservoir! 🥵", f"{measured['water_temp'].max():.1f}C")
        with col3:
            st.metric("Total Readings Taken", len(measured))

        # Display: Debug panel (always visible)
        display_debug_panel(temperatures, forecaster, hourly_air_temps)

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
