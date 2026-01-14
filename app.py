"""West Reservoir Temperature Tracker - Streamlit App"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data import load_water_temps, load_historical_air_temps, load_forecast_air_temps, DataLoadError
from forecaster import WaterTempForecaster


st.set_page_config(
    page_title="West Reservoir Water Temperature Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def display_debug_panel(temperatures: pd.DataFrame, forecaster: WaterTempForecaster):
    """Display comprehensive debug information."""
    with st.expander("Debug Information", expanded=True):
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
            st.metric("Total Rows", len(temperatures))

        # Section 2: Model Parameters
        st.subheader("Model Parameters")
        st.write(f"**Heat transfer coefficient (k)**: {forecaster.k:.4f} day⁻¹")
        st.write("**Physics equation**: dT/dt = k × (T_air_yesterday - T_water)")

        # Section 3: Tomorrow's Calculation (if available)
        has_predictions = any(temperatures["source"] == "PREDICTED")
        if has_predictions:
            st.subheader("Tomorrow's Calculation Breakdown")
            latest = temperatures[temperatures["source"] == "MEASURED"].iloc[-1]
            tomorrow = temperatures[temperatures["source"] == "PREDICTED"].iloc[0]

            explanation = forecaster.explain_prediction(
                current_water_temp=latest["water_temp"],
                yesterday_air_temp=latest["air_temp"],
            )

            st.code(
                f"""
Current water temperature:    {explanation['current_water_temp']:.2f}°C
Yesterday's air temperature:  {explanation['yesterday_air_temp']:.2f}°C
Temperature difference:       {explanation['temperature_difference']:.2f}°C
Heat transfer rate (k):       {explanation['heat_transfer_coefficient']:.4f}
Temperature change:           {explanation['temperature_change']:.2f}°C
────────────────────────────────────────────
Tomorrow's predicted temp:    {explanation['predicted_water_temp']:.2f}°C
            """
            )

        # Section 4: Raw DataFrame
        st.subheader("Raw Data (Last 10 Rows)")
        display_df = temperatures.tail(10)[["date", "water_temp", "air_temp", "source"]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True)


def create_temperature_chart(temperatures: pd.DataFrame) -> go.Figure:
    """Create temperature chart with measured and predicted data."""
    fig = go.Figure()

    # Split data by source
    measured = temperatures[temperatures["source"] == "MEASURED"]
    predicted = temperatures[temperatures["source"] == "PREDICTED"]

    # Measured water temperature
    fig.add_trace(
        go.Scatter(
            x=measured["date"],
            y=measured["water_temp"],
            mode="lines+markers",
            name="Measured Water Temp",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )

    # Predicted water temperature
    if not predicted.empty:
        fig.add_trace(
            go.Scatter(
                x=predicted["date"],
                y=predicted["water_temp"],
                mode="lines+markers",
                name="Predicted Water Temp",
                line=dict(color="orange", width=2, dash="dash"),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="West Reservoir Water Temperature",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        height=500,
    )

    return fig


def main():
    """Main application."""
    st.title("West Reservoir Water Temperature Tracker")
    st.markdown("Simple, transparent water temperature tracking and prediction")

    try:
        # Step 1: Load water temperature measurements
        water_temps = load_water_temps()

        # Step 2: Load historical air temperatures
        start_date = water_temps["date"].min()
        end_date = datetime.now()
        air_temps_hist = load_historical_air_temps(start_date, end_date)

        # Step 3: Merge into main temperatures DataFrame
        temperatures = pd.merge(water_temps, air_temps_hist, on="date", how="outer")
        temperatures = temperatures.sort_values("date").reset_index(drop=True)

        # Step 4: Mark sources
        temperatures["source"] = "MEASURED"
        temperatures.loc[temperatures["water_temp"].isna(), "source"] = "AIR_ONLY"

        # Step 5: Add forecast dates
        try:
            forecast_air = load_forecast_air_temps(days=5)
            forecast_rows = forecast_air.copy()
            forecast_rows["water_temp"] = None
            forecast_rows["source"] = "AIR_ONLY"
            temperatures = pd.concat([temperatures, forecast_rows], ignore_index=True)
            temperatures = temperatures.sort_values("date").reset_index(drop=True)
        except DataLoadError as e:
            st.warning(f"Weather forecast unavailable: {e}")
            st.info("Showing historical data only")

        # Step 6: Train forecaster
        forecaster = WaterTempForecaster()
        forecaster.fit(temperatures[temperatures["source"] == "MEASURED"])

        # Step 7: Generate predictions
        temperatures = forecaster.predict(temperatures)

        # Display: Current temperature with clear date labeling
        today = datetime.now().date()
        st.header("Current Temperature")

        # Find today's data
        today_data = temperatures[temperatures["date"].dt.date == today]

        if not today_data.empty:
            latest = today_data.iloc[0]
            st.metric("Today's Temperature", f"{latest['water_temp']:.1f}°C")
        else:
            # No data for today, show latest reading
            measured_data = temperatures[temperatures["source"] == "MEASURED"]
            if not measured_data.empty:
                latest = measured_data.iloc[-1]
                latest_date = latest["date"].strftime("%Y-%m-%d")
                st.warning(f"No measurement for today yet. Showing latest reading from {latest_date}")
                st.metric(
                    f"Latest Reading ({latest_date})",
                    f"{latest['water_temp']:.1f}°C",
                )

        # Display: Temperature chart
        st.header("Temperature History and Forecast")
        chart = create_temperature_chart(temperatures)
        st.plotly_chart(chart, use_container_width=True)

        # Display: Summary statistics
        st.header("Summary Statistics")
        measured = temperatures[temperatures["source"] == "MEASURED"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{measured['water_temp'].mean():.1f}°C")
        with col2:
            st.metric("Minimum", f"{measured['water_temp'].min():.1f}°C")
        with col3:
            st.metric("Maximum", f"{measured['water_temp'].max():.1f}°C")
        with col4:
            st.metric("Total Readings", len(measured))

        # Display: Debug panel (always visible)
        display_debug_panel(temperatures, forecaster)

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
