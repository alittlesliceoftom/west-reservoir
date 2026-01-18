"""Training Explorer - Analyze forecaster model performance"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats

from data import (
    load_water_temps,
    load_hourly_air_temps,
    DataLoadError,
)
from forecaster import WaterTempForecaster

# Constants matching forecaster.py
SUMMER_SOLSTICE_DOY = 172

st.set_page_config(
    page_title="Training Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def compute_training_predictions(
    measured_data: pd.DataFrame,
    hourly_air_temps: pd.DataFrame,
    k: float,
    k_seasonal: float = 0.0,
    measurement_hour: int = 7,
    max_gap_days: int = 3,
) -> pd.DataFrame:
    """
    Compute predictions for all measurement pairs within max_gap_days.

    Uses all pairs within the gap (not just consecutive), e.g., with max_gap_days=3:
    - Sunday -> Monday (1 day)
    - Sunday -> Tuesday (2 days)
    - Sunday -> Wednesday (3 days)

    Applies seasonal adjustment to k based on date.

    Returns DataFrame with: date, actual, predicted, error, abs_error, days_gap, etc.
    """
    hourly_indexed = hourly_air_temps.set_index("datetime").sort_index()
    results = []

    for i in range(1, len(measured_data)):
        curr_row = measured_data.iloc[i]
        curr_date = curr_row["date"]
        end_dt = pd.Timestamp(curr_date).replace(hour=measurement_hour)

        # Look back at all previous measurements within the max gap
        for j in range(i - 1, -1, -1):
            prev_row = measured_data.iloc[j]
            prev_date = prev_row["date"]
            start_dt = pd.Timestamp(prev_date).replace(hour=measurement_hour)

            days_gap = (end_dt - start_dt).days
            if days_gap > max_gap_days:
                break  # No point looking further back

            if days_gap < 1:
                continue  # Same day, skip

            # Get hourly temps for this period
            mask = (hourly_indexed.index >= start_dt) & (hourly_indexed.index < end_dt)
            hourly_temps = hourly_indexed.loc[mask, "air_temp"].tolist()

            if len(hourly_temps) >= 20:
                # Calculate seasonal-adjusted k
                if isinstance(curr_date, pd.Timestamp):
                    day_of_year = curr_date.dayofyear
                else:
                    day_of_year = curr_date.timetuple().tm_yday

                seasonal_factor = 1 + k_seasonal * math.sin(
                    2 * math.pi * (day_of_year - SUMMER_SOLSTICE_DOY) / 365
                )
                k_effective = k * seasonal_factor

                # Simulate
                water_temp = prev_row["water_temp"]
                for air_temp in hourly_temps:
                    water_temp += k_effective * (air_temp - water_temp)

                predicted = water_temp
                actual = curr_row["water_temp"]
                error = predicted - actual

                results.append({
                    "date": curr_date,
                    "prev_date": prev_date,
                    "days_gap": days_gap,
                    "prev_water_temp": prev_row["water_temp"],
                    "actual": actual,
                    "predicted": predicted,
                    "error": error,
                    "abs_error": abs(error),
                    "hours_simulated": len(hourly_temps),
                    "avg_air_temp": np.mean(hourly_temps),
                    "seasonal_factor": seasonal_factor,
                    "k_effective": k_effective,
                })

    return pd.DataFrame(results)


def main():
    st.title("Training Explorer")
    st.markdown("Analyze model training and prediction errors")

    try:
        # Load data
        water_temps = load_water_temps()
        start_date = water_temps["date"].min()
        end_date = datetime.now()
        hourly_air_temps = load_hourly_air_temps(start_date, end_date)

        # Prepare measured data
        measured_data = water_temps.sort_values("date").reset_index(drop=True)

        # Sidebar controls
        st.sidebar.header("Training Parameters")

        # Training window
        training_window = st.sidebar.selectbox(
            "Training Window",
            ["All Data", "Last 90 days", "Last 60 days", "Last 30 days", "Last 14 days"],
            index=0,
        )

        window_days = {
            "All Data": None,
            "Last 90 days": 90,
            "Last 60 days": 60,
            "Last 30 days": 30,
            "Last 14 days": 14,
        }

        # Max gap for training pairs
        max_gap_days = st.sidebar.slider(
            "Max Gap Between Measurements (days)",
            min_value=1,
            max_value=7,
            value=3,
            help="Use all measurement pairs within this gap. E.g., 3 means Sun->Mon, Sun->Tue, Sun->Wed all count as training pairs.",
        )

        days = window_days[training_window]
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            training_data = measured_data[measured_data["date"] >= cutoff].copy()
        else:
            training_data = measured_data.copy()

        # Train forecaster
        forecaster = WaterTempForecaster()
        forecaster.set_hourly_air_temps(hourly_air_temps)

        # Create temp df for training
        training_df = training_data.copy()
        training_df["source"] = "MEASURED"
        forecaster.fit(training_df)

        k = forecaster.k
        k_seasonal = forecaster.k_seasonal
        daily_response = 1 - (1 - k) ** 24

        # Display trained values
        st.sidebar.divider()
        st.sidebar.subheader("Trained Model")
        st.sidebar.metric("k (per hour)", f"{k:.4f}")
        st.sidebar.metric("k_seasonal (amplitude)", f"{k_seasonal:.3f}")
        st.sidebar.metric("Daily Response (base)", f"{daily_response:.1%}")
        st.sidebar.caption(
            f"Seasonal adjustment: k varies ±{abs(k_seasonal)*100:.0f}% "
            f"from winter to summer"
        )

        # Manual k override
        st.sidebar.divider()
        use_custom_k = st.sidebar.checkbox("Override k manually")
        if use_custom_k:
            custom_k = st.sidebar.slider(
                "Custom k",
                min_value=0.001,
                max_value=0.1,
                value=k,
                step=0.001,
                format="%.4f",
            )
            custom_k_seasonal = st.sidebar.slider(
                "Custom k_seasonal",
                min_value=-0.5,
                max_value=0.5,
                value=k_seasonal,
                step=0.01,
                format="%.3f",
            )
            k = custom_k
            k_seasonal = custom_k_seasonal
            daily_response = 1 - (1 - k) ** 24
            st.sidebar.metric("Custom Daily Response", f"{daily_response:.1%}")

        # Compute predictions for all data (using selected k, k_seasonal, and max_gap)
        predictions_df = compute_training_predictions(
            measured_data, hourly_air_temps, k, k_seasonal=k_seasonal, max_gap_days=max_gap_days
        )

        if predictions_df.empty:
            st.error("Not enough data to compute predictions")
            return

        # Filter predictions to training window for metrics
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            window_predictions = predictions_df[predictions_df["date"] >= cutoff]
        else:
            window_predictions = predictions_df

        # === Summary Statistics ===
        st.header("Summary Statistics")

        mae = window_predictions["abs_error"].mean()
        rmse = np.sqrt((window_predictions["error"] ** 2).mean())
        me = window_predictions["error"].mean()  # Mean error (bias)
        std_error = window_predictions["error"].std()

        # R² calculation
        ss_res = ((window_predictions["actual"] - window_predictions["predicted"]) ** 2).sum()
        ss_tot = ((window_predictions["actual"] - window_predictions["actual"].mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("MAE", f"{mae:.2f}°C")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}°C")
        with col3:
            st.metric("Mean Error (Bias)", f"{me:+.2f}°C")
        with col4:
            st.metric("Std Dev of Error", f"{std_error:.2f}°C")
        with col5:
            st.metric("R²", f"{r2:.3f}")

        st.caption(
            "MAE = Mean Absolute Error, RMSE = Root Mean Square Error, "
            "Bias = positive means model over-predicts"
        )

        # === Predicted vs Actual Plot ===
        st.header("Predicted vs Actual")

        fig_pva = go.Figure()

        # Perfect prediction line
        min_val = min(window_predictions["actual"].min(), window_predictions["predicted"].min())
        max_val = max(window_predictions["actual"].max(), window_predictions["predicted"].max())

        fig_pva.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        ))

        fig_pva.add_trace(go.Scatter(
            x=window_predictions["actual"],
            y=window_predictions["predicted"],
            mode="markers",
            name="Predictions",
            marker=dict(size=8, color="blue", opacity=0.6),
            text=window_predictions["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Date: %{text}<br>Actual: %{x:.1f}°C<br>Predicted: %{y:.1f}°C<extra></extra>",
        ))

        fig_pva.update_layout(
            xaxis_title="Actual Temperature (°C)",
            yaxis_title="Predicted Temperature (°C)",
            height=500,
        )

        st.plotly_chart(fig_pva, use_container_width=True)

        # === Residuals Plots ===
        st.header("Residuals Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Residuals vs Predicted
            fig_rvp = go.Figure()

            fig_rvp.add_hline(y=0, line_dash="dash", line_color="red")

            fig_rvp.add_trace(go.Scatter(
                x=window_predictions["predicted"],
                y=window_predictions["error"],
                mode="markers",
                marker=dict(size=8, color="blue", opacity=0.6),
                text=window_predictions["date"].dt.strftime("%Y-%m-%d"),
                hovertemplate="Date: %{text}<br>Predicted: %{x:.1f}°C<br>Error: %{y:.2f}°C<extra></extra>",
            ))

            fig_rvp.update_layout(
                title="Residuals vs Predicted",
                xaxis_title="Predicted Temperature (°C)",
                yaxis_title="Residual (Predicted - Actual)",
                height=400,
            )

            st.plotly_chart(fig_rvp, use_container_width=True)

        with col2:
            # Residuals Distribution (Histogram + KDE)
            fig_hist = go.Figure()

            # Histogram
            fig_hist.add_trace(go.Histogram(
                x=window_predictions["error"],
                nbinsx=20,
                name="Error Distribution",
                opacity=0.7,
                histnorm="probability density",
            ))

            # KDE
            errors = window_predictions["error"].values
            kde_x = np.linspace(errors.min() - 0.5, errors.max() + 0.5, 100)
            kde = stats.gaussian_kde(errors)
            kde_y = kde(kde_x)

            fig_hist.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode="lines",
                name="KDE",
                line=dict(color="red", width=2),
            ))

            fig_hist.add_vline(x=0, line_dash="dash", line_color="green")

            fig_hist.update_layout(
                title="Error Distribution (Histogram + KDE)",
                xaxis_title="Error (°C)",
                yaxis_title="Density",
                height=400,
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        # === Residuals over Time ===
        st.header("Residuals Over Time")

        # Aggregate errors by date (multiple training pairs per date with max_gap_days > 1)
        daily_errors = window_predictions.groupby("date")["error"].mean().reset_index()

        fig_time = go.Figure()

        fig_time.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_time.add_trace(go.Bar(
            x=daily_errors["date"],
            y=daily_errors["error"],
            marker_color=["red" if e > 0 else "blue" for e in daily_errors["error"]],
            hovertemplate="Date: %{x}<br>Mean Error: %{y:.2f}°C<extra></extra>",
        ))

        fig_time.update_layout(
            xaxis_title="Date",
            yaxis_title="Mean Residual (°C)",
            height=400,
        )

        st.plotly_chart(fig_time, use_container_width=True)

        # === Temperature and Error Timeline ===
        st.header("Temperature Timeline")

        fig_timeline = go.Figure()

        fig_timeline.add_trace(go.Scatter(
            x=window_predictions["date"],
            y=window_predictions["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue"),
        ))

        fig_timeline.add_trace(go.Scatter(
            x=window_predictions["date"],
            y=window_predictions["predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="orange", dash="dash"),
        ))

        fig_timeline.add_trace(go.Scatter(
            x=window_predictions["date"],
            y=window_predictions["avg_air_temp"],
            mode="lines",
            name="Avg Air Temp",
            line=dict(color="red", width=1),
            opacity=0.5,
        ))

        fig_timeline.update_layout(
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            height=500,
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

        # === Raw Data Table ===
        st.header("Training Data Details")

        display_df = window_predictions[[
            "date", "days_gap", "prev_water_temp", "actual", "predicted", "error", "abs_error", "avg_air_temp"
        ]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df = display_df.rename(columns={
            "date": "Date",
            "days_gap": "Gap (days)",
            "prev_water_temp": "Start Water (°C)",
            "actual": "Actual (°C)",
            "predicted": "Predicted (°C)",
            "error": "Error (°C)",
            "abs_error": "Abs Error (°C)",
            "avg_air_temp": "Avg Air (°C)",
        })

        st.dataframe(
            display_df.style.format({
                "Gap (days)": "{:d}",
                "Start Water (°C)": "{:.1f}",
                "Actual (°C)": "{:.1f}",
                "Predicted (°C)": "{:.1f}",
                "Error (°C)": "{:+.2f}",
                "Abs Error (°C)": "{:.2f}",
                "Avg Air (°C)": "{:.1f}",
            }).background_gradient(subset=["Error (°C)"], cmap="RdYlGn_r", vmin=-2, vmax=2),
            use_container_width=True,
            height=400,
        )

        # === Error by Month ===
        st.header("Error Analysis by Month")

        monthly = predictions_df.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_stats = monthly.groupby("month").agg({
            "error": ["mean", "std", "count"],
            "abs_error": "mean",
        }).round(2)
        monthly_stats.columns = ["Mean Error", "Std Error", "Count", "MAE"]
        monthly_stats = monthly_stats.reset_index()

        fig_monthly = go.Figure()

        fig_monthly.add_trace(go.Bar(
            x=monthly_stats["month"],
            y=monthly_stats["Mean Error"],
            name="Mean Error (Bias)",
            marker_color=["red" if x > 0 else "blue" for x in monthly_stats["Mean Error"]],
        ))

        fig_monthly.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Mean Error (°C)",
            height=400,
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

        st.dataframe(monthly_stats, use_container_width=True)

    except DataLoadError as e:
        st.error(f"Cannot load data: {e}")


if __name__ == "__main__":
    main()
