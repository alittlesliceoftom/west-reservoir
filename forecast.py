import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import requests
from io import StringIO
from meteostat import Point, Daily


class WaterTemperatureModel:
    """
    Multi-component heat transfer model for water temperature prediction

    dT/dt = k_avg * (T_air_avg_yesterday - T_water) + 
            k_max * (T_air_max_yesterday - T_water) + 
            k_min * (T_air_min_yesterday - T_water)

    Where:
    - k_avg: heat transfer coefficient for average temperature (day^-1)
    - k_max: heat transfer coefficient for peak temperature (day^-1) 
    - k_min: heat transfer coefficient for minimum temperature (day^-1)
    """

    def __init__(self):
        self.k_avg = 0.03  # Heat transfer coefficient for average temperature
        self.k_max = 0.02  # Heat transfer coefficient for peak temperature (heating)
        self.k_min = 0.01  # Heat transfer coefficient for minimum temperature (cooling)
        self.baseline_temp = 12.0  # Annual mean water temperature


    def predict_temperature(self, air_temps_avg, air_temps_min, air_temps_max, initial_water_temp, dates):
        """
        Predict water temperature time series using multi-component heat transfer

        Parameters:
        -----------
        air_temps_avg : array-like
            Daily average air temperatures (°C)
        air_temps_min : array-like
            Daily minimum air temperatures (°C)
        air_temps_max : array-like
            Daily maximum air temperatures (°C)
        initial_water_temp : float
            Starting water temperature (°C)
        dates : array-like
            Corresponding dates for each temperature

        Returns:
        --------
        water_temps : np.array
            Predicted water temperatures
        """
        n_days = len(air_temps_avg)
        water_temps = np.zeros(n_days)
        water_temps[0] = initial_water_temp

        for i in range(1, n_days):
            # Multi-component heat transfer using yesterday's temperatures
            temp_diff_avg = air_temps_avg[i - 1] - water_temps[i - 1]
            temp_diff_max = air_temps_max[i - 1] - water_temps[i - 1]  # Peak heating effect
            temp_diff_min = air_temps_min[i - 1] - water_temps[i - 1]  # Cooling effect
            
            # Combined heat transfer rate
            dT_dt = (self.k_avg * temp_diff_avg + 
                    self.k_max * temp_diff_max + 
                    self.k_min * temp_diff_min)

            water_temps[i] = water_temps[i - 1] + dT_dt

            # Physical constraints
            water_temps[i] = max(0.1, water_temps[i])  # Above freezing

        return water_temps

    def fit_parameters(self, air_temps_avg, air_temps_min, air_temps_max, observed_water_temps, dates):
        """
        Fit model parameters to observed data using least squares

        Parameters:
        -----------
        air_temps_avg : array-like
            Historical average air temperatures
        air_temps_min : array-like
            Historical minimum air temperatures
        air_temps_max : array-like
            Historical maximum air temperatures
        observed_water_temps : array-like
            Historical water temperatures
        dates : array-like
            Corresponding dates
        """

        def objective(params):
            self.k_avg, self.k_max, self.k_min = params
            predicted = self.predict_temperature(
                air_temps_avg, air_temps_min, air_temps_max, observed_water_temps[0], dates
            )
            return np.sum((predicted - observed_water_temps) ** 2)

        # Parameter bounds: k_avg, k_max, k_min (0.001-0.2 each)
        bounds = [(0.001, 0.2), (0.001, 0.2), (0.001, 0.2)]
        initial_guess = [self.k_avg, self.k_max, self.k_min]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            self.k_avg, self.k_max, self.k_min = result.x
            return result
        else:
            print("Warning: Optimization failed")
            return None

    def forecast(
        self, air_temp_avg_forecast, air_temp_min_forecast, air_temp_max_forecast, 
        current_water_temp, start_date, forecast_days=14, 
        today_air_temp_avg=None, today_air_temp_min=None, today_air_temp_max=None
    ):
        """
        Generate water temperature forecast using multi-component heat transfer

        Parameters:
        -----------
        air_temp_avg_forecast : array-like
            Forecasted average air temperatures (starting from tomorrow)
        air_temp_min_forecast : array-like
            Forecasted minimum air temperatures (starting from tomorrow)
        air_temp_max_forecast : array-like
            Forecasted maximum air temperatures (starting from tomorrow)
        current_water_temp : float
            Current water temperature
        start_date : datetime
            Start date for forecast (tomorrow)
        forecast_days : int
            Number of days to forecast
        today_air_temp_avg : float, optional
            Today's average air temperature (needed for i-1 indexing)
        today_air_temp_min : float, optional
            Today's minimum air temperature (needed for i-1 indexing)
        today_air_temp_max : float, optional
            Today's maximum air temperature (needed for i-1 indexing)
        """
        dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
        
        # If today's air temps are provided, prepend them to the forecast arrays
        # This ensures air_temps[i-1] works correctly for tomorrow's prediction
        if all(temp is not None for temp in [today_air_temp_avg, today_air_temp_min, today_air_temp_max]):
            # Prepend today's temperatures to the forecast arrays
            air_temps_avg_with_today = [today_air_temp_avg] + list(air_temp_avg_forecast[:forecast_days])
            air_temps_min_with_today = [today_air_temp_min] + list(air_temp_min_forecast[:forecast_days])
            air_temps_max_with_today = [today_air_temp_max] + list(air_temp_max_forecast[:forecast_days])
            # Prepend today's date to the dates array
            today_date = start_date - timedelta(days=1)
            dates_with_today = [today_date] + dates
            
            # Predict temperatures using multi-component model
            predictions = self.predict_temperature(
                air_temps_avg_with_today, air_temps_min_with_today, air_temps_max_with_today,
                current_water_temp, dates_with_today
            )
            # Return only the forecast predictions (skip today's prediction)
            return predictions[1:]
        else:
            # Fallback: use average temps for all components if today's data not fully available
            return self.predict_temperature(
                air_temp_avg_forecast[:forecast_days], 
                air_temp_avg_forecast[:forecast_days],  # Use avg as fallback for min
                air_temp_avg_forecast[:forecast_days],  # Use avg as fallback for max
                current_water_temp, dates
            )


# Data loading functions
SHEET_URL = "https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/export?format=csv&gid=0"
RESERVOIR_LOCATION = Point(51.566938, -0.090492)  # West Reservoir coordinates


def load_reservoir_data():
    """Load reservoir temperature data from Google Sheets"""
    try:
        print("Loading reservoir data from Google Sheets...")
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        # Clean data similar to Streamlit app
        df.columns = ["Date", "Temperature"] + list(df.columns[2:])
        df = df[["Date", "Temperature"]]

        # Convert and clean
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
        df = df.dropna().sort_values("Date").reset_index(drop=True)

        print(f"✅ Loaded {len(df)} reservoir temperature readings")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        return df

    except Exception as e:
        print(f"❌ Error loading reservoir data: {e}")
        return pd.DataFrame()


def load_weather_data(start_date, end_date):
    """Load weather data for the specified date range"""
    try:
        print(f"Loading weather data from {start_date.date()} to {end_date.date()}...")

        weather_data = Daily(RESERVOIR_LOCATION, start_date, end_date)
        weather_df = weather_data.fetch()

        if not weather_df.empty:
            weather_df = weather_df.reset_index()
            weather_df = weather_df[["time", "tavg", "tmin", "tmax"]].copy()
            weather_df.columns = ["Date", "Air_Temperature", "Air_Temp_Min", "Air_Temp_Max"]
            weather_df = weather_df.dropna()

            print(f"✅ Loaded {len(weather_df)} weather readings")
            return weather_df
        else:
            print("❌ No weather data available")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error loading weather data: {e}")
        return pd.DataFrame()


def prepare_real_data():
    """Load and prepare real data for analysis"""
    # Load reservoir data
    reservoir_df = load_reservoir_data()
    if reservoir_df.empty:
        return None, None, None

    # Get weather data for the same period (with some buffer)
    start_date = reservoir_df["Date"].min() - timedelta(days=30)
    end_date = reservoir_df["Date"].max() + timedelta(days=30)

    weather_df = load_weather_data(start_date, end_date)
    if weather_df.empty:
        return None, None, None

    # Merge datasets
    merged_df = pd.merge(reservoir_df, weather_df, on="Date", how="inner")
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    if len(merged_df) < 30:
        print(f"❌ Not enough overlapping data: {len(merged_df)} points")
        return None, None, None

    print(
        f"✅ Merged dataset: {len(merged_df)} points with both water and air temperature"
    )

    dates = merged_df["Date"].dt.to_pydatetime()
    air_temps_avg = merged_df["Air_Temperature"].values
    air_temps_min = merged_df["Air_Temp_Min"].values
    air_temps_max = merged_df["Air_Temp_Max"].values
    water_temps = merged_df["Temperature"].values

    return dates, air_temps_avg, air_temps_min, air_temps_max, water_temps


# Example usage and testing
def generate_synthetic_data():
    """Generate synthetic data for testing"""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    n_days = len(dates)

    # Synthetic air temperature with seasonal cycle + noise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    air_temps_avg = (
        12
        + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + np.random.normal(0, 3, n_days)
    )
    
    # Generate min/max based on average with realistic daily variation
    daily_range = 3 + np.random.normal(0, 1, n_days)  # Variable daily temperature range
    air_temps_max = air_temps_avg + daily_range
    air_temps_min = air_temps_avg - daily_range * 0.6  # Min is closer to average than max

    # "True" water temperatures (for testing) - more damped, delayed
    water_temps = (
        12
        + 5 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
        + np.random.normal(0, 1, n_days)
    )

    return dates, air_temps_avg, air_temps_min, air_temps_max, water_temps


def demo_model(use_real_data=True):
    """Demonstrate the model with real or synthetic data"""

    if use_real_data:
        print("🌊 WEST RESERVOIR TEMPERATURE ANALYSIS")
        print("=" * 50)

        # Try to load real data
        data_result = prepare_real_data()
        if data_result[0] is not None:
            dates, air_temps, water_temps = data_result
            data_source = "Google Sheets"
            print(f"📊 Using real data from {data_source}")
        else:
            print("⚠️  Real data unavailable, falling back to synthetic data")
            dates, air_temps, water_temps = generate_synthetic_data()
            data_source = "Synthetic"
    else:
        dates, air_temps, water_temps = generate_synthetic_data()
        data_source = "Synthetic"
        print(f"📊 Using {data_source} data for testing")

    print(f"📈 Dataset: {len(dates)} data points")
    print(
        f"📅 Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Initialize and train model
    model = WaterTemperatureModel()

    # Split data: train on first 75%, test on last 25%
    split_idx = int(len(dates) * 0.75)

    train_dates = dates[:split_idx]
    train_air = air_temps[:split_idx]
    train_water = water_temps[:split_idx]

    test_dates = dates[split_idx:]
    test_air = air_temps[split_idx:]
    test_water = water_temps[split_idx:]

    print(f"\n🧠 TRAINING PHYSICS MODEL")
    print(f"Training period: {len(train_dates)} points")
    print(f"Testing period: {len(test_dates)} points")

    # Fit model
    print("\nFitting model parameters...")
    result = model.fit_parameters(train_air, train_air, train_air, train_water, train_dates)

    if result and result.success:
        print(f"\n✅ MODEL PARAMETERS (Optimized):")
        print(f"  🔧 Heat transfer coefficient (k_avg): {model.k_avg:.4f} day⁻¹")
        print(f"  🔧 Heat transfer coefficient (k_max): {model.k_max:.4f} day⁻¹")
        print(f"  🔧 Heat transfer coefficient (k_min): {model.k_min:.4f} day⁻¹")
        print(f"  🎯 Optimization success: {result.success}")
        print(f"  📉 Final cost: {result.fun:.2f}")
    else:
        print("⚠️  Optimization failed, using default parameters")

    # Generate predictions
    predicted_train = model.predict_temperature(train_air, train_air, train_air, train_water[0], train_dates)
    predicted_test = model.predict_temperature(
        test_air, test_air, test_air, predicted_train[-1], test_dates
    )

    # Calculate performance metrics
    train_rmse = np.sqrt(np.mean((predicted_train - train_water) ** 2))
    test_rmse = np.sqrt(np.mean((predicted_test - test_water) ** 2))
    train_mae = np.mean(np.abs(predicted_train - train_water))
    test_mae = np.mean(np.abs(predicted_test - test_water))

    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"  📈 Training RMSE: {train_rmse:.2f} °C")
    print(f"  📉 Testing RMSE: {test_rmse:.2f} °C")
    print(f"  📈 Training MAE: {train_mae:.2f} °C")
    print(f"  📉 Testing MAE: {test_mae:.2f} °C")

    # Additional statistics
    temp_range = np.max(water_temps) - np.min(water_temps)
    print(f"  🌡️  Temperature range in data: {temp_range:.1f} °C")
    print(f"  📊 Test RMSE as % of range: {(test_rmse / temp_range) * 100:.1f}%")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Main time series plot
    plt.subplot(3, 1, 1)
    plt.plot(
        train_dates, train_air, "b-", alpha=0.6, linewidth=1, label="Air Temperature"
    )
    plt.plot(train_dates, train_water, "ro", markersize=2, label="Observed Water Temp")
    plt.plot(
        train_dates, predicted_train, "g-", linewidth=2, label="Predicted Water Temp"
    )
    plt.title(
        f"Training Period - {data_source} Data (RMSE: {train_rmse:.2f}°C)", fontsize=14
    )
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(
        test_dates, test_air, "b-", alpha=0.6, linewidth=1, label="Air Temperature"
    )
    plt.plot(test_dates, test_water, "ro", markersize=2, label="Observed Water Temp")
    plt.plot(
        test_dates, predicted_test, "g-", linewidth=2, label="Predicted Water Temp"
    )
    plt.title(
        f"Testing Period - Model Validation (RMSE: {test_rmse:.2f}°C)", fontsize=14
    )
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scatter plot of predictions vs observations
    plt.subplot(3, 1, 3)
    plt.scatter(test_water, predicted_test, alpha=0.6, s=30, label="Test Data")
    plt.scatter(train_water, predicted_train, alpha=0.4, s=20, label="Train Data")

    # Perfect prediction line
    min_temp = min(np.min(water_temps), np.min(predicted_test))
    max_temp = max(np.max(water_temps), np.max(predicted_test))
    plt.plot(
        [min_temp, max_temp],
        [min_temp, max_temp],
        "r--",
        alpha=0.8,
        label="Perfect Prediction",
    )

    plt.xlabel("Observed Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title("Predicted vs Observed Water Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":
    # Run demonstration with real data
    model = demo_model(use_real_data=True)

    # Get the latest data for forecasting context
    data_result = prepare_real_data()
    if data_result[0] is not None:
        dates, air_temps, water_temps = data_result
        current_water_temp = water_temps[-1]
        latest_date = dates[-1]

        print(f"\n🔮 EXAMPLE FORECAST")
        print("=" * 30)
        print(f"📅 Starting from: {latest_date.strftime('%Y-%m-%d')}")
        print(f"🌡️  Current water temp: {current_water_temp:.1f}°C")

        # Generate example future air temperatures (seasonal appropriate)
        day_of_year = latest_date.timetuple().tm_yday
        base_temp = 12 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        future_air_temps = [base_temp + np.random.normal(0, 2) for _ in range(14)]

        start_date = latest_date + timedelta(days=1)
        forecast = model.forecast(future_air_temps, future_air_temps, future_air_temps, current_water_temp, start_date, 14)

        print(f"\n📈 14-Day Water Temperature Forecast:")
        print("Date        | Air Temp | Water Temp")
        print("-" * 36)

        for i, temp in enumerate(forecast):
            date = start_date + timedelta(days=i)
            air_temp = future_air_temps[i]
            print(f"{date.strftime('%Y-%m-%d')} | {air_temp:8.1f} | {temp:10.1f}")

        print(f"\n📊 Forecast Summary:")
        print(f"  🔥 Max predicted: {np.max(forecast):.1f}°C")
        print(f"  🧊 Min predicted: {np.min(forecast):.1f}°C")
        print(f"  📊 Avg predicted: {np.mean(forecast):.1f}°C")
        print(
            f"  📈 Temperature change: {forecast[-1] - current_water_temp:+.1f}°C over 14 days"
        )

    else:
        print("\n❌ Cannot generate realistic forecast - no real data available")

        # Fallback example
        print("\n🔮 EXAMPLE FORECAST (Synthetic)")
        print("=" * 35)
        future_air_temps = [15, 16, 14, 12, 13, 15, 17, 16, 15, 13, 12, 14, 16, 18]
        current_water_temp = 11.5
        start_date = datetime(2024, 6, 17)

        forecast = model.forecast(future_air_temps, future_air_temps, future_air_temps, current_water_temp, start_date, 14)

        print("Date        | Air Temp | Water Temp")
        print("-" * 36)

        for i, temp in enumerate(forecast):
            date = start_date + timedelta(days=i)
            air_temp = future_air_temps[i]
            print(f"{date.strftime('%Y-%m-%d')} | {air_temp:8.1f} | {temp:10.1f}")

    print("\n✅ Analysis complete!")
