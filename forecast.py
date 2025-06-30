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
    Degree-day accumulation model for water temperature prediction

    dT/dt = k * (T_air - T_water) + seasonal_offset(DOY)

    Where:
    - k: heat transfer coefficient (day^-1)
    - seasonal_offset: sinusoidal adjustment for solar/seasonal effects
    """

    def __init__(self):
        self.k = 0.05  # Heat transfer coefficient
        self.seasonal_amp = 2.0  # Amplitude of seasonal variation (°C)
        self.seasonal_phase = 100  # Phase shift (day of year)
        self.baseline_temp = 12.0  # Annual mean water temperature

    def seasonal_offset(self, day_of_year):
        """Calculate seasonal temperature offset based on day of year"""
        # Peak warming around day 180 (late June), cooling around day 360
        return self.seasonal_amp * np.sin(
            2 * np.pi * (day_of_year - self.seasonal_phase) / 365
        )

    def predict_temperature(self, air_temps, initial_water_temp, dates):
        """
        Predict water temperature time series

        Parameters:
        -----------
        air_temps : array-like
            Daily air temperatures (°C)
        initial_water_temp : float
            Starting water temperature (°C)
        dates : array-like
            Corresponding dates for each temperature

        Returns:
        --------
        water_temps : np.array
            Predicted water temperatures
        """
        n_days = len(air_temps)
        water_temps = np.zeros(n_days)
        water_temps[0] = initial_water_temp

        for i in range(1, n_days):
            day_of_year = dates[i].timetuple().tm_yday

            # Seasonal offset calculation
            seasonal = self.seasonal_offset(day_of_year)

            # Core differential equation: dT/dt = k*(T_air - T_water) + seasonal
            temp_diff = air_temps[i - 1] - water_temps[i - 1]
            dT_dt = self.k * temp_diff + seasonal / 365  # Daily rate

            water_temps[i] = water_temps[i - 1] + dT_dt

            # Physical constraints
            water_temps[i] = max(0.1, water_temps[i])  # Above freezing

        return water_temps

    def fit_parameters(self, air_temps, observed_water_temps, dates):
        """
        Fit model parameters to observed data using least squares

        Parameters:
        -----------
        air_temps : array-like
            Historical air temperatures
        observed_water_temps : array-like
            Historical water temperatures
        dates : array-like
            Corresponding dates
        """

        def objective(params):
            self.k, self.seasonal_amp, self.seasonal_phase = params
            predicted = self.predict_temperature(
                air_temps, observed_water_temps[0], dates
            )
            return np.sum((predicted - observed_water_temps) ** 2)

        # Parameter bounds: k (0.01-0.2), seasonal_amp (0-5), phase (0-365)
        bounds = [(0.01, 0.5), (0, 5), (0, 365)]
        initial_guess = [self.k, self.seasonal_amp, self.seasonal_phase]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            self.k, self.seasonal_amp, self.seasonal_phase = result.x
            return result
        else:
            print("Warning: Optimization failed")
            return None

    def forecast(
        self, air_temp_forecast, current_water_temp, start_date, forecast_days=14
    ):
        """
        Generate water temperature forecast

        Parameters:
        -----------
        air_temp_forecast : array-like
            Forecasted air temperatures
        current_water_temp : float
            Current water temperature
        start_date : datetime
            Start date for forecast
        forecast_days : int
            Number of days to forecast
        """
        dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
        return self.predict_temperature(
            air_temp_forecast[:forecast_days], current_water_temp, dates
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
            weather_df = weather_df[["time", "tavg"]].copy()
            weather_df.columns = ["Date", "Air_Temperature"]
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
    air_temps = merged_df["Air_Temperature"].values
    water_temps = merged_df["Temperature"].values

    return dates, air_temps, water_temps


# Example usage and testing
def generate_synthetic_data():
    """Generate synthetic data for testing"""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    n_days = len(dates)

    # Synthetic air temperature with seasonal cycle + noise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    air_temps = (
        12
        + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + np.random.normal(0, 3, n_days)
    )

    # "True" water temperatures (for testing) - more damped, delayed
    water_temps = (
        12
        + 5 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
        + np.random.normal(0, 1, n_days)
    )

    return dates, air_temps, water_temps


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
    result = model.fit_parameters(train_air, train_water, train_dates)

    if result and result.success:
        print(f"\n✅ MODEL PARAMETERS (Optimized):")
        print(f"  🔧 Heat transfer coefficient (k): {model.k:.4f} day⁻¹")
        print(f"  🌊 Seasonal amplitude: {model.seasonal_amp:.2f} °C")
        print(f"  📅 Seasonal phase shift: {model.seasonal_phase:.0f} days")
        print(f"  🎯 Optimization success: {result.success}")
        print(f"  📉 Final cost: {result.fun:.2f}")
    else:
        print("⚠️  Optimization failed, using default parameters")

    # Generate predictions
    predicted_train = model.predict_temperature(train_air, train_water[0], train_dates)
    predicted_test = model.predict_temperature(
        test_air, predicted_train[-1], test_dates
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
        forecast = model.forecast(future_air_temps, current_water_temp, start_date, 14)

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

        forecast = model.forecast(future_air_temps, current_water_temp, start_date, 14)

        print("Date        | Air Temp | Water Temp")
        print("-" * 36)

        for i, temp in enumerate(forecast):
            date = start_date + timedelta(days=i)
            air_temp = future_air_temps[i]
            print(f"{date.strftime('%Y-%m-%d')} | {air_temp:8.1f} | {temp:10.1f}")

    print("\n✅ Analysis complete!")
