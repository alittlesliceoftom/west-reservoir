"""Debug script to check today's data in the DataFrame"""
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/Users/asliceoftom/Documents/projects/claude/project-1-west-res')

# Import after path is set
from data import load_water_temps, load_historical_air_temps, load_hourly_air_temps, load_forecast_air_temps, DataLoadError
from forecaster import WaterTempForecaster

today = datetime.now().date()
print(f"Today: {today}\n")

# Replicate app's exact data flow
water_temps = load_water_temps()
start_date = pd.Timestamp(water_temps["date"].min()).normalize()
end_date = pd.Timestamp.now().normalize()
air_temps_hist = load_historical_air_temps(start_date, end_date)
hourly_air_temps = load_hourly_air_temps(start_date, end_date)

# Fill missing daily temps from hourly
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

try:
    forecast = load_forecast_air_temps(days=5)
    forecast["source"] = "AIR_ONLY"
    print(f"Forecast first date: {forecast['date'].min().date()}")
    temperatures = pd.concat([temperatures, forecast], ignore_index=True)
    temperatures = temperatures.sort_values("date").reset_index(drop=True)
except DataLoadError as e:
    print(f"Forecast load failed: {e}\n")

# Train and predict
forecaster = WaterTempForecaster()
forecaster.set_hourly_air_temps(hourly_air_temps)
forecaster.fit(temperatures[temperatures["source"] == "MEASURED"])
temperatures = forecaster.predict(temperatures)

# Check today's row in final DataFrame
today_data = temperatures[temperatures["date"].dt.date == today]
print("="*80)
print(f"TODAY'S DATA IN FINAL DATAFRAME ({today}):")
print("="*80)
if today_data.empty:
    print("❌ NO DATA FOR TODAY!")
else:
    print(today_data[["date", "water_temp", "air_temp", "air_temp_min", "air_temp_max", "source"]].to_string())
    print()
    
    # Check if min/max exist
    has_min = today_data["air_temp_min"].notna().any()
    has_max = today_data["air_temp_max"].notna().any()
    
    if has_min and has_max:
        print("✅ Today has air_temp_min and air_temp_max")
        print(f"   Range: {today_data['air_temp_min'].values[0]:.1f}°C to {today_data['air_temp_max'].values[0]:.1f}°C")
    else:
        print(f"❌ Missing min/max: has_min={has_min}, has_max={has_max}")
        print("   This is why error bars don't show!")

print("\n" + "="*80)
print("LAST 5 ROWS OF DATAFRAME:")
print("="*80)
print(temperatures.tail()[["date", "water_temp", "air_temp", "air_temp_min", "air_temp_max", "source"]].to_string())
