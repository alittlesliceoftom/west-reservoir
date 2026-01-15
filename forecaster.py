"""Water temperature forecasting using hourly physics simulation"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class WaterTempForecaster:
    """
    Physics-based water temperature forecaster using hourly simulation.

    Uses heat transfer equation applied hour-by-hour:
        T_water(t+1h) = T_water(t) + k * (T_air(t) - T_water(t))

    Water temperature measurements are at 7am, so the model simulates
    the 24 hours from 7am to 7am to predict the next day's reading.
    """

    MEASUREMENT_HOUR = 7  # Water temp is measured at 7am

    def __init__(self, heat_transfer_coeff: float = 0.02):
        """
        Initialize forecaster with hourly heat transfer coefficient.

        Args:
            heat_transfer_coeff: Heat transfer coefficient k per hour
                                 (default: 0.02, meaning ~40% daily response)
        """
        self.k = heat_transfer_coeff
        self.hourly_air_temps: Optional[pd.DataFrame] = None

    def set_hourly_air_temps(self, hourly_air_temps: pd.DataFrame) -> None:
        """
        Set the hourly air temperature data for simulation.

        Args:
            hourly_air_temps: DataFrame with 'datetime' and 'air_temp' columns
        """
        self.hourly_air_temps = hourly_air_temps.copy()
        self.hourly_air_temps = self.hourly_air_temps.set_index("datetime").sort_index()

    def _get_hourly_temps_for_period(
        self, start_dt: datetime, end_dt: datetime
    ) -> List[float]:
        """
        Get hourly air temperatures for a time period.

        Args:
            start_dt: Start datetime (inclusive)
            end_dt: End datetime (exclusive)

        Returns:
            List of hourly air temperatures
        """
        if self.hourly_air_temps is None:
            return []

        # Get hourly temps between start and end
        mask = (self.hourly_air_temps.index >= start_dt) & (
            self.hourly_air_temps.index < end_dt
        )
        temps = self.hourly_air_temps.loc[mask, "air_temp"].tolist()
        return temps

    def _simulate_24h(
        self, start_water_temp: float, hourly_air_temps: List[float]
    ) -> float:
        """
        Simulate water temperature change over a period using hourly air temps.

        Args:
            start_water_temp: Starting water temperature
            hourly_air_temps: List of hourly air temperatures

        Returns:
            Final water temperature after simulation
        """
        water_temp = start_water_temp

        for air_temp in hourly_air_temps:
            temp_diff = air_temp - water_temp
            water_temp += self.k * temp_diff

        return water_temp

    def fit(self, temperatures: pd.DataFrame) -> None:
        """
        Train the model on measured water temperatures using hourly simulation.

        Optimizes the heat transfer coefficient k to minimize prediction error.

        Args:
            temperatures: DataFrame with columns: date, water_temp, source
                         Only rows with source == 'MEASURED' will be used.
        """
        if self.hourly_air_temps is None:
            return

        # Filter to measured data only
        training_data = temperatures[temperatures["source"] == "MEASURED"].copy()

        if len(training_data) < 10:
            return

        # Sort by date
        training_data = training_data.sort_values("date").reset_index(drop=True)

        # Build training pairs: (start_water_temp, hourly_airs, actual_end_temp)
        training_pairs = []

        for i in range(1, len(training_data)):
            prev_row = training_data.iloc[i - 1]
            curr_row = training_data.iloc[i]

            prev_date = prev_row["date"]
            curr_date = curr_row["date"]

            # Get 7am datetime for each measurement
            start_dt = pd.Timestamp(prev_date).replace(hour=self.MEASUREMENT_HOUR)
            end_dt = pd.Timestamp(curr_date).replace(hour=self.MEASUREMENT_HOUR)

            # Get hourly temps for this period
            hourly_temps = self._get_hourly_temps_for_period(start_dt, end_dt)

            if len(hourly_temps) >= 20:  # Need at least ~20 hours of data
                training_pairs.append(
                    {
                        "start_water": prev_row["water_temp"],
                        "hourly_airs": hourly_temps,
                        "actual_end": curr_row["water_temp"],
                    }
                )

        if len(training_pairs) < 5:
            return

        def objective(params):
            """Objective function: minimize sum of squared errors"""
            k = params[0]
            total_error = 0

            for pair in training_pairs:
                # Simulate with this k value
                water_temp = pair["start_water"]
                for air_temp in pair["hourly_airs"]:
                    water_temp += k * (air_temp - water_temp)

                # Add squared error
                total_error += (water_temp - pair["actual_end"]) ** 2

            return total_error

        # Optimize k (bounds: 0.001 to 0.1 per hour)
        bounds = [(0.001, 0.1)]
        initial_guess = [self.k]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            self.k = result.x[0]

    def predict_next_day(
        self, current_water_temp: float, hourly_air_temps: List[float]
    ) -> float:
        """
        Predict tomorrow's water temperature using hourly simulation.

        Args:
            current_water_temp: Today's water temperature at 7am
            hourly_air_temps: List of hourly air temps from 7am to 7am (24 values)

        Returns:
            Predicted water temperature for tomorrow at 7am
        """
        return self._simulate_24h(current_water_temp, hourly_air_temps)

    def explain_prediction(
        self, current_water_temp: float, hourly_air_temps: List[float]
    ) -> Dict:
        """
        Returns a detailed breakdown of the 24-hour simulation.

        Args:
            current_water_temp: Today's water temperature at 7am
            hourly_air_temps: List of hourly air temps from 7am to 7am

        Returns:
            dict: Dictionary containing simulation details
        """
        if not hourly_air_temps:
            return {
                "current_water_temp": current_water_temp,
                "hours_simulated": 0,
                "predicted_water_temp": current_water_temp,
                "hourly_breakdown": [],
            }

        # Simulate and track each hour
        water_temp = current_water_temp
        hourly_breakdown = []

        for i, air_temp in enumerate(hourly_air_temps):
            hour = (self.MEASUREMENT_HOUR + i) % 24
            temp_diff = air_temp - water_temp
            temp_change = self.k * temp_diff
            new_water_temp = water_temp + temp_change

            hourly_breakdown.append(
                {
                    "hour": hour,
                    "air_temp": air_temp,
                    "water_temp_before": water_temp,
                    "temp_change": temp_change,
                    "water_temp_after": new_water_temp,
                }
            )

            water_temp = new_water_temp

        return {
            "current_water_temp": current_water_temp,
            "hours_simulated": len(hourly_air_temps),
            "air_temp_avg": sum(hourly_air_temps) / len(hourly_air_temps),
            "air_temp_min": min(hourly_air_temps),
            "air_temp_max": max(hourly_air_temps),
            "heat_transfer_coefficient": self.k,
            "total_temp_change": water_temp - current_water_temp,
            "predicted_water_temp": water_temp,
            "hourly_breakdown": hourly_breakdown,
        }

    def predict(self, temperatures: pd.DataFrame) -> pd.DataFrame:
        """
        Predict water temps for all rows where source == 'AIR_ONLY'.

        Uses hourly simulation for each day.

        Args:
            temperatures: DataFrame with columns: date, water_temp, air_temp, source

        Returns:
            pd.DataFrame: Updated DataFrame with predictions filled in
        """
        result = temperatures.copy()
        result = result.sort_values("date").reset_index(drop=True)

        for i in range(len(result)):
            if result.loc[i, "source"] == "AIR_ONLY":
                if i == 0:
                    continue

                prev_row = result.iloc[i - 1]
                curr_date = result.loc[i, "date"]

                # Get start water temp from previous day
                current_water_temp = prev_row["water_temp"]

                # Get 7am to 7am period
                start_dt = pd.Timestamp(prev_row["date"]).replace(
                    hour=self.MEASUREMENT_HOUR
                )
                end_dt = pd.Timestamp(curr_date).replace(hour=self.MEASUREMENT_HOUR)

                # Get hourly temps for this period
                hourly_temps = self._get_hourly_temps_for_period(start_dt, end_dt)

                if hourly_temps:
                    predicted = self._simulate_24h(current_water_temp, hourly_temps)
                else:
                    # Fallback: use daily air temp with equivalent daily k
                    daily_k = 1 - (1 - self.k) ** 24
                    air_temp = result.loc[i, "air_temp"]
                    if pd.notna(air_temp):
                        predicted = current_water_temp + daily_k * (
                            air_temp - current_water_temp
                        )
                    else:
                        predicted = current_water_temp

                result.loc[i, "water_temp"] = predicted
                result.loc[i, "source"] = "PREDICTED"

        return result
