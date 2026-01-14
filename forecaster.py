"""Water temperature forecasting using simple physics model"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict


class WaterTempForecaster:
    """
    Simple physics-based water temperature forecaster.

    Uses heat transfer equation: dT/dt = k * (T_air_yesterday - T_water)

    Where:
    - k: heat transfer coefficient (day^-1)
    - T_air_yesterday: Previous day's air temperature
    - T_water: Current water temperature
    """

    def __init__(self, heat_transfer_coeff: float = 0.05):
        """
        Initialize forecaster with heat transfer coefficient.

        Args:
            heat_transfer_coeff: Heat transfer coefficient k (default: 0.05 day^-1)
        """
        self.k = heat_transfer_coeff

    def fit(self, temperatures: pd.DataFrame) -> None:
        """
        Train the model on measured water temperatures.

        Optimizes the heat transfer coefficient k to minimize prediction error.

        Args:
            temperatures: DataFrame with columns: date, water_temp, air_temp, source
                         Only rows with source == 'MEASURED' will be used for training.
        """
        # Filter to measured data only
        training_data = temperatures[temperatures["source"] == "MEASURED"].copy()

        if len(training_data) < 10:
            # Not enough data to train, use default coefficient
            return

        # Sort by date
        training_data = training_data.sort_values("date").reset_index(drop=True)

        # Extract arrays for training
        air_temps = training_data["air_temp"].values
        water_temps = training_data["water_temp"].values

        def objective(params):
            """Objective function: minimize sum of squared errors"""
            k = params[0]

            # Predict water temps using current k
            predicted = np.zeros(len(water_temps))
            predicted[0] = water_temps[0]  # Start with first measurement

            for i in range(1, len(water_temps)):
                # Use yesterday's air temp (i-1) to predict today's water temp
                temp_diff = air_temps[i - 1] - predicted[i - 1]
                temp_change = k * temp_diff
                predicted[i] = predicted[i - 1] + temp_change

            # Return sum of squared errors
            return np.sum((predicted - water_temps) ** 2)

        # Optimize k (bounds: 0.01 to 0.5)
        bounds = [(0.01, 0.5)]
        initial_guess = [self.k]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            self.k = result.x[0]

    def predict_next_day(
        self, current_water_temp: float, yesterday_air_temp: float
    ) -> float:
        """
        Predict tomorrow's water temperature.

        Args:
            current_water_temp: Today's water temperature (°C)
            yesterday_air_temp: Yesterday's air temperature (°C)

        Returns:
            float: Predicted water temperature for tomorrow (°C)
        """
        # Calculate temperature difference
        temp_diff = yesterday_air_temp - current_water_temp

        # Calculate temperature change
        temp_change = self.k * temp_diff

        # Predicted temperature (no clamping)
        predicted = current_water_temp + temp_change

        return predicted

    def explain_prediction(
        self, current_water_temp: float, yesterday_air_temp: float
    ) -> Dict[str, float]:
        """
        Returns a breakdown of tomorrow's prediction for debugging.

        Args:
            current_water_temp: Today's water temperature (°C)
            yesterday_air_temp: Yesterday's air temperature (°C)

        Returns:
            dict: Dictionary containing:
                - current_water_temp: Current water temperature
                - yesterday_air_temp: Yesterday's air temperature
                - temperature_difference: Difference between air and water
                - heat_transfer_coefficient: Model's k value
                - temperature_change: Predicted change in water temp
                - predicted_water_temp: Tomorrow's predicted temperature
        """
        temp_diff = yesterday_air_temp - current_water_temp
        temp_change = self.k * temp_diff
        predicted = current_water_temp + temp_change

        return {
            "current_water_temp": current_water_temp,
            "yesterday_air_temp": yesterday_air_temp,
            "temperature_difference": temp_diff,
            "heat_transfer_coefficient": self.k,
            "temperature_change": temp_change,
            "predicted_water_temp": predicted,
        }

    def predict(self, temperatures: pd.DataFrame) -> pd.DataFrame:
        """
        Predict water temps for all rows where source == 'AIR_ONLY'.

        Iterates day by day, using each prediction as input for the next day.
        Updates the DataFrame in place.

        Args:
            temperatures: DataFrame with columns: date, water_temp, air_temp, source

        Returns:
            pd.DataFrame: Updated DataFrame with predictions filled in
        """
        result = temperatures.copy()

        # Sort by date to ensure correct temporal order
        result = result.sort_values("date").reset_index(drop=True)

        # Find where predictions start
        for i in range(len(result)):
            if result.loc[i, "source"] == "AIR_ONLY":
                # Get the previous row (most recent measured or predicted value)
                if i == 0:
                    # Can't predict if this is the first row
                    continue

                prev_row = result.iloc[i - 1]
                current_water_temp = prev_row["water_temp"]

                # Get yesterday's air temp (from previous row)
                yesterday_air_temp = prev_row["air_temp"]

                # Predict tomorrow's water temp
                predicted_water_temp = self.predict_next_day(
                    current_water_temp, yesterday_air_temp
                )

                # Update the DataFrame
                result.loc[i, "water_temp"] = predicted_water_temp
                result.loc[i, "source"] = "PREDICTED"

        return result
