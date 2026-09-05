"""Water temperature forecasting using hourly physics simulation"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Sequence
from datetime import datetime, timedelta


WEATHER_COLUMNS = ("air_temp", "shortwave_radiation", "cloud_cover")


class WaterTempForecaster:
    """
    Physics-based water temperature forecaster using hourly simulation.

    Per hour, water temperature is updated by three additive terms:

        clearness(t) = 1 - cloud_cover(t) / 100
        T_water(t+1h) = T_water(t)
                      + k_air   * (T_air(t)   - T_water(t))   # conduction/convection
                      + k_solar *  I(t)                        # shortwave heating
                      - k_cool  *  clearness(t)                # longwave cooling to clear sky

    Water temperature measurements are at 7am, so each training/prediction
    period spans 7am-to-7am (24 hours).

    When solar/cloud data is unavailable (e.g. via the legacy
    `set_hourly_air_temps` path), shortwave_radiation defaults to 0 and
    cloud_cover defaults to 100% (fully overcast → no radiative cooling).
    In that case the model collapses to the original single-term physics.
    """

    MEASUREMENT_HOUR = 7  # Water temp is measured at 7am

    def __init__(
        self,
        k_air: float = 0.02,
        k_solar: float = 5e-4,
        k_cool: float = 0.01,
        heat_transfer_coeff: Optional[float] = None,
    ):
        """
        Args:
            k_air: Air/water heat-transfer coefficient per hour
            k_solar: Solar heating coefficient (°C per W/m² per hour)
            k_cool: Clear-sky radiative cooling rate (°C per hour at 100% clearness)
            heat_transfer_coeff: Back-compat alias for k_air (legacy single-term init)
        """
        if heat_transfer_coeff is not None:
            k_air = heat_transfer_coeff
        self.k_air = k_air
        self.k_solar = k_solar
        self.k_cool = k_cool
        self.hourly_weather: Optional[pd.DataFrame] = None

    @property
    def k(self) -> float:
        """Back-compat alias for k_air."""
        return self.k_air

    def set_hourly_weather(self, hourly_weather: pd.DataFrame) -> None:
        """
        Set the hourly weather data for simulation.

        Args:
            hourly_weather: DataFrame with 'datetime' and the columns:
                            air_temp, shortwave_radiation, cloud_cover
        """
        df = hourly_weather.copy()
        for col in WEATHER_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"hourly_weather missing required column '{col}'")
        df = df[["datetime"] + list(WEATHER_COLUMNS)].copy()
        df = df.set_index("datetime").sort_index()
        # Forward-fill short gaps so an isolated missing hour doesn't kill a period
        df = df.ffill(limit=3)
        self.hourly_weather = df

    def set_hourly_air_temps(self, hourly_air_temps: pd.DataFrame) -> None:
        """
        Legacy entry point: accept air temps only and assume zero solar / full cloud.

        Args:
            hourly_air_temps: DataFrame with 'datetime' and 'air_temp' columns
        """
        df = hourly_air_temps[["datetime", "air_temp"]].copy()
        df["shortwave_radiation"] = 0.0
        df["cloud_cover"] = 100.0  # fully overcast → no radiative cooling term
        self.set_hourly_weather(df)

    def _get_weather_for_period(
        self, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        """Return weather rows in [start_dt, end_dt). Empty DataFrame if none."""
        if self.hourly_weather is None:
            return pd.DataFrame(columns=list(WEATHER_COLUMNS))

        mask = (self.hourly_weather.index >= start_dt) & (
            self.hourly_weather.index < end_dt
        )
        return self.hourly_weather.loc[mask].copy()

    def _get_hourly_temps_for_period(
        self, start_dt: datetime, end_dt: datetime
    ) -> List[float]:
        """Back-compat helper: return just hourly air temps for a period."""
        return self._get_weather_for_period(start_dt, end_dt)["air_temp"].tolist()

    @staticmethod
    def _step(
        water: float,
        air_temp: float,
        irradiance: float,
        cloud_cover: float,
        k_air: float,
        k_solar: float,
        k_cool: float,
    ) -> float:
        """Single hour update."""
        clearness = 1.0 - cloud_cover / 100.0
        return (
            water
            + k_air * (air_temp - water)
            + k_solar * irradiance
            - k_cool * clearness
        )

    def _simulate_period(
        self,
        start_water_temp: float,
        weather_slice: pd.DataFrame,
    ) -> float:
        """
        Run the 3-term hourly simulation across the supplied weather rows.
        """
        water = start_water_temp
        if weather_slice.empty:
            return water

        airs = weather_slice["air_temp"].to_numpy()
        sols = weather_slice["shortwave_radiation"].to_numpy()
        clouds = weather_slice["cloud_cover"].to_numpy()
        for i in range(len(airs)):
            water = self._step(
                water, airs[i], sols[i], clouds[i],
                self.k_air, self.k_solar, self.k_cool,
            )
        return water

    def _simulate_24h(
        self, start_water_temp: float, hourly_air_temps: Sequence[float]
    ) -> float:
        """
        Back-compat: simulate using only an air-temp list (zero solar, full cloud).
        """
        water = start_water_temp
        for air_temp in hourly_air_temps:
            water = self._step(
                water, air_temp, 0.0, 100.0,
                self.k_air, self.k_solar, self.k_cool,
            )
        return water

    def fit(self, temperatures: pd.DataFrame) -> None:
        """
        Fit k_air, k_solar, k_cool against measured water temperatures.

        Args:
            temperatures: DataFrame with columns: date, water_temp, source
        """
        if self.hourly_weather is None:
            return

        training_data = temperatures[temperatures["source"] == "MEASURED"].copy()
        if len(training_data) < 10:
            return

        training_data = training_data.sort_values("date").reset_index(drop=True)

        # Build training pairs as plain NumPy arrays for fast inner loop
        training_pairs = []
        for i in range(1, len(training_data)):
            prev_row = training_data.iloc[i - 1]
            curr_row = training_data.iloc[i]

            start_dt = pd.Timestamp(prev_row["date"]).replace(hour=self.MEASUREMENT_HOUR)
            end_dt = pd.Timestamp(curr_row["date"]).replace(hour=self.MEASUREMENT_HOUR)

            slice_df = self._get_weather_for_period(start_dt, end_dt)
            if len(slice_df) >= 20:
                training_pairs.append({
                    "start_water": float(prev_row["water_temp"]),
                    "airs": slice_df["air_temp"].to_numpy(),
                    "sols": slice_df["shortwave_radiation"].to_numpy(),
                    "clouds": slice_df["cloud_cover"].to_numpy(),
                    "actual_end": float(curr_row["water_temp"]),
                })

        if len(training_pairs) < 5:
            return

        def objective(params):
            k_air, k_solar, k_cool = params
            total_error = 0.0
            for pair in training_pairs:
                water = pair["start_water"]
                airs = pair["airs"]
                sols = pair["sols"]
                clouds = pair["clouds"]
                for i in range(len(airs)):
                    clearness = 1.0 - clouds[i] / 100.0
                    water += (
                        k_air * (airs[i] - water)
                        + k_solar * sols[i]
                        - k_cool * clearness
                    )
                total_error += (water - pair["actual_end"]) ** 2
            return total_error

        bounds = [
            (0.001, 0.1),       # k_air per hour
            (1e-5, 1e-3),       # k_solar °C per W/m² per hour
            (0.0, 0.1),         # k_cool °C per hour (≥ 0 cooling only)
        ]
        initial_guess = [self.k_air, self.k_solar, self.k_cool]
        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            self.k_air, self.k_solar, self.k_cool = (float(x) for x in result.x)

    def predict_next_day(
        self,
        current_water_temp: float,
        hourly_air_temps_or_weather,
    ) -> float:
        """
        Predict tomorrow's water temperature (7am to 7am).

        Args:
            current_water_temp: Today's water temperature at 7am
            hourly_air_temps_or_weather: Either a list of 24 hourly air temps
                (legacy path, assumes zero solar / full cloud) or a DataFrame
                with columns air_temp, shortwave_radiation, cloud_cover.

        Returns:
            Predicted water temperature 24h later.
        """
        if isinstance(hourly_air_temps_or_weather, pd.DataFrame):
            return self._simulate_period(current_water_temp, hourly_air_temps_or_weather)
        return self._simulate_24h(current_water_temp, hourly_air_temps_or_weather)

    def predict_forward(
        self,
        start_datetime,
        start_water_temp: float,
        targets=1,
    ) -> pd.DataFrame:
        """
        Forecast water temperature forward from a known starting point.

        Runs ONE continuous simulation, checkpointed at each target, rather
        than re-simulating per target.

        Args:
            start_datetime: Anchor time. Normalised to MEASUREMENT_HOUR (7am),
                            since water temps are measured at 7am.
            start_water_temp: Known water temperature at the anchor.
            targets: Either an int n (forecast 1..n days ahead), or a sequence
                     of dates/datetimes. Duplicates are kept as zero-length
                     legs rather than de-duplicated, so callers get exactly one
                     row per target they asked for.

        Returns:
            DataFrame with columns:
                target_datetime (Timestamp, at 7am)
                horizon_days    (int, days from the anchor)
                water_temp      (float, NaN where weather coverage is missing)
                has_weather     (bool, whether that leg had any weather data)
        """
        columns = ["target_datetime", "horizon_days", "water_temp", "has_weather"]

        start_dt = pd.Timestamp(start_datetime).normalize() + pd.Timedelta(
            hours=self.MEASUREMENT_HOUR
        )

        if isinstance(targets, (int, np.integer)):
            target_dts = [
                start_dt + pd.Timedelta(days=i) for i in range(1, int(targets) + 1)
            ]
        else:
            target_dts = sorted(
                pd.Timestamp(t).normalize() + pd.Timedelta(hours=self.MEASUREMENT_HOUR)
                for t in targets
            )

        target_dts = [t for t in target_dts if t >= start_dt]

        if not target_dts:
            return pd.DataFrame({c: [] for c in columns}).astype(
                {"horizon_days": "int64", "water_temp": "float64", "has_weather": "bool"}
            )

        rows = []
        water = float(start_water_temp)
        cursor = start_dt

        for target_dt in target_dts:
            weather_slice = self._get_weather_for_period(cursor, target_dt)
            has_weather = not weather_slice.empty

            if has_weather:
                water = self._simulate_period(water, weather_slice)
            else:
                # No data for this leg: refuse to fabricate, and stay NaN onward.
                water = float("nan")

            rows.append({
                "target_datetime": target_dt,
                "horizon_days": int((target_dt - start_dt) / pd.Timedelta(days=1)),
                "water_temp": water,
                "has_weather": has_weather,
            })
            cursor = target_dt

        return pd.DataFrame(rows, columns=columns)

    def explain_prediction(
        self,
        current_water_temp: float,
        weather_slice=None,
        hourly_air_temps: Optional[Sequence[float]] = None,
    ) -> Dict:
        """
        Returns a per-hour breakdown of the simulation including the
        contribution of each physics term.

        Pass either `weather_slice` (preferred, full 3-term model) or
        `hourly_air_temps` (legacy, assumes zero solar / full cloud).
        For back-compat, a list passed positionally as `weather_slice`
        is treated as `hourly_air_temps`.
        """
        # Back-compat: positional list goes to legacy path
        if weather_slice is not None and not isinstance(weather_slice, pd.DataFrame):
            hourly_air_temps = weather_slice
            weather_slice = None

        if weather_slice is None and hourly_air_temps is not None:
            n = len(hourly_air_temps)
            weather_slice = pd.DataFrame({
                "air_temp": list(hourly_air_temps),
                "shortwave_radiation": [0.0] * n,
                "cloud_cover": [100.0] * n,
            })

        if weather_slice is None or weather_slice.empty:
            return {
                "current_water_temp": current_water_temp,
                "hours_simulated": 0,
                "predicted_water_temp": current_water_temp,
                "hourly_breakdown": [],
            }

        airs = weather_slice["air_temp"].to_numpy()
        sols = weather_slice["shortwave_radiation"].to_numpy()
        clouds = weather_slice["cloud_cover"].to_numpy()

        water = current_water_temp
        breakdown = []
        for i in range(len(airs)):
            hour = (self.MEASUREMENT_HOUR + i) % 24
            clearness = 1.0 - clouds[i] / 100.0
            dT_air = self.k_air * (airs[i] - water)
            dT_solar = self.k_solar * sols[i]
            dT_cool = -self.k_cool * clearness
            total_change = dT_air + dT_solar + dT_cool
            new_water = water + total_change
            breakdown.append({
                "hour": hour,
                "air_temp": float(airs[i]),
                "shortwave_radiation": float(sols[i]),
                "cloud_cover": float(clouds[i]),
                "water_temp_before": float(water),
                "dT_air": float(dT_air),
                "dT_solar": float(dT_solar),
                "dT_cool": float(dT_cool),
                "temp_change": float(total_change),
                "water_temp_after": float(new_water),
            })
            water = new_water

        return {
            "current_water_temp": current_water_temp,
            "hours_simulated": len(airs),
            "air_temp_avg": float(airs.mean()),
            "air_temp_min": float(airs.min()),
            "air_temp_max": float(airs.max()),
            "solar_avg": float(sols.mean()),
            "solar_max": float(sols.max()),
            "cloud_avg": float(clouds.mean()),
            "k_air": self.k_air,
            "k_solar": self.k_solar,
            "k_cool": self.k_cool,
            "heat_transfer_coefficient": self.k_air,  # back-compat alias
            "total_temp_change": float(water - current_water_temp),
            "predicted_water_temp": float(water),
            "hourly_breakdown": breakdown,
        }

    def fill_predictions(self, temperatures: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in predicted water temps for rows where source == 'AIR_ONLY'.

        Each maximal run of consecutive AIR_ONLY rows is anchored on the row
        immediately before it and forecast in one pass via predict_forward.
        Equivalent to chaining day by day, because each AIR_ONLY row was
        already chained from the previous row's value.
        """
        result = temperatures.copy()
        result = result.sort_values("date").reset_index(drop=True)

        i = 0
        while i < len(result):
            if result.loc[i, "source"] != "AIR_ONLY" or i == 0:
                i += 1
                continue

            # Collect this maximal run of consecutive AIR_ONLY rows.
            run_start = i
            while i < len(result) and result.loc[i, "source"] == "AIR_ONLY":
                i += 1
            run_end = i  # exclusive

            anchor = result.iloc[run_start - 1]
            predictions = self.predict_forward(
                start_datetime=anchor["date"],
                start_water_temp=anchor["water_temp"],
                targets=list(result.loc[run_start:run_end - 1, "date"]),
            )

            for offset, row_idx in enumerate(range(run_start, run_end)):
                prediction = predictions.iloc[offset]
                # Legs without weather stay untouched, as before.
                if prediction["has_weather"]:
                    result.loc[row_idx, "water_temp"] = prediction["water_temp"]
                    result.loc[row_idx, "source"] = "PREDICTED"

        return result
