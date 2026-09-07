"""
Check that a dependency or interpreter change has not moved the forecast.

Run it under the current environment to capture a baseline, then under the
candidate environment and compare. Inputs are fetched once and frozen to disk,
so the only thing that varies between runs is Python and the libraries:

    env/bin/python debug/forecast_equivalence.py capture   # fetch, freeze, run
    env-new/bin/python debug/forecast_equivalence.py run   # reuse the inputs
    python debug/forecast_equivalence.py compare

Expect the simulation to agree exactly and the fitted coefficients not to.
`fit()` runs a scipy optimiser, and a different scipy converges to a slightly
different point in the basin, which carries into the forecast. Agreement to
around 1e-5 C is normal and is far below both the 0.1 C measurement resolution
and the model's own error. A difference large enough to see on the dashboard
means something real changed.

Artefacts land in debug/equivalence/, which is gitignored.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Running as debug/forecast_equivalence.py puts debug/ on sys.path, not the
# repo root, so the project modules are not importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).parent / "equivalence"
INPUT_WEATHER = OUT / "hourly_weather.csv"
INPUT_MEASURED = OUT / "measured.csv"

# %.17g round-trips an IEEE double exactly, so freezing the inputs through CSV
# cannot itself introduce a difference and make the comparison lie.
FLOAT_FMT = "%.17g"

FORECAST_DAYS = 5


def capture():
    """Fetch the real inputs and freeze them to disk."""
    from data import (
        build_hourly_weather,
        build_temperatures_frame,
        combine_hourly_temps,
        load_forecast_weather,
        load_historical_air_temps,
        load_historical_solar_cloud,
        load_hourly_air_temps,
        load_water_temps,
    )

    water_temps = load_water_temps()
    start = pd.Timestamp(water_temps["date"].min()).normalize()
    end = pd.Timestamp.now().normalize()

    air_hist = load_historical_air_temps(start, end)
    hourly_air = load_hourly_air_temps(start, end)
    solar_hist = load_historical_solar_cloud(start, end)
    forecast_weather = load_forecast_weather(days=FORECAST_DAYS)

    temperatures = build_temperatures_frame(water_temps, air_hist)

    # Archive and forecast air are combined before the weather frame is built.
    # Passing the archive alone makes build_hourly_weather's inner join drop
    # every future hour, and the forecast goes NaN past day one - vacuous
    # exactly where the physics runs.
    combined = combine_hourly_temps(hourly_air, forecast_weather)
    hourly_weather = build_hourly_weather(combined, solar_hist, forecast_weather)

    measured = temperatures[temperatures["source"] == "MEASURED"].copy()

    OUT.mkdir(exist_ok=True)
    hourly_weather.to_csv(INPUT_WEATHER, index=False, float_format=FLOAT_FMT)
    measured.to_csv(INPUT_MEASURED, index=False, float_format=FLOAT_FMT)
    print(f"captured {len(hourly_weather)} weather hours, {len(measured)} measurements")


def run():
    """Fit and forecast over the frozen inputs, and record the numbers."""
    from forecaster import WaterTempForecaster

    if not INPUT_WEATHER.exists():
        raise SystemExit("No frozen inputs. Run 'capture' first.")

    hourly_weather = pd.read_csv(INPUT_WEATHER, parse_dates=["datetime"])
    measured = pd.read_csv(INPUT_MEASURED, parse_dates=["date"])

    forecaster = WaterTempForecaster()
    forecaster.set_hourly_weather(hourly_weather)
    forecaster.fit(measured)

    anchor = measured["date"].max()
    anchor_temp = float(measured.loc[measured["date"] == anchor, "water_temp"].iloc[0])
    predictions = forecaster.predict_forward(
        anchor, anchor_temp, days_ahead=FORECAST_DAYS
    )

    # repr() of a float is exact, so a comparison of these strings is a
    # comparison of the bits, not of a rounded rendering.
    result = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "rows_weather": int(len(hourly_weather)),
        "rows_measured": int(len(measured)),
        "anchor": str(anchor),
        "anchor_temp": repr(anchor_temp),
        "k_air": repr(forecaster.k_air),
        "k_solar": repr(forecaster.k_solar),
        "k_cool": repr(forecaster.k_cool),
        "predictions": [
            {"horizon_days": int(r.horizon_days), "water_temp": repr(float(r.water_temp))}
            for r in predictions.itertuples()
        ],
    }

    out = OUT / f"results-{sys.version.split()[0]}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


def compare():
    """Diff every results-*.json in the output directory."""
    files = sorted(OUT.glob("results-*.json"))
    if len(files) < 2:
        raise SystemExit(f"Need two result files to compare, found {len(files)}.")

    results = [json.loads(f.read_text()) for f in files]
    a, b = results[0], results[-1]

    print(f"{'':16} {a['python'] + ' / pd ' + a['pandas']:>24}"
          f" {b['python'] + ' / pd ' + b['pandas']:>24}    delta")

    for key in ("k_air", "k_solar", "k_cool"):
        delta = float(b[key]) - float(a[key])
        print(f"{key:16} {a[key]:>24} {b[key]:>24}   {delta:+.3e}")

    print()
    worst = 0.0
    for pa, pb in zip(a["predictions"], b["predictions"]):
        delta = float(pb["water_temp"]) - float(pa["water_temp"])
        worst = max(worst, abs(delta))
        exact = "exact" if pa["water_temp"] == pb["water_temp"] else ""
        print(f"h={pa['horizon_days']}  {pa['water_temp']:>22}"
              f" {pb['water_temp']:>22}   {delta:+.2e}  {exact}")

    print(f"\nlargest forecast difference: {worst:.2e} C")
    if worst == 0:
        print("identical.")
    elif worst < 1e-3:
        print("below measurement resolution (0.1 C) - the optimiser landed "
              "somewhere marginally different.")
    else:
        print("LARGE ENOUGH TO MATTER. Investigate before changing the pins.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "capture":
        capture()
        run()
    elif mode == "compare":
        compare()
    else:
        run()
