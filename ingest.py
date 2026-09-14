"""
Fetch observed weather from Open-Meteo and store it in MotherDuck.

Run on a schedule so the dashboard never has to call Open-Meteo itself. The
archive hangs intermittently, and a hang in a cron job is a retry in an hour
rather than a blank page.

    python ingest.py              # trailing window, the hourly job
    python ingest.py --days 700   # backfill
"""

import argparse
import sys

import pandas as pd

from data import DataLoadError, load_historical_weather, today_utc
from forecast_storage import ForecastStorage, ForecastStorageError

# Open-Meteo's archive is ERA5T. Measured 2026-09-14: final reanalysis
# (models=era5) ran 6 days behind the data the default endpoint serves, so
# anything newer than that is preliminary and gets revised later. Re-fetching
# 10 days leaves 4 days of margin; appending instead of overwriting would
# freeze the preliminary values forever.
ERA5_PRELIMINARY_DAYS = 6
TRAILING_WINDOW_DAYS = 10


def weather_actuals_frame(start_date, end_date) -> pd.DataFrame:
    """The archive's hourly air, solar and cloud as one frame."""
    archive = load_historical_weather(start_date, end_date)
    merged = archive["hourly_air"].merge(
        archive["solar_cloud"], on="datetime", how="outer"
    )
    return merged.sort_values("datetime").reset_index(drop=True)


def ingest(days: int = TRAILING_WINDOW_DAYS, storage=None) -> int:
    """
    Fetch the trailing window and store it. Returns the number of hours stored.
    """
    end_date = today_utc()
    start_date = end_date - pd.Timedelta(days=days)

    frame = weather_actuals_frame(start_date, end_date)

    storage = storage or ForecastStorage()
    storage.initialize_schema()
    storage.store_weather_actuals(frame)

    return len(frame)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=TRAILING_WINDOW_DAYS,
        help=f"How many days back to fetch (default {TRAILING_WINDOW_DAYS}).",
    )
    args = parser.parse_args(argv)

    try:
        stored = ingest(days=args.days)
    except (DataLoadError, ForecastStorageError) as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        return 1

    print(f"Stored {stored} hours covering the last {args.days} days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
