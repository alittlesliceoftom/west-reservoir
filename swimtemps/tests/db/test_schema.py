from datetime import datetime, timezone

import pandas as pd
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from db import queries
from db.models import Base, Water
from tests.factories import add_reading, make_user, make_water

T0 = pd.Timestamp("2026-06-01 00:00", tz="UTC")


def weather(values, source_hours=3, start=T0):
    return pd.DataFrame({
        "valid_at": pd.date_range(start, periods=source_hours, freq="h"),
        "air_temp": values,
        "shortwave": [100.0] * source_hours,
        "cloud_cover": [50.0] * source_hours,
    })


def stored(db, water):
    return db.execute(text(
        "SELECT valid_at, source, air_temp FROM weather_hourly WHERE water_id = :w ORDER BY valid_at"
    ), {"w": water.id}).all()


def test_the_migration_matches_the_models(engine):
    with engine.connect() as conn:
        diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diffs == []


def test_archive_overwrites_forecast(db):
    water = make_water(db, make_user(db))
    queries.upsert_weather(db, water.id, weather([10.0, 11.0, 12.0]), "forecast")
    queries.upsert_weather(db, water.id, weather([9.0, 9.5, 9.9]), "archive")
    db.commit()
    rows = stored(db, water)
    assert [(r.source, r.air_temp) for r in rows] == [("archive", 9.0), ("archive", 9.5), ("archive", 9.9)]


def test_forecast_never_overwrites_archive(db):
    water = make_water(db, make_user(db))
    queries.upsert_weather(db, water.id, weather([9.0, 9.5, 9.9]), "archive")
    queries.upsert_weather(db, water.id, weather([10.0, 11.0, 12.0]), "forecast")
    db.commit()
    assert [r.air_temp for r in stored(db, water)] == [9.0, 9.5, 9.9]


def test_forecast_refreshes_forecast(db):
    water = make_water(db, make_user(db))
    queries.upsert_weather(db, water.id, weather([10.0, 11.0, 12.0]), "forecast")
    queries.upsert_weather(db, water.id, weather([13.0, 14.0, 15.0]), "forecast")
    db.commit()
    assert [r.air_temp for r in stored(db, water)] == [13.0, 14.0, 15.0]


def test_null_archive_hours_are_not_stored_over_a_forecast(db):
    water = make_water(db, make_user(db))
    queries.upsert_weather(db, water.id, weather([10.0, 11.0, 12.0]), "forecast")
    queries.upsert_weather(db, water.id, weather([9.0, None, None]), "archive")
    db.commit()
    assert [(r.source, r.air_temp) for r in stored(db, water)] == [
        ("archive", 9.0), ("forecast", 11.0), ("forecast", 12.0)]


def test_weather_frame_reads_back_what_was_stored(db):
    water = make_water(db, make_user(db))
    queries.upsert_weather(db, water.id, weather([10.0, 11.0, 12.0]), "forecast")
    db.commit()
    frame = queries.weather_frame(db, water.id)
    assert list(frame.columns) == ["valid_at", "air_temp", "shortwave", "cloud_cover"]
    assert frame["valid_at"].iloc[0] == T0
    assert frame["air_temp"].tolist() == [10.0, 11.0, 12.0]


def test_active_readings_exclude_soft_deleted_rows(db):
    user = make_user(db)
    water = make_water(db, user)
    add_reading(db, water, user, "2026-06-01 07:00", 14.2)
    gone = add_reading(db, water, user, "2026-06-02 07:00", 15.0)
    gone.deleted_at = datetime.now(timezone.utc)
    db.commit()
    frame = queries.active_readings(db, water.id)
    assert frame["temp_c"].tolist() == [14.2]
    assert isinstance(frame["temp_c"].iloc[0], float)
    assert str(frame["measured_at"].dt.tz) == "UTC"


def test_water_type_is_constrained(db):
    user = make_user(db)
    with pytest.raises(IntegrityError):
        make_water(db, user, water_type="sea")


def test_new_waters_start_dirty(db):
    water = make_water(db, make_user(db))
    assert db.get(Water, water.id).needs_recompute is True


def test_readings_outside_minus_two_to_forty_are_refused_by_the_database(db):
    user = make_user(db)
    water = make_water(db, user)
    with pytest.raises(IntegrityError):
        add_reading(db, water, user, "2026-06-02 07:00", 40.1)
