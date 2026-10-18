import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select, text

from db.models import ForecastPoint, ForecastRun, Heartbeat, ModelFit, Snapshot, Water
from forecast.openmeteo import OpenMeteoError, parse_hourly
from tests.factories import add_reading, make_user, make_water
from worker import cycle

FIXTURES = Path(__file__).parents[1] / "fixtures"
# The recorded forecast was fetched at 2026-10-17 23:55 UTC: it covers
# 2026-10-10 00:00 to 2026-10-23 23:00 UTC.
NOW = pd.Timestamp("2026-10-18 01:00", tz="UTC").to_pydatetime()


class FakeOpenMeteo:
    def __init__(self, fail=False):
        self.forecast_calls = []
        self.archive_calls = []
        self.fail = fail
        self.recorded = parse_hourly(json.loads((FIXTURES / "openmeteo_forecast.json").read_text()))

    def forecast(self, lat, lon):
        self.forecast_calls.append((lat, lon))
        if self.fail:
            raise OpenMeteoError("Open-Meteo request failed: HTTP 502")
        return self.recorded

    def archive(self, lat, lon, start, end):
        self.archive_calls.append((lat, lon, start, end))
        hours = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23), freq="h")
        return pd.DataFrame({"valid_at": hours, "air_temp": 12.0, "shortwave": 0.0, "cloud_cover": 80.0})


def daily_readings(db, water, user, temps, last="2026-10-17 06:00"):
    end = pd.Timestamp(last, tz="UTC")
    for i, temp in enumerate(temps):
        add_reading(db, water, user, (end - pd.Timedelta(days=len(temps) - 1 - i)).to_pydatetime(), temp)


@pytest.fixture
def owner(db):
    return make_user(db)


def snapshot(db, water) -> dict:
    db.expire_all()
    return db.get(Snapshot, water.id).payload


def test_a_dirty_water_gets_weather_fit_run_points_and_a_snapshot(db, session_factory, owner):
    water = make_water(db, owner, name="Test Lake")
    daily_readings(db, water, owner, [13.0, 12.8, 12.7, 12.5])
    client = FakeOpenMeteo()

    assert cycle.drain_dirty(session_factory, client, NOW) == 1

    db.expire_all()
    assert db.get(Water, water.id).needs_recompute is False
    fit = db.scalars(select(ModelFit)).one()
    assert (fit.method, fit.n_legs) == ("scale", 3)
    run = db.scalars(select(ForecastRun)).one()
    assert run.anchor_temp == 12.5
    assert run.anchor_at == pd.Timestamp("2026-10-17 06:00", tz="UTC")
    points = db.scalars(select(ForecastPoint).order_by(ForecastPoint.valid_at)).all()
    assert len(points) == 5
    assert points[0].valid_at == pd.Timestamp("2026-10-18 06:00", tz="UTC")  # today, 07:00 BST
    assert all(p.lower < p.water_temp < p.upper for p in points)

    payload = snapshot(db, water)
    assert payload["latest_reading"] == {"temp": 12.5, "at": "2026-10-17T06:00:00+00:00"}
    assert 5 < payload["estimate_now"]["temp"] < 20
    assert payload["estimate_now"]["at"] == "2026-10-18T01:00:00+00:00"
    assert len(payload["forecast"]) == 5
    assert payload["forecast"][0]["horizon"] == 1  # anchor 17th 06:00 UTC, first target 18th 06:00 UTC
    assert payload["calibration"] == {"method": "scale", "n_readings": 4, "n_legs": 3, "scale": pytest.approx(fit.scale)}
    assert payload["water"]["name"] == "Test Lake"
    assert len(payload["readings"]) == 4
    assert payload["air"] and payload["trajectory"]
    assert payload["weather_fetched_at"] is not None


def test_a_water_with_no_readings_gets_a_snapshot_saying_so(db, session_factory, owner):
    water = make_water(db, owner)
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW)
    payload = snapshot(db, water)
    assert payload["latest_reading"] is None
    assert payload["estimate_now"] is None
    assert payload["forecast"] == []
    assert payload["calibration"]["method"] == "prior"
    assert db.scalar(select(func.count()).select_from(ForecastRun)) == 0


def test_a_reading_saved_during_a_compute_still_triggers_a_recompute(db, session_factory, owner, monkeypatch):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8])
    original = cycle.compute_water

    def compute_while_a_reading_arrives(session, w, client, now, **kwargs):
        with session_factory() as other:
            other.execute(text("UPDATE waters SET needs_recompute = true WHERE id = :id"), {"id": w.id})
            other.commit()
        return original(session, w, client, now, **kwargs)

    monkeypatch.setattr(cycle, "compute_water", compute_while_a_reading_arrives)
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW)
    db.expire_all()
    assert db.get(Water, water.id).needs_recompute is True


def test_a_failed_compute_leaves_the_water_dirty_and_the_others_run(db, session_factory, owner, monkeypatch):
    broken = make_water(db, owner, name="Broken", lat=10.0, lon=10.0)
    fine = make_water(db, owner, name="Fine")
    original = cycle.compute_water

    def fail_for_broken(session, w, client, now, **kwargs):
        if w.id == broken.id:
            raise RuntimeError("boom")
        return original(session, w, client, now, **kwargs)

    monkeypatch.setattr(cycle, "compute_water", fail_for_broken)
    assert cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW) == 1
    db.expire_all()
    assert db.get(Water, broken.id).needs_recompute is True
    assert db.get(Water, fine.id).needs_recompute is False
    assert db.get(Snapshot, fine.id) is not None


def test_open_meteo_down_still_builds_a_snapshot_from_stored_weather(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8, 12.7])
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW)
    add_reading(db, water, owner, pd.Timestamp("2026-10-17 18:00", tz="UTC").to_pydatetime(), 12.9)
    db.execute(text("UPDATE waters SET needs_recompute = true"))
    db.commit()

    later = NOW + timedelta(hours=2)
    cycle.sweep(session_factory, FakeOpenMeteo(fail=True), later)

    payload = snapshot(db, water)
    assert payload["latest_reading"]["temp"] == 12.9
    assert payload["built_at"] == later.isoformat()


def test_waters_within_the_same_rounded_coordinates_share_one_forecast_call(db, session_factory, owner):
    a = make_water(db, owner, name="North Pond", lat=51.4500, lon=-2.6000)
    b = make_water(db, owner, name="South Pond", lat=51.4505, lon=-2.6010)
    client = FakeOpenMeteo()
    cycle.sweep(session_factory, client, NOW)
    assert client.forecast_calls == [(51.45, -2.6)]
    for water in (a, b):
        stored = db.scalar(text("SELECT count(*) FROM weather_hourly WHERE water_id = :w"), {"w": water.id})
        assert stored == 336


def test_readings_older_than_seven_days_trigger_an_archive_backfill(db, session_factory, owner):
    water = make_water(db, owner)
    add_reading(db, water, owner, pd.Timestamp("2026-09-20 06:00", tz="UTC").to_pydatetime(), 16.0)
    client = FakeOpenMeteo()
    cycle.drain_dirty(session_factory, client, NOW)
    assert len(client.archive_calls) == 1
    _, _, start, end = client.archive_calls[0]
    assert start <= pd.Timestamp("2026-09-20").date()
    assert end >= pd.Timestamp("2026-10-10").date()
    sources = dict(db.execute(text(
        "SELECT source, count(*) FROM weather_hourly WHERE water_id = :w GROUP BY source"), {"w": water.id}).all())
    assert sources["archive"] > 0 and sources["forecast"] > 0


def test_recent_readings_need_no_archive_call(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8])
    client = FakeOpenMeteo()
    cycle.drain_dirty(session_factory, client, NOW)
    assert client.archive_calls == []


def test_the_fit_is_reused_until_readings_change(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8, 12.7])
    client = FakeOpenMeteo()
    cycle.sweep(session_factory, client, NOW)
    cycle.sweep(session_factory, client, NOW + timedelta(hours=1))
    assert db.scalar(select(func.count()).select_from(ModelFit)) == 1
    assert db.scalar(select(func.count()).select_from(ForecastRun)) == 2

    add_reading(db, water, owner, pd.Timestamp("2026-10-17 18:00", tz="UTC").to_pydatetime(), 12.9,
                created_at=NOW + timedelta(minutes=90))
    cycle.sweep(session_factory, client, NOW + timedelta(hours=2))
    assert db.scalar(select(func.count()).select_from(ModelFit)) == 2


def test_a_day_old_fit_is_refreshed(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8, 12.7])
    client = FakeOpenMeteo()
    cycle.sweep(session_factory, client, NOW)
    cycle.sweep(session_factory, client, NOW + timedelta(hours=25))
    assert db.scalar(select(func.count()).select_from(ModelFit)) == 2


def test_a_dirty_drain_right_after_a_sweep_does_not_refetch_the_forecast(db, session_factory, owner):
    water = make_water(db, owner)
    client = FakeOpenMeteo()
    cycle.sweep(session_factory, client, NOW)
    db.execute(text("UPDATE waters SET needs_recompute = true WHERE id = :id"), {"id": water.id})
    db.commit()
    cycle.drain_dirty(session_factory, client, NOW + timedelta(minutes=5))
    assert len(client.forecast_calls) == 1


def test_archived_waters_are_skipped(db, session_factory, owner):
    water = make_water(db, owner, archived_at=NOW)
    client = FakeOpenMeteo()
    cycle.sweep(session_factory, client, NOW)
    assert client.forecast_calls == []
    assert db.get(Snapshot, water.id) is None


def test_each_cycle_writes_a_heartbeat_and_prunes_old_ones(db, session_factory):
    db.add(Heartbeat(at=NOW - timedelta(days=30)))
    db.commit()
    cycle.write_heartbeat(session_factory, NOW)
    beats = db.scalars(select(Heartbeat.at)).all()
    assert beats == [NOW]


# --- final review fixes ---------------------------------------------------

def test_a_new_water_gets_a_snapshot_even_when_open_meteo_is_down(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8])
    client = FakeOpenMeteo(fail=True)
    assert cycle.drain_dirty(session_factory, client, NOW) == 1
    payload = snapshot(db, water)
    assert payload["latest_reading"]["temp"] == 12.8
    assert payload["estimate_now"] is None and payload["forecast"] == []
    assert db.get(Water, water.id).needs_recompute is False


def test_a_reading_a_few_minutes_ahead_of_the_worker_clock_is_used(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8])
    add_reading(db, water, owner, NOW + timedelta(minutes=5), 12.4)
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW)
    assert snapshot(db, water)["latest_reading"]["temp"] == 12.4


class GarbledOpenMeteo(FakeOpenMeteo):
    def forecast(self, lat, lon):
        self.forecast_calls.append((lat, lon))
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_an_unexpected_weather_error_does_not_stop_the_sweep(db, session_factory, owner):
    water = make_water(db, owner)
    daily_readings(db, water, owner, [13.0, 12.8])
    assert cycle.sweep(session_factory, GarbledOpenMeteo(), NOW) == 1
    assert snapshot(db, water)["latest_reading"]["temp"] == 12.8


def test_the_archive_backfill_goes_back_at_most_three_years(db, session_factory, owner):
    water = make_water(db, owner)
    add_reading(db, water, owner, pd.Timestamp("2021-06-01 06:00", tz="UTC").to_pydatetime(), 16.0)
    client = FakeOpenMeteo()
    cycle.drain_dirty(session_factory, client, NOW)
    assert client.archive_calls == []

    add_reading(db, water, owner, pd.Timestamp("2026-09-20 06:00", tz="UTC").to_pydatetime(), 16.0)
    db.execute(text("UPDATE waters SET needs_recompute = true"))
    db.commit()
    cycle.drain_dirty(session_factory, client, NOW + timedelta(hours=1))
    assert len(client.archive_calls) == 1
    assert client.archive_calls[0][2] >= pd.Timestamp("2023-10-01").date()
