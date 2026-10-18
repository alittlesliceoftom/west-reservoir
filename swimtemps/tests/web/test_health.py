from datetime import timedelta

from db.models import Heartbeat
from db import queries
from tests.factories import make_user, make_water
from tests.web.conftest import NOW, store_snapshot
from tests.db.test_schema import weather


def healthy_water(db, at=NOW):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water, built_at=at - timedelta(minutes=20))
    queries.upsert_weather(db, water.id, weather([10.0, 11.0, 12.0]), "forecast", fetched_at=at - timedelta(minutes=20))
    db.add(Heartbeat(at=at - timedelta(seconds=15)))
    db.commit()
    return water


def test_healthy(client, db):
    healthy_water(db)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_a_snapshot_older_than_three_hours_fails(client, db):
    water = healthy_water(db)
    store_snapshot(db, water, built_at=NOW - timedelta(hours=3, minutes=1))
    response = client.get("/healthz")
    assert response.status_code == 503
    assert "snapshot" in response.json()["failures"][0]


def test_an_active_water_with_no_snapshot_fails(client, db):
    healthy_water(db)
    make_water(db, make_user(db, email="b@example.com"), name="Never Built")
    assert client.get("/healthz").status_code == 503


def test_open_meteo_silent_for_three_hours_fails(client, db):
    healthy_water(db, at=NOW)
    db.execute(queries.text("UPDATE weather_hourly SET fetched_at = :t"), {"t": NOW - timedelta(hours=4)})
    db.commit()
    response = client.get("/healthz")
    assert response.status_code == 503
    assert any("Open-Meteo" in f for f in response.json()["failures"])


def test_a_silent_worker_fails(client, db):
    healthy_water(db)
    db.execute(queries.text("UPDATE heartbeat SET at = :t"), {"t": NOW - timedelta(hours=4)})
    db.commit()
    assert any("worker" in f for f in client.get("/healthz").json()["failures"])


def test_no_waters_only_needs_the_worker(client, db):
    db.add(Heartbeat(at=NOW))
    db.commit()
    assert client.get("/healthz").status_code == 200


def feed_water(db, last_good, state="on"):
    from tests.factories import make_feed
    water = healthy_water(db)
    water.feed_mode = "feed"
    make_feed(db, water, make_user(db, email="f@example.com"), state=state, last_good_reading_at=last_good)
    db.commit()
    return water


def test_a_feed_without_a_good_reading_for_three_days_fails(client, db):
    feed_water(db, NOW - timedelta(days=3, minutes=1))
    response = client.get("/healthz")
    assert response.status_code == 503
    assert "feed for Test Lake has had no good reading for 3 days" in response.json()["failures"]


def test_a_feed_with_a_recent_good_reading_is_healthy(client, db):
    feed_water(db, NOW - timedelta(days=2, hours=23))
    assert client.get("/healthz").status_code == 200


def test_a_paused_feed_does_not_fail(client, db):
    feed_water(db, NOW - timedelta(days=30), state="paused")
    assert client.get("/healthz").status_code == 200


def test_a_feed_water_switched_to_manual_does_not_fail(client, db):
    water = feed_water(db, NOW - timedelta(days=30), state="off")
    water.feed_mode = "manual"
    db.commit()
    assert client.get("/healthz").status_code == 200
