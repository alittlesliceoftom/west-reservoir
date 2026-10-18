from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import insert

from db.models import Snapshot
from web.app import create_app
from web.config import Settings

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
JWT_SECRET = "test-secret-that-is-long-enough-for-hs256-signing"


@pytest.fixture
def settings():
    return Settings(
        env="test", site_url="http://testserver", supabase_url="https://example.supabase.co",
        supabase_anon_key="anon", supabase_jwt_secret=JWT_SECRET, dev_login=False,
    )


@pytest.fixture
def app(settings, session_factory):
    return create_app(settings=settings, session_factory=session_factory, clock=lambda: NOW)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def payload(**overrides) -> dict:
    """A snapshot payload as the worker writes it."""
    base = {
        "built_at": "2026-10-18T11:46:00+00:00",
        "water": {"name": "Test Lake", "slug": "test-lake", "water_type": "still", "depth_class": "lake",
                  "timezone": "Europe/London", "reading_time": "07:00"},
        "latest_reading": {"temp": 13.6, "at": "2026-10-18T06:00:00+00:00"},
        "estimate_now": {"temp": 13.8, "at": "2026-10-18T12:00:00+00:00"},
        "forecast": [
            {"at": f"2026-10-{d}T{'06' if d < 25 else '07'}:00:00+00:00", "temp": t, "lower": t - 0.7,
             "upper": t + 0.7, "horizon": d - 18}
            for d, t in zip(range(19, 24), (13.5, 13.4, 13.2, 13.1, 13.0))
        ],
        "trajectory": [["2026-10-18T06:00:00+00:00", 13.6], ["2026-10-18T09:00:00+00:00", 13.7]],
        "readings": [["2026-10-17T06:00:00+00:00", 13.9], ["2026-10-18T06:00:00+00:00", 13.6]],
        "air": [["2026-10-18T06:00:00+00:00", 9.1], ["2026-10-18T07:00:00+00:00", 9.8]],
        "calibration": {"method": "scale", "n_readings": 12, "n_legs": 11, "scale": 1.1},
        "train_mae": 0.21,
        "weather_fetched_at": "2026-10-18T11:45:00+00:00",
    }
    base.update(overrides)
    return base


def store_snapshot(db, water, data=None, built_at=None):
    data = data or payload()
    built = built_at or datetime.fromisoformat(data["built_at"])
    db.execute(insert(Snapshot).values(water_id=water.id, built_at=built, payload=data)
               .on_conflict_do_update(index_elements=["water_id"], set_={"built_at": built, "payload": data}))
    db.commit()
