from datetime import timedelta

from sqlalchemy import func, select

from db.models import Heartbeat, Reading, Snapshot, Water
from tests.factories import make_user, make_water
from tests.worker.test_cycle import NOW, FakeOpenMeteo
from worker import cycle, main, seed


def test_the_first_pass_sweeps_and_writes_a_heartbeat(db, session_factory):
    make_water(db, make_user(db))
    client = FakeOpenMeteo()
    assert main.run_once(session_factory, client, NOW, last_sweep=None) == NOW
    assert len(client.forecast_calls) == 1
    assert db.scalar(select(func.count()).select_from(Heartbeat)) == 1


def test_between_sweeps_only_dirty_waters_are_touched(db, session_factory):
    make_water(db, make_user(db), needs_recompute=False)
    client = FakeOpenMeteo()
    later = NOW + timedelta(minutes=10)
    assert main.run_once(session_factory, client, later, last_sweep=NOW) == NOW
    assert client.forecast_calls == []
    assert db.scalar(select(func.count()).select_from(Heartbeat)) == 1


def test_an_hour_after_the_last_sweep_it_sweeps_again(db, session_factory):
    make_water(db, make_user(db), needs_recompute=False)
    client = FakeOpenMeteo()
    later = NOW + timedelta(hours=1)
    assert main.run_once(session_factory, client, later, last_sweep=NOW) == later
    assert len(client.forecast_calls) == 1


def test_seed_creates_west_reservoir_once_and_it_computes(db, session_factory):
    assert seed.seed(session_factory).startswith("seeded West Reservoir with 416")
    assert seed.seed(session_factory) == "west-reservoir already exists"
    water = db.scalar(select(Water).where(Water.slug == "west-reservoir"))
    assert db.scalar(select(func.count()).select_from(Reading)) == 416

    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW)
    payload = db.get(Snapshot, water.id).payload
    assert payload["calibration"]["method"] == "full"
    assert payload["calibration"]["n_readings"] == 416
    assert len(payload["forecast"]) == 5
