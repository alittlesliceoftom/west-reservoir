import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from feeds.clean import DEFAULT_THRESHOLDS, clean
from feeds.recipe import read_json_table
from feeds.thin import thin_hourly

FIXTURES = Path(__file__).parents[1] / "fixtures" / "feeds"
CLEVEDON = {"orientation": "columns", "time_path": "labels", "value_path": "datasets[0].data", "time_format": "epoch_ms"}
FAULT_DAY = datetime(2026, 10, 18, tzinfo=timezone.utc)
NO_AIR = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def sensor(sensor_id):
    payload = json.loads((FIXTURES / f"clevedon_{sensor_id}_168h_2026-10-18.json").read_text())
    return thin_hourly(read_json_table(payload, CLEVEDON, "Europe/London", None).rows)


@pytest.fixture(scope="module")
def air():
    hourly = json.loads((FIXTURES / "clevedon_air_2026-10-18.json").read_text())["hourly"]
    return pd.Series(hourly["temperature_2m"], index=pd.to_datetime(hourly["time"]).tz_localize("UTC"))


def flags(rows, air):
    """Each row's reason, keyed by its clock hour (the sensor stamps a few seconds past)."""
    return dict(zip([t.replace(minute=0, second=0) for t, _ in rows], clean(rows, [], air)))


def hours(day, first, last):
    return [day + timedelta(hours=h) for h in range(first, last + 1)]


@pytest.mark.parametrize("sensor_id", ["266367", "666332"])
def test_nothing_is_flagged_in_the_clean_days(sensor_id, air):
    result = flags(sensor(sensor_id), air)
    assert [at for at, reason in result.items() if at < FAULT_DAY and reason] == []


def test_every_faulty_hour_is_flagged_on_the_deep_sensor(air):
    result = flags(sensor("266367"), air)
    assert all(result[at] for at in hours(FAULT_DAY, 4, 8))


def test_the_deep_sensors_recovery_is_accepted(air):
    assert flags(sensor("266367"), air)[FAULT_DAY + timedelta(hours=9)] is None


def test_every_faulty_hour_is_flagged_on_the_shallow_sensor(air):
    result = flags(sensor("666332"), air)
    assert all(result[at] for at in hours(FAULT_DAY, 4, 9))


def test_a_reason_says_when_a_value_matches_the_air(air):
    reasons = [r for r in flags(sensor("266367"), air).values() if r]
    assert any("air temperature" in r for r in reasons)


def test_a_value_repeated_for_more_than_a_day_is_a_flatline():
    t0 = FAULT_DAY
    rows = [(t0 + timedelta(hours=h), 13.5) for h in range(26)]
    reasons = clean(rows, [], NO_AIR)
    assert reasons[:25] == [None] * 25
    assert "flatline" in reasons[25].lower()


def test_a_genuine_step_is_accepted_once_a_day_has_passed():
    t0 = FAULT_DAY
    history = [(t0 + timedelta(hours=h), 12.0) for h in range(-24, 0, 1)]
    history = [(at, 12.0 + 0.01 * i) for i, (at, _) in enumerate(history)]
    after = [(t0 + timedelta(hours=h), 15.0 + 0.01 * h) for h in range(0, 30)]
    reasons = clean(after, history, NO_AIR)
    assert reasons[0] is not None
    accepted = [i for i, r in enumerate(reasons) if r is None]
    assert accepted and accepted[0] <= 24
    assert all(r is None for r in reasons[accepted[0]:])


def test_good_history_is_the_reference():
    t0 = FAULT_DAY
    history = [(t0 + timedelta(hours=h), 14.0) for h in range(-5, 0)]
    assert clean([(t0, 10.0)], history, NO_AIR)[0] is not None
    assert clean([(t0, 14.3)], history, NO_AIR)[0] is None


def test_thresholds_can_be_overridden():
    t0 = FAULT_DAY
    history = [(t0 - timedelta(hours=1), 14.0)]
    assert clean([(t0, 16.5)], history, NO_AIR)[0] is not None
    loose = {**DEFAULT_THRESHOLDS, "jump_base": 5.0, "day_limit": 5.0}
    assert clean([(t0, 16.5)], history, NO_AIR, thresholds=loose)[0] is None
    assert clean([(t0, 16.5)], history, NO_AIR, thresholds={"jump_base": 5.0, "day_limit": 5.0})[0] is None


def test_defaults():
    assert DEFAULT_THRESHOLDS == {"jump_base": 2.0, "jump_per_hour": 0.1, "day_limit": 2.5,
                                  "flatline_hours": 24, "air_margin": 1.0}
