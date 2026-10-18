import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select

from db import queries
from db.models import Feed, ForecastRun, Reading, Snapshot, Water
from feeds.fetch import FetchError
from feeds.recipe import read_json_table
from feeds.thin import thin_hourly
from tests.factories import add_feed_reading, make_feed, make_user, make_water
from tests.worker.test_cycle import FakeOpenMeteo
from worker import cycle
from worker.feeds import poll_feed, poll_feeds

FIXTURES = Path(__file__).parents[1] / "fixtures" / "feeds"
CLEVEDON = {"url": "https://lake.example/temps", "orientation": "columns", "time_path": "labels",
            "value_path": "datasets[0].data", "time_format": "epoch_ms"}
NOW = pd.Timestamp("2026-10-18 09:30", tz="UTC").to_pydatetime()
LAST_GOOD = pd.Timestamp("2026-10-18 09:00:03", tz="UTC")


def deep():
    return json.loads((FIXTURES / "clevedon_266367_168h_2026-10-18.json").read_text())


def columns(*rows):
    """A Clevedon-shaped payload from (UTC time string, °C) pairs."""
    return {"labels": [int(pd.Timestamp(at, tz="UTC").timestamp() * 1000) for at, _ in rows],
            "datasets": [{"data": [t for _, t in rows]}]}


def fetching(payload):
    calls = []

    def fetch(url):
        calls.append(url)
        return payload() if callable(payload) else payload

    fetch.calls = calls
    return fetch


@pytest.fixture
def owner(db):
    return make_user(db)


@pytest.fixture
def feed(db, owner):
    water = make_water(db, owner, feed_mode="feed", needs_recompute=False)
    return make_feed(db, water, owner, config=CLEVEDON)


def standing(db, feed):
    return db.scalars(select(Reading).where(Reading.feed_id == feed.id, Reading.deleted_at.is_(None))
                      .order_by(Reading.measured_at)).all()


def fresh(db, model, key):
    db.expire_all()
    return db.get(model, key)


def test_the_first_poll_stores_hourly_rows_with_their_flags(db, feed):
    expected = thin_hourly(read_json_table(deep(), CLEVEDON, "Europe/London", None).rows)
    result = poll_feed(db, feed, NOW, fetch=fetching(deep()))
    db.commit()
    rows = standing(db, feed)
    assert (result.new, result.corrected) == (len(expected), 0)
    assert [r.measured_at for r in rows] == [at for at, _ in expected]
    assert all(r.source == "feed" and r.submitted_by is None for r in rows)
    flagged = [r for r in rows if r.flagged]
    assert result.flagged == len(flagged) == 5
    assert all(r.flag_reason for r in flagged)
    stored = fresh(db, Feed, feed.id)
    assert stored.last_polled_at == stored.last_success_at == NOW
    assert stored.last_error is None
    assert pd.Timestamp(stored.last_good_reading_at) == LAST_GOOD
    assert fresh(db, Water, feed.water_id).needs_recompute is True


def test_a_second_identical_poll_adds_nothing(db, feed):
    poll_feed(db, feed, NOW, fetch=fetching(deep()))
    db.commit()
    db.execute(Water.__table__.update().values(needs_recompute=False))
    db.commit()
    result = poll_feed(db, feed, NOW + timedelta(hours=1), fetch=fetching(deep()))
    db.commit()
    assert (result.new, result.corrected) == (0, 0)
    assert fresh(db, Water, feed.water_id).needs_recompute is False


def test_a_new_minute_in_an_hour_already_stored_adds_nothing(db, feed):
    poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 08:00", 12.4))))
    db.commit()
    result = poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 08:05", 12.3), ("2026-10-18 08:45", 12.5))))
    db.commit()
    assert result.new == 0
    assert [float(r.temp_c) for r in standing(db, feed)] == [12.4]


def test_a_changed_value_is_stored_as_a_correction(db, feed):
    poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 07:00", 12.4), ("2026-10-18 08:00", 12.5))))
    db.commit()
    result = poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 07:00", 12.4), ("2026-10-18 08:00", 12.7))))
    db.commit()
    assert (result.new, result.corrected) == (0, 1)
    old = db.scalar(select(Reading).where(Reading.feed_id == feed.id, Reading.deleted_at.is_not(None)))
    new = db.get(Reading, old.replaced_by)
    assert float(old.temp_c) == 12.5 and old.deleted_at == NOW
    assert float(new.temp_c) == 12.7 and new.deleted_at is None and new.measured_at == old.measured_at


def test_rows_that_fail_the_manual_checks_are_not_stored(db, feed):
    payload = columns(("1999-12-31 07:00", 12.0), ("2026-10-18 06:00", 45.0), ("2026-10-18 07:00", 12.4),
                      ("2026-10-18 11:00", 12.4))
    result = poll_feed(db, feed, NOW, fetch=fetching(payload))
    db.commit()
    assert result.new == 1 and len(result.errors) == 3
    assert [float(r.temp_c) for r in standing(db, feed)] == [12.4]


def test_values_are_stored_to_a_tenth(db, feed):
    poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 07:00", 12.44))))
    db.commit()
    result = poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 07:00", 12.41))))
    assert result.corrected == 0


def test_a_feed_sending_only_flagged_rows_does_not_advance_its_last_good_reading(db, feed):
    good = [(f"2026-10-17 {h:02d}:00", 12.7) for h in range(10, 20)]
    poll_feed(db, feed, NOW, fetch=fetching(columns(*good)))
    db.commit()
    result = poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 05:00", 8.0), ("2026-10-18 06:00", 8.1))))
    db.commit()
    assert result.new == 2 and result.flagged == 2
    assert pd.Timestamp(fresh(db, Feed, feed.id).last_good_reading_at) == pd.Timestamp("2026-10-17 19:00", tz="UTC")


def test_cleaning_compares_with_good_readings_already_stored(db, feed):
    poll_feed(db, feed, NOW, fetch=fetching(columns(*[(f"2026-10-17 {h:02d}:00", 12.7) for h in range(10, 20)])))
    db.commit()
    result = poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-17 20:00", 9.0))))
    assert result.flagged == 1


def test_cleaning_notes_a_value_matching_stored_air(db, feed):
    frame = pd.DataFrame({"valid_at": pd.date_range("2026-10-17 10:00", periods=12, freq="h", tz="UTC"),
                          "air_temp": 9.2, "shortwave": 0.0, "cloud_cover": 50.0})
    queries.upsert_weather(db, feed.water_id, frame, "forecast")
    db.commit()
    poll_feed(db, feed, NOW, fetch=fetching(columns(*[(f"2026-10-17 {h:02d}:00", 12.7) for h in range(10, 20)])))
    db.commit()
    poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-17 20:00", 9.0))))
    db.commit()
    reading = db.scalar(select(Reading).where(Reading.flagged.is_(True)))
    assert "air temperature" in reading.flag_reason


def test_a_recipe_error_is_recorded_and_the_next_feed_still_polls(db, session_factory, owner, feed):
    other_water = make_water(db, owner, name="Other Lake", feed_mode="feed")
    other = make_feed(db, other_water, owner, config={**CLEVEDON, "url": "https://other.example/temps"})

    def fetch(url):
        return {"wrong": 1} if url == CLEVEDON["url"] else deep()

    assert poll_feeds(session_factory, NOW, fetch=fetch) == 1
    broken = fresh(db, Feed, feed.id)
    assert "labels" in broken.last_error and broken.last_polled_at == NOW and broken.last_success_at is None
    assert db.scalar(select(func.count()).select_from(Reading).where(Reading.feed_id == other.id)) > 100
    assert fresh(db, Feed, other.id).last_error is None


def test_a_fetch_error_is_recorded(db, session_factory, feed):
    def fetch(url):
        raise FetchError("The feed answered 503 instead of 200.")

    poll_feeds(session_factory, NOW, fetch=fetch)
    assert fresh(db, Feed, feed.id).last_error == "The feed answered 503 instead of 200."


def test_an_unexpected_error_in_one_feed_never_stops_the_others(db, session_factory, owner, feed):
    other_water = make_water(db, owner, name="Other Lake", feed_mode="feed")
    other = make_feed(db, other_water, owner, config={**CLEVEDON, "url": "https://other.example/temps"})

    def fetch(url):
        if url == CLEVEDON["url"]:
            raise RuntimeError("boom")
        return deep()

    assert poll_feeds(session_factory, NOW, fetch=fetch) == 1
    assert fresh(db, Feed, feed.id).last_error
    assert fresh(db, Feed, other.id).last_success_at == NOW


def test_a_success_clears_the_last_error(db, feed):
    feed.last_error = "The feed answered 503 instead of 200."
    db.commit()
    poll_feed(db, feed, NOW, fetch=fetching(deep()))
    db.commit()
    assert fresh(db, Feed, feed.id).last_error is None


@pytest.mark.parametrize("state, mode, archived", [("paused", "feed", False), ("off", "manual", False),
                                                    ("on", "manual", False), ("on", "feed", True)])
def test_only_feeds_that_are_on_for_a_feed_water_are_polled(db, session_factory, feed, state, mode, archived):
    feed.state = state
    water = db.get(Water, feed.water_id)
    water.feed_mode = mode
    water.archived_at = NOW if archived else None
    db.commit()
    fetch = fetching(deep())
    assert poll_feeds(session_factory, NOW, fetch=fetch) == 0
    assert fetch.calls == []


def test_the_sweep_polls_feeds_before_computing(db, session_factory, feed):
    fetch = fetching(deep())
    cycle.sweep(session_factory, FakeOpenMeteo(), NOW, fetch=fetch)
    assert fetch.calls == [CLEVEDON["url"]]
    payload = db.get(Snapshot, feed.water_id).payload
    assert payload["latest_reading"]["at"].startswith("2026-10-18T09:00")


def test_a_drain_polls_a_feed_that_has_never_been_polled(db, session_factory, feed):
    fetch = fetching(deep())
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW, fetch=fetch)
    assert fetch.calls == [CLEVEDON["url"]]
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), NOW + timedelta(minutes=1), fetch=fetch)
    assert len(fetch.calls) == 1


# --- quiet and paused --------------------------------------------------------

def compute(db, feed, now):
    water = db.get(Water, feed.water_id)
    payload = cycle.compute_water(db, water, FakeOpenMeteo(), now)
    db.commit()
    return payload


@pytest.fixture
def polled(db, feed):
    poll_feed(db, feed, NOW, fetch=fetching(deep()))
    db.commit()
    return feed


def test_a_fresh_feed_is_not_quiet(db, polled):
    payload = compute(db, polled, NOW)
    assert payload["feed"] == {
        "source_name": "Example Lake", "source_link": "https://example-lake.org/live/", "state": "on",
        "last_good_reading_at": LAST_GOOD.isoformat(), "quiet": False, "paused_forecast": False,
    }
    assert payload["estimate_now"] is not None and payload["forecast"]


def test_a_feed_is_quiet_after_twelve_hours_and_still_forecasts(db, polled):
    payload = compute(db, polled, (LAST_GOOD + timedelta(hours=13)).to_pydatetime())
    assert payload["feed"]["quiet"] is True and payload["feed"]["paused_forecast"] is False
    assert payload["estimate_now"] is not None and payload["forecast"]


def test_the_forecast_pauses_after_five_days_without_a_good_reading(db, polled):
    runs = db.scalar(select(func.count()).select_from(ForecastRun))
    payload = compute(db, polled, (LAST_GOOD + timedelta(days=5, minutes=1)).to_pydatetime())
    assert payload["feed"]["paused_forecast"] is True
    assert payload["estimate_now"] is None and payload["forecast"] == [] and payload["trajectory"] == []
    assert payload["latest_reading"] is not None and payload["readings"]
    assert db.scalar(select(func.count()).select_from(ForecastRun)) == runs


def test_a_paused_feed_has_no_forecast(db, polled):
    polled.state = "paused"
    db.commit()
    payload = compute(db, polled, NOW)
    assert payload["feed"]["state"] == "paused" and payload["feed"]["paused_forecast"] is True
    assert payload["estimate_now"] is None and payload["forecast"] == []


def test_chart_readings_carry_flag_reasons(db, polled):
    payload = compute(db, polled, NOW)
    reasons = [r[2] for r in payload["readings"]]
    assert sum(r is not None for r in reasons) == 5
    assert all(len(r) == 4 for r in payload["readings"])


def test_an_accepted_flag_is_noted_on_the_chart(db, polled):
    reading = db.scalar(select(Reading).where(Reading.flagged.is_(True)).order_by(Reading.measured_at))
    reading.flag_override = "accept"
    db.commit()
    payload = compute(db, polled, NOW)
    item = next(r for r in payload["readings"] if r[0] == pd.Timestamp(reading.measured_at).tz_convert("UTC").isoformat())
    assert item[2] is None and "accepted by the water's admin" in item[3]


def test_the_anchor_is_the_latest_usable_reading(db, polled):
    payload = compute(db, polled, NOW)
    assert payload["latest_reading"] == {"temp": 12.2, "at": LAST_GOOD.isoformat()}


def test_a_manual_water_has_no_feed(db, owner):
    water = make_water(db, owner, name="Manual Lake")
    payload = cycle.compute_water(db, water, FakeOpenMeteo(), NOW)
    assert payload["feed"] is None


def test_a_water_switched_back_to_manual_has_no_feed_block(db, polled):
    water = db.get(Water, polled.water_id)
    water.feed_mode, polled.state = "manual", "off"
    db.commit()
    payload = compute(db, polled, NOW + timedelta(days=10))
    assert payload["feed"] is None and payload["readings"]


def test_poll_feed_uses_the_waters_reading_time_for_dates(db, owner):
    water = make_water(db, owner, name="Date Lake", feed_mode="feed")
    config = {"url": "https://lake.example/rows", "orientation": "rows", "rows_path": "$", "header_row": True,
              "time_field": 0, "value_field": 1, "time_format": "%d-%b-%Y"}
    feed = make_feed(db, water, owner, config=config)
    poll_feed(db, feed, NOW, fetch=fetching([["date", "temp"], ["17-Oct-2026", "12.5"]]))
    db.commit()
    assert [r.measured_at for r in standing(db, feed)] == [pd.Timestamp("2026-10-17 06:00", tz="UTC")]


def test_an_existing_feed_reading_keeps_its_overrule(db, feed):
    reading = add_feed_reading(db, feed, "2026-10-18 07:00:00", 12.4, flag_override="reject")
    poll_feed(db, feed, NOW, fetch=fetching(columns(("2026-10-18 07:00", 12.4))))
    db.commit()
    assert fresh(db, Reading, reading.id).flag_override == "reject"


# --- a feed that reads nothing usable says why --------------------------------

ROWS = {"url": "https://lake.example/rows", "orientation": "rows", "rows_path": "$", "time_field": "time",
        "value_field": "temp", "time_format": "iso"}


@pytest.fixture
def rows_feed(db, owner):
    water = make_water(db, owner, name="Rows Lake", feed_mode="feed")
    return make_feed(db, water, owner, config=ROWS)


def test_a_renamed_key_records_an_error_instead_of_a_silent_success(db, rows_feed):
    payload = [{"time": "2026-10-18T07:00Z", "temperature": 12.4}, {"time": "2026-10-18T08:00Z", "temperature": 12.5}]
    result = poll_feed(db, rows_feed, NOW, fetch=fetching(payload))
    db.commit()
    stored = fresh(db, Feed, rows_feed.id)
    assert result.ok is False
    assert "0 of 2 rows" in stored.last_error and "row 1" in stored.last_error
    assert stored.last_success_at is None and stored.last_polled_at == NOW


def test_an_empty_feed_records_an_error(db, rows_feed):
    poll_feed(db, rows_feed, NOW, fetch=fetching([]))
    db.commit()
    assert fresh(db, Feed, rows_feed.id).last_error == "The feed returned no readings."


def test_a_feed_whose_rows_all_fail_the_checks_records_an_error(db, rows_feed):
    payload = [{"time": "2026-10-18T07:00Z", "temp": 54.4}, {"time": "2026-10-18T08:00Z", "temp": 54.5}]
    poll_feed(db, rows_feed, NOW, fetch=fetching(payload))
    db.commit()
    error = fresh(db, Feed, rows_feed.id).last_error
    assert "0 of 2 rows" in error and "outside -2 to 40" in error


def test_a_feed_whose_rows_are_all_stored_already_is_a_success(db, rows_feed):
    payload = [{"time": "2026-10-18T07:00Z", "temp": 12.4}]
    poll_feed(db, rows_feed, NOW, fetch=fetching(payload))
    result = poll_feed(db, rows_feed, NOW, fetch=fetching(payload))
    db.commit()
    assert result.ok and fresh(db, Feed, rows_feed.id).last_error is None


def test_a_correction_in_the_same_poll_does_not_unguard_new_rows(db, feed):
    start = pd.Timestamp("2026-10-15 00:00", tz="UTC")
    steady = [((start + timedelta(hours=h)).isoformat(), 13.2 + 0.1 * (h % 3)) for h in range(80)]
    poll_feed(db, feed, NOW, fetch=fetching(columns(*steady)))
    db.commit()
    corrected = [(at, 13.0 if i == 30 else t) for i, (at, t) in enumerate(steady)]
    result = poll_feed(db, feed, NOW, fetch=fetching(columns(*corrected, ((start + timedelta(hours=80)).isoformat(), 9.0))))
    db.commit()
    assert (result.new, result.corrected, result.flagged) == (1, 1, 1)


# --- date-only feeds match on the local date -----------------------------------

@pytest.fixture
def dated_feed(db, owner):
    water = make_water(db, owner, name="Dated Lake", feed_mode="feed")
    config = {"url": "https://lake.example/rows", "orientation": "rows", "rows_path": "$", "header_row": True,
              "time_field": 0, "value_field": 1, "time_format": "%d-%b-%Y"}
    return make_feed(db, water, owner, config=config)


DATED = [["date", "temp"], ["16-Oct-2026", "12.5"], ["17-Oct-2026", "12.4"]]


def test_a_new_reading_time_does_not_duplicate_a_date_only_history(db, dated_feed):
    poll_feed(db, dated_feed, NOW, fetch=fetching(DATED))
    db.commit()
    water = db.get(Water, dated_feed.water_id)
    water.default_reading_time = pd.Timestamp("09:00").time()
    db.commit()
    result = poll_feed(db, dated_feed, NOW, fetch=fetching(DATED))
    db.commit()
    assert (result.new, result.corrected) == (0, 0)
    assert len(standing(db, dated_feed)) == 2


def test_a_date_only_value_change_is_a_correction_at_the_current_reading_time(db, dated_feed):
    poll_feed(db, dated_feed, NOW, fetch=fetching(DATED))
    db.commit()
    db.get(Water, dated_feed.water_id).default_reading_time = pd.Timestamp("09:00").time()
    db.commit()
    changed = [DATED[0], DATED[1], ["17-Oct-2026", "12.9"]]
    result = poll_feed(db, dated_feed, NOW, fetch=fetching(changed))
    db.commit()
    assert (result.new, result.corrected) == (0, 1)
    rows = standing(db, dated_feed)
    assert [(r.measured_at, float(r.temp_c)) for r in rows] == [
        (pd.Timestamp("2026-10-16 06:00", tz="UTC"), 12.5), (pd.Timestamp("2026-10-17 08:00", tz="UTC"), 12.9)]
