from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from db import queries
from db.models import Feed, FeedRequest, Reading, Water
from tests.factories import add_feed_reading, add_reading, make_feed, make_user, make_water

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def feed_water(db):
    user = make_user(db)
    water = make_water(db, user, feed_mode="feed")
    return user, water, make_feed(db, water, user)


def test_waters_start_in_manual_mode(db):
    water = make_water(db, make_user(db))
    assert db.get(Water, water.id).feed_mode == "manual"


def test_a_second_standing_row_for_the_same_feed_time_is_refused(db, feed_water):
    _, _, feed = feed_water
    add_feed_reading(db, feed, "2026-10-18 06:00", 13.5)
    with pytest.raises(IntegrityError):
        add_feed_reading(db, feed, "2026-10-18 06:00", 13.6)


def test_a_feed_time_can_be_stored_again_once_the_first_is_soft_deleted(db, feed_water):
    _, _, feed = feed_water
    old = add_feed_reading(db, feed, "2026-10-18 06:00", 13.5)
    old.deleted_at = NOW
    db.commit()
    new = add_feed_reading(db, feed, "2026-10-18 06:00", 13.6)
    old.replaced_by = new.id
    db.commit()
    assert queries.active_readings(db, feed.water_id)["temp_c"].tolist() == [13.6]


def test_manual_readings_still_need_a_submitter(db):
    user = make_user(db)
    water = make_water(db, user)
    db.add(Reading(water_id=water.id, measured_at=NOW, temp_c=13.0, source="manual"))
    with pytest.raises(IntegrityError):
        db.commit()


def test_feed_readings_need_a_feed(db, feed_water):
    _, water, _ = feed_water
    db.add(Reading(water_id=water.id, measured_at=NOW, temp_c=13.0, source="feed"))
    with pytest.raises(IntegrityError):
        db.commit()


@pytest.mark.parametrize("column, value", [
    ("flag_override", "maybe"), ("source", "robot"),
])
def test_reading_enums_are_constrained(db, feed_water, column, value):
    _, _, feed = feed_water
    with pytest.raises(IntegrityError):
        add_feed_reading(db, feed, "2026-10-18 06:00", 13.5, **{column: value})


def test_feed_state_is_constrained(db):
    user = make_user(db)
    water = make_water(db, user)
    with pytest.raises(IntegrityError):
        make_feed(db, water, user, state="sleeping")


def test_one_feed_per_water(db, feed_water):
    user, water, _ = feed_water
    with pytest.raises(IntegrityError):
        make_feed(db, water, user)


def test_feed_requests_store_their_fields(db):
    user = make_user(db)
    db.add(FeedRequest(page_url="https://lake.example/live", notes="a sensor", ai_block="swimtemps-feed v1\nend",
                       requested_by=user.id))
    db.commit()
    request = db.query(FeedRequest).one()
    assert request.status == "open" and request.water_id is None and request.created_at is not None


def test_feed_request_status_is_constrained(db):
    user = make_user(db)
    db.add(FeedRequest(page_url="https://lake.example/live", requested_by=user.id, status="pending"))
    with pytest.raises(IntegrityError):
        db.commit()


def test_feed_defaults(db, feed_water):
    _, _, feed = feed_water
    stored = db.get(Feed, feed.id)
    assert stored.thresholds == {} and stored.last_error is None and stored.created_at is not None


def test_active_readings_carry_their_source(db, feed_water):
    user, water, feed = feed_water
    add_reading(db, water, user, "2026-10-17 06:00", 13.9)
    add_feed_reading(db, feed, "2026-10-18 06:00", 13.5)
    frame = queries.active_readings(db, water.id)
    assert frame["source"].tolist() == ["manual", "feed"]


def test_active_readings_leave_out_a_flagged_row_until_it_is_accepted(db, feed_water):
    _, water, feed = feed_water
    add_feed_reading(db, feed, "2026-10-18 05:00", 13.5)
    flagged = add_feed_reading(db, feed, "2026-10-18 06:00", 9.1, flagged=True, flag_reason="jump")
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == [13.5]
    flagged.flag_override = "accept"
    db.commit()
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == [13.5, 9.1]


def test_active_readings_leave_out_an_unflagged_row_marked_wrong(db, feed_water):
    _, water, feed = feed_water
    add_feed_reading(db, feed, "2026-10-18 05:00", 13.5)
    add_feed_reading(db, feed, "2026-10-18 06:00", 13.4, flag_override="reject")
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == [13.5]


def test_chart_readings_return_every_standing_row_with_its_usability(db, feed_water):
    _, water, feed = feed_water
    add_feed_reading(db, feed, "2026-10-18 04:00", 13.5)
    add_feed_reading(db, feed, "2026-10-18 05:00", 9.1, flagged=True, flag_reason="Jumped 4.4 °C")
    add_feed_reading(db, feed, "2026-10-18 06:00", 13.4, flag_override="reject")
    gone = add_feed_reading(db, feed, "2026-10-18 07:00", 13.3)
    gone.deleted_at = NOW
    db.commit()
    frame = queries.chart_readings(db, water.id, since=None)
    assert list(frame.columns) == ["measured_at", "temp_c", "usable", "reason", "note"]
    assert frame["temp_c"].tolist() == [13.5, 9.1, 13.4]
    assert frame["usable"].tolist() == [True, False, False]
    assert frame["reason"].iloc[0] is None
    assert frame["reason"].iloc[1] == "Jumped 4.4 °C"
    assert "admin" in frame["reason"].iloc[2]


def test_chart_readings_start_at_since(db, feed_water):
    _, water, feed = feed_water
    add_feed_reading(db, feed, "2026-10-01 06:00", 13.5)
    add_feed_reading(db, feed, "2026-10-18 06:00", 13.4)
    frame = queries.chart_readings(db, water.id, since=pd.Timestamp("2026-10-10", tz="UTC").to_pydatetime())
    assert frame["temp_c"].tolist() == [13.4]


def test_an_overrule_counts_as_a_reading_change(db, feed_water):
    _, water, feed = feed_water
    reading = add_feed_reading(db, feed, "2026-10-18 06:00", 9.1, flagged=True, flag_reason="jump")
    reading.created_at = pd.Timestamp("2026-10-18 06:05", tz="UTC").to_pydatetime()
    db.commit()
    since = pd.Timestamp("2026-10-18 07:00", tz="UTC").to_pydatetime()
    assert not queries.readings_changed_since(db, water.id, since)
    reading.flag_override, reading.override_at = "accept", NOW
    db.commit()
    assert queries.readings_changed_since(db, water.id, since)


def test_chart_readings_note_a_flag_an_admin_accepted(db, feed_water):
    _, water, feed = feed_water
    add_feed_reading(db, feed, "2026-10-18 04:00", 13.5)
    add_feed_reading(db, feed, "2026-10-18 05:00", 9.1, flagged=True, flag_reason="Jumped 4.4 °C.",
                     flag_override="accept")
    frame = queries.chart_readings(db, water.id, since=None)
    assert frame["usable"].tolist() == [True, True]
    assert frame["note"].iloc[0] is None
    assert frame["note"].iloc[1] == "Flagged (Jumped 4.4 °C.), accepted by the water's admin"
