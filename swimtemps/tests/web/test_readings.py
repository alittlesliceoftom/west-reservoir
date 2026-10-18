import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from db.models import Reading, Water
from tests.factories import add_reading, make_user, make_water
from tests.web.conftest import NOW, store_snapshot
from tests.web.helpers import post, sign_in

# NOW is 2026-10-18 12:00 UTC (13:00 BST); the reading time is 07:00 local.


@pytest.fixture
def owner(db):
    return make_user(db, email="owner@example.com", name="Owner")


@pytest.fixture
def swimmer(db):
    return make_user(db, email="swimmer@example.com", name="Swimmer")


@pytest.fixture
def water(db, owner):
    w = make_water(db, owner, needs_recompute=False)
    store_snapshot(db, w)  # estimate now 13.8
    return w


def readings(db):
    db.expire_all()
    return db.scalars(select(Reading).where(Reading.deleted_at.is_(None)).order_by(Reading.measured_at)).all()


def text_of(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))


# --- single reading ---------------------------------------------------------

def test_the_form_defaults_to_today_at_the_reading_time(client, db, water, swimmer):
    sign_in(client, swimmer)
    page = client.get("/w/test-lake/readings").text
    assert 'value="2026-10-18T07:00"' in page
    assert 'inputmode="decimal"' in page


def test_the_form_defaults_to_now_when_the_reading_time_is_still_to_come(client, db, owner, swimmer):
    from datetime import time
    make_water(db, owner, name="Late Lake", reading_time=time(18, 0))
    sign_in(client, swimmer)
    assert 'value="2026-10-18T13:00"' in client.get("/w/late-lake/readings").text


def test_saving_a_reading_flags_the_water_and_returns_to_its_page(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": "13.9", "measured_at": "2026-10-18T07:00"})
    assert response.status_code == 303 and response.headers["location"] == "/w/test-lake"
    saved = readings(db)
    assert [(r.temp_c, r.measured_at, r.submitted_by) for r in saved] == [
        (13.9, datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc), swimmer.id)]
    assert db.get(Water, water.id).needs_recompute is True
    assert "Updating forecast..." in client.get("/w/test-lake").text


@pytest.mark.parametrize("temp", ["40.1", "-2.1", "hot", ""])
def test_the_hard_limits_refuse(client, db, water, swimmer, temp):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": temp, "measured_at": "2026-10-18T07:00"})
    assert response.status_code == 422
    assert "between -2 and 40" in response.text or "a number" in response.text
    assert readings(db) == []


def test_the_limits_themselves_are_allowed(client, db, water, swimmer):
    sign_in(client, swimmer)
    post(client, "/w/test-lake/readings", {"temp": "40", "measured_at": "2026-10-18T07:00", "confirm": "1"})
    post(client, "/w/test-lake/readings", {"temp": "-2", "measured_at": "2026-10-17T07:00", "confirm": "1"})
    assert len(readings(db)) == 2


def test_a_future_time_is_refused(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": "13.9", "measured_at": "2026-10-18T19:00"})
    assert response.status_code == 422 and "in the future" in response.text


def test_far_from_the_estimate_asks_for_confirmation(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": "18.0", "measured_at": "2026-10-18T07:00"})
    assert response.status_code == 200
    assert "4.2 °C away from the current estimate of 13.8 °C" in text_of(response.text)
    assert 'name="confirm" value="1"' in response.text
    assert readings(db) == []

    confirmed = post(client, "/w/test-lake/readings", {"temp": "18.0", "measured_at": "2026-10-18T07:00", "confirm": "1"})
    assert confirmed.status_code == 303
    assert len(readings(db)) == 1


def test_within_three_degrees_needs_no_confirmation(client, db, water, swimmer):
    sign_in(client, swimmer)
    assert post(client, "/w/test-lake/readings", {"temp": "16.7", "measured_at": "2026-10-18T07:00"}).status_code == 303


def test_a_water_with_no_estimate_skips_the_soft_check(client, db, owner, swimmer):
    make_water(db, owner, name="New Lake")  # no snapshot at all
    sign_in(client, swimmer)
    assert post(client, "/w/new-lake/readings", {"temp": "25.0", "measured_at": "2026-10-18T07:00"}).status_code == 303


def test_an_old_reading_is_not_compared_with_todays_estimate(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": "20.0", "measured_at": "2026-08-01T07:00"})
    assert response.status_code == 303


def test_a_second_reading_in_the_same_hour_asks_first(client, db, water, swimmer):
    add_reading(db, water, swimmer, "2026-10-18 06:10", 13.7)
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": "13.8", "measured_at": "2026-10-18T07:30"})
    assert response.status_code == 200
    assert "already a reading at 07:10" in text_of(response.text)
    assert len(readings(db)) == 1


def test_the_thirty_first_reading_in_an_hour_is_refused(client, db, water, swimmer):
    for i in range(30):
        add_reading(db, water, swimmer, datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=i), 14.0,
                    created_at=NOW - timedelta(minutes=30))
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings", {"temp": "13.9", "measured_at": "2026-10-18T07:00"})
    assert response.status_code == 429
    assert "30 readings an hour" in response.text


def test_readings_older_than_an_hour_do_not_count(client, db, water, swimmer):
    for i in range(30):
        add_reading(db, water, swimmer, datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=i), 14.0,
                    created_at=NOW - timedelta(minutes=61))
    sign_in(client, swimmer)
    assert post(client, "/w/test-lake/readings", {"temp": "13.9", "measured_at": "2026-10-18T07:00"}).status_code == 303


def test_unknown_water_404s(client, db, swimmer):
    sign_in(client, swimmer)
    assert post(client, "/w/nope/readings", {"temp": "13.9"}).status_code == 404


# --- paste several ---------------------------------------------------------

PASTE = "Date, Temp\n15/10/2026 14.1\n16/10/2026 14.0\n17/10 07:00, 13.9\n18/10/2026 13.8\n"


def test_preview_shows_each_row_without_saving(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings/preview", {"rows": PASTE})
    page = text_of(response.text)
    assert "4 readings ready to save" in page
    assert "Sat 17 Oct 07:00" in page
    assert readings(db) == []


def test_preview_flags_duplicates_in_the_database_and_in_the_paste(client, db, water, swimmer):
    add_reading(db, water, swimmer, "2026-10-16 06:20", 14.0)
    sign_in(client, swimmer)
    page = text_of(post(client, "/w/test-lake/readings/preview",
                        {"rows": "16/10/2026 14.0\n17/10/2026 13.9\n17/10/2026 07:30 13.8"}).text)
    assert page.count("Already a reading this hour") == 1
    assert page.count("Repeated in this paste") == 1
    assert "1 reading ready to save" in page
    assert "2 skipped" in page


def test_preview_flags_outliers_against_their_neighbours(client, db, water, swimmer):
    sign_in(client, swimmer)
    page = text_of(post(client, "/w/test-lake/readings/preview",
                        {"rows": "14/10/2026 14.1\n15/10/2026 14.0\n16/10/2026 19.5\n17/10/2026 13.9"}).text)
    assert page.count("Check this") == 1
    assert "4 readings ready to save" in page


def test_preview_lists_unreadable_lines_and_blocks_saving(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings/preview", {"rows": "18/10/2026 13.8\n31/02/2026 13\n18/10/2026"})
    page = text_of(response.text)
    assert "Line 2: not a real date" in page
    assert "Line 3: no temperature" in page
    assert "Fix the lines above" in page
    assert 'name="confirm_paste"' not in response.text


def test_saving_a_paste_stores_new_rows_and_skips_duplicates(client, db, water, swimmer):
    add_reading(db, water, swimmer, "2026-10-16 06:00", 14.0)
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings/paste", {"rows": PASTE})
    assert response.status_code == 303
    assert [r.temp_c for r in readings(db)] == [14.1, 14.0, 13.9, 13.8]
    assert db.get(Water, water.id).needs_recompute is True


def test_a_paste_with_errors_is_not_saved(client, db, water, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/w/test-lake/readings/paste", {"rows": "18/10/2026 13.8\nnonsense 12"})
    assert response.status_code == 422
    assert readings(db) == []


def test_a_paste_over_the_hourly_limit_is_refused_whole(client, db, water, swimmer):
    for i in range(25):
        add_reading(db, water, swimmer, datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i), 5.0,
                    created_at=NOW - timedelta(minutes=10))
    sign_in(client, swimmer)
    rows = "\n".join(f"{d}/09/2026 15.0" for d in range(1, 7))
    response = post(client, "/w/test-lake/readings/paste", {"rows": rows})
    assert response.status_code == 429
    assert "You can add 5 more readings to this water in the next hour" in text_of(response.text)
    assert db.scalar(select(func.count()).select_from(Reading)) == 25


def test_the_waters_admin_can_paste_its_history_past_the_limit(client, db, water, owner):
    sign_in(client, owner)
    rows = "\n".join(f"{d}/{m}/2026 15.0" for m in (7, 8) for d in range(1, 29))
    assert post(client, "/w/test-lake/readings/paste", {"rows": rows}).status_code == 303
    assert len(readings(db)) == 56


def test_the_page_offers_paste_several(client, db, water, swimmer):
    sign_in(client, swimmer)
    page = client.get("/w/test-lake/readings").text
    assert 'hx-post="/w/test-lake/readings/preview"' in page
    assert "dd/mm/yyyy" in page
