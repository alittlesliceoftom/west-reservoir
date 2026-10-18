import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from tests.factories import make_user, make_water
from tests.web.conftest import NOW, payload, store_snapshot
from tests.worker.test_cycle import FakeOpenMeteo
from worker import cycle


def visible(html: str) -> str:
    """The page's text as a reader sees it: tags removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))


def test_water_page_headline_and_last_reading(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    page = client.get("/w/test-lake").text
    assert "About 13.8 °C now" in page
    assert "Last reading 13.6 °C, Sun 18 Oct 07:00" in page


def test_water_page_five_day_strip(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    page = visible(client.get("/w/test-lake").text)
    assert "Mon 07:00 — 13.5 °C (12.8–14.2)" in page
    assert "Fri 07:00 — 13.0 °C (12.3–13.7)" in page


def test_calibrated_and_default_parameter_copy(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    assert "Calibrated from 12 readings" in client.get("/w/test-lake").text

    store_snapshot(db, water, payload(calibration={"method": "prior", "n_readings": 1, "n_legs": 0, "scale": 1.0}))
    page = client.get("/w/test-lake").text
    assert "Using default parameters — forecasts tighten as readings are added" in page


def test_river_and_harbour_waters_carry_the_reliability_note(client, db):
    user = make_user(db)
    still = make_water(db, user, name="Still", needs_recompute=False)
    river = make_water(db, user, name="River", water_type="river", needs_recompute=False)
    store_snapshot(db, still)
    store_snapshot(db, river)
    assert "less reliable" not in client.get("/w/still").text
    assert "less reliable" in client.get("/w/river").text


def test_footer_has_the_update_age_and_open_meteo_credit(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    page = client.get("/w/test-lake").text
    assert "Forecast updated 14 min ago" in page
    assert "Weather data by" in page and "Open-Meteo.com" in page


def test_a_stale_snapshot_says_how_old_it_is(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water, payload(built_at="2026-10-18T06:00:00+00:00"))
    assert "Forecast updated 6 hours ago" in client.get("/w/test-lake").text


def test_a_water_without_a_snapshot_is_computing(client, db):
    make_water(db, make_user(db))
    page = client.get("/w/test-lake").text
    assert "Computing your first forecast..." in page
    assert 'hx-get="/w/test-lake/status' in page


def test_a_dirty_water_shows_updating_and_polls(client, db):
    water = make_water(db, make_user(db), needs_recompute=True)
    store_snapshot(db, water)
    page = client.get("/w/test-lake").text
    assert "Updating forecast..." in page
    assert "About 13.8 °C now" in page


def test_a_water_with_no_readings_invites_the_first(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water, payload(latest_reading=None, estimate_now=None, forecast=[], readings=[],
                                      trajectory=[], calibration={"method": "prior", "n_readings": 0, "n_legs": 0, "scale": 1.0}))
    page = client.get("/w/test-lake").text
    assert "No readings yet" in page
    assert "About" not in page.split("<main")[1].split("No readings yet")[0]


def test_a_reading_without_an_estimate_still_shows_the_reading(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water, payload(estimate_now=None))
    page = client.get("/w/test-lake").text
    assert "Last reading 13.6 °C, Sun 18 Oct 07:00" in page
    assert "About 13.8" not in page


def test_status_poll_refreshes_once_the_snapshot_is_newer(client, db):
    water = make_water(db, make_user(db), needs_recompute=True)
    store_snapshot(db, water)
    since = "2026-10-18T11:46:00+00:00"
    waiting = client.get("/w/test-lake/status", params={"since": since})
    assert waiting.status_code == 200 and "HX-Refresh" not in waiting.headers
    store_snapshot(db, water, payload(built_at="2026-10-18T11:59:00+00:00"))
    db.execute(text("UPDATE waters SET needs_recompute = false"))
    db.commit()
    done = client.get("/w/test-lake/status", params={"since": since})
    assert done.headers["HX-Refresh"] == "true"


def test_status_poll_for_a_first_forecast(client, db):
    water = make_water(db, make_user(db))
    assert "HX-Refresh" not in client.get("/w/test-lake/status", params={"since": ""}).headers
    store_snapshot(db, water)
    assert client.get("/w/test-lake/status", params={"since": ""}).headers["HX-Refresh"] == "true"


def test_unknown_water_is_404(client):
    response = client.get("/w/nowhere")
    assert response.status_code == 404
    assert "No water called" in response.text


def test_chart_data_is_embedded_as_json(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    page = client.get("/w/test-lake").text
    assert '<script id="chart-data" type="application/json">' in page
    assert '"forecast"' in page


def test_home_lists_active_waters_with_estimate_and_last_reading(client, db):
    user = make_user(db)
    water = make_water(db, user, needs_recompute=False)
    store_snapshot(db, water)
    make_water(db, user, name="Old Pond", archived_at=NOW)
    make_water(db, user, name="Brand New")
    page = client.get("/").text
    assert "Test Lake" in page and "13.8 °C" in page and "Sun 18 Oct" in page
    assert "Brand New" in page and "Computing first forecast" in page
    assert "Old Pond" not in page
    assert 'href="/new"' in page and "Add your water" in page


def test_archived_water_page_is_gone(client, db):
    make_water(db, make_user(db), archived_at=NOW)
    assert client.get("/w/test-lake").status_code == 404


def test_a_real_worker_snapshot_renders(client, db, session_factory):
    user = make_user(db)
    water = make_water(db, user)
    from tests.factories import add_reading
    for i, t in enumerate([13.0, 12.8, 12.7, 12.5]):
        add_reading(db, water, user, (pd.Timestamp("2026-10-14 06:00", tz="UTC") + pd.Timedelta(days=i)).to_pydatetime(), t)
    cycle.drain_dirty(session_factory, FakeOpenMeteo(), datetime(2026, 10, 18, 1, 0, tzinfo=timezone.utc))
    page = client.get("/w/test-lake").text
    assert "Last reading 12.5 °C, Sat 17 Oct 07:00" in page
    assert page.count(" — ") >= 5
    assert "Calibrated from 4 readings" in page


def test_pages_never_call_open_meteo(client, db, monkeypatch):
    from forecast.openmeteo import OpenMeteoClient

    def refuse(*args, **kwargs):
        raise AssertionError("a page load called Open-Meteo")

    monkeypatch.setattr(OpenMeteoClient, "_get", refuse)
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    assert client.get("/w/test-lake").status_code == 200
    assert client.get("/").status_code == 200


def test_pages_render_well_under_a_second(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    client.get("/w/test-lake")
    start = datetime.now()
    client.get("/w/test-lake")
    assert datetime.now() - start < timedelta(milliseconds=300)


def test_a_pending_update_does_not_claim_there_are_no_readings(client, db):
    water = make_water(db, make_user(db), needs_recompute=True)
    store_snapshot(db, water, payload(latest_reading=None, estimate_now=None, forecast=[], readings=[], trajectory=[]))
    page = client.get("/w/test-lake").text
    assert "Updating forecast..." in page
    assert "No readings yet" not in page


def test_a_reading_without_an_estimate_says_why(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water, payload(estimate_now=None))
    assert "No current estimate" in client.get("/w/test-lake").text


# --- feed waters -------------------------------------------------------------

def feed_block(**overrides):
    return {"source_name": "Example Lake", "source_link": "https://example-lake.org/live/", "state": "on",
            "last_good_reading_at": "2026-10-18T11:00:00+00:00", "quiet": False, "paused_forecast": False,
            **overrides}


def feed_water(db, data=None):
    from tests.factories import make_feed
    owner = make_user(db)
    water = make_water(db, owner, needs_recompute=False, feed_mode="feed")
    make_feed(db, water, owner)
    store_snapshot(db, water, data or payload(feed=feed_block()))
    return water


def test_a_feed_water_credits_its_source(client, db):
    feed_water(db)
    page = client.get("/w/test-lake").text
    assert re.search(r'Readings from <a href="https://example-lake.org/live/"[^>]*>Example Lake</a>', page)


def test_a_feed_water_has_no_add_a_reading_button(client, db):
    feed_water(db)
    page = client.get("/w/test-lake").text
    assert "/w/test-lake/readings" not in page and "Add a reading" not in page


def test_a_manual_water_keeps_its_add_a_reading_button(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    assert "/w/test-lake/readings" in client.get("/w/test-lake").text


def test_a_quiet_feed_says_since_when(client, db):
    feed_water(db, payload(feed=feed_block(quiet=True, last_good_reading_at="2026-10-17T23:00:00+00:00")))
    assert "Feed quiet since Sun 18 Oct 00:00" in visible(client.get("/w/test-lake").text)


def test_a_feed_silent_for_five_days_pauses_the_forecast(client, db):
    data = payload(feed=feed_block(quiet=True, paused_forecast=True, last_good_reading_at="2026-10-12T09:00:00+00:00"),
                   estimate_now=None, forecast=[], trajectory=[])
    feed_water(db, data)
    page = visible(client.get("/w/test-lake").text)
    assert "Forecast paused: no data from Example Lake since Mon 12 Oct" in page
    assert "Feed quiet since" not in page
    assert "weather data" not in page
    assert "Next 5 days" not in page


def test_a_feed_paused_by_its_admin_says_so(client, db):
    data = payload(feed=feed_block(state="paused", paused_forecast=True), estimate_now=None, forecast=[], trajectory=[])
    feed_water(db, data)
    page = visible(client.get("/w/test-lake").text)
    assert "Feed paused by the venue" in page
    assert "Forecast paused: no data" not in page


def test_a_feed_with_no_good_reading_yet(client, db):
    data = payload(feed=feed_block(quiet=True, paused_forecast=True, last_good_reading_at=None),
                   latest_reading=None, estimate_now=None, forecast=[], trajectory=[],
                   readings=[["2026-10-18T06:00:00+00:00", 8.4, "Jumped 4.3 °C from the last good reading (12.7 °C)."]])
    feed_water(db, data)
    page = visible(client.get("/w/test-lake").text)
    assert "No good readings from Example Lake yet" in page
    assert "Add the first one" not in page
    assert 'id="chart-data"' in client.get("/w/test-lake").text


def test_chart_data_carries_flag_reasons(client, db):
    readings = [["2026-10-18T05:00:00+00:00", 13.6, None],
                ["2026-10-18T06:00:00+00:00", 8.4, "Jumped 4.3 °C from the last good reading (12.7 °C)."]]
    feed_water(db, payload(feed=feed_block(), readings=readings))
    html = client.get("/w/test-lake").text
    data = re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    import json
    assert json.loads(data)["readings"] == readings


def test_feed_water_pages_never_call_the_feed(client, app, db):
    def refuse(*args, **kwargs):
        raise AssertionError("a page load contacted a feed")

    app.state.feed_fetch = refuse
    feed_water(db)
    for path in ("/w/test-lake", "/", "/w/test-lake/status"):
        assert client.get(path).status_code == 200, path
