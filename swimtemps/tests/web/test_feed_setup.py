import hashlib
import json
import re
from pathlib import Path

import pytest
from sqlalchemy import func, select

from db import queries
from db.models import Feed, Reading, Water
from feeds.fetch import FetchError
from tests.factories import add_feed_reading, add_reading, make_feed, make_user, make_water
from tests.feeds.test_block import EXAMPLE
from tests.web.conftest import NOW, payload, store_snapshot
from tests.web.helpers import csrf, post, sign_in
from worker.feeds import poll_feeds

FIXTURES = Path(__file__).parents[1] / "fixtures" / "feeds"
STATEMENT = "I have the venue's permission to share this data, or it is publicly published."
FIELDS = {
    "url": "https://lake.example/temps", "orientation": "columns", "time_path": "labels",
    "value_path": "datasets[0].data", "time_format": "epoch_ms", "timezone": "", "unit": "C",
    "source_name": "Example Lake", "source_link": "https://lake.example/live/",
}
CONFIG = {"url": "https://lake.example/temps", "orientation": "columns", "time_path": "labels",
          "value_path": "datasets[0].data", "time_format": "epoch_ms", "unit": "C"}


def deep():
    return json.loads((FIXTURES / "clevedon_266367_168h_2026-10-18.json").read_text())


class FakeFetch:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = [], result, error

    def __call__(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        return self.result if self.result is not None else deep()


@pytest.fixture
def fetch(app):
    fake = FakeFetch()
    app.state.feed_fetch = fake
    return fake


@pytest.fixture
def admin(db):
    return make_user(db, email="admin@example.com", name="Admin")


@pytest.fixture
def swimmer(db):
    return make_user(db, email="swimmer@example.com", name="Swimmer")


@pytest.fixture
def water(db, admin):
    return make_water(db, admin, needs_recompute=False)


def fresh(db, model, key):
    db.expire_all()
    return db.get(model, key)


def token_from_test(client, fields=FIELDS):
    page = post(client, "/w/test-lake/feed/test", fields)
    match = re.search(r'name="tested" value="([0-9a-f]+)"', page.text)
    assert match, page.text
    return match.group(1)


def enable(client, fields=FIELDS, **extra):
    return post(client, "/w/test-lake/feed/enable", {**fields, "tested": token_from_test(client, fields),
                                                      "attest": "yes", **extra})


# --- who may -----------------------------------------------------------------

def test_signed_out_visitors_are_sent_to_sign_in(client, water):
    response = client.get("/w/test-lake/feed", follow_redirects=False)
    assert response.headers["location"] == "/auth/login?next=/w/test-lake/feed"


def test_contributors_are_refused_everywhere(client, db, water, admin, swimmer, fetch):
    feed = make_feed(db, water, admin)
    reading = add_feed_reading(db, feed, "2026-10-18 06:00", 13.5)
    sign_in(client, swimmer)
    assert client.get("/w/test-lake/feed").status_code == 403
    for path in ("paste", "test", "enable", "pause", "resume", "disable"):
        assert post(client, f"/w/test-lake/feed/{path}", FIELDS).status_code == 403, path
    assert post(client, f"/w/test-lake/admin/readings/{reading.id}/override", {"decision": "reject"}).status_code == 403
    assert fetch.calls == []
    assert fresh(db, Reading, reading.id).flag_override is None


def test_a_superuser_may_set_up_a_feed(client, db, water, fetch):
    sign_in(client, make_user(db, email="root@example.com", superuser=True))
    assert client.get("/w/test-lake/feed").status_code == 200


# --- the setup page ----------------------------------------------------------

def test_the_setup_page_explains_and_offers_the_prompt(client, water, admin):
    sign_in(client, admin)
    page = client.get("/w/test-lake/feed").text
    assert "swimtemps-feed v1" in page and "&lt;PASTE THE PAGE ADDRESS&gt;" in page
    assert "data-copy" in page
    assert "/feeds/request?water=test-lake" in page


def test_the_setup_page_and_a_paste_never_fetch(client, water, admin, fetch):
    sign_in(client, admin)
    client.get("/w/test-lake/feed")
    post(client, "/w/test-lake/feed/paste", {"block": EXAMPLE})
    assert fetch.calls == []


def test_pasting_the_prompts_example_fills_the_fields(client, water, admin):
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/paste", {"block": "Here you go:\n```\n" + EXAMPLE + "\n```"}).text
    assert 'value="https://example-lake.org/wp-json/lake/v1/temps?hours=168"' in page
    assert 'value="datasets[0].data"' in page
    assert re.search(r'<option value="columns"\s+selected', page)
    assert 'value="Example Lake"' in page


def test_a_paste_lists_what_is_missing_unknown_and_invalid(client, water, admin):
    sign_in(client, admin)
    block = EXAMPLE.replace("time_format: epoch_ms", "time_format: unknown").replace("unit: C", "unit: K\ndepth: 1m")
    page = post(client, "/w/test-lake/feed/paste", {"block": block}).text
    assert "time_format" in page and "depth" in page and "unit: &#39;K&#39; is not C or F" in page


def test_pasting_a_manually_updated_source_explains_it_should_stay_manual(client, water, admin):
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/paste", {"block": EXAMPLE.replace("automated: yes", "automated: no")}).text
    assert "updated by hand" in page and "manual water" in page
    assert "/feeds/request?water=test-lake" in page


def test_pasting_terms_that_forbid_access_explains_why_not(client, water, admin):
    block = EXAMPLE.replace("terms: unknown", "terms: forbids automated access")
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/paste", {"block": block}).text
    assert "agreement" in page and "/feeds/request?water=test-lake" in page


def test_pasting_without_a_block_asks_for_it(client, water, admin):
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/paste", {"block": "No API found, sorry."}).text
    assert "starts with swimtemps-feed v1" in page


# --- Test --------------------------------------------------------------------

def test_a_test_shows_what_would_be_stored_and_stores_nothing(client, db, water, admin, fetch):
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/test", FIELDS)
    assert page.status_code == 200
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", page.text))
    assert "353 rows read" in text and "169 hourly readings ready" in text and "5 would be flagged" in text
    assert "Jumped" in text
    assert fetch.calls == ["https://lake.example/temps"]
    assert db.scalar(select(func.count()).select_from(Reading)) == 0
    assert db.scalar(select(func.count()).select_from(Feed)) == 0


def test_the_permission_box_is_ticked_by_default_with_its_statement(client, water, admin, fetch):
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/test", FIELDS).text
    assert re.search(r'<input type="checkbox" name="attest" value="yes" checked', page)
    assert "I have the venue&#39;s permission to share this data, or it is publicly published." in page


def test_a_test_with_bad_fields_does_not_fetch(client, water, admin, fetch):
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/test", {**FIELDS, "orientation": "sideways", "time_path": ""})
    assert page.status_code == 422
    assert "sideways" in page.text and "time_path" in page.text
    assert fetch.calls == []


def test_a_feed_at_a_private_address_is_refused(client, water, admin):
    sign_in(client, admin)   # the real fetcher: an address literal needs no lookup
    page = post(client, "/w/test-lake/feed/test", {**FIELDS, "url": "http://127.0.0.1:8000/temps"})
    assert page.status_code == 422
    assert "private or internal address" in page.text


def test_a_feed_whose_shape_does_not_match_says_so(client, water, admin, fetch):
    fetch.result = {"labels": [1, 2]}
    sign_in(client, admin)
    page = post(client, "/w/test-lake/feed/test", FIELDS)
    assert page.status_code == 422 and "datasets" in page.text


# --- Turn on -----------------------------------------------------------------

def test_turning_on_without_a_test_is_refused(client, db, water, admin, fetch):
    sign_in(client, admin)
    response = post(client, "/w/test-lake/feed/enable", {**FIELDS, "attest": "yes"})
    assert response.status_code == 422 and "Test" in response.text
    assert db.scalar(select(func.count()).select_from(Feed)) == 0


def test_turning_on_a_config_changed_since_its_test_is_refused(client, db, water, admin, fetch):
    sign_in(client, admin)
    token = token_from_test(client)
    response = post(client, "/w/test-lake/feed/enable", {**FIELDS, "time_path": "times", "tested": token, "attest": "yes"})
    assert response.status_code == 422
    assert db.scalar(select(func.count()).select_from(Feed)) == 0


def test_turning_on_needs_the_permission_box(client, db, water, admin, fetch):
    sign_in(client, admin)
    response = post(client, "/w/test-lake/feed/enable", {**FIELDS, "tested": token_from_test(client)})
    assert response.status_code == 422 and "permission" in response.text
    assert db.scalar(select(func.count()).select_from(Feed)) == 0


def test_turning_on_retests_and_refuses_a_feed_that_now_fails(client, db, water, admin, fetch):
    sign_in(client, admin)
    token = token_from_test(client)
    fetch.error = FetchError("The feed answered 503 instead of 200.")
    response = post(client, "/w/test-lake/feed/enable", {**FIELDS, "tested": token, "attest": "yes"})
    assert response.status_code == 422 and "503" in response.text
    assert db.scalar(select(func.count()).select_from(Feed)) == 0


def test_turning_on_stores_the_feed_and_its_attestation(client, db, water, admin, fetch):
    sign_in(client, admin)
    response = enable(client)
    assert response.status_code == 303
    feed = db.scalar(select(Feed))
    assert feed.config == CONFIG and feed.recipe == "json_table"
    assert (feed.source_name, feed.source_link) == ("Example Lake", "https://lake.example/live/")
    assert (feed.state, feed.attested_by, feed.attested_statement) == ("on", admin.id, STATEMENT)
    assert feed.attested_at == NOW and feed.last_polled_at is None
    stored = fresh(db, Water, water.id)
    assert stored.feed_mode == "feed" and stored.needs_recompute is True
    assert db.scalar(select(func.count()).select_from(Reading)) == 0


def test_the_tested_token_is_the_configs_hash(client, water, admin, fetch):
    sign_in(client, admin)
    expected = hashlib.sha256(json.dumps({**CONFIG, "source_name": "Example Lake",
                                          "source_link": "https://lake.example/live/"}, sort_keys=True).encode()).hexdigest()
    assert token_from_test(client) == expected


def test_the_setup_page_shows_a_feed_that_is_on(client, db, water, admin, fetch):
    sign_in(client, admin)
    enable(client)
    page = client.get("/w/test-lake/feed").text
    assert "Pause" in page and "Switch to manual readings" in page
    assert 'value="https://lake.example/temps"' in page


# --- pause, resume, switch ---------------------------------------------------

def test_pause_and_resume(client, db, water, admin, fetch):
    sign_in(client, admin)
    enable(client)
    db.execute(Water.__table__.update().values(needs_recompute=False))
    db.commit()
    assert post(client, "/w/test-lake/feed/pause").status_code == 303
    assert db.scalar(select(Feed.state)) == "paused"
    assert fresh(db, Water, water.id).needs_recompute is True
    assert post(client, "/w/test-lake/feed/resume").status_code == 303
    feed = db.scalar(select(Feed))
    db.refresh(feed)
    assert feed.state == "on" and feed.last_polled_at is None


def test_switching_to_manual_keeps_feed_readings_and_stops_polling(client, db, session_factory, water, admin, fetch):
    sign_in(client, admin)
    enable(client)
    feed = db.scalar(select(Feed))
    add_feed_reading(db, feed, "2026-10-18 06:00", 13.5)
    assert post(client, "/w/test-lake/feed/disable").status_code == 303
    stored = fresh(db, Water, water.id)
    feed = db.scalar(select(Feed))
    assert stored.feed_mode == "manual" and feed.state == "off" and feed.config == CONFIG
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == [13.5]
    assert queries.chart_readings(db, water.id, since=None)["temp_c"].tolist() == [13.5]
    polled = FakeFetch()
    assert poll_feeds(session_factory, NOW, fetch=polled) == 0 and polled.calls == []
    store_snapshot(db, water, payload())
    assert "/w/test-lake/readings" in client.get("/w/test-lake").text
    assert post(client, "/w/test-lake/readings", {"temp": "13.4", "measured_at": "2026-10-18T09:00"}).status_code == 303


def test_a_feed_water_refuses_typed_readings(client, db, water, admin, swimmer, fetch):
    sign_in(client, admin)
    enable(client)
    store_snapshot(db, water, payload())
    sign_in(client, swimmer)
    assert "/w/test-lake/readings" not in client.get("/w/test-lake").text
    added = post(client, "/w/test-lake/readings", {"temp": "13.4", "measured_at": "2026-10-18T07:00"})
    assert added.status_code == 409 and "come from Example Lake" in added.text
    pasted = post(client, "/w/test-lake/readings/paste", {"rows": "18/10/2026 13.4"})
    assert pasted.status_code == 409
    assert client.get("/w/test-lake/readings").status_code == 409
    assert db.scalar(select(func.count()).select_from(Reading)) == 0


def test_switching_back_to_a_feed_reuses_its_config(client, db, water, admin, fetch):
    sign_in(client, admin)
    enable(client)
    post(client, "/w/test-lake/feed/disable")
    add_reading(db, water, admin, "2026-10-18 07:00", 13.4)
    assert enable(client).status_code == 303
    assert db.scalar(select(func.count()).select_from(Feed)) == 1
    assert fresh(db, Water, water.id).feed_mode == "feed"
    assert queries.active_readings(db, water.id)["source"].tolist() == ["manual"]


# --- overrules ---------------------------------------------------------------

@pytest.fixture
def flagged(db, water, admin):
    feed = make_feed(db, water, admin)
    water.feed_mode = "feed"
    db.commit()
    add_feed_reading(db, feed, "2026-10-18 05:00", 13.5)
    return add_feed_reading(db, feed, "2026-10-18 06:00", 9.1, flagged=True, flag_reason="Jumped 4.4 °C.")


def test_accepting_a_flag_makes_the_reading_usable(client, db, water, admin, flagged):
    sign_in(client, admin)
    response = post(client, f"/w/test-lake/admin/readings/{flagged.id}/override", {"decision": "accept"})
    assert response.status_code == 303
    reading = fresh(db, Reading, flagged.id)
    assert (reading.flag_override, reading.override_by, reading.override_at) == ("accept", admin.id, NOW)
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == [13.5, 9.1]
    assert fresh(db, Water, water.id).needs_recompute is True


def test_rejecting_and_clearing_an_overrule(client, db, water, admin, flagged):
    sign_in(client, admin)
    good = db.scalar(select(Reading).where(Reading.temp_c == 13.5))
    post(client, f"/w/test-lake/admin/readings/{good.id}/override", {"decision": "reject"})
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == []
    post(client, f"/w/test-lake/admin/readings/{good.id}/override", {"decision": "clear"})
    assert queries.active_readings(db, water.id)["temp_c"].tolist() == [13.5]


def test_a_bad_decision_is_refused(client, db, water, admin, flagged):
    sign_in(client, admin)
    assert post(client, f"/w/test-lake/admin/readings/{flagged.id}/override", {"decision": "maybe"}).status_code == 422


def test_feed_readings_offer_overrules_not_corrections(client, db, water, admin, flagged):
    sign_in(client, admin)
    page = client.get("/w/test-lake/admin").text
    assert f"/admin/readings/{flagged.id}/override" in page
    assert f"/admin/readings/{flagged.id}/correct" not in page and f"/admin/readings/{flagged.id}/delete" not in page
    assert "Jumped 4.4 °C." in page
    assert post(client, f"/w/test-lake/admin/readings/{flagged.id}/delete").status_code == 422
    assert post(client, f"/w/test-lake/admin/readings/{flagged.id}/correct",
                {"temp": "13.0", "measured_at": "2026-10-18T07:00"}).status_code == 422
    assert fresh(db, Reading, flagged.id).deleted_at is None


def test_the_admin_page_links_to_feed_setup(client, water, admin):
    sign_in(client, admin)
    assert "/w/test-lake/feed" in client.get("/w/test-lake/admin").text


# --- /me ---------------------------------------------------------------------

def test_me_lists_feed_waters_with_a_switch_to_manual(client, db, water, admin):
    feed = make_feed(db, water, admin, last_good_reading_at=NOW.replace(hour=2), last_error="The feed answered 503 instead of 200.")
    water.feed_mode = "feed"
    db.commit()
    sign_in(client, admin)
    page = re.sub(r"\s+", " ", client.get("/me").text)
    assert "Example Lake" in page and "on" in page
    assert "10 hours ago" in page
    assert "503" in page
    assert 'action="/w/test-lake/feed/disable"' in page
    assert feed.id
    assert csrf(client)
