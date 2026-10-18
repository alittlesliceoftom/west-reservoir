import re

import pytest
from sqlalchemy import select

from db.models import FeedRequest
from tests.factories import make_user, make_water
from tests.web.conftest import NOW
from tests.web.helpers import post, sign_in


@pytest.fixture
def swimmer(db):
    return make_user(db, email="swimmer@example.com", name="Swimmer")


@pytest.fixture
def root(db):
    return make_user(db, email="root@example.com", name="Root", superuser=True)


def request_row(db, user, **extra):
    row = FeedRequest(page_url="https://lake.example/live", notes="sensor at 1 m", requested_by=user.id, **extra)
    db.add(row)
    db.commit()
    return row


def test_signed_out_visitors_are_sent_to_sign_in(client):
    response = client.get("/feeds/request?water=test-lake", follow_redirects=False)
    assert response.headers["location"].startswith("/auth/login?next=/feeds/request")


def test_the_form_names_the_water_it_is_for(client, db, swimmer):
    make_water(db, swimmer)
    sign_in(client, swimmer)
    page = client.get("/feeds/request?water=test-lake").text
    assert "Test Lake" in page and 'name="water" value="test-lake"' in page


def test_a_request_stores_its_fields(client, db, swimmer):
    water = make_water(db, swimmer)
    sign_in(client, swimmer)
    response = post(client, "/feeds/request", {"page_url": "https://lake.example/live", "notes": "half-hourly",
                                                "ai_block": "swimtemps-feed v1\nurl: unknown\nend", "water": "test-lake"})
    assert response.status_code == 303
    row = db.scalar(select(FeedRequest))
    assert (row.page_url, row.notes, row.ai_block) == ("https://lake.example/live", "half-hourly",
                                                       "swimtemps-feed v1\nurl: unknown\nend")
    assert (row.requested_by, row.water_id, row.status) == (swimmer.id, water.id, "open")
    assert "Thanks" in client.get(response.headers["location"]).text


def test_a_request_needs_a_web_address(client, db, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/feeds/request", {"page_url": "the lake by the pub"})
    assert response.status_code == 422 and "web address" in response.text
    assert db.scalar(select(FeedRequest)) is None


def test_a_request_for_an_unknown_water_is_stored_without_one(client, db, swimmer):
    sign_in(client, swimmer)
    post(client, "/feeds/request", {"page_url": "https://lake.example/live", "water": "nowhere"})
    assert db.scalar(select(FeedRequest)).water_id is None


def test_overlong_fields_are_refused(client, db, swimmer):
    sign_in(client, swimmer)
    response = post(client, "/feeds/request", {"page_url": "https://lake.example/live", "notes": "x" * 5001})
    assert response.status_code == 422


def test_only_superusers_see_the_requests(client, db, swimmer):
    row = request_row(db, swimmer)
    sign_in(client, swimmer)
    assert client.get("/admin/feed-requests").status_code == 403
    assert post(client, f"/admin/feed-requests/{row.id}", {"status": "done"}).status_code == 403
    db.refresh(row)
    assert row.status == "open"


def test_superusers_see_open_requests(client, db, swimmer, root):
    request_row(db, swimmer, ai_block="swimtemps-feed v1\nend")
    sign_in(client, root)
    page = client.get("/admin/feed-requests").text
    assert "https://lake.example/live" in page and "sensor at 1 m" in page and "swimtemps-feed v1" in page
    assert "Swimmer" in page


def test_a_superuser_marks_a_request_done(client, db, swimmer, root):
    row = request_row(db, swimmer)
    sign_in(client, root)
    assert post(client, f"/admin/feed-requests/{row.id}", {"status": "done"}).status_code == 303
    db.refresh(row)
    assert row.status == "done" and row.updated_at == NOW


def test_a_bad_status_is_refused(client, db, swimmer, root):
    row = request_row(db, swimmer)
    sign_in(client, root)
    assert post(client, f"/admin/feed-requests/{row.id}", {"status": "maybe"}).status_code == 422


def test_superusers_find_the_requests_from_me(client, db, root):
    sign_in(client, root)
    assert re.search(r'href="/admin/feed-requests"', client.get("/me").text)
