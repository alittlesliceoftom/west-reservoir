import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from db.models import Feed
from forecast.openmeteo import OpenMeteoClient
from web.routes.feeds import config_token, form_config
from tests.factories import make_user, make_water
from tests.web.helpers import post, sign_in
from tests.web.test_feed_setup import FIELDS, FakeFetch
from web.ratelimit import LIMITS, RateLimiter

FIXTURES = Path(__file__).parents[1] / "fixtures"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- the limiter -----------------------------------------------------------

def test_the_chosen_limits():
    assert LIMITS == {"geocode": (30, 60), "point": (30, 60), "feed_fetch": (10, 60)}


def test_a_key_gets_its_limit_then_is_refused():
    limiter = RateLimiter({"search": (3, 60)}, clock=Clock())
    assert [limiter.allow("search", 1) for _ in range(4)] == [True, True, True, False]


def test_the_window_slides():
    clock = Clock()
    limiter = RateLimiter({"search": (2, 60)}, clock=clock)
    limiter.allow("search", 1)
    clock.now += 30
    limiter.allow("search", 1)
    assert not limiter.allow("search", 1)
    clock.now += 31   # the first hit is now more than 60 s old
    assert limiter.allow("search", 1)
    assert not limiter.allow("search", 1)


def test_refused_hits_do_not_extend_the_wait():
    clock = Clock()
    limiter = RateLimiter({"search": (1, 60)}, clock=clock)
    limiter.allow("search", 1)
    for _ in range(5):
        clock.now += 10
        limiter.allow("search", 1)
    clock.now += 11
    assert limiter.allow("search", 1)


def test_keys_and_buckets_are_counted_apart():
    limiter = RateLimiter({"a": (1, 60), "b": (1, 60)}, clock=Clock())
    assert limiter.allow("a", 1) and limiter.allow("a", 2) and limiter.allow("b", 1)
    assert not limiter.allow("a", 1)


def test_reset_forgets_every_hit():
    limiter = RateLimiter({"search": (1, 60)}, clock=Clock())
    limiter.allow("search", 1)
    limiter.reset()
    assert limiter.allow("search", 1)


# --- the routes ------------------------------------------------------------

@pytest.fixture
def limits(app):
    """Small limits, so a test reaches them in a few requests."""
    limiter = RateLimiter({"geocode": (2, 60), "point": (2, 60), "feed_fetch": (2, 60)}, clock=Clock())
    app.state.rate_limiter = limiter
    return limiter


@pytest.fixture
def openmeteo(app):
    calls = []

    def handler(request):
        calls.append(request.url)
        name = "openmeteo_geocode.json" if "geocoding" in request.url.host else "openmeteo_point.json"
        return httpx.Response(200, json=json.loads((FIXTURES / name).read_text()))

    app.state.openmeteo = OpenMeteoClient(http=httpx.Client(transport=httpx.MockTransport(handler)))
    return calls


def test_the_app_has_a_limiter_with_the_chosen_limits(app):
    assert isinstance(app.state.rate_limiter, RateLimiter) and app.state.rate_limiter.limits == LIMITS


@pytest.mark.parametrize("path, params", [
    ("/api/geocode", {"q": "Bristol"}),
    ("/api/point", {"lat": 51.45, "lon": -2.6}),
])
def test_wizard_lookups_over_the_limit_get_429_with_a_message(client, db, limits, openmeteo, path, params):
    sign_in(client, make_user(db))
    assert [client.get(path, params=params).status_code for _ in range(2)] == [200, 200]
    response = client.get(path, params=params)
    assert response.status_code == 429
    assert "minute" in response.json()["detail"]
    assert len(openmeteo) == 2


def test_wizard_limits_are_per_user(client, db, limits, openmeteo):
    first, second = make_user(db, email="a@example.com"), make_user(db, email="b@example.com")
    sign_in(client, first)
    for _ in range(2):
        client.get("/api/geocode", params={"q": "Bristol"})
    assert client.get("/api/geocode", params={"q": "Bristol"}).status_code == 429
    sign_in(client, second)
    assert client.get("/api/geocode", params={"q": "Bristol"}).status_code == 200


def test_feed_tests_over_the_limit_show_a_message_and_fetch_nothing(client, app, db, limits):
    admin = make_user(db, email="admin@example.com")
    make_water(db, admin, needs_recompute=False)
    fetch = FakeFetch()
    app.state.feed_fetch = fetch
    sign_in(client, admin)
    assert [post(client, "/w/test-lake/feed/test", FIELDS).status_code for _ in range(2)] == [200, 200]
    response = post(client, "/w/test-lake/feed/test", FIELDS)
    assert response.status_code == 429
    assert 'role="alert"' in response.text and "minute" in response.text
    assert len(fetch.calls) == 2


def test_turning_on_shares_the_feed_limit_and_stores_nothing_over_it(client, app, db, limits):
    admin = make_user(db, email="admin@example.com")
    make_water(db, admin, needs_recompute=False)
    fetch = FakeFetch()
    app.state.feed_fetch = fetch
    sign_in(client, admin)
    config, _ = form_config(FIELDS)
    for _ in range(2):
        post(client, "/w/test-lake/feed/test", FIELDS)
    response = post(client, "/w/test-lake/feed/enable", {**FIELDS, "tested": config_token(config), "attest": "yes"})
    assert response.status_code == 429
    assert 'role="alert"' in response.text and "minute" in response.text
    assert len(fetch.calls) == 2
    assert db.scalar(select(func.count()).select_from(Feed)) == 0
