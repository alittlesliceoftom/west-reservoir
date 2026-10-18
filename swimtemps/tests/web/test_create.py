import json
from datetime import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from db import queries
from db.models import Water, WaterMember
from forecast.openmeteo import OpenMeteoClient
from tests.factories import make_user, make_water
from tests.web.conftest import NOW
from tests.web.helpers import post, sign_in
from web.slugs import slugify, slugify_unique

FIXTURES = Path(__file__).parents[1] / "fixtures"

DETAILS = {
    "lat": "51.4552", "lon": "-2.5967", "timezone": "Europe/London",
    "name": "Bristol Harbourside Lido", "water_type": "still", "depth_class": "shallow", "reading_time": "08:00",
}


@pytest.fixture
def openmeteo(app):
    calls = []

    def handler(request):
        calls.append(request.url)
        if "geocoding" in request.url.host:
            return httpx.Response(200, json=json.loads((FIXTURES / "openmeteo_geocode.json").read_text()))
        return httpx.Response(200, json=json.loads((FIXTURES / "openmeteo_point.json").read_text()))

    app.state.openmeteo = OpenMeteoClient(http=httpx.Client(transport=httpx.MockTransport(handler)))
    return calls


@pytest.fixture
def creator(db, client):
    user = make_user(db, email="maker@example.com", name="Maker")
    sign_in(client, user)
    return user


# --- slugs -----------------------------------------------------------------

def test_slugify_is_url_safe_and_readable():
    assert slugify("Bristol Harbourside Lido") == "bristol-harbourside-lido"
    assert slugify("  Lac d'Annecy (south) ") == "lac-d-annecy-south"
    assert slugify("Ærø Strand") == "aero-strand"
    assert slugify("!!!") == "water"


def test_identical_names_get_distinct_slugs(db):
    user = make_user(db)
    make_water(db, user, name="Lido", slug="lido")
    assert slugify_unique(db, "Lido") == "lido-2"
    make_water(db, user, name="Lido", slug="lido-2")
    assert slugify_unique(db, "Lido") == "lido-3"


def test_archived_waters_keep_their_slug(db):
    make_water(db, make_user(db), name="Lido", slug="lido", archived_at=NOW)
    assert slugify_unique(db, "Lido") == "lido-2"


def test_reserved_paths_are_not_slugs(db):
    assert slugify_unique(db, "New") != "new"


# --- nearby -----------------------------------------------------------------

def test_nearby_finds_waters_within_300_metres_nearest_first(db):
    user = make_user(db)
    make_water(db, user, name="Near", lat=51.4560, lon=-2.5967)   # ~90 m north
    make_water(db, user, name="Edge", lat=51.4578, lon=-2.5967)   # ~290 m
    make_water(db, user, name="Far", lat=51.4600, lon=-2.5967)    # ~530 m
    make_water(db, user, name="Gone", lat=51.4553, lon=-2.5967, archived_at=NOW)
    names = [w.name for w in queries.nearby_waters(db, 51.4552, -2.5967)]
    assert names == ["Near", "Edge"]


# --- the wizard -------------------------------------------------------------

def test_creating_needs_sign_in(client):
    response = client.get("/new", follow_redirects=False)
    assert response.headers["location"] == "/auth/login?next=/new"


def test_step_two_has_search_and_a_map(client, creator):
    page = client.get("/new").text
    assert 'id="map"' in page and "leaflet" in page
    assert 'name="q"' in page


def test_geocode_proxies_open_meteo_with_country_and_region(client, creator, openmeteo):
    results = client.get("/api/geocode", params={"q": "Bristol"}).json()
    assert results[0] == {"name": "Bristol", "lat": 51.45523, "lon": -2.59665, "region": "England",
                          "country": "United Kingdom", "timezone": "Europe/London"}


def test_point_preview_shows_weather_elevation_timezone_and_nearby(client, db, creator, openmeteo):
    make_water(db, creator, name="Existing Lido", lat=51.4553, lon=-2.5968)
    body = client.get("/api/point", params={"lat": 51.4552, "lon": -2.5967}).json()
    assert body["air_temp"] == 8.8 and body["elevation"] == 13.0 and body["timezone"] == "Europe/London"
    assert body["nearby"][0]["name"] == "Existing Lido"
    assert body["nearby"][0]["slug"] == "existing-lido"


def test_point_preview_rejects_impossible_coordinates(client, creator, openmeteo):
    assert client.get("/api/point", params={"lat": 91, "lon": 0}).status_code == 422
    assert openmeteo == []


def test_point_preview_reports_open_meteo_trouble_without_crashing(client, creator, app):
    app.state.openmeteo = OpenMeteoClient(http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    response = client.get("/api/point", params={"lat": 51.45, "lon": -2.6})
    assert response.status_code == 502
    assert "weather" in response.json()["detail"].lower()


def test_details_step_carries_the_location(client, creator):
    response = post(client, "/new/details", {"lat": "51.4552", "lon": "-2.5967", "timezone": "Europe/London"})
    assert response.status_code == 200
    page = response.text
    assert 'name="lat" value="51.4552"' in page and 'name="timezone" value="Europe/London"' in page
    assert 'name="reading_time" value="07:00"' in page
    for label in ("Lake or reservoir", "River", "Harbour", "Shallow (under 2 m)", "Not sure"):
        assert label in page


def test_creating_a_water_makes_the_creator_its_admin_and_flags_it(client, db, creator):
    response = post(client, "/new", DETAILS)
    assert response.status_code == 303
    assert response.headers["location"] == "/w/bristol-harbourside-lido/first-readings"
    water = db.scalar(select(Water).where(Water.slug == "bristol-harbourside-lido"))
    assert (water.name, water.lat, water.lon, water.timezone) == ("Bristol Harbourside Lido", 51.4552, -2.5967, "Europe/London")
    assert (water.water_type, water.depth_class, water.default_reading_time) == ("still", "shallow", time(8, 0))
    assert water.needs_recompute is True and water.created_by == creator.id
    member = db.get(WaterMember, (water.id, creator.id))
    assert member.role == "admin"


@pytest.mark.parametrize("field, value", [
    ("lat", "95"), ("lon", "-200"), ("lat", "north"), ("timezone", "Mars/Olympus"),
    ("water_type", "sea"), ("depth_class", "bottomless"), ("name", "   "), ("reading_time", "25:00"),
    ("name", "x" * 81),
])
def test_bad_details_are_refused_with_a_reason(client, db, creator, field, value):
    response = post(client, "/new", {**DETAILS, field: value})
    assert response.status_code == 422
    assert db.scalar(select(Water)) is None


def test_first_readings_page_offers_paste_and_skip(client, db, creator):
    post(client, "/new", DETAILS)
    page = client.get("/w/bristol-harbourside-lido/first-readings").text
    assert 'hx-post="/w/bristol-harbourside-lido/readings/preview"' in page
    assert 'href="/w/bristol-harbourside-lido"' in page and "Skip" in page


def test_after_skipping_the_water_page_is_computing(client, db, creator):
    post(client, "/new", DETAILS)
    assert "Computing your first forecast..." in client.get("/w/bristol-harbourside-lido").text


def test_the_new_water_appears_on_the_home_page(client, db, creator):
    post(client, "/new", DETAILS)
    assert "Bristol Harbourside Lido" in client.get("/").text
