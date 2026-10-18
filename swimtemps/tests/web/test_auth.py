import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from db.models import User
from tests.factories import make_user, make_water
from tests.web.conftest import NOW
from tests.web.helpers import FakeSupabase, csrf, post, sign_in, token_for
from web.app import create_app
from web.auth import ACCESS_COOKIE, PKCE_COOKIE, REFRESH_COOKIE, SupabaseAuth
from web.config import ConfigError, Settings


def make_client(settings, session_factory, supabase=None):
    app = create_app(settings=settings, session_factory=session_factory, clock=lambda: NOW, supabase=supabase)
    return TestClient(app)


def test_a_valid_token_signs_the_user_in(client, db):
    user = make_user(db, name="Alice")
    sign_in(client, user)
    page = client.get("/").text
    assert "Alice" in page and "Sign out" in page


def test_a_token_with_a_bad_signature_is_anonymous_and_cleared(client, db):
    user = make_user(db)
    client.cookies.set(ACCESS_COOKIE, token_for(user, secret="some-other-secret-that-is-also-long-enough"))
    response = client.get("/")
    assert "Sign in" in response.text
    assert f'{ACCESS_COOKIE}=""' in response.headers.get("set-cookie", "")


def test_an_expired_token_is_refreshed_once(settings, session_factory, db):
    user = make_user(db, name="Alice")
    fake = FakeSupabase(user=user)
    with make_client(settings, session_factory, fake) as c:
        sign_in(c, user, expires_in=-60)
        response = c.get("/")
    assert "Alice" in response.text
    assert fake.refreshed == [f"refresh-{user.id}"]
    assert ACCESS_COOKIE in response.headers["set-cookie"]


def test_an_expired_token_whose_refresh_fails_signs_out(settings, session_factory, db):
    user = make_user(db)
    with make_client(settings, session_factory, FakeSupabase(user=user, fail=True)) as c:
        sign_in(c, user, expires_in=-60)
        response = c.get("/")
    assert "Sign in" in response.text
    assert f'{REFRESH_COOKIE}=""' in response.headers["set-cookie"]


def test_an_expired_token_without_a_refresh_token_signs_out(client, db):
    user = make_user(db)
    client.cookies.set(ACCESS_COOKIE, token_for(user, expires_in=-60))
    assert "Sign in" in client.get("/").text


def test_a_first_sign_in_creates_the_user_row(client, db):
    ghost = User(id=uuid.uuid4(), email="new@example.com", display_name="New Person", google_sub="g-123")
    sign_in(client, ghost)
    client.get("/")
    row = db.scalar(select(User).where(User.email == "new@example.com"))
    assert (row.id, row.google_sub, row.display_name) == (ghost.id, "g-123", "New Person")


def test_login_redirects_to_supabase_google_with_pkce(settings, session_factory):
    with make_client(settings, session_factory, SupabaseAuth(settings)) as c:
        response = c.get("/auth/login", params={"next": "/w/test-lake"}, follow_redirects=False)
    assert response.status_code == 303
    location = urlparse(response.headers["location"])
    query = parse_qs(location.query)
    assert (location.netloc, location.path) == ("example.supabase.co", "/auth/v1/authorize")
    assert query["provider"] == ["google"]
    assert query["redirect_to"] == ["http://testserver/auth/callback"]
    assert query["code_challenge_method"] == ["s256"]
    cookie = response.headers["set-cookie"]
    assert PKCE_COOKIE in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie


def test_callback_exchanges_the_code_and_lands_on_next(settings, session_factory, db):
    user = User(id=uuid.uuid4(), email="b@example.com", display_name="Bea", google_sub="g-9")
    fake = FakeSupabase(user=user)
    with make_client(settings, session_factory, fake) as c:
        c.get("/auth/login", params={"next": "/w/test-lake"}, follow_redirects=False)
        response = c.get("/auth/callback", params={"code": "abc"}, follow_redirects=False)
    assert fake.exchanged == [("abc", "verifier-123")]
    assert response.status_code == 303 and response.headers["location"] == "/w/test-lake"
    assert ACCESS_COOKIE in response.headers["set-cookie"]
    assert db.scalar(select(User).where(User.email == "b@example.com")).google_sub == "g-9"


@pytest.mark.parametrize("bad_next", ["//evil.example", "https://evil.example/", "javascript:alert(1)"])
def test_callback_never_redirects_off_site(settings, session_factory, db, bad_next):
    user = User(id=uuid.uuid4(), email="c@example.com", display_name="C", google_sub="g")
    with make_client(settings, session_factory, FakeSupabase(user=user)) as c:
        c.get("/auth/login", params={"next": bad_next}, follow_redirects=False)
        response = c.get("/auth/callback", params={"code": "abc"}, follow_redirects=False)
    assert response.headers["location"] == "/"


def test_callback_without_a_verifier_explains(client):
    response = client.get("/auth/callback", params={"code": "abc"})
    assert response.status_code == 400
    assert "sign-in" in response.text.lower()


def test_callback_with_a_provider_error_explains(client):
    response = client.get("/auth/callback", params={"error": "access_denied", "error_description": "User cancelled"})
    assert response.status_code == 400 and "User cancelled" in response.text


def test_dev_login_is_404_unless_enabled(client):
    assert client.get("/auth/dev-login").status_code == 404


def test_dev_login_signs_in_when_enabled(session_factory, db, settings):
    dev = Settings(**{**settings.__dict__, "env": "development", "dev_login": True, "supabase_url": "", "supabase_anon_key": ""})
    with make_client(dev, session_factory) as c:
        assert c.get("/auth/login", follow_redirects=False).headers["location"].startswith("/auth/dev-login")
        response = post(c, "/auth/dev-login", {"email": "dev@example.com", "name": "Dev", "next": "/"})
        assert response.status_code == 303
        assert "Dev" in c.get("/").text
    assert db.scalar(select(User).where(User.email == "dev@example.com")) is not None


def test_dev_login_is_refused_in_production():
    with pytest.raises(ConfigError):
        Settings(env="production", dev_login=True, supabase_url="https://x.supabase.co", supabase_anon_key="k").validate()


def test_a_post_without_the_csrf_token_is_refused(client, db):
    user = make_user(db)
    make_water(db, user)
    sign_in(client, user)
    csrf(client)
    response = client.post("/w/test-lake/readings", data={"temp": "14.0"}, follow_redirects=False)
    assert response.status_code == 403


def test_a_csrf_header_is_accepted_for_htmx(client, db):
    user = make_user(db)
    make_water(db, user)
    sign_in(client, user)
    token = csrf(client)
    response = client.post("/w/test-lake/readings/preview", data={"rows": "18/10/2026 14.0"},
                           headers={"X-CSRF-Token": token, "HX-Request": "true"})
    assert response.status_code == 200


def test_sign_out_clears_the_session(client, db):
    user = make_user(db)
    sign_in(client, user)
    response = post(client, "/auth/logout")
    assert response.status_code == 303
    assert f'{ACCESS_COOKIE}=""' in response.headers["set-cookie"]


def test_signed_out_visitors_are_sent_to_sign_in(client, db):
    make_water(db, make_user(db))
    response = client.get("/w/test-lake/readings", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/auth/login?next=/w/test-lake/readings"


def test_a_jwks_outage_does_not_sign_everyone_out(settings, session_factory, db, monkeypatch):
    import jwt as pyjwt
    user = make_user(db)
    with make_client(settings, session_factory) as c:
        def unreachable(token):
            raise pyjwt.PyJWKClientConnectionError("Supabase is not answering")
        c.app.state.verifier.verify = unreachable
        sign_in(c, user)
        response = c.get("/")
    assert "Sign in" in response.text
    assert ACCESS_COOKIE not in response.headers.get("set-cookie", "")
