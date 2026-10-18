import pytest
from fastapi.testclient import TestClient

from tests.web.conftest import NOW
from web.app import create_app
from web.auth import CSRF_COOKIE
from web.config import ConfigError, Settings


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SITE_URL", "https://swimtemps.example/")
    monkeypatch.setenv("SUPABASE_URL", "https://ref.supabase.co/")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.delenv("DEV_LOGIN", raising=False)
    s = Settings.from_env()
    assert (s.env, s.site_url, s.supabase_url, s.dev_login) == (
        "production", "https://swimtemps.example", "https://ref.supabase.co", False)
    assert s.secure_cookies is True


def test_production_without_supabase_refuses_to_start(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    with pytest.raises(ConfigError, match="SUPABASE_URL"):
        Settings.from_env()


def test_cookies_are_secure_on_https(session_factory):
    settings = Settings(env="production", site_url="https://swimtemps.example",
                        supabase_url="https://ref.supabase.co", supabase_anon_key="anon")
    with TestClient(create_app(settings=settings, session_factory=session_factory, clock=lambda: NOW)) as c:
        cookie = c.get("/").headers["set-cookie"]
    assert CSRF_COOKIE in cookie and "Secure" in cookie and "HttpOnly" in cookie
