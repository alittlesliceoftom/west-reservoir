"""Signing in and posting forms in web tests, through the real middleware."""

import time

import jwt

from tests.web.conftest import JWT_SECRET
from web.auth import ACCESS_COOKIE, CSRF_COOKIE, REFRESH_COOKIE


def token_for(user, secret=JWT_SECRET, expires_in=3600, **claims) -> str:
    now = int(time.time())
    body = {
        "sub": str(user.id), "email": user.email, "aud": "authenticated", "role": "authenticated",
        "iat": now, "exp": now + expires_in,
        "user_metadata": {"full_name": user.display_name, "provider_id": user.google_sub},
        **claims,
    }
    return jwt.encode(body, secret, algorithm="HS256")


def sign_in(client, user, **kwargs):
    client.cookies.set(ACCESS_COOKIE, token_for(user, **kwargs))
    client.cookies.set(REFRESH_COOKIE, "refresh-" + str(user.id))


def csrf(client) -> str:
    if CSRF_COOKIE not in client.cookies:
        client.get("/")
    return client.cookies[CSRF_COOKIE]


def post(client, url, data=None, **kwargs):
    return client.post(url, data={"csrf_token": csrf(client), **(data or {})}, follow_redirects=False, **kwargs)


class FakeSession:
    def __init__(self, access_token, refresh_token="new-refresh", user=None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.user = user


class FakeSupabase:
    """Stands in for the two Supabase network calls: code exchange and refresh."""

    def __init__(self, user=None, fail=False):
        self.user = user
        self.fail = fail
        self.refreshed = []
        self.exchanged = []

    def login_url(self, redirect_to):
        return "https://example.supabase.co/auth/v1/authorize?provider=google", "verifier-123"

    def exchange(self, code, verifier):
        self.exchanged.append((code, verifier))
        if self.fail:
            raise RuntimeError("exchange failed")
        return FakeSession(token_for(self.user), user=self.user)

    def refresh(self, refresh_token):
        self.refreshed.append(refresh_token)
        if self.fail:
            raise RuntimeError("refresh failed")
        return FakeSession(token_for(self.user))
