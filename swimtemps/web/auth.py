"""
Sign-in: Supabase Auth with Google, server-side PKCE.

The browser never talks to Supabase. `/auth/login` asks Supabase for the
Google authorize URL and keeps the PKCE verifier in a short-lived httpOnly
cookie; `/auth/callback` exchanges the code for a session. The access and
refresh tokens live in httpOnly cookies. Every request verifies the access
token locally (the project's JWKS, or its HS256 secret for a legacy project);
an expired token is refreshed once through Supabase, otherwise the visitor is
signed out. Authorisation is this app's job: row-level security is not used.
"""

import hmac
import secrets
import uuid
from functools import cached_property
from urllib.parse import quote, urlparse

import jwt
from fastapi import Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from supabase_auth import SyncGoTrueClient, SyncMemoryStorage

from db.models import User
from web.config import Settings

ACCESS_COOKIE = "st_access"
REFRESH_COOKIE = "st_refresh"
PKCE_COOKIE = "st_pkce"
NEXT_COOKIE = "st_next"
CSRF_COOKIE = "st_csrf"
SESSION_DAYS = 30
AUDIENCE = "authenticated"


class LoginRequired(Exception):
    """Raised by `require_user`; answered with a redirect to sign in."""


class TokenVerifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def jwks(self) -> jwt.PyJWKClient:
        return jwt.PyJWKClient(f"{self.settings.supabase_url}/auth/v1/.well-known/jwks.json", cache_keys=True)

    def verify(self, token: str) -> dict:
        """Claims of a valid token. Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError."""
        algorithm = jwt.get_unverified_header(token).get("alg")
        if algorithm == "HS256":
            if not self.settings.supabase_jwt_secret:
                raise jwt.InvalidTokenError("HS256 token but no SUPABASE_JWT_SECRET configured")
            return jwt.decode(token, self.settings.supabase_jwt_secret, algorithms=["HS256"], audience=AUDIENCE)
        key = self.jwks.get_signing_key_from_jwt(token)
        return jwt.decode(token, key, algorithms=["ES256", "RS256"], audience=AUDIENCE)


class SupabaseAuth:
    """The three Supabase Auth calls this app makes, via supabase-auth (pinned 2.33.0)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _client(self, storage=None) -> SyncGoTrueClient:
        key = self.settings.supabase_anon_key
        return SyncGoTrueClient(
            url=f"{self.settings.supabase_url}/auth/v1",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            flow_type="pkce", persist_session=False, auto_refresh_token=False,
            storage=storage or SyncMemoryStorage(),
        )

    def login_url(self, redirect_to: str) -> tuple[str, str]:
        """The Google authorize URL and the PKCE verifier that must come back with the code."""
        storage = SyncMemoryStorage()
        response = self._client(storage).sign_in_with_oauth(
            {"provider": "google", "options": {"redirect_to": redirect_to}}
        )
        verifier = next(v for k, v in storage.storage.items() if k.endswith("-code-verifier"))
        return response.url, verifier

    def exchange(self, code: str, verifier: str):
        return self._client().exchange_code_for_session({"auth_code": code, "code_verifier": verifier}).session

    def refresh(self, refresh_token: str):
        return self._client().refresh_session(refresh_token).session


def google_sub(claims: dict, identities=None) -> str | None:
    """Google's own subject id, so identity survives a move off Supabase Auth."""
    for identity in identities or []:
        if getattr(identity, "provider", None) == "google":
            data = getattr(identity, "identity_data", None) or {}
            if data.get("sub"):
                return data["sub"]
    metadata = claims.get("user_metadata") or {}
    return metadata.get("provider_id") or metadata.get("sub")


def upsert_user(session: Session, claims: dict, identities=None) -> User:
    user = session.get(User, uuid.UUID(claims["sub"]))
    metadata = claims.get("user_metadata") or {}
    name = metadata.get("full_name") or metadata.get("name") or (claims.get("email") or "").split("@")[0]
    if user is None:
        user = User(id=uuid.UUID(claims["sub"]), email=claims.get("email"), display_name=name,
                    google_sub=google_sub(claims, identities))
        session.add(user)
    else:
        user.email = claims.get("email") or user.email
        user.google_sub = user.google_sub or google_sub(claims, identities)
    session.commit()
    return user


def cookie_options(settings: Settings, max_age: int, path: str = "/") -> dict:
    return {"httponly": True, "secure": settings.secure_cookies, "samesite": "lax", "max_age": max_age, "path": path}


def set_session_cookies(response, settings: Settings, access_token: str, refresh_token: str) -> None:
    age = SESSION_DAYS * 86400
    response.set_cookie(ACCESS_COOKIE, access_token, **cookie_options(settings, age))
    response.set_cookie(REFRESH_COOKIE, refresh_token, **cookie_options(settings, age))


def clear_session_cookies(response, settings: Settings) -> None:
    for name in (ACCESS_COOKIE, REFRESH_COOKIE):
        response.delete_cookie(name, path="/", secure=settings.secure_cookies, httponly=True, samesite="lax")


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Puts `request.state.user` (a User or None) and `request.state.csrf_token`
    on every request, refreshing or clearing the session cookies as needed.
    """

    async def dispatch(self, request: Request, call_next):
        app = request.app
        settings: Settings = app.state.settings
        request.state.user = None
        request.state.csrf_token = request.cookies.get(CSRF_COOKIE) or secrets.token_urlsafe(32)

        new_tokens, clear = None, False
        token = request.cookies.get(ACCESS_COOKIE)
        if token:
            claims = None
            try:
                claims = app.state.verifier.verify(token)
            except jwt.ExpiredSignatureError:
                refresh_token = request.cookies.get(REFRESH_COOKIE)
                if refresh_token:
                    try:
                        session = await run_in_threadpool(app.state.supabase.refresh, refresh_token)
                        claims = app.state.verifier.verify(session.access_token)
                        new_tokens = (session.access_token, session.refresh_token)
                    except Exception:
                        claims = None
            except (jwt.InvalidTokenError, jwt.PyJWKClientError):
                claims = None
            if claims is None:
                clear = True
            else:
                request.state.user = await run_in_threadpool(self._load_user, app, claims)

        response = await call_next(request)
        if new_tokens:
            set_session_cookies(response, settings, *new_tokens)
        if clear:
            clear_session_cookies(response, settings)
        if request.cookies.get(CSRF_COOKIE) != request.state.csrf_token:
            response.set_cookie(CSRF_COOKIE, request.state.csrf_token,
                                **cookie_options(settings, SESSION_DAYS * 86400))
        return response

    @staticmethod
    def _load_user(app, claims: dict) -> User:
        with app.state.session_factory() as session:
            return upsert_user(session, claims)


def current_user(request: Request) -> User | None:
    return request.state.user


def require_user(request: Request) -> User:
    if request.state.user is None:
        raise LoginRequired()
    return request.state.user


def login_redirect(request: Request) -> RedirectResponse:
    if request.method == "GET":
        path = request.url.path
    else:
        referer = urlparse(request.headers.get("referer", ""))
        path = referer.path if referer.netloc == request.url.netloc else "/"
    target = f"/auth/login?next={quote(safe_next(path), safe='/')}"
    response = RedirectResponse(target, status_code=303)
    if request.headers.get("HX-Request"):
        response.headers["HX-Redirect"] = target
    return response


def safe_next(value: str | None) -> str:
    """Only same-site paths; anything else goes home."""
    if not value:
        return "/"
    if not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    return value


async def verify_csrf(request: Request) -> None:
    """Double-submit check for every state-changing request."""
    from fastapi import HTTPException

    expected = request.cookies.get(CSRF_COOKIE, "")
    sent = request.headers.get("X-CSRF-Token")
    if sent is None:
        form = await request.form()
        sent = form.get("csrf_token", "")
    if not expected or not hmac.compare_digest(str(sent), expected):
        raise HTTPException(403, "This form has expired. Go back, reload the page and try again.")
