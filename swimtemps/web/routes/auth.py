"""Sign in with Google via Supabase, sign out, and a development-only sign-in."""

import time
import uuid

import jwt
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from web.auth import (
    AUDIENCE,
    NEXT_COOKIE,
    PKCE_COOKIE,
    clear_session_cookies,
    cookie_options,
    safe_next,
    set_session_cookies,
    upsert_user,
    verify_csrf,
)
from web.templating import render

router = APIRouter(prefix="/auth")

PKCE_SECONDS = 600
DEV_NAMESPACE = uuid.UUID("6f1c4a52-61f7-4c39-9a51-3c1f5b1b0d10")


def dev_login_enabled(request: Request) -> bool:
    settings = request.app.state.settings
    return settings.dev_login and not settings.production


@router.get("/login")
def login(request: Request, next: str = "/"):
    settings = request.app.state.settings
    target = safe_next(next)
    if not settings.supabase_url:
        if dev_login_enabled(request):
            return RedirectResponse(f"/auth/dev-login?next={target}", status_code=303)
        raise HTTPException(503, "Sign-in is not configured on this server yet.")
    url, verifier = request.app.state.supabase.login_url(f"{settings.site_url}/auth/callback")
    response = RedirectResponse(url, status_code=303)
    response.set_cookie(PKCE_COOKIE, verifier, **cookie_options(settings, PKCE_SECONDS, path="/auth"))
    response.set_cookie(NEXT_COOKIE, target, **cookie_options(settings, PKCE_SECONDS, path="/auth"))
    return response


@router.get("/callback", response_class=HTMLResponse)
async def callback(request: Request, code: str = "", error: str = "", error_description: str = ""):
    settings = request.app.state.settings
    if error:
        return render(request, "error.html", {"status": 400, "message": f"Sign-in did not complete: {error_description or error}."},
                      status_code=400)
    verifier = request.cookies.get(PKCE_COOKIE)
    if not code or not verifier:
        return render(request, "error.html", {
            "status": 400, "message": "This sign-in link has expired or was opened in another browser. Please sign in again.",
        }, status_code=400)
    try:
        session = await run_in_threadpool(request.app.state.supabase.exchange, code, verifier)
        claims = request.app.state.verifier.verify(session.access_token)
    except Exception:
        return render(request, "error.html", {"status": 400, "message": "Sign-in failed. Please try again."}, status_code=400)

    identities = getattr(session.user, "identities", None)
    with request.app.state.session_factory() as db:
        await run_in_threadpool(upsert_user, db, claims, identities)

    response = RedirectResponse(safe_next(request.cookies.get(NEXT_COOKIE)), status_code=303)
    set_session_cookies(response, settings, session.access_token, session.refresh_token)
    response.delete_cookie(PKCE_COOKIE, path="/auth")
    response.delete_cookie(NEXT_COOKIE, path="/auth")
    return response


@router.post("/logout", dependencies=[Depends(verify_csrf)])
def logout(request: Request):
    response = RedirectResponse("/", status_code=303)
    clear_session_cookies(response, request.app.state.settings)
    return response


@router.get("/dev-login", response_class=HTMLResponse)
def dev_login_form(request: Request, next: str = "/"):
    if not dev_login_enabled(request):
        raise HTTPException(404, "Not found.")
    return render(request, "dev_login.html", {"next": safe_next(next)})


@router.post("/dev-login", dependencies=[Depends(verify_csrf)])
def dev_login(request: Request, email: str = Form(...), name: str = Form(""), next: str = Form("/")):
    """A signed token for any email, using the configured HS256 secret. Development only."""
    if not dev_login_enabled(request):
        raise HTTPException(404, "Not found.")
    settings = request.app.state.settings
    if not settings.supabase_jwt_secret:
        raise HTTPException(503, "Set SUPABASE_JWT_SECRET to any long string to use the development sign-in.")
    email = email.strip().lower()
    now = int(time.time())
    claims = {
        "sub": str(uuid.uuid5(DEV_NAMESPACE, email)), "email": email, "aud": AUDIENCE, "role": "authenticated",
        "iat": now, "exp": now + 3600, "user_metadata": {"full_name": name.strip() or email.split("@")[0]},
    }
    token = jwt.encode(claims, settings.supabase_jwt_secret, algorithm="HS256")
    with request.app.state.session_factory() as db:
        upsert_user(db, claims)
    response = RedirectResponse(safe_next(next), status_code=303)
    set_session_cookies(response, settings, token, "dev-login-has-no-refresh-token")
    return response
