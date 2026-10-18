"""
The web app. Server-rendered Jinja with HTMX; it reads snapshots and writes
readings, and never computes a forecast.

    uvicorn web.app:app
"""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from db.session import session_factory as default_session_factory
from web.auth import AuthMiddleware, LoginRequired, SupabaseAuth, TokenVerifier, login_redirect
from web.config import Settings
from feeds.fetch import fetch_json
from forecast.openmeteo import OpenMeteoClient
from web.routes import admin, auth, create, feeds, health, pages, readings
from web.templating import render


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_app(settings: Settings | None = None, session_factory=None, clock=None, supabase=None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    app = FastAPI(title="swimtemps", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.session_factory = session_factory or default_session_factory()
    app.state.clock = clock or utcnow
    app.state.verifier = TokenVerifier(settings)
    app.state.supabase = supabase or SupabaseAuth(settings)
    app.state.openmeteo = OpenMeteoClient()
    # Only the feed setup's Test and Turn-on POSTs call it.
    app.state.feed_fetch = fetch_json
    app.add_middleware(AuthMiddleware)

    app.mount("/static", StaticFiles(directory=str(Path(__file__).with_name("static"))), name="static")
    app.include_router(pages.router)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(readings.router)
    app.include_router(create.router)
    app.include_router(admin.router)
    app.include_router(feeds.router)

    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, exc: LoginRequired):
        return login_redirect(request)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith(("/api/", "/healthz")) or exc.status_code < 400:
            from fastapi.exception_handlers import http_exception_handler
            return await http_exception_handler(request, exc)
        message = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
        return render(request, "error.html", {"status": exc.status_code, "message": message},
                      status_code=exc.status_code, headers=getattr(exc, "headers", None))

    return app


def __getattr__(name):
    # `uvicorn web.app:app` builds the app on first access, from the environment.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
