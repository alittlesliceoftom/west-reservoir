"""
Feed setup for a water's admins: paste an AI's `swimtemps-feed v1` block or
fill in the fields, Test (one fetch, nothing stored), confirm permission and
Turn on. Also pause, resume, switching back to manual, and overruling the
cleaning's flags. The Test and Turn-on POSTs are the only web requests that
contact a feed; pages never do.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import queries
from db.models import Feed, Reading, User, Water
from feeds.block import PROMPT, RECIPE_KEYS, REQUIRED_FOR, check_value, missing_keys, parse_block
from feeds.clean import clean
from feeds.fetch import FetchError
from feeds.recipe import RecipeError, read_feed
from feeds.thin import thin_hourly
from web.auth import require_user, verify_csrf
from web.deps import get_db
from web.routes.admin import admin_water, back, standing_reading
from web.templating import render
from worker.feeds import checked

router = APIRouter()

ATTESTATION = "I have the venue's permission to share this data, or it is publicly published."
CREDIT_KEYS = ("source_name", "source_link")
SHOWN_ROWS = 20
STOPS = {
    "not_automated": (
        "This looks like a source that is updated by hand. swimtemps takes feeds only from sensors and "
        "loggers, so this should stay a manual water, where anyone can add readings. If the AI got that "
        "wrong, request a feed and tell us why."),
    "terms_forbid": (
        "The venue's or vendor's terms forbid automated access, so swimtemps cannot fetch this feed without "
        "their agreement. Request a feed and we can ask them."),
    "no_block": "Paste the block that starts with swimtemps-feed v1 and ends with end.",
    "bad_version": "Paste the block that starts with swimtemps-feed v1 and ends with end.",
}


def form_config(form: dict) -> tuple[dict, list[str]]:
    """The recipe and credit fields from a form, checked as the block parser checks them."""
    config, errors = {}, []
    for key in RECIPE_KEYS:
        value = str(form.get(key, "")).strip()
        if not value:
            continue
        try:
            config[key] = check_value(key, value)
        except ValueError as error:
            errors.append(str(error))
    unused = [k for o, keys in REQUIRED_FOR.items() if o != config.get("orientation") for k in keys]
    unused += ["rows_path", "header_row"] if config.get("orientation") != "rows" else []
    for key in unused:
        config.pop(key, None)
    errors += [f"{key} is needed" for key in missing_keys(config)]
    return config, errors


def config_token(config: dict) -> str:
    """Names the exact configuration a Test ran on, so Turn on can require one."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def field_values(config: dict) -> dict:
    values = {k: v for k, v in config.items() if k in RECIPE_KEYS}
    if "header_row" in values:
        values["header_row"] = "yes" if values["header_row"] else "no"
    return {k: str(v) for k, v in values.items()}


@dataclass
class TestResult:
    read: int = 0
    rows: list = field(default_factory=list)          # (at, temp, reason) after checks and thinning
    errors: list[str] = field(default_factory=list)   # rows that could not be read or failed the checks
    failure: str | None = None                        # the fetch or the recipe failed

    @property
    def flagged(self) -> list:
        return [r for r in self.rows if r[2]]


def run_test(request: Request, db: Session, water: Water, config: dict) -> TestResult:
    """One fetch, read, checked, thinned and cleaned in memory. Stores nothing."""
    now = request.app.state.clock()
    recipe_config = {k: v for k, v in config.items() if k not in CREDIT_KEYS}
    try:
        payload = request.app.state.feed_fetch(config["url"])
        parsed = read_feed("json_table", payload, recipe_config, water.timezone, water.default_reading_time)
    except (FetchError, RecipeError) as error:
        return TestResult(failure=str(error))
    rows, rejected = checked(parsed.rows, now)
    rows = thin_hourly(rows)
    air = queries.weather_frame(db, water.id)
    reasons = clean(rows, [], pd.Series(air["air_temp"].to_numpy(), index=pd.DatetimeIndex(air["valid_at"])))
    return TestResult(read=len(parsed.rows) + len(parsed.errors), errors=parsed.errors + rejected,
                      rows=[(at, temp, reason) for (at, temp), reason in zip(rows, reasons)])


def current_feed(db: Session, water: Water) -> Feed | None:
    return db.scalar(select(Feed).where(Feed.water_id == water.id))


def setup_page(request, db, water, *, status_code=200, fields=None, block="", parsed=None, stop=None,
               errors=None, test=None, config=None):
    feed = current_feed(db, water)
    if fields is None:
        fields = field_values({**(feed.config if feed else {}),
                               **({"source_name": feed.source_name, "source_link": feed.source_link} if feed else {})})
    return render(request, "feed_setup.html", {
        "water": water, "feed": feed, "prompt": PROMPT, "fields": fields, "block": block, "parsed": parsed,
        "stop": STOPS.get(stop) if stop else None, "errors": errors or [], "test": test,
        "token": config_token(config) if config and test and not test.failure else "",
        "attestation": ATTESTATION, "shown": SHOWN_ROWS,
        "message": {"on": "The feed is on. Readings arrive within a minute.", "paused": "The feed is paused.",
                    "resumed": "The feed is on again.", "manual": "This water is back to manual readings. "
                    "Its feed readings are kept, and the feed can be turned on again here."}.get(
                        request.query_params.get("done", "")),
    }, status_code=status_code)


def _form(request_form) -> dict:
    return {k: str(v) for k, v in request_form.items()}


@router.get("/w/{slug}/feed", response_class=HTMLResponse)
def setup(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    return setup_page(request, db, admin_water(db, slug, user))


@router.post("/w/{slug}/feed/paste", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
def paste(request: Request, slug: str, block: str = Form(""), db: Session = Depends(get_db),
          user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    parsed = parse_block(block)
    if parsed.stop:
        return setup_page(request, db, water, block=block, stop=parsed.stop)
    return setup_page(request, db, water, block=block, parsed=parsed, fields=field_values(parsed.config))


@router.post("/w/{slug}/feed/test", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def test(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    form = _form(await request.form())
    config, errors = form_config(form)
    if errors:
        return setup_page(request, db, water, status_code=422, fields=form, errors=errors)
    result = run_test(request, db, water, config)
    return setup_page(request, db, water, status_code=422 if result.failure else 200, fields=form,
                      test=result, config=config)


@router.post("/w/{slug}/feed/enable", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def turn_on(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    form = _form(await request.form())
    config, errors = form_config(form)
    if errors:
        return setup_page(request, db, water, status_code=422, fields=form, errors=errors)
    if form.get("tested") != config_token(config):
        return setup_page(request, db, water, status_code=422, fields=form,
                          errors=["Press Test first, and turn the feed on from its result."])
    result = run_test(request, db, water, config)
    if form.get("attest") != "yes":
        return setup_page(request, db, water, status_code=422, fields=form, test=result, config=config,
                          errors=["Tick the box to confirm you have permission to share this data."])
    if result.failure or not result.rows:
        return setup_page(request, db, water, status_code=422, fields=form, test=result, config=config,
                          errors=["The feed did not pass its test this time, so it was not turned on."])

    now = request.app.state.clock()
    feed = current_feed(db, water) or Feed(water_id=water.id, created_by=user.id)
    feed.recipe = "json_table"
    feed.config = {k: v for k, v in config.items() if k not in CREDIT_KEYS}
    feed.source_name, feed.source_link = config["source_name"], config["source_link"]
    feed.state, feed.last_polled_at, feed.last_error = "on", None, None
    feed.attested_by, feed.attested_at, feed.attested_statement = user.id, now, ATTESTATION
    db.add(feed)
    water.feed_mode = "feed"
    queries.mark_dirty(db, water.id)
    db.commit()
    return RedirectResponse(f"/w/{water.slug}/feed?done=on", status_code=303)


def _feed_or_404(db: Session, water: Water) -> Feed:
    feed = current_feed(db, water)
    if feed is None:
        raise HTTPException(404, "This water has no feed.")
    return feed


def _change(request, db, water, done, **changes):
    feed = _feed_or_404(db, water)
    for key, value in changes.items():
        setattr(feed, key, value)
    queries.mark_dirty(db, water.id)
    db.commit()
    return RedirectResponse(f"/w/{water.slug}/feed?done={done}", status_code=303)


@router.post("/w/{slug}/feed/pause", dependencies=[Depends(verify_csrf)])
def pause(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    return _change(request, db, admin_water(db, slug, user), "paused", state="paused")


@router.post("/w/{slug}/feed/resume", dependencies=[Depends(verify_csrf)])
def resume(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    # Never-polled feeds are polled within a minute by the worker's drain.
    return _change(request, db, admin_water(db, slug, user), "resumed", state="on", last_polled_at=None)


@router.post("/w/{slug}/feed/disable", dependencies=[Depends(verify_csrf)])
def disable(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    water.feed_mode = "manual"
    return _change(request, db, water, "manual", state="off")


@router.post("/w/{slug}/admin/readings/{reading_id}/override", dependencies=[Depends(verify_csrf)])
def overrule(request: Request, slug: str, reading_id: int, decision: str = Form(""),
             db: Session = Depends(get_db), user: User = Depends(require_user)):
    """An admin's call on a feed reading: `accept` (it was real), `reject` (it is wrong) or `clear`."""
    water = admin_water(db, slug, user)
    reading: Reading = standing_reading(db, water, reading_id)
    if decision not in ("accept", "reject", "clear"):
        raise HTTPException(422, "Choose accept, reject or clear.")
    if reading.source != "feed":
        raise HTTPException(422, "Typed readings are corrected or deleted instead.")
    now: datetime = request.app.state.clock()
    reading.flag_override = None if decision == "clear" else decision
    reading.override_by, reading.override_at = user.id, now
    queries.mark_dirty(db, water.id)
    db.commit()
    return back(water, "overruled")
