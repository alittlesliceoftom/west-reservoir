"""Public pages. They read stored rows only: never the weather API, never the model."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import queries
from db.models import Snapshot, Water
from web.deps import get_db
from web.templating import render

router = APIRouter()


def active_water(db: Session, slug: str) -> Water:
    water = db.scalar(select(Water).where(Water.slug == slug, Water.archived_at.is_(None)))
    if water is None:
        raise HTTPException(404, f"No water called '{slug}'.")
    return water


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    return render(request, "home.html", {"waters": queries.home_listing(db)})


@router.get("/w/{slug}", response_class=HTMLResponse)
def water_page(request: Request, slug: str, db: Session = Depends(get_db)):
    water = active_water(db, slug)
    snapshot = db.get(Snapshot, water.id)
    payload = snapshot.payload if snapshot else None
    chart = None
    if payload and payload["latest_reading"]:
        chart = {k: payload[k] for k in ("readings", "air", "forecast", "trajectory", "latest_reading", "estimate_now")}
        chart["timezone"] = water.timezone
        chart["now"] = request.app.state.clock().isoformat()
    return render(request, "water.html", {
        "water": water,
        "p": payload,
        "chart": chart,
        "updating": water.needs_recompute or snapshot is None,
        "first": snapshot is None,
        "since": snapshot.built_at.isoformat() if snapshot else "",
    })


@router.get("/w/{slug}/status", response_class=HTMLResponse)
def water_status(request: Request, slug: str, since: str = "", db: Session = Depends(get_db)):
    """HTMX poll while a recompute is pending: tells the page to reload once a newer snapshot exists."""
    water = active_water(db, slug)
    snapshot = db.get(Snapshot, water.id)
    newer = snapshot is not None and (not since or snapshot.built_at > datetime.fromisoformat(since))
    if newer:
        return HTMLResponse("", headers={"HX-Refresh": "true"})
    return render(request, "_status.html", {"water": water, "since": since, "first": snapshot is None})
