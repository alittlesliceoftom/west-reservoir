"""
Create a water, in steps: sign in, place it on the map, describe it, add its
first readings. The location step is the only place the web app calls
Open-Meteo, to show the weather at the chosen spot.
"""

from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from db import queries
from db.models import DEPTH_CLASSES, WATER_TYPES, User, Water, WaterMember
from forecast.openmeteo import OpenMeteoError
from web.auth import require_user, verify_csrf
from web.deps import get_db
from web.routes.pages import active_water
from web.slugs import slugify_unique
from web.templating import render

router = APIRouter()

NAME_MAX = 80
NEARBY_METRES = 300


def check_location(lat: str, lon: str, timezone: str) -> tuple[dict, list[str]]:
    errors, values = [], {}
    try:
        values["lat"], values["lon"] = float(lat), float(lon)
        if not (-90 <= values["lat"] <= 90 and -180 <= values["lon"] <= 180):
            errors.append("That location is off the map. Place the pin again.")
    except ValueError:
        errors.append("Choose a location on the map.")
    try:
        ZoneInfo(timezone)
        values["timezone"] = timezone
    except (ZoneInfoNotFoundError, ValueError):
        errors.append("The timezone for this spot could not be found. Place the pin again.")
    return values, errors


def check_details(form: dict) -> tuple[dict, list[str]]:
    values, errors = check_location(form.get("lat", ""), form.get("lon", ""), form.get("timezone", ""))
    name = " ".join(form.get("name", "").split())
    if not name:
        errors.append("Give the water a name.")
    elif len(name) > NAME_MAX:
        errors.append(f"Keep the name under {NAME_MAX} characters.")
    values["name"] = name
    if form.get("water_type") not in WATER_TYPES:
        errors.append("Choose what kind of water it is.")
    if form.get("depth_class") not in DEPTH_CLASSES:
        errors.append("Choose a rough depth, or 'Not sure'.")
    values["water_type"], values["depth_class"] = form.get("water_type"), form.get("depth_class")
    try:
        values["reading_time"] = time.fromisoformat(form.get("reading_time", ""))
    except ValueError:
        errors.append("Enter the usual reading time as hours and minutes, like 07:00.")
    return values, errors


@router.get("/new", response_class=HTMLResponse)
def new_location(request: Request, user: User = Depends(require_user)):
    return render(request, "new_location.html")


@router.get("/api/geocode")
async def geocode(request: Request, q: str = Query(min_length=2, max_length=80), user: User = Depends(require_user)):
    try:
        return await run_in_threadpool(request.app.state.openmeteo.geocode, q)
    except OpenMeteoError:
        return JSONResponse({"detail": "Place search is not answering. Try again, or move the map by hand."}, 502)


@router.get("/api/point")
async def point(
    request: Request,
    lat: float = Query(ge=-90, le=90),
    lon: float = Query(ge=-180, le=180),
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    try:
        preview = await run_in_threadpool(request.app.state.openmeteo.point_preview, lat, lon)
    except OpenMeteoError:
        return JSONResponse({"detail": "Could not fetch the weather for this spot. Try again in a minute."}, 502)
    nearby = queries.nearby_waters(db, lat, lon, NEARBY_METRES)
    return {**preview, "nearby": [{"name": w.name, "slug": w.slug, "distance_m": round(w.distance_m)} for w in nearby]}


@router.post("/new/details", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
def new_details(request: Request, lat: str = Form(""), lon: str = Form(""), timezone: str = Form(""),
                user: User = Depends(require_user)):
    values, errors = check_location(lat, lon, timezone)
    if errors:
        return render(request, "new_location.html", {"errors": errors}, status_code=422)
    return render(request, "new_details.html", {"v": {**values, "reading_time": time(7, 0)}, "errors": []})


@router.post("/new", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def create_water(request: Request, db: Session = Depends(get_db), user: User = Depends(require_user)):
    form = {k: str(v) for k, v in (await request.form()).items()}
    values, errors = check_details(form)
    if errors:
        return render(request, "new_details.html", {"v": {**form, **values}, "errors": errors}, status_code=422)
    water = Water(
        slug=slugify_unique(db, values["name"]), name=values["name"], lat=values["lat"], lon=values["lon"],
        timezone=values["timezone"], water_type=values["water_type"], depth_class=values["depth_class"],
        default_reading_time=values["reading_time"], created_by=user.id, needs_recompute=True,
    )
    db.add(water)
    db.flush()
    db.add(WaterMember(water_id=water.id, user_id=user.id, role="admin"))
    db.commit()
    return RedirectResponse(f"/w/{water.slug}/first-readings", status_code=303)


@router.get("/w/{slug}/first-readings", response_class=HTMLResponse)
def first_readings(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = active_water(db, slug)
    return render(request, "new_readings.html", {"water": water, "rows": "", "preview": None})


