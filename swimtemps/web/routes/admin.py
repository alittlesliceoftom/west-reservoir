"""
Water admin: correct and delete readings, edit details, promote admins,
archive. Open to the water's admins and to superusers. Nothing is destroyed:
deletion is a timestamp, and a correction is a deletion plus a new reading
that the old one points to.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import Reading, User, Water, WaterMember
from web.auth import require_user, verify_csrf
from web.deps import get_db
from web.paste import MAX_TEMP, MIN_TEMP
from web.permissions import can_admin
from web.routes.create import check_about
from web.routes.pages import active_water
from web.routes.readings import FUTURE_SLACK, local_input_value
from web.templating import render

router = APIRouter()

MESSAGES = {
    "corrected": "Reading corrected. The forecast is updating.",
    "deleted": "Reading deleted. The forecast is updating.",
    "details": "Details saved. The forecast is updating.",
    "promoted": "Admin added.",
}
SHOWN_READINGS = 200


def admin_water(db: Session, slug: str, user: User) -> Water:
    water = active_water(db, slug)
    if not can_admin(db, user, water):
        raise HTTPException(403, "Only this water's admins can do that.")
    return water


def standing_reading(db: Session, water: Water, reading_id: int) -> Reading:
    reading = db.get(Reading, reading_id)
    if reading is None or reading.water_id != water.id or reading.deleted_at is not None:
        raise HTTPException(404, "That reading does not exist, or has already been changed.")
    return reading


def admin_page(request, db, water, *, status_code=200, error=None, form=None):
    rows = db.execute(
        select(Reading, User.display_name, User.email)
        .join(User, User.id == Reading.submitted_by)
        .where(Reading.water_id == water.id, Reading.deleted_at.is_(None))
        .order_by(Reading.measured_at.desc()).limit(SHOWN_READINGS)
    ).all()
    total = db.scalar(select(func.count()).select_from(Reading).where(
        Reading.water_id == water.id, Reading.deleted_at.is_(None)))
    members = db.execute(
        select(User.display_name, User.email, WaterMember.role)
        .join(WaterMember, WaterMember.user_id == User.id)
        .where(WaterMember.water_id == water.id).order_by(User.display_name)
    ).all()
    return render(request, "admin.html", {
        "water": water, "rows": rows, "total": total, "members": members, "error": error,
        "message": MESSAGES.get(request.query_params.get("done", "")),
        "form": form or {}, "input_value": lambda at: local_input_value(at, water.timezone),
    }, status_code=status_code)


def back(water: Water, done: str) -> RedirectResponse:
    return RedirectResponse(f"/w/{water.slug}/admin?done={done}", status_code=303)


@router.get("/w/{slug}/admin", response_class=HTMLResponse)
def admin(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    return admin_page(request, db, admin_water(db, slug, user))


@router.post("/w/{slug}/admin/readings/{reading_id}/correct", dependencies=[Depends(verify_csrf)])
def correct(request: Request, slug: str, reading_id: int, temp: str = Form(""), measured_at: str = Form(""),
            db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    old = standing_reading(db, water, reading_id)
    now = request.app.state.clock()
    try:
        value = float(temp.replace(",", "."))
        if not MIN_TEMP <= value <= MAX_TEMP:
            raise ValueError
    except ValueError:
        return admin_page(request, db, water, status_code=422, error="The temperature must be a number between -2 and 40 °C.")
    try:
        at = datetime.fromisoformat(measured_at).replace(tzinfo=ZoneInfo(water.timezone)).astimezone(timezone.utc)
    except ValueError:
        return admin_page(request, db, water, status_code=422, error="Enter the date and time the reading was taken.")
    if at > now + FUTURE_SLACK:
        return admin_page(request, db, water, status_code=422, error="That time is in the future.")

    new = Reading(water_id=water.id, measured_at=at, temp_c=value, submitted_by=user.id)
    db.add(new)
    db.flush()
    old.deleted_at, old.deleted_by, old.replaced_by = now, user.id, new.id
    water.needs_recompute = True
    db.commit()
    return back(water, "corrected")


@router.post("/w/{slug}/admin/readings/{reading_id}/delete", dependencies=[Depends(verify_csrf)])
def delete(request: Request, slug: str, reading_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    reading = standing_reading(db, water, reading_id)
    reading.deleted_at, reading.deleted_by = request.app.state.clock(), user.id
    water.needs_recompute = True
    db.commit()
    return back(water, "deleted")


@router.post("/w/{slug}/admin/details", dependencies=[Depends(verify_csrf)])
async def details(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    form = {k: str(v) for k, v in (await request.form()).items()}
    values, errors = check_about(form)
    if errors:
        return admin_page(request, db, water, status_code=422, error=" ".join(errors), form=form)
    water.name, water.water_type, water.depth_class = values["name"], values["water_type"], values["depth_class"]
    water.default_reading_time = values["reading_time"]
    water.needs_recompute = True
    db.commit()
    return back(water, "details")


@router.post("/w/{slug}/admin/members", dependencies=[Depends(verify_csrf)])
def promote(request: Request, slug: str, email: str = Form(""), db: Session = Depends(get_db),
            user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    member = db.scalar(select(User).where(func.lower(User.email) == email.strip().lower()))
    if member is None:
        return admin_page(request, db, water, status_code=422, error=(
            f"{email.strip()} has not signed in to swimtemps yet. Ask them to sign in once, then add them."))
    db.execute(insert(WaterMember).values(water_id=water.id, user_id=member.id, role="admin")
               .on_conflict_do_update(index_elements=["water_id", "user_id"], set_={"role": "admin"}))
    db.commit()
    return back(water, "promoted")


@router.post("/w/{slug}/admin/archive", dependencies=[Depends(verify_csrf)])
def archive(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = admin_water(db, slug, user)
    water.archived_at = request.app.state.clock()
    db.commit()
    return RedirectResponse("/", status_code=303)


@router.get("/me", response_class=HTMLResponse)
def me(request: Request, db: Session = Depends(get_db), user: User = Depends(require_user)):
    waters = db.scalars(
        select(Water).where(Water.archived_at.is_(None)).where(
            (Water.created_by == user.id)
            | Water.id.in_(select(WaterMember.water_id).where(WaterMember.user_id == user.id))
        ).order_by(Water.name)
    ).all()
    readings = db.execute(
        select(Reading, Water.name, Water.slug, Water.timezone)
        .join(Water, Water.id == Reading.water_id)
        .where(Reading.submitted_by == user.id, Reading.deleted_at.is_(None))
        .order_by(Reading.created_at.desc()).limit(50)
    ).all()
    return render(request, "me.html", {"waters": waters, "readings": readings})
