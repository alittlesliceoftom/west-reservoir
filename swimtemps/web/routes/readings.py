"""Adding readings: one at a time, or several pasted at once."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import median
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import queries
from db.models import Reading, Snapshot, User, Water
from web.auth import require_user, verify_csrf
from web.deps import get_db
from web.paste import EARLIEST, MAX_TEMP, MIN_TEMP, ParsedRow, parse_paste
from web.permissions import can_admin
from web.routes.pages import active_water
from web.templating import render

router = APIRouter()

RATE_LIMIT = 30                       # readings per user per water per rolling hour
SOFT_LIMIT = 3.0                      # °C from the estimate before asking "are you sure?"
ESTIMATE_WINDOW = timedelta(hours=12) # only readings this close to now are compared with "now"
NEIGHBOUR_WINDOW = timedelta(days=3)  # pasted rows are compared with readings this close
FUTURE_SLACK = timedelta(minutes=10)


def local_input_value(at: datetime, tz: str) -> str:
    return at.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%dT%H:%M")


def default_reading_time(water: Water, now: datetime) -> str:
    """Today at the water's reading time, or now if that is still to come."""
    zone = ZoneInfo(water.timezone)
    local_now = now.astimezone(zone)
    usual = datetime.combine(local_now.date(), water.default_reading_time, tzinfo=zone)
    return local_input_value(usual if usual <= local_now else local_now, water.timezone)


def current_estimate(db: Session, water: Water) -> float | None:
    snapshot = db.get(Snapshot, water.id)
    if snapshot and snapshot.payload.get("estimate_now"):
        return float(snapshot.payload["estimate_now"]["temp"])
    return None


def allowance(db: Session, user: User, water: Water, now: datetime) -> int | None:
    """Readings this user may still add to this water this hour; None means no limit (its admins)."""
    if can_admin(db, user, water):
        return None
    return max(0, RATE_LIMIT - queries.recent_submissions(db, user.id, water.id, now - timedelta(hours=1)))


def hour_bounds(at: datetime, tz: str) -> tuple[datetime, datetime]:
    """The local clock hour containing `at`, as UTC bounds."""
    local = at.astimezone(ZoneInfo(tz)).replace(minute=0, second=0, microsecond=0)
    start = local.astimezone(timezone.utc)
    return start, start + timedelta(hours=1)


def reading_in_same_hour(db: Session, water: Water, at: datetime) -> Reading | None:
    start, end = hour_bounds(at, water.timezone)
    return db.scalar(select(Reading).where(
        Reading.water_id == water.id, Reading.deleted_at.is_(None),
        Reading.measured_at >= start, Reading.measured_at < end,
    ).limit(1))


def save(db: Session, water: Water, user: User, rows: list[tuple[datetime, float]]) -> None:
    db.add_all(Reading(water_id=water.id, measured_at=at, temp_c=temp, submitted_by=user.id, source="manual")
               for at, temp in rows)
    queries.mark_dirty(db, water.id)
    db.commit()


def form_page(request, water, *, status_code=200, **context):
    now = request.app.state.clock()
    base = {
        "water": water,
        "estimate": None,
        "measured_at": default_reading_time(water, now),
        "temp": "",
        "error": None,
        "warnings": [],
        "rows": "",
        "preview": None,
    }
    return render(request, "add_reading.html", {**base, **context}, status_code=status_code)


@router.get("/w/{slug}/readings", response_class=HTMLResponse)
def reading_form(request: Request, slug: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    water = active_water(db, slug)
    return form_page(request, water, estimate=current_estimate(db, water))


@router.post("/w/{slug}/readings", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
def add_reading(
    request: Request, slug: str, temp: str = Form(""), measured_at: str = Form(""), confirm: str = Form(""),
    db: Session = Depends(get_db), user: User = Depends(require_user),
):
    water = active_water(db, slug)
    now = request.app.state.clock()
    estimate = current_estimate(db, water)
    page = dict(estimate=estimate, temp=temp, measured_at=measured_at or default_reading_time(water, now))

    try:
        value = float(temp.replace(",", ".").strip())
    except ValueError:
        return form_page(request, water, status_code=422, error="Enter the temperature as a number, like 13.5.", **page)
    if not MIN_TEMP <= value <= MAX_TEMP:
        return form_page(request, water, status_code=422, error="The temperature must be between -2 and 40 °C.", **page)
    try:
        local = datetime.fromisoformat(page["measured_at"])
        at = local.replace(tzinfo=ZoneInfo(water.timezone)).astimezone(timezone.utc)
    except ValueError:
        return form_page(request, water, status_code=422, error="Enter the date and time the reading was taken.", **page)
    if at > now + FUTURE_SLACK:
        return form_page(request, water, status_code=422, error="That time is in the future.", **page)
    if at.date() < EARLIEST:
        return form_page(request, water, status_code=422, error="That date is before 2000. Check the year.", **page)

    remaining = allowance(db, user, water, now)
    if remaining == 0:
        return form_page(request, water, status_code=429, error=(
            f"You have added {RATE_LIMIT} readings an hour to this water, the most allowed. Please try again later."), **page)

    if not confirm:
        warnings = []
        if estimate is not None and abs(at - now) <= ESTIMATE_WINDOW and abs(value - estimate) > SOFT_LIMIT:
            warnings.append(
                f"{abs(value - estimate):.1f} °C away from the current estimate of {estimate:.1f} °C. "
                "Worth a double check: a wrong reading pulls the forecast with it.")
        existing = reading_in_same_hour(db, water, at)
        if existing is not None:
            existing_local = existing.measured_at.astimezone(ZoneInfo(water.timezone))
            warnings.append(
                f"There is already a reading at {existing_local:%H:%M} ({existing.temp_c:.1f} °C) in this hour.")
        if warnings:
            return form_page(request, water, warnings=warnings, **page)

    save(db, water, user, [(at, value)])
    return RedirectResponse(f"/w/{water.slug}", status_code=303)


@dataclass
class PreviewRow:
    row: ParsedRow
    status: str = "ok"     # ok | outlier | duplicate | repeat | error
    note: str = ""


@dataclass
class Preview:
    rows: list[PreviewRow] = field(default_factory=list)

    @property
    def errors(self) -> list[PreviewRow]:
        return [r for r in self.rows if r.status == "error"]

    @property
    def to_save(self) -> list[PreviewRow]:
        return [r for r in self.rows if r.status in ("ok", "outlier")]

    @property
    def skipped(self) -> int:
        return sum(r.status in ("duplicate", "repeat") for r in self.rows)


def analyse_paste(db: Session, water: Water, text: str, now: datetime) -> Preview:
    """
    Parse and flag pasted rows. Repeats of a stored reading's local hour, or of
    an earlier pasted row's, are skipped; values more than 3 °C from the median
    of readings within 3 days (stored or pasted) are flagged but kept.
    """
    zone = ZoneInfo(water.timezone)
    parsed = parse_paste(text, water.timezone, water.default_reading_time, now.astimezone(zone).date(),
                         now=now + FUTURE_SLACK)
    stored = queries.active_readings(db, water.id)
    stored_points = list(zip(stored.measured_at.dt.to_pydatetime(), stored.temp_c))
    stored_hours = {hour_bounds(at, water.timezone)[0] for at, _ in stored_points}
    estimate = current_estimate(db, water)

    preview, seen_hours = Preview(), set()
    valid = [r for r in parsed if r.error is None]
    for row in parsed:
        if row.error:
            preview.rows.append(PreviewRow(row, "error", row.error))
            continue
        hour = hour_bounds(row.measured_at, water.timezone)[0]
        if hour in stored_hours:
            preview.rows.append(PreviewRow(row, "duplicate", "Already a reading this hour"))
            continue
        if hour in seen_hours:
            preview.rows.append(PreviewRow(row, "repeat", "Repeated in this paste"))
            continue
        seen_hours.add(hour)
        neighbours = [t for at, t in stored_points if abs(at - row.measured_at) <= NEIGHBOUR_WINDOW]
        neighbours += [o.temp for o in valid if o is not row and abs(o.measured_at - row.measured_at) <= NEIGHBOUR_WINDOW]
        reference = median(neighbours) if neighbours else (
            estimate if estimate is not None and abs(row.measured_at - now) <= ESTIMATE_WINDOW else None)
        if reference is not None and abs(row.temp - reference) > SOFT_LIMIT:
            preview.rows.append(PreviewRow(row, "outlier", f"Check this: {abs(row.temp - reference):.1f} °C from nearby readings"))
        else:
            preview.rows.append(PreviewRow(row, "ok"))
    return preview


def preview_response(request, water, preview, rows_text, status_code=200, error=None):
    context = {"preview": preview, "rows": rows_text, "paste_error": error}
    if request.headers.get("HX-Request"):
        return render(request, "_paste_preview.html", {"water": water, **context}, status_code=status_code)
    return form_page(request, water, status_code=status_code, **context)


@router.post("/w/{slug}/readings/preview", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
def preview_paste(request: Request, slug: str, rows: str = Form(""), db: Session = Depends(get_db),
                  user: User = Depends(require_user)):
    water = active_water(db, slug)
    return preview_response(request, water, analyse_paste(db, water, rows, request.app.state.clock()), rows)


@router.post("/w/{slug}/readings/paste", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
def save_paste(request: Request, slug: str, rows: str = Form(""), db: Session = Depends(get_db),
               user: User = Depends(require_user)):
    water = active_water(db, slug)
    now = request.app.state.clock()
    preview = analyse_paste(db, water, rows, now)
    if preview.errors:
        return preview_response(request, water, preview, rows, status_code=422)
    if not preview.to_save:
        return preview_response(request, water, preview, rows, status_code=422, error="There are no new readings to save.")
    remaining = allowance(db, user, water, now)
    if remaining is not None and len(preview.to_save) > remaining:
        plural = "reading" if remaining == 1 else "readings"
        return preview_response(request, water, preview, rows, status_code=429, error=(
            f"You can add {remaining} more {plural} to this water in the next hour. "
            "Paste fewer rows, or ask the water's admin to add them."))
    save(db, water, user, [(r.row.measured_at, r.row.temp) for r in preview.to_save])
    return RedirectResponse(f"/w/{water.slug}", status_code=303)


