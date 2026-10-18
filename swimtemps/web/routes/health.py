"""/healthz, for an uptime monitor."""

from datetime import timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db import queries
from web.deps import get_db

router = APIRouter()

MAX_AGE = timedelta(hours=3)


@router.get("/healthz")
def healthz(request: Request, db: Session = Depends(get_db)):
    """
    503 when any active water's snapshot is over 3 hours old (or missing),
    Open-Meteo has not answered in 3 hours, or the worker has not written a
    heartbeat in 3 hours.
    """
    now = request.app.state.clock()
    state = queries.health(db)
    failures = []
    if state["active_waters"]:
        oldest = state["oldest_snapshot"]
        if state["missing_snapshots"] or oldest is None or now - oldest > MAX_AGE:
            failures.append("a water's snapshot is missing or over 3 hours old")
        if state["last_weather"] is None or now - state["last_weather"] > MAX_AGE:
            failures.append("Open-Meteo has not answered in 3 hours")
    if state["last_heartbeat"] is None or now - state["last_heartbeat"] > MAX_AGE:
        failures.append("the worker has not written a heartbeat in 3 hours")
    body = {"ok": not failures, "failures": failures, "active_waters": state["active_waters"]}
    return JSONResponse(body, status_code=503 if failures else 200)
