"""
One worker cycle's units of work: refresh weather, fit, forecast, snapshot.

`sweep` runs over every active water; `drain_dirty` over those the web has
flagged with `needs_recompute`. Each water is processed in its own
transaction, and its flag is cleared *before* its data is read: a reading
saved mid-compute sets the flag again and is picked up next time, and a
failed compute sets it back.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from db import queries
from db.models import ForecastPoint, ForecastRun, Heartbeat, ModelFit, Snapshot, Water
from forecast.band import halfwidth, horizon_days
from forecast.model import (
    FitResult,
    estimate_at,
    fit_water,
    forecast_points,
    forecast_targets,
    prepare_weather,
    run_forward,
)
from forecast.openmeteo import PAST_DAYS, OpenMeteoError
from forecast.physics import Coefficients, load_params
from worker.snapshot import build_payload, iso

log = logging.getLogger("swimtemps.worker")

FIT_MAX_AGE = timedelta(hours=24)
FORECAST_FRESH = timedelta(minutes=50)
HEARTBEAT_KEEP = timedelta(days=7)
# Weather is backfilled for readings up to this old. Older readings stay in the
# record but form no training legs, which keeps a mistyped year from pulling
# decades of hourly weather into a free-tier database.
ARCHIVE_MAX = timedelta(days=3 * 365)


def coordinate_key(water: Water) -> tuple[float, float]:
    """Waters this close share one Open-Meteo grid call."""
    return (round(water.lat, 2), round(water.lon, 2))


def last_forecast_fetch(session: Session, water_id: int) -> datetime | None:
    return session.execute(text(
        "SELECT max(fetched_at) FROM weather_hourly WHERE water_id = :w AND source = 'forecast'"
    ), {"w": water_id}).scalar()


def _forecast_is_fresh(session: Session, water_id: int, now: datetime) -> bool:
    fetched = last_forecast_fetch(session, water_id)
    return fetched is not None and fetched > now - FORECAST_FRESH


def refresh_weather(session: Session, waters: list[Water], client, now: datetime, skip_fresh: bool = False) -> None:
    """
    Store the forecast for each coordinate group, one call per group, and
    backfill from the archive any reading between 7 days and 3 years old that
    has no weather. A failed call is logged and skipped: the stored weather
    stands, and one group's failure never stops another's.
    """
    groups: dict[tuple, list[Water]] = defaultdict(list)
    for water in waters:
        groups[coordinate_key(water)].append(water)

    archive_before = (pd.Timestamp(now) - timedelta(days=PAST_DAYS)).tz_convert("UTC")
    archive_from = (pd.Timestamp(now) - ARCHIVE_MAX).tz_convert("UTC").to_pydatetime()
    for (lat, lon), members in groups.items():
        fresh = skip_fresh and all(_forecast_is_fresh(session, w.id, now) for w in members)
        if not fresh:
            try:
                frame = client.forecast(lat, lon)
            except OpenMeteoError as error:
                log.warning("forecast for %s,%s failed: %s", lat, lon, error)
            except Exception:
                log.exception("forecast for %s,%s failed unexpectedly", lat, lon)
            else:
                for water in members:
                    queries.upsert_weather(session, water.id, frame, "forecast", fetched_at=now)

        missing = [
            m for w in members
            if (m := queries.earliest_reading_without_weather(session, w.id, since=archive_from)) is not None
            and pd.Timestamp(m) < archive_before
        ]
        if missing:
            start = (pd.Timestamp(min(missing)).tz_convert("UTC") - timedelta(days=1)).date()
            end = archive_before.date()
            try:
                frame = client.archive(lat, lon, start, end)
            except OpenMeteoError as error:
                log.warning("archive for %s,%s failed: %s", lat, lon, error)
            except Exception:
                log.exception("archive for %s,%s failed unexpectedly", lat, lon)
            else:
                for water in members:
                    queries.upsert_weather(session, water.id, frame, "archive", fetched_at=now)
        session.flush()


def _latest_fit(session: Session, water_id: int) -> ModelFit | None:
    return session.scalars(
        select(ModelFit).where(ModelFit.water_id == water_id).order_by(ModelFit.fitted_at.desc(), ModelFit.id.desc()).limit(1)
    ).first()


def _fit(session: Session, water: Water, readings, weather, now, params) -> FitResult:
    """Refit when readings changed, there is no fit, or the fit is a day old; else reuse."""
    last = _latest_fit(session, water.id)
    stale = (
        last is None
        or last.fitted_at < now - FIT_MAX_AGE
        or queries.readings_changed_since(session, water.id, last.fitted_at)
    )
    if not stale:
        return FitResult(
            Coefficients(last.k_air, last.k_solar, last.k_cool),
            last.scale, last.method, last.n_legs,
            float("nan") if last.train_mae is None else last.train_mae,
        )
    fit = fit_water(readings, weather, water.depth_class, params=params)
    session.add(ModelFit(
        water_id=water.id, fitted_at=now, k_air=fit.coefficients.k_air, k_solar=fit.coefficients.k_solar,
        k_cool=fit.coefficients.k_cool, scale=fit.scale, n_legs=fit.n_legs, method=fit.method,
        train_mae=None if pd.isna(fit.train_mae) else fit.train_mae,
    ))
    session.flush()
    return fit


def compute_water(session: Session, water: Water, client, now: datetime, refresh: bool = True,
                  params: dict | None = None) -> Snapshot:
    """Weather (optionally), fit, forecast run and snapshot for one water."""
    params = params or load_params()
    if refresh:
        refresh_weather(session, [water], client, now, skip_fresh=True)

    readings = queries.active_readings(session, water.id)
    weather = prepare_weather(queries.weather_frame(session, water.id))
    fit = _fit(session, water, readings, weather, now, params)
    fit_row = _latest_fit(session, water.id)

    anchor = readings.iloc[-1] if len(readings) else None
    estimate, points, trajectory = None, [], None
    if anchor is not None:
        targets = forecast_targets(now, water.timezone, water.default_reading_time)
        estimate = estimate_at(anchor.measured_at, anchor.temp_c, weather, fit.coefficients, at=now)
        trajectory = run_forward(anchor.measured_at, anchor.temp_c, weather, fit.coefficients, until=targets[-1])
        own = queries.own_errors(session, water.id, params["own_errors_min_points"])
        run = ForecastRun(
            water_id=water.id, created_at=now, fit_id=fit_row.id, anchor_reading_id=int(anchor.id),
            anchor_at=anchor.measured_at.to_pydatetime(), anchor_temp=float(anchor.temp_c),
        )
        session.add(run)
        session.flush()
        for valid_at, value in forecast_points(anchor.measured_at, anchor.temp_c, weather, fit.coefficients, targets):
            if value is None:
                continue
            horizon = horizon_days(anchor.measured_at, valid_at)
            width = halfwidth(horizon, fit.method, water.water_type, own=own, params=params)
            session.add(ForecastPoint(run_id=run.id, valid_at=valid_at.to_pydatetime(),
                                      water_temp=value, lower=value - width, upper=value + width))
            points.append({
                "at": iso(valid_at), "temp": round(value, 1), "lower": round(value - width, 1),
                "upper": round(value + width, 1), "horizon": horizon,
            })

    payload = build_payload(
        water=water, now=now, readings=readings, fit=fit, anchor=anchor, estimate=estimate,
        points=points, trajectory=trajectory, weather=weather,
        weather_fetched_at=last_forecast_fetch(session, water.id),
    )
    session.execute(
        insert(Snapshot).values(water_id=water.id, built_at=now, payload=payload)
        .on_conflict_do_update(index_elements=["water_id"], set_={"built_at": now, "payload": payload})
    )
    return payload


def process(factory: sessionmaker, water_id: int, client, now: datetime, refresh: bool) -> bool:
    """Clear the flag, compute, commit. On failure the flag is set back. Returns success."""
    with factory() as session:
        session.execute(update(Water).where(Water.id == water_id).values(needs_recompute=False))
        session.commit()
    try:
        with factory() as session:
            water = session.get(Water, water_id)
            compute_water(session, water, client, now, refresh=refresh)
            session.commit()
        return True
    except Exception:
        log.exception("compute failed for water %s", water_id)
        with factory() as session:
            session.execute(update(Water).where(Water.id == water_id).values(needs_recompute=True))
            session.commit()
        return False


def _active_ids(session: Session, dirty_only: bool) -> list[int]:
    query = select(Water.id).where(Water.archived_at.is_(None)).order_by(Water.id)
    if dirty_only:
        query = query.where(Water.needs_recompute.is_(True))
    return list(session.scalars(query))


def sweep(factory: sessionmaker, client, now: datetime) -> int:
    """Refresh weather for every active water (one call per coordinate group), then recompute each."""
    with factory() as session:
        ids = _active_ids(session, dirty_only=False)
        if not ids:
            return 0
        waters = list(session.scalars(select(Water).where(Water.id.in_(ids))))
        refresh_weather(session, waters, client, now)
        session.commit()
    return sum(process(factory, water_id, client, now, refresh=False) for water_id in ids)


def drain_dirty(factory: sessionmaker, client, now: datetime) -> int:
    """Recompute every flagged water. Returns how many succeeded."""
    with factory() as session:
        ids = _active_ids(session, dirty_only=True)
    return sum(process(factory, water_id, client, now, refresh=True) for water_id in ids)


def write_heartbeat(factory: sessionmaker, now: datetime) -> None:
    with factory() as session:
        session.execute(delete(Heartbeat).where(Heartbeat.at < now - HEARTBEAT_KEEP))
        session.add(Heartbeat(at=now))
        session.commit()
