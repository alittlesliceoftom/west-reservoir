"""Analytical reads and bulk writes, as SQL."""

from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

MEASURES = ("air_temp", "shortwave", "cloud_cover")
MIN_POINTS_PER_HORIZON = 10

# A reading the fit, the estimate and scoring may use: unflagged, or flagged
# and accepted by an admin, and not marked wrong. Expects the alias `r`.
USABLE_SQL = "(r.flag_override = 'accept' OR (r.flag_override IS NULL AND NOT r.flagged))"
REJECTED_REASON = "Marked as wrong by the water's admin"
ACCEPTED_NOTE = "Flagged ({reason}), accepted by the water's admin"

# Archive values overwrite forecast values; a forecast never overwrites an
# archive value. A null in the incoming row keeps what is stored.
UPSERT_WEATHER = text("""
    INSERT INTO weather_hourly (water_id, valid_at, source, air_temp, shortwave, cloud_cover, fetched_at)
    VALUES (:water_id, :valid_at, :source, :air_temp, :shortwave, :cloud_cover, COALESCE(:fetched_at, now()))
    ON CONFLICT (water_id, valid_at) DO UPDATE SET
        source = excluded.source,
        air_temp = COALESCE(excluded.air_temp, weather_hourly.air_temp),
        shortwave = COALESCE(excluded.shortwave, weather_hourly.shortwave),
        cloud_cover = COALESCE(excluded.cloud_cover, weather_hourly.cloud_cover),
        fetched_at = excluded.fetched_at
    WHERE weather_hourly.source = 'forecast' OR excluded.source = 'archive'
""")


def _none_if_nan(value):
    return None if pd.isna(value) else float(value)


def upsert_weather(session: Session, water_id: int, frame: pd.DataFrame, source: str,
                   fetched_at: datetime | None = None) -> int:
    """Store hourly weather for one water. Rows with no air temperature are skipped."""
    rows = [
        {
            "water_id": water_id,
            "valid_at": row.valid_at.to_pydatetime(),
            "source": source,
            "fetched_at": fetched_at,
            **{m: _none_if_nan(getattr(row, m)) for m in MEASURES},
        }
        for row in frame.itertuples(index=False)
        if not pd.isna(row.air_temp)
    ]
    if rows:
        session.execute(UPSERT_WEATHER, rows)
    return len(rows)


def weather_frame(session: Session, water_id: int, since: datetime | None = None) -> pd.DataFrame:
    sql = text("""
        SELECT valid_at, air_temp, shortwave, cloud_cover
        FROM weather_hourly
        WHERE water_id = :water_id AND (CAST(:since AS timestamptz) IS NULL OR valid_at >= :since)
        ORDER BY valid_at
    """)
    rows = session.execute(sql, {"water_id": water_id, "since": since}).all()
    frame = pd.DataFrame(rows, columns=["valid_at", *MEASURES])
    frame["valid_at"] = pd.to_datetime(frame["valid_at"], utc=True)
    return frame.astype({m: float for m in MEASURES})


def active_readings(session: Session, water_id: int) -> pd.DataFrame:
    """
    Usable readings that stand, oldest first: id, measured_at (UTC),
    temp_c (float), source ('manual' | 'feed').
    """
    sql = text(f"""
        SELECT r.id, r.measured_at, r.temp_c::float8 AS temp_c, r.source
        FROM readings r
        WHERE r.water_id = :water_id AND r.deleted_at IS NULL AND {USABLE_SQL}
        ORDER BY r.measured_at, r.id
    """)
    rows = session.execute(sql, {"water_id": water_id}).all()
    frame = pd.DataFrame(rows, columns=["id", "measured_at", "temp_c", "source"])
    frame["measured_at"] = pd.to_datetime(frame["measured_at"], utc=True)
    return frame.astype({"temp_c": float, "source": str})


def chart_readings(session: Session, water_id: int, since: datetime | None) -> pd.DataFrame:
    """
    Every standing reading from `since`, usable or not, oldest first:
    measured_at (UTC), temp_c, usable (bool), reason (why it is not usable;
    None when it is) and note (for a flag an admin accepted; else None).
    """
    sql = text(f"""
        SELECT r.measured_at, r.temp_c::float8 AS temp_c, {USABLE_SQL} AS usable,
               CASE WHEN {USABLE_SQL} THEN NULL
                    ELSE COALESCE(r.flag_reason, :rejected) END AS reason,
               CASE WHEN r.flagged AND r.flag_override = 'accept' THEN r.flag_reason END AS accepted
        FROM readings r
        WHERE r.water_id = :water_id AND r.deleted_at IS NULL
          AND (CAST(:since AS timestamptz) IS NULL OR r.measured_at >= :since)
        ORDER BY r.measured_at, r.id
    """)
    rows = session.execute(sql, {"water_id": water_id, "since": since, "rejected": REJECTED_REASON}).all()
    frame = pd.DataFrame(rows, columns=["measured_at", "temp_c", "usable", "reason", "accepted"])
    frame["measured_at"] = pd.to_datetime(frame["measured_at"], utc=True)
    frame = frame.astype({"temp_c": float, "usable": bool, "reason": object})
    frame["reason"] = frame["reason"].where(frame["reason"].notna(), None)
    notes = [None if pd.isna(a) else ACCEPTED_NOTE.format(reason=a) for a in frame.pop("accepted")]
    frame["note"] = pd.Series(notes, index=frame.index, dtype=object)
    return frame


def mark_dirty(session: Session, water_id: int) -> None:
    """
    Ask the worker to recompute a water. Always an explicit UPDATE: setting the
    ORM attribute is skipped when the loaded value is already true, and the
    worker may have cleared it since that load.
    """
    session.execute(text("UPDATE waters SET needs_recompute = true WHERE id = :id"), {"id": water_id})


def readings_changed_since(session: Session, water_id: int, since: datetime) -> bool:
    """Whether any reading was added, deleted, corrected or overruled after `since`."""
    sql = text("""
        SELECT EXISTS (
            SELECT 1 FROM readings
            WHERE water_id = :water_id
              AND (created_at > :since OR deleted_at > :since OR override_at > :since)
        )
    """)
    return bool(session.execute(sql, {"water_id": water_id, "since": since}).scalar())


def earliest_reading_without_weather(session: Session, water_id: int, since: datetime) -> datetime | None:
    """The oldest standing reading since `since` whose hour has no stored weather."""
    sql = text("""
        SELECT min(r.measured_at)
        FROM readings r
        LEFT JOIN weather_hourly w
          ON w.water_id = r.water_id AND w.valid_at = date_trunc('hour', r.measured_at)
        WHERE r.water_id = :water_id AND r.deleted_at IS NULL AND w.valid_at IS NULL
          AND r.measured_at >= :since
    """)
    return session.execute(sql, {"water_id": water_id, "since": since}).scalar()


def own_errors(session: Session, water_id: int, min_points: int) -> dict[int, float] | None:
    """
    This water's 80th-percentile absolute forecast error by horizon, or None
    until it has `min_points` scored points. A horizon with fewer than
    MIN_POINTS_PER_HORIZON of its own is left out, so the base band is used there.

    A point is scored against a usable reading in the same hour. Each anchor's
    forecast is scored as first issued: hourly re-runs from the same anchor
    have seen more of the real weather, and scoring them would make the band
    look tighter than the forecasts people actually read.
    """
    sql = text(f"""
        WITH scored AS (
            SELECT DISTINCT ON (r.id, fr.anchor_reading_id)
                r.id AS reading_id,
                GREATEST(1, CEIL(EXTRACT(EPOCH FROM (p.valid_at - fr.anchor_at)) / 86400 - 1e-9))::int AS horizon,
                abs(p.water_temp - r.temp_c::float8) AS abs_error
            FROM forecast_points p
            JOIN forecast_runs fr ON fr.id = p.run_id
            JOIN readings r
              ON r.water_id = fr.water_id
             AND r.deleted_at IS NULL
             AND {USABLE_SQL}
             AND date_trunc('hour', r.measured_at + interval '30 minutes') = p.valid_at
            WHERE fr.water_id = :water_id AND fr.created_at < r.measured_at
            ORDER BY r.id, fr.anchor_reading_id, fr.created_at
        )
        SELECT horizon, count(*) AS n,
               percentile_cont(0.8) WITHIN GROUP (ORDER BY abs_error) AS p80
        FROM scored GROUP BY horizon ORDER BY horizon
    """)
    rows = session.execute(sql, {"water_id": water_id}).all()
    if sum(r.n for r in rows) < min_points:
        return None
    return {int(r.horizon): float(r.p80) for r in rows if r.n >= MIN_POINTS_PER_HORIZON}


def recent_submissions(session: Session, user_id, water_id: int, since: datetime) -> int:
    """Readings this user has submitted to this water since `since`, deleted ones included."""
    sql = text("""
        SELECT count(*) FROM readings
        WHERE submitted_by = :user_id AND water_id = :water_id AND created_at > :since
    """)
    return int(session.execute(sql, {"user_id": user_id, "water_id": water_id, "since": since}).scalar())


def nearby_waters(session: Session, lat: float, lon: float, metres: float = 300) -> list:
    """Non-archived waters within `metres`, nearest first (haversine)."""
    sql = text("""
        SELECT * FROM (
            SELECT id, slug, name, lat, lon,
                   2 * 6371000 * asin(sqrt(
                       power(sin(radians(lat - :lat) / 2), 2)
                       + cos(radians(:lat)) * cos(radians(lat)) * power(sin(radians(lon - :lon) / 2), 2)
                   )) AS distance_m
            FROM waters WHERE archived_at IS NULL
        ) w
        WHERE distance_m <= :metres
        ORDER BY distance_m
    """)
    return session.execute(sql, {"lat": lat, "lon": lon, "metres": metres}).all()


def home_listing(session: Session) -> list:
    """Every active water with its snapshot payload, if built, by name."""
    sql = text("""
        SELECT w.slug, w.name, w.timezone, w.water_type, s.built_at, s.payload
        FROM waters w LEFT JOIN snapshots s ON s.water_id = w.id
        WHERE w.archived_at IS NULL
        ORDER BY lower(w.name)
    """)
    return session.execute(sql).all()


def health(session: Session) -> dict:
    """
    The figures /healthz judges, plus each polled feed's name and the time of
    its last good reading (its creation, before it has one).
    """
    sql = text("""
        SELECT
            (SELECT count(*) FROM waters WHERE archived_at IS NULL) AS active_waters,
            (SELECT min(s.built_at)
               FROM waters w JOIN snapshots s ON s.water_id = w.id
              WHERE w.archived_at IS NULL) AS oldest_snapshot,
            (SELECT count(*)
               FROM waters w LEFT JOIN snapshots s ON s.water_id = w.id
              WHERE w.archived_at IS NULL AND s.water_id IS NULL) AS missing_snapshots,
            (SELECT max(fetched_at) FROM weather_hourly) AS last_weather,
            (SELECT max(at) FROM heartbeat) AS last_heartbeat
    """)
    state = dict(session.execute(sql).mappings().one())
    state["feeds"] = session.execute(text("""
        SELECT w.name, COALESCE(f.last_good_reading_at, f.created_at) AS since
        FROM feeds f JOIN waters w ON w.id = f.water_id
        WHERE f.state = 'on' AND w.feed_mode = 'feed' AND w.archived_at IS NULL
        ORDER BY w.name
    """)).all()
    return state

