"""Analytical reads and bulk writes, as SQL."""

from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

MEASURES = ("air_temp", "shortwave", "cloud_cover")

# Archive values overwrite forecast values; a forecast never overwrites an
# archive value. A null in the incoming row keeps what is stored.
UPSERT_WEATHER = text("""
    INSERT INTO weather_hourly (water_id, valid_at, source, air_temp, shortwave, cloud_cover, fetched_at)
    VALUES (:water_id, :valid_at, :source, :air_temp, :shortwave, :cloud_cover, now())
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


def upsert_weather(session: Session, water_id: int, frame: pd.DataFrame, source: str) -> int:
    """Store hourly weather for one water. Rows with no air temperature are skipped."""
    rows = [
        {
            "water_id": water_id,
            "valid_at": row.valid_at.to_pydatetime(),
            "source": source,
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
    """Readings that stand, oldest first: id, measured_at (UTC), temp_c (float)."""
    sql = text("""
        SELECT id, measured_at, temp_c::float8 AS temp_c
        FROM readings
        WHERE water_id = :water_id AND deleted_at IS NULL
        ORDER BY measured_at, id
    """)
    rows = session.execute(sql, {"water_id": water_id}).all()
    frame = pd.DataFrame(rows, columns=["id", "measured_at", "temp_c"])
    frame["measured_at"] = pd.to_datetime(frame["measured_at"], utc=True)
    return frame.astype({"temp_c": float})


def readings_changed_since(session: Session, water_id: int, since: datetime) -> bool:
    """Whether any reading was added, deleted or corrected after `since`."""
    sql = text("""
        SELECT EXISTS (
            SELECT 1 FROM readings
            WHERE water_id = :water_id AND (created_at > :since OR deleted_at > :since)
        )
    """)
    return bool(session.execute(sql, {"water_id": water_id, "since": since}).scalar())


def earliest_reading_without_weather(session: Session, water_id: int) -> datetime | None:
    """The oldest standing reading whose hour has no stored weather."""
    sql = text("""
        SELECT min(r.measured_at)
        FROM readings r
        LEFT JOIN weather_hourly w
          ON w.water_id = r.water_id AND w.valid_at = date_trunc('hour', r.measured_at)
        WHERE r.water_id = :water_id AND r.deleted_at IS NULL AND w.valid_at IS NULL
    """)
    return session.execute(sql, {"water_id": water_id}).scalar()


def own_errors(session: Session, water_id: int, min_points: int) -> dict[int, float] | None:
    """
    This water's 80th-percentile absolute forecast error by horizon, or None
    until it has `min_points` scored points.

    A point is scored against a reading in the same hour. For each reading and
    horizon only the latest run made before the reading counts, so hourly
    re-runs do not multiply one miss into many.
    """
    sql = text("""
        WITH scored AS (
            SELECT DISTINCT ON (r.id, horizon)
                r.id AS reading_id,
                GREATEST(1, CEIL(EXTRACT(EPOCH FROM (p.valid_at - fr.anchor_at)) / 86400 - 1e-9))::int AS horizon,
                abs(p.water_temp - r.temp_c::float8) AS abs_error
            FROM forecast_points p
            JOIN forecast_runs fr ON fr.id = p.run_id
            JOIN readings r
              ON r.water_id = fr.water_id
             AND r.deleted_at IS NULL
             AND date_trunc('hour', r.measured_at + interval '30 minutes') = p.valid_at
            WHERE fr.water_id = :water_id AND fr.created_at < r.measured_at
            ORDER BY r.id, horizon, fr.created_at DESC
        )
        SELECT horizon, count(*) AS n,
               percentile_cont(0.8) WITHIN GROUP (ORDER BY abs_error) AS p80
        FROM scored GROUP BY horizon ORDER BY horizon
    """)
    rows = session.execute(sql, {"water_id": water_id}).all()
    if sum(r.n for r in rows) < min_points:
        return None
    return {int(r.horizon): float(r.p80) for r in rows if r.n >= 3}


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
    sql = text("""
        SELECT
            (SELECT count(*) FROM waters WHERE archived_at IS NULL) AS active_waters,
            (SELECT min(coalesce(s.built_at, '-infinity'))
               FROM waters w LEFT JOIN snapshots s ON s.water_id = w.id
              WHERE w.archived_at IS NULL) AS oldest_snapshot,
            (SELECT max(fetched_at) FROM weather_hourly) AS last_weather,
            (SELECT max(at) FROM heartbeat) AS last_heartbeat
    """)
    return dict(session.execute(sql).mappings().one())

