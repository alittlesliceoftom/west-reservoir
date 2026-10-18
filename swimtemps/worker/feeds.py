"""
Polling feeds: fetch, read, check, thin, store, clean.

Rows are matched to the feed's standing readings by measured time. A new
time is inserted; the same value is left alone, so re-fetching a full
history adds nothing; a different value is the venue's correction and is
stored the way an admin correction is (old row soft-deleted, new row, old
`replaced_by` new). A standing row already in the same clock hour wins over
a new row at another minute, so thinning stays stable from poll to poll.

New and corrected rows are cleaned against the feed's good readings and
the stored air temperature. A poll with no usable row (the venue renamed a
key, or the feed is empty) is a failure with the reason in `last_error`.
Each feed is polled in its own transaction, and one failure never stops
the others.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select, text, update
from sqlalchemy.orm import Session, sessionmaker

from db import queries
from db.models import Feed, Reading, Water
from feeds.clean import clean
from feeds.fetch import FetchError, fetch_json
from feeds.recipe import RecipeError, read_feed
from feeds.thin import thin_hourly
from web.paste import EARLIEST, MAX_TEMP, MIN_TEMP

log = logging.getLogger("swimtemps.worker")

FUTURE_SLACK = timedelta(minutes=10)   # as for manual readings
CLEAN_HISTORY = timedelta(hours=48)
HOUR = timedelta(hours=1)


@dataclass
class PollResult:
    new: int = 0
    corrected: int = 0
    flagged: int = 0
    errors: list[str] = field(default_factory=list)
    ok: bool = True     # False when the fetch or the recipe failed


def _utc(at: datetime) -> datetime:
    return at.astimezone(timezone.utc)


def _hour(at: datetime) -> datetime:
    return _utc(at).replace(minute=0, second=0, microsecond=0)


def checked(rows: list[tuple[datetime, float]], now: datetime) -> tuple[list[tuple[datetime, float]], list[str]]:
    """The rows that pass the manual-reading checks, to 0.1 °C, and why the others did not."""
    kept, errors = [], []
    for at, temp in rows:
        value = round(temp, 1)
        if not MIN_TEMP <= value <= MAX_TEMP:
            errors.append(f"{at:%Y-%m-%d %H:%M} UTC: {value:.1f} °C is outside -2 to 40 °C")
        elif at > now + FUTURE_SLACK:
            errors.append(f"{at:%Y-%m-%d %H:%M} UTC: in the future")
        elif at.date() < EARLIEST:
            errors.append(f"{at:%Y-%m-%d %H:%M} UTC: before 2000")
        else:
            kept.append((at, value))
    return kept, errors


def _good_history(session: Session, feed: Feed, written: list[Reading]) -> list[tuple[datetime, float]]:
    """
    The feed's usable standing readings from 48 h before the first written
    row to the last, leaving out the written rows themselves.
    """
    rows = session.execute(text(f"""
        SELECT r.measured_at, r.temp_c::float8 FROM readings r
        WHERE r.feed_id = :feed AND r.deleted_at IS NULL AND {queries.USABLE_SQL}
          AND r.measured_at >= :since AND r.measured_at <= :until AND NOT (r.id = ANY(:written))
        ORDER BY r.measured_at
    """), {"feed": feed.id, "since": written[0].measured_at - CLEAN_HISTORY, "until": written[-1].measured_at,
           "written": [r.id for r in written]}).all()
    return [(_utc(at), temp) for at, temp in rows]


def _air(session: Session, water_id: int, since: datetime) -> pd.Series:
    weather = queries.weather_frame(session, water_id, since=since)
    return pd.Series(weather["air_temp"].to_numpy(), index=pd.DatetimeIndex(weather["valid_at"]))


def last_good_reading_at(session: Session, feed_id: int) -> datetime | None:
    return session.execute(text(f"""
        SELECT max(r.measured_at) FROM readings r
        WHERE r.feed_id = :feed AND r.deleted_at IS NULL AND {queries.USABLE_SQL}
    """), {"feed": feed_id}).scalar()


def store(session: Session, feed: Feed, rows: list[tuple[datetime, float]], now: datetime) -> tuple[list[Reading], int]:
    """Insert or correct thinned rows. Returns the readings written, and how many were corrections."""
    if not rows:
        return [], 0
    existing = session.scalars(select(Reading).where(
        Reading.feed_id == feed.id, Reading.deleted_at.is_(None),
        Reading.measured_at >= _hour(rows[0][0]), Reading.measured_at < _hour(rows[-1][0]) + HOUR,
    )).all()
    by_time = {_utc(r.measured_at): r for r in existing}
    taken_hours = {_hour(r.measured_at) for r in existing}

    written, corrected = [], 0
    for at, temp in rows:
        old = by_time.get(at)
        if old is None and _hour(at) in taken_hours:
            continue
        if old is not None and abs(float(old.temp_c) - temp) < 0.05:
            continue
        if old is not None:
            old.deleted_at = now
            session.flush()   # the partial unique index allows one standing row per time
            corrected += 1
        reading = Reading(water_id=feed.water_id, measured_at=at, temp_c=temp, source="feed", feed_id=feed.id)
        session.add(reading)
        if old is not None:
            session.flush()
            old.replaced_by = reading.id
        taken_hours.add(_hour(at))
        written.append(reading)
    return written, corrected


def poll_feed(session: Session, feed: Feed, now: datetime, fetch=fetch_json) -> PollResult:
    """One poll of one feed. The caller commits."""
    water = session.get(Water, feed.water_id)
    feed.last_polled_at = now
    try:
        payload = fetch(feed.config["url"])
        parsed = read_feed(feed.recipe, payload, feed.config, water.timezone, water.default_reading_time)
    except (FetchError, RecipeError) as error:
        feed.last_error = str(error)
        log.warning("feed %s for water %s: %s", feed.id, water.id, error)
        return PollResult(errors=[str(error)], ok=False)

    rows, rejected = checked(parsed.rows, now)
    problems = parsed.errors + rejected
    if not rows:
        total = len(parsed.rows) + len(parsed.errors)
        feed.last_error = (f"0 of {total} rows could be used, e.g. {problems[0]}." if problems
                           else "The feed returned no readings.")
        log.warning("feed %s for water %s: %s", feed.id, water.id, feed.last_error)
        return PollResult(errors=problems, ok=False)
    written, corrected = store(session, feed, thin_hourly(rows), now)
    flagged = 0
    if written:
        session.flush()
        written.sort(key=lambda r: r.measured_at)
        first = written[0].measured_at
        reasons = clean([(r.measured_at, float(r.temp_c)) for r in written], _good_history(session, feed, written),
                        _air(session, water.id, first - CLEAN_HISTORY), feed.thresholds)
        for reading, reason in zip(written, reasons):
            reading.flagged, reading.flag_reason = reason is not None, reason
        flagged = sum(reason is not None for reason in reasons)
        session.flush()
        queries.mark_dirty(session, water.id)

    feed.last_success_at = now
    feed.last_error = None
    feed.last_good_reading_at = last_good_reading_at(session, feed.id)
    result = PollResult(new=len(written) - corrected, corrected=corrected, flagged=flagged, errors=problems)
    if result.errors:
        log.info("feed %s: %d rows not stored, e.g. %s", feed.id, len(result.errors), result.errors[0])
    return result


def poll_feeds(factory: sessionmaker, now: datetime, fetch=fetch_json, unpolled_only: bool = False) -> int:
    """
    Poll every feed that is on, for a water in feed mode that is not
    archived; with `unpolled_only`, just those never polled. Returns how many
    polls succeeded.
    """
    query = (select(Feed.id).join(Water, Water.id == Feed.water_id)
             .where(Feed.state == "on", Water.feed_mode == "feed", Water.archived_at.is_(None)).order_by(Feed.id))
    if unpolled_only:
        query = query.where(Feed.last_polled_at.is_(None))
    with factory() as session:
        ids = list(session.scalars(query))

    succeeded = 0
    for feed_id in ids:
        try:
            with factory() as session:
                result = poll_feed(session, session.get(Feed, feed_id), now, fetch)
                session.commit()
            succeeded += result.ok
        except Exception as error:
            log.exception("polling feed %s failed unexpectedly", feed_id)
            with factory() as session:
                session.execute(update(Feed).where(Feed.id == feed_id).values(
                    last_polled_at=now, last_error=f"Unexpected error while polling ({type(error).__name__})."))
                session.commit()
    return succeeded
