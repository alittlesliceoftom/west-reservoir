"""Builders for test rows."""

import uuid
from datetime import time

import pandas as pd

from db.models import Reading, User, Water, WaterMember


def make_user(db, email="alice@example.com", superuser=False, name=None) -> User:
    user = User(
        id=uuid.uuid4(), email=email, google_sub=f"g-{email}",
        display_name=name or email.split("@")[0].title(), is_superuser=superuser,
    )
    db.add(user)
    db.commit()
    return user


def make_water(db, creator, name="Test Lake", slug=None, lat=51.45, lon=-2.6,
               timezone="Europe/London", water_type="still", depth_class="lake",
               reading_time=time(7, 0), admin=True, **extra) -> Water:
    water = Water(
        slug=slug or name.lower().replace(" ", "-"), name=name, lat=lat, lon=lon,
        timezone=timezone, water_type=water_type, depth_class=depth_class,
        default_reading_time=reading_time, created_by=creator.id, **extra,
    )
    db.add(water)
    db.flush()
    if admin:
        db.add(WaterMember(water_id=water.id, user_id=creator.id, role="admin"))
    db.commit()
    return water


def add_reading(db, water, user, at, temp, **extra) -> Reading:
    reading = Reading(
        water_id=water.id, measured_at=pd.Timestamp(at, tz="UTC").to_pydatetime() if isinstance(at, str) else at,
        temp_c=temp, submitted_by=user.id, **extra,
    )
    db.add(reading)
    db.commit()
    return reading


def make_feed(db, water, user, state="on", config=None, **extra):
    from db.models import Feed

    feed = Feed(
        water_id=water.id, recipe="json_table",
        config=config or {"url": "https://example-lake.org/temps", "orientation": "columns",
                          "time_path": "labels", "value_path": "datasets[0].data", "time_format": "epoch_ms"},
        source_name="Example Lake", source_link="https://example-lake.org/live/",
        state=state, attested_by=user.id, attested_at=pd.Timestamp("2026-10-01", tz="UTC").to_pydatetime(),
        attested_statement="I have the venue's permission to share this data, or it is publicly published.",
        created_by=user.id, **extra,
    )
    db.add(feed)
    db.commit()
    return feed


def add_feed_reading(db, feed, at, temp, **extra) -> Reading:
    reading = Reading(
        water_id=feed.water_id, measured_at=pd.Timestamp(at, tz="UTC").to_pydatetime() if isinstance(at, str) else at,
        temp_c=temp, **{"source": "feed", "feed_id": feed.id, **extra},
    )
    db.add(reading)
    db.commit()
    return reading
