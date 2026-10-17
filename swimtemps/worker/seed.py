"""
Development seed: West Reservoir with its full reading history and archive
weather, so a local site has a real water with a real fit on it.

    python -m worker.seed

Idempotent: an existing `west-reservoir` water is left alone.
"""

import sys
import uuid
from datetime import time

from sqlalchemy import select

from db import queries
from db.models import Reading, User, Water, WaterMember
from db.session import session_factory, session_scope
from forecast.replay import DATA, load_westres

import pandas as pd

SEED_EMAIL = "seed@swimtemps.local"


def seed(factory) -> str:
    with session_scope(factory) as session:
        if session.scalar(select(Water).where(Water.slug == "west-reservoir")):
            return "west-reservoir already exists"
        user = session.scalar(select(User).where(User.email == SEED_EMAIL))
        if user is None:
            user = User(id=uuid.uuid4(), email=SEED_EMAIL, display_name="Seed data")
            session.add(user)
            session.flush()
        water = Water(
            slug="west-reservoir", name="West Reservoir", lat=51.566938, lon=-0.090492,
            timezone="Europe/London", water_type="still", depth_class="lake",
            default_reading_time=time(7, 0), created_by=user.id,
        )
        session.add(water)
        session.flush()
        session.add(WaterMember(water_id=water.id, user_id=user.id, role="admin"))
        readings, _ = load_westres()
        session.add_all(
            Reading(water_id=water.id, measured_at=r.measured_at.to_pydatetime(), temp_c=float(r.temp_c),
                    submitted_by=user.id)
            for r in readings.itertuples()
        )
        weather = pd.read_csv(DATA / "westres_weather.csv", parse_dates=["valid_at"])
        queries.upsert_weather(session, water.id, weather, "archive")
        return f"seeded West Reservoir with {len(readings)} readings"


if __name__ == "__main__":
    print(seed(session_factory()))
    sys.exit(0)
