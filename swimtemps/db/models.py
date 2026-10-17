"""
Tables. All timestamps are timestamptz in UTC and all temperatures Celsius.

Readings are never destroyed: a deletion sets `deleted_at`, and a correction
also points `replaced_by` at the new row. Every read filters
`deleted_at IS NULL`.
"""

import uuid
from datetime import datetime, time

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    Time,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

WATER_TYPES = ("still", "river", "harbour")
DEPTH_CLASSES = ("shallow", "lake", "deep", "unknown")
ROLES = ("admin", "contributor")
FIT_METHODS = ("prior", "scale", "full")
WEATHER_SOURCES = ("archive", "forecast")


def _one_of(column: str, values: tuple[str, ...]) -> CheckConstraint:
    quoted = ", ".join(f"'{v}'" for v in values)
    return CheckConstraint(f"{column} IN ({quoted})", name=f"{column}_valid")


class Base(DeclarativeBase):
    pass


Timestamp = DateTime(timezone=True)


class User(Base):
    __tablename__ = "users"

    # The Supabase auth user id.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    email: Mapped[str] = mapped_column(Text, unique=True)
    # Kept so identity survives a move off Supabase Auth.
    google_sub: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    is_superuser: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())


class Water(Base):
    __tablename__ = "waters"
    __table_args__ = (
        _one_of("water_type", WATER_TYPES),
        _one_of("depth_class", DEPTH_CLASSES),
        CheckConstraint("lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180", name="lat_lon_valid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    lat: Mapped[float] = mapped_column(Double)
    lon: Mapped[float] = mapped_column(Double)
    timezone: Mapped[str] = mapped_column(Text)
    water_type: Mapped[str] = mapped_column(Text)
    depth_class: Mapped[str] = mapped_column(Text, server_default="unknown")
    default_reading_time: Mapped[time] = mapped_column(Time, server_default=text("'07:00'"))
    needs_recompute: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())
    archived_at: Mapped[datetime | None] = mapped_column(Timestamp)


class WaterMember(Base):
    __tablename__ = "water_members"
    __table_args__ = (_one_of("role", ROLES),)

    water_id: Mapped[int] = mapped_column(ForeignKey("waters.id"), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), primary_key=True)
    role: Mapped[str] = mapped_column(Text)


class Reading(Base):
    __tablename__ = "readings"
    __table_args__ = (
        CheckConstraint("temp_c BETWEEN -2 AND 40", name="temp_c_in_range"),
        Index("readings_water_measured", "water_id", "measured_at", postgresql_where=text("deleted_at IS NULL")),
        Index("readings_submitter_recent", "submitted_by", "water_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    water_id: Mapped[int] = mapped_column(ForeignKey("waters.id"))
    measured_at: Mapped[datetime] = mapped_column(Timestamp)
    temp_c: Mapped[float] = mapped_column(Numeric(4, 1, asdecimal=False))
    submitted_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(Timestamp)
    deleted_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    replaced_by: Mapped[int | None] = mapped_column(ForeignKey("readings.id"))


class WeatherHourly(Base):
    __tablename__ = "weather_hourly"
    __table_args__ = (_one_of("source", WEATHER_SOURCES),)

    water_id: Mapped[int] = mapped_column(ForeignKey("waters.id"), primary_key=True)
    valid_at: Mapped[datetime] = mapped_column(Timestamp, primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    air_temp: Mapped[float | None] = mapped_column(Double)
    shortwave: Mapped[float | None] = mapped_column(Double)
    cloud_cover: Mapped[float | None] = mapped_column(Double)
    fetched_at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())


class ModelFit(Base):
    __tablename__ = "model_fits"
    __table_args__ = (_one_of("method", FIT_METHODS),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    water_id: Mapped[int] = mapped_column(ForeignKey("waters.id"), index=True)
    fitted_at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())
    k_air: Mapped[float] = mapped_column(Double)
    k_solar: Mapped[float] = mapped_column(Double)
    k_cool: Mapped[float] = mapped_column(Double)
    scale: Mapped[float] = mapped_column(Double)
    n_legs: Mapped[int] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(Text)
    train_mae: Mapped[float | None] = mapped_column(Double)


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    water_id: Mapped[int] = mapped_column(ForeignKey("waters.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())
    fit_id: Mapped[int] = mapped_column(ForeignKey("model_fits.id"))
    anchor_reading_id: Mapped[int | None] = mapped_column(ForeignKey("readings.id"))
    anchor_at: Mapped[datetime] = mapped_column(Timestamp)
    anchor_temp: Mapped[float] = mapped_column(Double)


class ForecastPoint(Base):
    __tablename__ = "forecast_points"

    run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), primary_key=True)
    valid_at: Mapped[datetime] = mapped_column(Timestamp, primary_key=True)
    water_temp: Mapped[float] = mapped_column(Double)
    lower: Mapped[float] = mapped_column(Double)
    upper: Mapped[float] = mapped_column(Double)


class Snapshot(Base):
    """Exactly what the water page renders. The page reads this one row."""
    __tablename__ = "snapshots"

    water_id: Mapped[int] = mapped_column(ForeignKey("waters.id"), primary_key=True)
    built_at: Mapped[datetime] = mapped_column(Timestamp)
    payload: Mapped[dict] = mapped_column(JSONB)


class Heartbeat(Base):
    """One row per worker cycle. Also keeps the Supabase project from pausing."""
    __tablename__ = "heartbeat"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    at: Mapped[datetime] = mapped_column(Timestamp, server_default=func.now())
