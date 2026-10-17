"""Engine and sessions. DATABASE_URL is the only setting."""

import os
from contextlib import contextmanager
from functools import cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


class ConfigError(RuntimeError):
    pass


def normalise_url(url: str) -> str:
    """Supabase hands out postgres:// URLs; SQLAlchemy needs the psycopg driver named."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError("DATABASE_URL is not set. See deploy/.env.example.")
    return normalise_url(url)


@cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(normalise_url(url) if url else database_url(), pool_pre_ping=True)


def session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(get_engine(url), expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]):
    """A session committed on success and rolled back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
