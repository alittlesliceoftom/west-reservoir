"""
Test database: a throwaway Postgres cluster, migrated with Alembic.

Set TEST_DATABASE_URL to use an existing server instead (CI does). Otherwise
`initdb` and `pg_ctl` must be on PATH or in PG_BIN; Homebrew's postgresql@17
is found automatically.
"""

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).parents[1]
HOMEBREW_PG = Path("/opt/homebrew/opt/postgresql@17/bin")


def _pg_bin() -> Path:
    if os.environ.get("PG_BIN"):
        return Path(os.environ["PG_BIN"])
    found = shutil.which("pg_ctl")
    return Path(found).parent if found else HOMEBREW_PG


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def migrate(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "db" / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def database_url(tmp_path_factory):
    if os.environ.get("TEST_DATABASE_URL"):
        url = os.environ["TEST_DATABASE_URL"]
        migrate(url)
        yield url
        return

    bin_dir = _pg_bin()
    data = tmp_path_factory.mktemp("pg") / "data"
    port = _free_port()
    subprocess.run(
        [bin_dir / "initdb", "-D", data, "-U", "postgres", "-A", "trust", "-E", "UTF8", "--no-locale"],
        check=True, capture_output=True,
    )
    options = f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories='' -c fsync=off"
    # macOS Postgres refuses to start without a valid locale in the environment.
    env = {**os.environ, "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8")}
    subprocess.run(
        [bin_dir / "pg_ctl", "-D", data, "-o", options, "-l", data / "log", "-w", "start"],
        check=True, capture_output=True, env=env,
    )
    try:
        admin = create_engine(f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres", isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            conn.execute(text("CREATE DATABASE swimtemps_test"))
        admin.dispose()
        url = f"postgresql+psycopg://postgres@127.0.0.1:{port}/swimtemps_test"
        migrate(url)
        yield url
    finally:
        subprocess.run([bin_dir / "pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)


@pytest.fixture(scope="session")
def engine(database_url):
    engine = create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine):
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory
    with engine.begin() as conn:
        tables = conn.execute(text(
            "SELECT string_agg(quote_ident(tablename), ', ') FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
        )).scalar()
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def db(session_factory):
    with session_factory() as session:
        yield session
