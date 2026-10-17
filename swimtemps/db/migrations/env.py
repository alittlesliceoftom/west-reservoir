from alembic import context
from sqlalchemy import create_engine, pool

from db.models import Base
from db.session import database_url, normalise_url

config = context.config
target_metadata = Base.metadata


def url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    return normalise_url(configured) if configured else database_url()


def run_migrations_offline() -> None:
    context.configure(url=url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
