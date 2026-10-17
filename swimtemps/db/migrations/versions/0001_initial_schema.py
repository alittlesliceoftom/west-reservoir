"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-18 00:48:48.227593
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('heartbeat',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email', sa.Text(), nullable=False),
    sa.Column('google_sub', sa.Text(), nullable=True),
    sa.Column('display_name', sa.Text(), nullable=True),
    sa.Column('is_superuser', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    op.create_table('waters',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('slug', sa.Text(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('lat', sa.Double(), nullable=False),
    sa.Column('lon', sa.Double(), nullable=False),
    sa.Column('timezone', sa.Text(), nullable=False),
    sa.Column('water_type', sa.Text(), nullable=False),
    sa.Column('depth_class', sa.Text(), server_default='unknown', nullable=False),
    sa.Column('default_reading_time', sa.Time(), server_default=sa.text("'07:00'"), nullable=False),
    sa.Column('needs_recompute', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_by', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("depth_class IN ('shallow', 'lake', 'deep', 'unknown')", name='depth_class_valid'),
    sa.CheckConstraint("water_type IN ('still', 'river', 'harbour')", name='water_type_valid'),
    sa.CheckConstraint('lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180', name='lat_lon_valid'),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('slug')
    )
    op.create_table('model_fits',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('fitted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('k_air', sa.Double(), nullable=False),
    sa.Column('k_solar', sa.Double(), nullable=False),
    sa.Column('k_cool', sa.Double(), nullable=False),
    sa.Column('scale', sa.Double(), nullable=False),
    sa.Column('n_legs', sa.Integer(), nullable=False),
    sa.Column('method', sa.Text(), nullable=False),
    sa.Column('train_mae', sa.Double(), nullable=True),
    sa.CheckConstraint("method IN ('prior', 'scale', 'full')", name='method_valid'),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_fits_water_id'), 'model_fits', ['water_id'], unique=False)
    op.create_table('readings',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('measured_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('temp_c', sa.Numeric(precision=4, scale=1, asdecimal=False), nullable=False),
    sa.Column('submitted_by', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('deleted_by', sa.UUID(), nullable=True),
    sa.Column('replaced_by', sa.BigInteger(), nullable=True),
    sa.CheckConstraint('temp_c BETWEEN -2 AND 40', name='temp_c_in_range'),
    sa.ForeignKeyConstraint(['deleted_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['replaced_by'], ['readings.id'], ),
    sa.ForeignKeyConstraint(['submitted_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('readings_submitter_recent', 'readings', ['submitted_by', 'water_id', 'created_at'], unique=False)
    op.create_index('readings_water_measured', 'readings', ['water_id', 'measured_at'], unique=False, postgresql_where=sa.text('deleted_at IS NULL'))
    op.create_table('snapshots',
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('built_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('water_id')
    )
    op.create_table('water_members',
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('role', sa.Text(), nullable=False),
    sa.CheckConstraint("role IN ('admin', 'contributor')", name='role_valid'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('water_id', 'user_id')
    )
    op.create_table('weather_hourly',
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('valid_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('source', sa.Text(), nullable=False),
    sa.Column('air_temp', sa.Double(), nullable=True),
    sa.Column('shortwave', sa.Double(), nullable=True),
    sa.Column('cloud_cover', sa.Double(), nullable=True),
    sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source IN ('archive', 'forecast')", name='source_valid'),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('water_id', 'valid_at')
    )
    op.create_table('forecast_runs',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('fit_id', sa.BigInteger(), nullable=False),
    sa.Column('anchor_reading_id', sa.BigInteger(), nullable=True),
    sa.Column('anchor_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('anchor_temp', sa.Double(), nullable=False),
    sa.ForeignKeyConstraint(['anchor_reading_id'], ['readings.id'], ),
    sa.ForeignKeyConstraint(['fit_id'], ['model_fits.id'], ),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_forecast_runs_water_id'), 'forecast_runs', ['water_id'], unique=False)
    op.create_table('forecast_points',
    sa.Column('run_id', sa.BigInteger(), nullable=False),
    sa.Column('valid_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('water_temp', sa.Double(), nullable=False),
    sa.Column('lower', sa.Double(), nullable=False),
    sa.Column('upper', sa.Double(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['forecast_runs.id'], ),
    sa.PrimaryKeyConstraint('run_id', 'valid_at')
    )


def downgrade() -> None:
    op.drop_table('forecast_points')
    op.drop_index(op.f('ix_forecast_runs_water_id'), table_name='forecast_runs')
    op.drop_table('forecast_runs')
    op.drop_table('weather_hourly')
    op.drop_table('water_members')
    op.drop_table('snapshots')
    op.drop_index('readings_water_measured', table_name='readings', postgresql_where=sa.text('deleted_at IS NULL'))
    op.drop_index('readings_submitter_recent', table_name='readings')
    op.drop_table('readings')
    op.drop_index(op.f('ix_model_fits_water_id'), table_name='model_fits')
    op.drop_table('model_fits')
    op.drop_table('waters')
    op.drop_table('users')
    op.drop_table('heartbeat')
