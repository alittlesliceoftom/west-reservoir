"""feeds: automated sources, feed requests, and reading source and flags

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-18 13:06:57.331564
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('feed_requests',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('page_url', sa.Text(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('ai_block', sa.Text(), nullable=True),
    sa.Column('requested_by', sa.UUID(), nullable=False),
    sa.Column('water_id', sa.Integer(), nullable=True),
    sa.Column('status', sa.Text(), server_default='open', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('open', 'done', 'declined')", name='status_valid'),
    sa.ForeignKeyConstraint(['requested_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('feeds',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('water_id', sa.Integer(), nullable=False),
    sa.Column('recipe', sa.Text(), nullable=False),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('source_name', sa.Text(), nullable=False),
    sa.Column('source_link', sa.Text(), nullable=False),
    sa.Column('thresholds', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('state', sa.Text(), server_default='on', nullable=False),
    sa.Column('attested_by', sa.UUID(), nullable=False),
    sa.Column('attested_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('attested_statement', sa.Text(), nullable=False),
    sa.Column('last_polled_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_good_reading_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("state IN ('on', 'paused', 'off')", name='state_valid'),
    sa.ForeignKeyConstraint(['attested_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['water_id'], ['waters.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('water_id')
    )

    op.add_column('waters', sa.Column('feed_mode', sa.Text(), server_default='manual', nullable=False))
    op.create_check_constraint('feed_mode_valid', 'waters', "feed_mode IN ('manual', 'feed')")

    op.add_column('readings', sa.Column('source', sa.Text(), server_default='manual', nullable=False))
    op.add_column('readings', sa.Column('feed_id', sa.Integer(), nullable=True))
    op.add_column('readings', sa.Column('flagged', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.add_column('readings', sa.Column('flag_reason', sa.Text(), nullable=True))
    op.add_column('readings', sa.Column('flag_override', sa.Text(), nullable=True))
    op.add_column('readings', sa.Column('override_by', sa.UUID(), nullable=True))
    op.add_column('readings', sa.Column('override_at', sa.DateTime(timezone=True), nullable=True))
    op.alter_column('readings', 'submitted_by', existing_type=sa.UUID(), nullable=True)
    op.create_foreign_key('readings_feed_id_fkey', 'readings', 'feeds', ['feed_id'], ['id'])
    op.create_foreign_key('readings_override_by_fkey', 'readings', 'users', ['override_by'], ['id'])
    op.create_check_constraint('source_valid', 'readings', "source IN ('manual', 'feed')")
    op.create_check_constraint('flag_override_valid', 'readings', "flag_override IN ('accept', 'reject')")
    op.create_check_constraint(
        'reading_origin_valid', 'readings',
        "(source = 'manual' AND submitted_by IS NOT NULL) OR (source = 'feed' AND feed_id IS NOT NULL)",
    )
    op.create_index('readings_feed_time', 'readings', ['feed_id', 'measured_at'], unique=True,
                    postgresql_where=sa.text('feed_id IS NOT NULL AND deleted_at IS NULL'))


def downgrade() -> None:
    op.drop_index('readings_feed_time', table_name='readings',
                  postgresql_where=sa.text('feed_id IS NOT NULL AND deleted_at IS NULL'))
    op.drop_constraint('reading_origin_valid', 'readings', type_='check')
    op.drop_constraint('flag_override_valid', 'readings', type_='check')
    op.drop_constraint('source_valid', 'readings', type_='check')
    op.drop_constraint('readings_override_by_fkey', 'readings', type_='foreignkey')
    op.drop_constraint('readings_feed_id_fkey', 'readings', type_='foreignkey')
    # Fails while feed readings exist, since they have no submitter.
    op.alter_column('readings', 'submitted_by', existing_type=sa.UUID(), nullable=False)
    op.drop_column('readings', 'override_at')
    op.drop_column('readings', 'override_by')
    op.drop_column('readings', 'flag_override')
    op.drop_column('readings', 'flag_reason')
    op.drop_column('readings', 'flagged')
    op.drop_column('readings', 'feed_id')
    op.drop_column('readings', 'source')
    op.drop_constraint('feed_mode_valid', 'waters', type_='check')
    op.drop_column('waters', 'feed_mode')
    op.drop_table('feeds')
    op.drop_table('feed_requests')
