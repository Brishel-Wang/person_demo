"""Add the report evidence hash on databases stamped by the empty migration."""

from alembic import op


revision = "55c2c82b4d17"
down_revision = "31fcc4be0f2a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS evidence_hash VARCHAR(64)")


def downgrade() -> None:
    # Some databases already had this column at the previous revision.
    # Preserve hashes written by either application version.
    pass
