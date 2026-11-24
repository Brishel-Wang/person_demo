"""Persist video knowledge retrieval and Qwen conversation turns."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "a88bd31a6f20"
down_revision = "55c2c82b4d17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "video_chat_turns",
        sa.Column("turn_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("video_chat_turns.turn_id")),
        sa.Column("source_asset_id", UUID(as_uuid=True), sa.ForeignKey("source_assets.source_asset_id"), nullable=False),
        sa.Column("search_task_id", UUID(as_uuid=True), sa.ForeignKey("search_tasks.task_id")),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("request", JSONB(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("snapshot", JSONB()),
        sa.Column("reply", JSONB()),
        sa.Column("provider", sa.String(24)),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_video_chat_turns_status", "video_chat_turns", ["status"])
    op.create_index("ix_video_chat_turns_created_at", "video_chat_turns", ["created_at"])


def downgrade() -> None:
    op.drop_table("video_chat_turns")
