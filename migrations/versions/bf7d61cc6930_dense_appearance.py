"""Version per-frame appearance ingestion and save the strongest matched frame."""

import sqlalchemy as sa
from alembic import op

revision = "bf7d61cc6930"
down_revision = "a88bd31a6f20"
branch_labels = None
depends_on = None

_RUN_KEY = ["source_asset_id", "window_start_us", "window_end_us", "analysis_fps",
            "detector_sha256", "feature_space_id"]


def upgrade() -> None:
    op.add_column("ingest_runs", sa.Column("vision_profile", sa.String(32), nullable=False, server_default="legacy"))
    op.drop_constraint("uq_ingest_same_config", "ingest_runs", type_="unique")
    op.create_unique_constraint("uq_ingest_same_config", "ingest_runs", [*_RUN_KEY, "vision_profile"])
    op.add_column("search_candidates", sa.Column("matched_local_us", sa.BigInteger))
    op.create_index("ix_observations_tracklet_us", "observations", ["tracklet_revision_id", "local_us"])


def downgrade() -> None:
    op.drop_index("ix_observations_tracklet_us", table_name="observations")
    op.drop_column("search_candidates", "matched_local_us")
    op.drop_constraint("uq_ingest_same_config", "ingest_runs", type_="unique")
    op.create_unique_constraint("uq_ingest_same_config", "ingest_runs", _RUN_KEY)
    op.drop_column("ingest_runs", "vision_profile")
