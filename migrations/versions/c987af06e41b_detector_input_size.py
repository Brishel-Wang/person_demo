"""Record detector input size and include it in ingest reuse."""

import sqlalchemy as sa
from alembic import op

revision = "c987af06e41b"
down_revision = "bf7d61cc6930"
branch_labels = None
depends_on = None

_RUN_KEY = ["source_asset_id", "window_start_us", "window_end_us", "analysis_fps",
            "detector_sha256", "feature_space_id", "vision_profile"]


def upgrade() -> None:
    op.add_column("ingest_runs", sa.Column("detector_imgsz", sa.Integer(), nullable=False, server_default="640"))
    op.drop_constraint("uq_ingest_same_config", "ingest_runs", type_="unique")
    op.create_unique_constraint("uq_ingest_same_config", "ingest_runs", [*_RUN_KEY, "detector_imgsz"])


def downgrade() -> None:
    op.drop_constraint("uq_ingest_same_config", "ingest_runs", type_="unique")
    op.create_unique_constraint("uq_ingest_same_config", "ingest_runs", _RUN_KEY)
    op.drop_column("ingest_runs", "detector_imgsz")
