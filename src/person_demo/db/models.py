"""ORM 模型：素材、camera、观测、tracklet、continuation、向量清单、覆盖、
任务、复核、报告与元数据版本（spec 4.1 三类存储中的 PostgreSQL 部分）。

所有时间字段 local_us 均为 asset_relative 片内微秒。
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from person_demo.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------- cameras / assets
class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    camera_code: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("dataset_id", "camera_code", name="uq_cameras_dataset_code"),)


class SourceAsset(Base):
    """不可变录像资产；source_asset_id 与 camera_id 分开（spec 2.2）。"""

    __tablename__ = "source_assets"

    source_asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cameras.camera_id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    # probe 结果
    codec_name: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    avg_fps: Mapped[float | None] = mapped_column(Float)
    is_vfr: Mapped[bool | None] = mapped_column(Boolean)
    duration_us: Mapped[int | None] = mapped_column(BigInteger)
    nb_frames: Mapped[int | None] = mapped_column(BigInteger)
    pts_anchor: Mapped[int | None] = mapped_column(BigInteger)
    time_base_num: Mapped[int | None] = mapped_column(Integer)
    time_base_den: Mapped[int | None] = mapped_column(Integer)
    timebase_status: Mapped[str] = mapped_column(String(32), default="unprobed")
    # resolved 处理窗口
    resolved_window_start_us: Mapped[int | None] = mapped_column(BigInteger)
    resolved_window_end_us: Mapped[int | None] = mapped_column(BigInteger)
    # 真实时间/地点可空（spec 10.1）
    recording_time_status: Mapped[str] = mapped_column(String(16), default="unknown")
    location_status: Mapped[str] = mapped_column(String(16), default="unknown")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    camera: Mapped[Camera] = relationship()


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    ingest_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_assets.source_asset_id"), nullable=False)
    window_start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    window_end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    analysis_fps: Mapped[int] = mapped_column(Integer, nullable=False)
    vision_profile: Mapped[str] = mapped_column(String(32), nullable=False, server_default="legacy")
    tracker_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detector_sha256: Mapped[str | None] = mapped_column(String(64))
    detector_imgsz: Mapped[int] = mapped_column(Integer, nullable=False, server_default="640")
    feature_space_id: Mapped[str | None] = mapped_column(String(64))
    aggregation_version: Mapped[str] = mapped_column(String(32), default="mean_l2_v1")
    tracker_config: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/analyzing/committing/publishing/ready/failed
    fence_token: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=_uuid)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint(
            "source_asset_id", "window_start_us", "window_end_us", "analysis_fps",
            "detector_sha256", "feature_space_id", "vision_profile", "detector_imgsz",
            name="uq_ingest_same_config",
        ),
    )


class Tracklet(Base):
    """不可变 chunk 版本（spec 3.1）。"""

    __tablename__ = "tracklets"

    tracklet_revision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    ingest_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ingest_runs.ingest_run_id"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_code: Mapped[str] = mapped_column(String(32), nullable=False)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_assets.source_asset_id"), nullable=False)
    local_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    continuation_of: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    continuation_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    t_start_local_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_end_local_us: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 左闭右开
    representative_local_us: Mapped[int | None] = mapped_column(BigInteger)
    quality: Mapped[float | None] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    feature_space_id: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregation_version: Mapped[str] = mapped_column(String(32), default="mean_l2_v1")
    vector_key: Mapped[str | None] = mapped_column(String(512))
    row_index_key: Mapped[str | None] = mapped_column(String(512))
    suspicious: Mapped[bool] = mapped_column(Boolean, default=False)
    suspicious_reason: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="committed")  # temp/committed/tombstoned
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    observations: Mapped[list["Observation"]] = relationship(back_populates="tracklet")

    __table_args__ = (
        UniqueConstraint("ingest_run_id", "local_track_id", "chunk_no", name="uq_tracklet_run_id_chunk"),
        Index("ix_tracklets_asset_time", "source_asset_id", "t_start_local_us"),
    )


class Observation(Base):
    __tablename__ = "observations"

    observation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tracklet_revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracklets.tracklet_revision_id"), nullable=False)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_assets.source_asset_id"), nullable=False)
    local_us: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    frame_pts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    time_base_num: Mapped[int] = mapped_column(Integer, nullable=False)
    time_base_den: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x2: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y2: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_w: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_h: Mapped[int] = mapped_column(Integer, nullable=False)
    crop_key: Mapped[str | None] = mapped_column(String(512))
    crop_status: Mapped[str] = mapped_column(String(32), default="valid")  # valid/empty_box/clip_fail/decode_fail
    laplacian_var: Mapped[float | None] = mapped_column(Float)
    is_representative: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    tracklet: Mapped[Tracklet] = relationship(back_populates="observations")

    __table_args__ = (
        Index("ix_observations_asset_us", "source_asset_id", "local_us"),
        Index("ix_observations_tracklet_us", "tracklet_revision_id", "local_us"),
    )


class VectorOutbox(Base):
    """SQL 事务提交后由 publisher 领取 → Milvus upsert → Strong 校验 → ready。"""

    __tablename__ = "vector_outbox"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    ingest_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tracklet_revision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    feature_space_id: Mapped[str] = mapped_column(String(64), nullable=False)
    publish_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/published/verified/ready
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        UniqueConstraint("tracklet_revision_id", "feature_space_id", name="uq_outbox_tracklet_space"),
        Index("ix_outbox_scope", "dataset_id", "source_asset_id", "ingest_run_id", "feature_space_id", "status"),
    )


class PublishWatermark(Base):
    """连续 ready 水位，只前进到连续序号，不跨空洞（spec 4.3）。"""

    __tablename__ = "publish_watermarks"

    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    ingest_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    feature_space_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    watermark_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=-1)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class CoverageSegment(Base):
    __tablename__ = "coverage_segments"

    segment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    ingest_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="published")  # analyzed/published
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_coverage_asset", "dataset_id", "source_asset_id"),)


class QueryAsset(Base):
    __tablename__ = "query_assets"

    asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # from_frame / upload
    source_asset_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    origin_observation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    origin_tracklet_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    origin_continuation_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    origin_camera_code: Mapped[str | None] = mapped_column(String(32))
    origin_status: Mapped[str] = mapped_column(String(32), default="verified")  # verified/unknown/unresolved
    bbox: Mapped[dict | None] = mapped_column(JSONB)  # [x1,y1,x2,y2]
    image_key: Mapped[str | None] = mapped_column(String(512))
    embedding_key: Mapped[str | None] = mapped_column(String(512))
    feature_space_id: Mapped[str | None] = mapped_column(String(64))
    local_us: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SearchTask(Base):
    __tablename__ = "search_tasks"

    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("query_assets.asset_id"), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), default="body_only")
    evaluation_scope: Mapped[str] = mapped_column(String(32), default="intra_camera_demo")
    scope_windows: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    search_revision: Mapped[int] = mapped_column(Integer, default=1)
    frozen_ingest_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    frozen_feature_space_id: Mapped[str | None] = mapped_column(String(64))
    frozen_watermark: Mapped[int | None] = mapped_column(BigInteger)
    frozen_coverage: Mapped[dict | None] = mapped_column(JSONB)
    snapshot: Mapped[dict | None] = mapped_column(JSONB)
    candidate_truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    recalled_count: Mapped[int] = mapped_column(Integer, default=0)
    kept_count: Mapped[int] = mapped_column(Integer, default=0)
    origin_exclusion_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    candidates: Mapped[list["SearchCandidate"]] = relationship(back_populates="task")


class SearchCandidate(Base):
    __tablename__ = "search_candidates"

    hit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    search_task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("search_tasks.task_id"), nullable=False)
    tracklet_revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracklets.tracklet_revision_id"), nullable=False)
    prototype_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    rerank_similarity: Mapped[float | None] = mapped_column(Float)
    rerank_status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/ok/rerank_failed
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    same_continuation_as_query: Mapped[bool] = mapped_column(Boolean, default=False)
    matched_local_us: Mapped[int | None] = mapped_column(BigInteger)
    matched_buckets: Mapped[list | None] = mapped_column(JSONB)
    review_status: Mapped[str] = mapped_column(String(32), default="pending_review")  # pending_review/confirmed/rejected
    review_revision: Mapped[int] = mapped_column(Integer, default=0)
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=_uuid)
    rank_position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    task: Mapped[SearchTask] = relationship(back_populates="candidates")
    tracklet: Mapped[Tracklet] = relationship()

    __table_args__ = (Index("ix_candidates_task", "search_task_id", "rank_position"),)


class Review(Base):
    __tablename__ = "reviews"

    review_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    hit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("search_candidates.hit_id"), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # confirm/reject
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("hit_id", "revision", name="uq_review_hit_revision"),)


class EvidenceBundle(Base):
    __tablename__ = "evidence_bundles"

    bundle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    search_task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("search_tasks.task_id"), nullable=False)
    search_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ReportJob(Base):
    __tablename__ = "report_jobs"

    report_job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    search_task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("search_tasks.task_id"), nullable=False)
    search_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    provider: Mapped[str | None] = mapped_column(String(32))
    prompt_version: Mapped[str] = mapped_column(String(64), default="single_video_report_v1")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    timing: Mapped[dict | None] = mapped_column(JSONB)
    resource: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    search_task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("search_tasks.task_id"), nullable=False)
    search_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    report_job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("report_jobs.report_job_id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="SUCCEEDED")
    evidence_hash: Mapped[str | None] = mapped_column(String(64))
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    raw_output: Mapped[dict | None] = mapped_column(JSONB)
    validation: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ReportClaim(Base):
    __tablename__ = "report_claims"

    claim_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.report_id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids: Mapped[list] = mapped_column(JSONB, nullable=False)
    hypothesis_id: Mapped[str | None] = mapped_column(String(128))


class MetadataRevision(Base):
    __tablename__ = "metadata_revisions"

    revision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)  # camera_location / asset_metadata
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("entity_type", "entity_id", "revision", name="uq_meta_rev"),)


class Tombstone(Base):
    __tablename__ = "tombstones"

    tombstone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(256))
    deleted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_tombstones_entity", "entity_type", "entity_id"),)


class AssetProxy(Base):
    __tablename__ = "asset_proxies"

    proxy_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_assets.source_asset_id"), nullable=False)
    window_start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    window_end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    time_mapping: Mapped[dict] = mapped_column(JSONB, nullable=False)  # {frame_idx: local_us}
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DatasetStatus(Base):
    """数据集级状态快照（幂等导入与状态接口用）。"""

    __tablename__ = "dataset_status"

    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="empty")  # empty/importing/indexing/ready/partial/failed
    last_ingest_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    detail: Mapped[dict | None] = mapped_column(JSONB)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


def now_utc() -> dt.datetime:
    return _now()


class VideoChatTurn(Base):
    __tablename__ = "video_chat_turns"

    turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("video_chat_turns.turn_id"))
    source_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_assets.source_asset_id"))
    search_task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("search_tasks.task_id"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    request: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="QUEUED", index=True)
    snapshot: Mapped[dict | None] = mapped_column(JSONB)
    reply: Mapped[dict | None] = mapped_column(JSONB)
    provider: Mapped[str | None] = mapped_column(String(24))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
