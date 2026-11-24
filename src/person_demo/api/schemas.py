"""API 请求/响应模型（OpenAPI 锁定）。"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = None
    request_id: str | None = None


# ---------------- 素材 ----------------
class AssetWindowIn(BaseModel):
    source_asset_id: uuid.UUID
    t_start_local_us: int
    t_end_local_us: int


class SearchScopeIn(BaseModel):
    asset_windows: list[AssetWindowIn]


class ManifestCameraIn(BaseModel):
    camera_code: str
    filename: str
    local_path: str
    source_kind: str = "file"
    sha256: str | None = None
    metadata_expected_bytes: int | None = None


class ImportManifestIn(BaseModel):
    dataset_id: str = "single_camera_demo_v1"
    dataset_name: str | None = None
    time_axis: str = "asset_relative"
    requested_window_us: dict | None = None
    cameras: list[ManifestCameraIn]


class ImportAccepted(BaseModel):
    import_job_id: uuid.UUID
    status: str
    source_asset_id: uuid.UUID | None = None
    note: str | None = None


class DatasetStatusOut(BaseModel):
    dataset_id: str
    status: str
    camera_code: str | None = None
    source_asset_id: uuid.UUID | None = None
    resolved_window_us: list[int] | None = None
    ingest_run_id: uuid.UUID | None = None
    ingest_status: str | None = None
    watermark: int | None = None
    coverage: dict | None = None
    tracklet_count: int = 0
    observation_count: int = 0
    detail: dict | None = None


# ---------------- 查询资产 ----------------
class QueryFromFrameIn(BaseModel):
    source_asset_id: uuid.UUID
    local_us: int | None = None
    observation_id: uuid.UUID | None = None
    target_bbox: list[int] | None = None  # [x1,y1,x2,y2]


class QueryAssetOut(BaseModel):
    asset_id: uuid.UUID
    kind: str
    origin_status: str
    origin_observation_id: uuid.UUID | None = None
    origin_tracklet_revision_id: uuid.UUID | None = None
    origin_continuation_group_id: uuid.UUID | None = None
    origin_camera_code: str | None = None
    source_asset_id: uuid.UUID | None = None
    local_us: int | None = None
    bbox: list[int] | None = None
    feature_space_id: str | None = None
    image_url: str | None = None


class SearchTaskIn(BaseModel):
    dataset_id: str = "single_camera_demo_v1"
    time_axis: str = "asset_relative"
    mode: str = "body_only"
    query_asset_id: uuid.UUID
    scope: SearchScopeIn
    evaluation_scope: str = "intra_camera_demo"
    coverage_policy: str = "wait"
    retrieval_profile: str = "single_body_mean_l2_v1"


class SearchTaskAccepted(BaseModel):
    task_id: uuid.UUID
    status: str


class CandidateOut(BaseModel):
    hit_id: uuid.UUID
    camera_id: uuid.UUID | None = None
    camera_code: str | None = None
    source_asset_id: uuid.UUID | None = None
    tracklet_revision_id: uuid.UUID
    t_start_local_us: int
    t_end_local_us: int
    prototype_similarity: float
    rerank_similarity: float | None = None
    rerank_status: str | None = None
    sample_count: int
    same_continuation_as_query: bool
    representative_images: list[dict] = []
    evidence_id: uuid.UUID
    matched_buckets: list[int] | None = None
    review_status: str
    review_revision: int
    rank_position: int


class SearchResultsOut(BaseModel):
    task_id: uuid.UUID
    status: str
    total: int
    page: int
    page_size: int
    items: list[CandidateOut]
    snapshot: dict | None = None
    report_job_id: uuid.UUID | None = None
    report_status: str | None = None


class ReviewIn(BaseModel):
    hit_id: uuid.UUID
    action: str  # confirm / reject
    expected_revision: int


class ReviewOut(BaseModel):
    hit_id: uuid.UUID
    action: str
    revision: int
    status: str


class ReportRequestIn(BaseModel):
    expected_search_revision: int = 1
    report_profile: str = "qwen_single_report_local_v1"
    force_new_version: bool = False


class ReportJobOut(BaseModel):
    report_job_id: uuid.UUID
    status: str
    provider: str | None = None
    error: str | None = None


class ReportOut(BaseModel):
    report_id: uuid.UUID
    search_task_id: uuid.UUID
    search_revision: int
    provider: str
    status: str
    content: dict
    validation: dict | None = None


class MetadataRevisionIn(BaseModel):
    expected_revision: int
    payload: dict
