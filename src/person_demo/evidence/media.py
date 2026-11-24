"""证据回放载荷（spec 6.1 GET /v1/evidence/{id}）：授权播放引用、PTS 映射和实际 bbox 序列。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from fractions import Fraction

from sqlalchemy import select
from sqlalchemy.orm import Session

from person_demo.config import get_settings
from person_demo.db.models import (
    AssetProxy,
    Observation,
    SearchCandidate,
    SearchTask,
    SourceAsset,
    Tracklet,
)
from person_demo.db.repo import is_tombstoned
from person_demo.errors import EvidenceExpired, NotFound


def build_evidence_payload(session: Session, evidence_id: uuid.UUID) -> dict:
    candidate = session.scalar(
        select(SearchCandidate).where(SearchCandidate.evidence_id == evidence_id)
    )
    if candidate is None:
        raise NotFound(f"evidence {evidence_id} not found")
    tracklet = session.get(Tracklet, candidate.tracklet_revision_id)
    task = session.get(SearchTask, candidate.search_task_id)
    if (tracklet is None or tracklet.status != "committed" or task is None
            or task.frozen_ingest_run_id != tracklet.ingest_run_id
            or is_tombstoned(session, "tracklet", candidate.tracklet_revision_id)):
        raise EvidenceExpired("evidence tracklet is unavailable for this task")
    asset = session.get(SourceAsset, tracklet.source_asset_id)
    if asset is None or asset.dataset_id != task.dataset_id:
        raise EvidenceExpired("evidence asset is unavailable for this task")
    path = Path(asset.local_path)
    if not path.is_file() or path.stat().st_size != asset.size_bytes:
        raise EvidenceExpired("original video is unavailable or changed")

    observations = session.scalars(
        select(Observation).where(Observation.tracklet_revision_id == tracklet.tracklet_revision_id)
        .order_by(Observation.local_us.asc())
    ).all()

    bbox_sequence = [
        {
            "frame_pts": o.frame_pts,
            "time_base_num": o.time_base_num,
            "time_base_den": o.time_base_den,
            "local_us": o.local_us,
            "bbox": [o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2],
            "frame_w": o.frame_w,
            "frame_h": o.frame_h,
            "is_representative": o.is_representative,
        }
        for o in observations
    ]

    representative_images = []
    for o in observations:
        if o.is_representative and o.crop_key:
            representative_images.append({
                "local_us": o.local_us,
                "storage_key": o.crop_key,
                "bbox": [o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2],
            })

    proxy = session.scalar(
        select(AssetProxy).where(AssetProxy.source_asset_id == tracklet.source_asset_id)
        .order_by(AssetProxy.created_at.desc()).limit(1)
    )
    proxy_info = None
    if (proxy is not None and proxy.window_start_us <= tracklet.t_start_local_us
            and proxy.window_end_us >= tracklet.t_end_local_us):
        mapping = proxy.time_mapping or {}
        frame_idx = None
        best = 1 << 62
        for idx, us in mapping.items():
            d = abs(int(us) - tracklet.t_start_local_us)
            if d < best:
                best, frame_idx = d, int(idx)
        proxy_info = {
            "proxy_storage_key": proxy.storage_key,
            "fps": proxy.fps,
            "window_start_us": proxy.window_start_us,
            "window_end_us": proxy.window_end_us,
            "start_frame_index": frame_idx,
            "time_mapping": mapping,
        }

    return {
        "evidence_id": str(candidate.evidence_id),
        "hit_id": str(candidate.hit_id),
        "search_task_id": str(candidate.search_task_id),
        "camera_code": tracklet.camera_code,
        "source_asset_id": str(tracklet.source_asset_id),
        "tracklet_revision_id": str(tracklet.tracklet_revision_id),
        "local_track_id": tracklet.local_track_id,
        "chunk_no": tracklet.chunk_no,
        "continuation_of": str(tracklet.continuation_of) if tracklet.continuation_of else None,
        "continuation_group_id": str(tracklet.continuation_group_id) if tracklet.continuation_group_id else None,
        "t_start_local_us": tracklet.t_start_local_us,
        "t_end_local_us": tracklet.t_end_local_us,
        "representative_local_us": tracklet.representative_local_us,
        "time_axis": "asset_relative",
        "recording_time_status": "unknown",
        "location_status": "unknown",
        "review_status": candidate.review_status,
        "review_revision": candidate.review_revision,
        "prototype_similarity": candidate.prototype_similarity,
        "rerank_similarity": candidate.rerank_similarity,
        "rerank_status": candidate.rerank_status,
        "same_continuation_as_query": candidate.same_continuation_as_query,
        "matched_buckets": candidate.matched_buckets,
        "rank_position": candidate.rank_position,
        "suspicious": tracklet.suspicious,
        "suspicious_reason": tracklet.suspicious_reason,
        "sample_count": tracklet.sample_count,
        "bbox_sequence": bbox_sequence,
        "representative_images": representative_images,
        "proxy": proxy_info,
        "original_video_url": f"/v1/evidence/{candidate.evidence_id}/video",
        "video_timeline_offset_us": round(
            Fraction(asset.pts_anchor * asset.time_base_num, asset.time_base_den) * 1_000_000
        ),
    }


def read_row_index(session: Session, tracklet: Tracklet) -> list[dict]:
    """读取不可变 JSON 行索引（重排产物可查）。"""
    from person_demo.storage import StorageLayout

    s = get_settings()
    storage = StorageLayout(s.storage_root)
    if tracklet.row_index_key is None:
        return []
    text = storage.read_bytes(tracklet.row_index_key).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
