"""EvidenceBundle 构建（spec 7.1）：冻结 search revision、元数据快照、查询来源、
候选片内时间、外观分、复核状态、continuation 关系、覆盖与预算缺口。

最多 14 条证据：先人工确认，再按既有重排取待复核项，排除 rejected 并记录数量；
单路仅一个 asset，按片内时间排序展示。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from person_demo.config import get_settings
from person_demo.db.repo import is_tombstoned
from person_demo.errors import EvidenceExpired, RevisionConflict
from person_demo.db.models import (
    EvidenceBundle,
    MetadataRevision,
    SearchCandidate,
    SearchTask,
    SourceAsset,
    Tracklet,
)


def build_bundle(session: Session, task_id: uuid.UUID, search_revision: int | None = None) -> EvidenceBundle:
    s = get_settings()
    task = session.get(SearchTask, task_id)
    if task is None:
        raise ValueError("search task not found")
    rev = search_revision if search_revision is not None else task.search_revision
    if rev != task.search_revision:
        raise RevisionConflict("search revision is no longer available")
    if task.status not in ("COMPLETED", "PARTIAL"):
        raise ValueError("search results are not available for reporting")

    candidates = session.scalars(
        select(SearchCandidate).where(SearchCandidate.search_task_id == task_id)
        .order_by(SearchCandidate.rank_position.asc())
    ).all()

    snapshot = task.snapshot or {}
    asset_id = snapshot.get("source_asset_id")
    asset = session.get(SourceAsset, uuid.UUID(asset_id)) if asset_id else None
    if asset is None or asset.dataset_id != task.dataset_id:
        raise EvidenceExpired("report source asset is unavailable")
    if not Path(asset.local_path).is_file():
        raise EvidenceExpired("report source video is unavailable")

    revisions = session.scalars(select(MetadataRevision).where(
        MetadataRevision.entity_id.in_([asset.source_asset_id, asset.camera_id])
    ).order_by(MetadataRevision.entity_type, MetadataRevision.revision.desc())).all()
    metadata = {}
    for entry in revisions:
        metadata.setdefault(entry.entity_type, entry.revision)

    available = []
    for candidate in candidates:
        tracklet = session.get(Tracklet, candidate.tracklet_revision_id)
        if (tracklet is not None and tracklet.status == "committed"
                and tracklet.source_asset_id == asset.source_asset_id
                and tracklet.ingest_run_id == task.frozen_ingest_run_id
                and not is_tombstoned(session, "tracklet", tracklet.tracklet_revision_id)):
            available.append((candidate, tracklet))
    confirmed = [pair for pair in available if pair[0].review_status == "confirmed"]
    pending = [pair for pair in available if pair[0].review_status == "pending_review"]
    rejected = [pair for pair in available if pair[0].review_status == "rejected"]
    selected = (confirmed + pending)[:s.evidence_max_items]
    truncated = len(confirmed) + len(pending) > s.evidence_max_items

    evidences = []
    for c, t in selected:
        evidences.append({
            "evidence_id": str(c.evidence_id),
            "camera_code": t.camera_code if t else None,
            "source_asset_id": str(t.source_asset_id) if t else None,
            "tracklet_revision_id": str(c.tracklet_revision_id),
            "t_start_local_us": t.t_start_local_us if t else None,
            "t_end_local_us": t.t_end_local_us if t else None,
            "time_axis": "asset_relative",
            "shared_time": None,
            "location": None,
            "recording_time_status": asset.recording_time_status if asset else "unknown",
            "location_status": asset.location_status if asset else "unknown",
            "review_status": c.review_status,
            "prototype_similarity": c.prototype_similarity,
            "rerank_similarity": c.rerank_similarity,
            "same_continuation_as_query": c.same_continuation_as_query,
            "rank_position": c.rank_position,
            "matched_buckets": c.matched_buckets,
        })
    # 展示顺序：片内时间
    evidences.sort(key=lambda e: (e["t_start_local_us"] or 0))

    payload = {
        "schema_version": "1.6.1",
        "data_origin": "real_video_processing_result",
        "task_id": str(task.task_id),
        "search_revision": rev,
        "metadata_snapshot_id": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        "metadata_revisions": metadata,
        "mode": "single_video_summary",
        "capability_status": {
            "single_asset_local_timeline_available": True,
            "absolute_capture_time_verified": False,
            "route_hypotheses_enabled": False,
        },
        "coverage_status": (task.frozen_coverage or {}).get("status", "unknown"),
        "total_candidates": len(available),
        "unavailable_evidence_count": len(candidates) - len(available),
        "coverage_window": (task.frozen_coverage or {}).get("window"),
        "confirmed_count": len(confirmed),
        "rejected_count": len(rejected),
        "included_evidence_count": len(evidences),
        "evidence_truncated": truncated,
        "omitted_count": max(0, (len(confirmed) + len(pending)) - s.evidence_max_items),
        "evidences": evidences,
        "route_candidates": [],
        "dataset_id": task.dataset_id,
        "query_origin": snapshot.get("query_origin") or {},
        "evaluation_scope": task.evaluation_scope,
        "candidate_truncated": task.candidate_truncated,
    }
    evidence_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    existing = session.scalar(
        select(EvidenceBundle).where(
            EvidenceBundle.search_task_id == task_id,
            EvidenceBundle.search_revision == rev,
            EvidenceBundle.evidence_hash == evidence_hash,
        )
    )
    if existing is not None:
        return existing
    bundle = EvidenceBundle(
        search_task_id=task_id, search_revision=rev,
        evidence_hash=evidence_hash, payload=payload,
    )
    session.add(bundle)
    session.commit()
    return bundle
