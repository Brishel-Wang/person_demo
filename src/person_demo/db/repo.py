"""数据库事务辅助：导入幂等、outbox 发布、水位推进、墓碑。"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from person_demo.db.models import (
    IngestRun,
    PublishWatermark,
    SourceAsset,
    Tombstone,
    Tracklet,
    VectorOutbox,
)


def find_asset_by_sha(session: Session, dataset_id: str, sha256: str) -> SourceAsset | None:
    return session.scalar(
        select(SourceAsset).where(SourceAsset.dataset_id == dataset_id, SourceAsset.sha256 == sha256)
    )


def find_ingest_run_same_config(
    session: Session,
    source_asset_id: uuid.UUID,
    window_start_us: int,
    window_end_us: int,
    analysis_fps: int,
    detector_sha256: str | None,
    feature_space_id: str | None,
    vision_profile: str = "legacy",
    *,
    detector_imgsz: int = 640,
) -> IngestRun | None:
    return session.scalar(
        select(IngestRun).where(
            IngestRun.source_asset_id == source_asset_id,
            IngestRun.window_start_us == window_start_us,
            IngestRun.window_end_us == window_end_us,
            IngestRun.analysis_fps == analysis_fps,
            IngestRun.detector_sha256 == detector_sha256,
            IngestRun.feature_space_id == feature_space_id,
            IngestRun.vision_profile == vision_profile,
            IngestRun.detector_imgsz == detector_imgsz,
        )
    )


def claim_outbox_batch(session: Session, scope: tuple[str, uuid.UUID, uuid.UUID, str], limit: int) -> list[VectorOutbox]:
    """领取同 scope 内最早一批 pending 行（发布必须按 scope 串行推进）。"""
    dataset_id, source_asset_id, ingest_run_id, feature_space_id = scope
    rows = session.scalars(
        select(VectorOutbox)
        .where(
            VectorOutbox.dataset_id == dataset_id,
            VectorOutbox.source_asset_id == source_asset_id,
            VectorOutbox.ingest_run_id == ingest_run_id,
            VectorOutbox.feature_space_id == feature_space_id,
            VectorOutbox.status.in_(["pending", "published", "dense_pending"]),
        )
        .order_by(VectorOutbox.publish_seq.asc())
        .limit(limit)
    ).all()
    return list(rows)


def advance_watermark(
    session: Session,
    dataset_id: str,
    source_asset_id: uuid.UUID,
    ingest_run_id: uuid.UUID,
    feature_space_id: str,
) -> int:
    """只前进到连续 ready 序号，不跨空洞（spec 4.3）。"""
    wm = session.get(PublishWatermark, (dataset_id, source_asset_id, ingest_run_id, feature_space_id))
    if wm is None:
        wm = PublishWatermark(
            dataset_id=dataset_id, source_asset_id=source_asset_id,
            ingest_run_id=ingest_run_id, feature_space_id=feature_space_id, watermark_seq=-1,
        )
        session.add(wm)
    contiguous = wm.watermark_seq
    ready = session.scalars(
        select(VectorOutbox.publish_seq)
        .where(
            VectorOutbox.dataset_id == dataset_id,
            VectorOutbox.source_asset_id == source_asset_id,
            VectorOutbox.ingest_run_id == ingest_run_id,
            VectorOutbox.feature_space_id == feature_space_id,
            VectorOutbox.status.in_(["ready", "skipped"]),
            VectorOutbox.publish_seq > contiguous,
        )
        .order_by(VectorOutbox.publish_seq.asc())
    ).all()
    for seq in ready:
        if seq == contiguous + 1:
            contiguous = seq
        elif seq > contiguous + 1:
            break
    wm.watermark_seq = contiguous
    return wm.watermark_seq


def is_tombstoned(session: Session, entity_type: str, entity_id: uuid.UUID) -> bool:
    return (
        session.scalar(
            select(Tombstone.tombstone_id).where(
                Tombstone.entity_type == entity_type, Tombstone.entity_id == entity_id
            )
        )
        is not None
    )


def tombstone_tracklet(session: Session, tracklet: Tracklet, reason: str) -> None:
    tracklet.status = "tombstoned"
    session.add(Tombstone(entity_type="tracklet", entity_id=tracklet.tracklet_revision_id, reason=reason))
