"""publisher：SQL outbox → Milvus upsert → Strong 校验 → SQL ready → 连续水位（spec 4.3）。

- 以稳定主键完整 upsert；写入超时可已成功，按同 ID 重试（幂等）
- Strong 读取校验主键、空间与发布数据后，SQL 再标 ready
- 发布水位仅前进到连续 ready 序号，不跨空洞
"""
from __future__ import annotations

import datetime as dt
from io import BytesIO
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from person_demo.config import get_settings
from person_demo.db import new_session
from person_demo.db.models import IngestRun, Observation, Tombstone, Tracklet, VectorOutbox
from person_demo.db.repo import advance_watermark, claim_outbox_batch
from person_demo.storage import StorageLayout, bundle_id, related_key
from person_demo.vectors import MilvusStore
from person_demo.vectors.observations import publish_frames
from person_demo.vision.ingest_worker import DENSE_PROFILE

log = logging.getLogger(__name__)


def load_tracklet_payload(tracklet: Tracklet, vecs: np.ndarray) -> dict:
    if vecs.dtype != np.float32 or vecs.shape != (tracklet.sample_count, 512) or not 1 <= len(vecs) <= 5:
        raise ValueError(f"bad representative vectors shape {vecs.shape} for {tracklet.tracklet_revision_id}")
    if not np.isfinite(vecs).all() or not np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-5):
        raise ValueError(f"non-finite representative vectors for {tracklet.tracklet_revision_id}")
    proto = vecs.mean(axis=0).astype(np.float32)  # mean_l2_v1：等权平均后 L2 归一
    pnorm = float(np.linalg.norm(proto))
    if pnorm <= 1e-12:
        raise ValueError(f"zero-norm prototype for {tracklet.tracklet_revision_id}")
    proto = proto / pnorm
    return {
        "embedding_id": _embedding_id_from(tracklet),
        "dataset_id": tracklet.dataset_id,
        "camera_code": tracklet.camera_code,
        "source_asset_id": str(tracklet.source_asset_id),
        "ingest_run_id": str(tracklet.ingest_run_id),
        "tracklet_revision_id": str(tracklet.tracklet_revision_id),
        "feature_space_id": tracklet.feature_space_id,
        "publish_seq": -1,  # 由调用方填
        "t_start_local_us": tracklet.t_start_local_us,
        "t_end_local_us": tracklet.t_end_local_us,
        "representative_local_us": tracklet.representative_local_us or tracklet.t_start_local_us,
        "quality": tracklet.quality if tracklet.quality is not None else 0.0,
        "sample_count": tracklet.sample_count,
        "embedding": proto.tolist(),
    }


def _embedding_id_from(tracklet: Tracklet) -> str:
    from person_demo.feature_space import embedding_id

    return embedding_id(
        dataset_id=tracklet.dataset_id,
        ingest_run_id=str(tracklet.ingest_run_id),
        tracklet_revision_id=str(tracklet.tracklet_revision_id),
        space_id=tracklet.feature_space_id,
    )


def publish_outbox_batch(
    store: MilvusStore,
    rows: list[VectorOutbox],
    session: Session,
) -> dict:
    """发布一批 outbox 行并推进水位。返回统计。"""
    if not rows:
        return {"published": 0, "ready": 0, "watermark": None}
    scope = (rows[0].dataset_id, rows[0].source_asset_id, rows[0].ingest_run_id, rows[0].feature_space_id)
    publication = []
    dense_ids = [row.tracklet_revision_id for row in rows if row.status == "dense_pending"]
    observations = {}
    if dense_ids:
        for obs in session.scalars(select(Observation).where(
            Observation.tracklet_revision_id.in_(dense_ids), Observation.crop_status == "valid"
        ).order_by(Observation.tracklet_revision_id, Observation.local_us)):
            observations.setdefault(obs.tracklet_revision_id, []).append(obs)
    frames_by_tracklet = {}
    storage = StorageLayout(get_settings().storage_root)
    track_ids = [row.tracklet_revision_id for row in rows]
    tracklets = {t.tracklet_revision_id: t for t in session.scalars(
        select(Tracklet).where(Tracklet.tracklet_revision_id.in_(track_ids))
    )}
    tombstoned = set(session.scalars(select(Tombstone.entity_id).where(
        Tombstone.entity_type == "tracklet", Tombstone.entity_id.in_(track_ids),
    )))
    eligible = []
    files = []
    for row in rows:
        rid = row.tracklet_revision_id
        tracklet = tracklets.get(rid)
        if tracklet is None or tracklet.status == "tombstoned" or rid in tombstoned:
            row.status = "skipped"
            row.last_error = "tracklet missing or tombstoned"
            continue
        eligible.append((row, tracklet))
        frame_key = (related_key(tracklet.vector_key, storage.frame_vectors_key(str(rid)))
                     if row.status == "dense_pending" else None)
        files.append((tracklet.vector_key, frame_key))

    by_bundle: dict[str, list[str]] = {}
    for pair in files:
        for key in pair:
            if key is None:
                continue
            group = bundle_id(key)
            if group is not None:
                by_bundle.setdefault(group, []).append(key)
    packed = {}
    for group, keys in by_bundle.items():
        packed.update(storage.read_bundle(group, keys))

    def read_array(key: str) -> np.ndarray:
        if key in packed:
            return np.load(BytesIO(packed[key]), allow_pickle=False)
        return np.load(storage.resolve(key), allow_pickle=False)

    def read_vectors(keys: tuple[str, str | None]) -> tuple[np.ndarray, np.ndarray | None]:
        representatives, frames = keys
        return read_array(representatives), read_array(frames) if frames is not None else None

    if dense_ids and not packed:
        # Shared storage reads can overlap. Keep SQL and row mutation on this thread.
        with ThreadPoolExecutor(max_workers=16) as pool:
            loaded = list(pool.map(read_vectors, files))
    else:
        loaded = list(map(read_vectors, files))

    for (row, tracklet), (representatives, vectors) in zip(eligible, loaded, strict=True):
        payload = load_tracklet_payload(tracklet, representatives)
        payload["publish_seq"] = row.publish_seq
        publication.append((row, payload))
        if vectors is not None:
            rid = str(tracklet.tracklet_revision_id)
            obs_rows = observations.get(tracklet.tracklet_revision_id, [])
            if (not obs_rows or vectors.dtype != np.float32 or vectors.shape != (len(obs_rows), 512)
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
                raise ValueError(f"invalid observation vectors for {rid}")
            frames_by_tracklet[rid] = [{
                "observation_id": str(obs.observation_id),
                "tracklet_revision_id": rid,
                "source_asset_id": payload["source_asset_id"], "ingest_run_id": payload["ingest_run_id"],
                "feature_space_id": payload["feature_space_id"], "publish_seq": row.publish_seq,
                "local_us": obs.local_us, "embedding": vector.tolist(),
            } for obs, vector in zip(obs_rows, vectors, strict=True)]
        else:
            row.status = "published"
        row.attempts += 1
        row.updated_at = dt.datetime.now(dt.timezone.utc)

    payloads = [payload for _, payload in publication]
    confirmed = 0
    try:
        if payloads:
            store.upsert_rows(payloads)
            frame_ids = publish_frames(store, scope[3], [
                frame for frames in frames_by_tracklet.values() for frame in frames
            ])
            stored = store.get_by_ids(scope[3], [p["embedding_id"] for p in payloads])
            for row, payload in publication:
                got = stored.get(payload["embedding_id"], {})
                frames_ready = all(f["observation_id"] in frame_ids for f in
                                   frames_by_tracklet.get(payload["tracklet_revision_id"], []))
                if frames_ready and all(got.get(k) == payload[k] for k in (
                        "embedding_id", "dataset_id", "source_asset_id", "ingest_run_id",
                        "tracklet_revision_id", "feature_space_id", "publish_seq",
                        "t_start_local_us", "t_end_local_us", "sample_count")):
                    row.status = "ready"
                    row.last_error = None
                    confirmed += 1
                else:
                    row.last_error = "Strong read did not confirm publication"
    except Exception as exc:
        session.commit()
        log.warning("Milvus publication failed and can be retried: %s", exc)
        return {"published": 0, "ready": confirmed, "error": str(exc)}

    session.flush()
    watermark = advance_watermark(session, *scope)
    session.commit()
    result = {"published": len(payloads), "ready": confirmed, "watermark": watermark}
    if confirmed != len(payloads):
        result["error"] = "Strong read did not confirm every publication"
    return result


def publish_loop_once(store: MilvusStore, session: Session, limit: int = 256) -> dict:
    """处理一个 scope 的一批 pending 行（跨 scope 轮询由调用方决定）。"""
    row = session.execute(
        select(VectorOutbox.dataset_id, VectorOutbox.source_asset_id,
               VectorOutbox.ingest_run_id, VectorOutbox.feature_space_id)
        .where(VectorOutbox.status.in_(["pending", "published", "dense_pending"]))
        .group_by(VectorOutbox.dataset_id, VectorOutbox.source_asset_id,
                  VectorOutbox.ingest_run_id, VectorOutbox.feature_space_id)
        .limit(1)
    ).first()
    if row is None:
        return {"scopes": 0}
    scope = tuple(row)
    locked = session.scalar(select(func.pg_try_advisory_xact_lock(
        func.hashtextextended(str(scope[2]), 2))))
    if not locked:
        return {"scopes": 1, "published": 0, "ready": 0, "busy": True}
    run = session.get(IngestRun, scope[2])
    # Verify larger dense batches together while still sending bounded Milvus requests.
    batch_limit = limit * 4 if run and run.vision_profile == DENSE_PROFILE else limit
    rows = claim_outbox_batch(session, scope, batch_limit)
    return {"scopes": 1, **publish_outbox_batch(store, rows, session)}
