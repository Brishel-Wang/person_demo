"""单路检索服务（D4）：冻结快照、分桶召回、SQL 回查、来源排除、重排、物化候选。"""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from time import perf_counter

import numpy as np
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm import Session

from person_demo.config import get_settings
from person_demo.db.models import (
    CoverageSegment,
    IngestRun,
    Observation,
    QueryAsset,
    PublishWatermark,
    SearchCandidate,
    SearchTask,
    SourceAsset,
    Tombstone,
    Tracklet,
    VectorOutbox,
)
from person_demo.db.repo import is_tombstoned
from person_demo.errors import (
    CapabilityDisabled,
    EvidenceExpired,
    FeatureSpaceMismatch,
    InvalidTimeRange,
    NotFound,
    OriginUnresolved,
    SingleSourceRequired,
    VectorBackendUnavailable,
)
from person_demo.retrieval.exclude import evaluate_exclusion
from person_demo.retrieval.rerank import rerank_many
from person_demo.storage import StorageLayout
from person_demo.vectors import MilvusStore
from person_demo.vectors.observations import search_frames
from person_demo.vision.ingest_worker import DENSE_PROFILE

DEFAULT_FRAME_SIMILARITY = 0.68

log = logging.getLogger(__name__)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def validate_and_freeze(session: Session, task: SearchTask) -> dict:
    """校验请求并冻结快照。返回快照 dict；失败抛 DemoError。"""
    s = get_settings()
    dataset_id = task.dataset_id

    # 单一路径：该数据集只允许一个 source_asset
    assets = session.scalars(select(SourceAsset).where(SourceAsset.dataset_id == dataset_id)).all()
    if len(assets) != 1:
        raise SingleSourceRequired(f"dataset {dataset_id} must have exactly 1 source_asset, got {len(assets)}")
    asset = assets[0]

    windows = task.scope_windows.get("asset_windows", [])
    if len(windows) != 1:
        raise SingleSourceRequired(f"asset_windows must have exactly 1 window, got {len(windows)}")
    win = windows[0]
    w_start, w_end = int(win["t_start_local_us"]), int(win["t_end_local_us"])
    if w_start < 0 or w_end <= w_start:
        raise InvalidTimeRange(f"invalid window {w_start}:{w_end}")
    resolved_start = asset.resolved_window_start_us
    resolved_end = asset.resolved_window_end_us
    if resolved_start is None or resolved_end is None or w_start < resolved_start or w_end > resolved_end:
        raise InvalidTimeRange(
            f"window [{w_start},{w_end}) outside resolved window [{resolved_start},{resolved_end})"
        )
    if str(win["source_asset_id"]) != str(asset.source_asset_id):
        raise SingleSourceRequired("window source_asset_id does not match the only registered asset")

    if task.mode != "body_only":
        raise CapabilityDisabled(f"mode {task.mode} not enabled; allowed: body_only")
    if task.evaluation_scope not in ("intra_camera_demo", "intra_camera_holdout"):
        raise CapabilityDisabled(f"evaluation_scope {task.evaluation_scope} not enabled")

    # 查询资产与特征空间
    qa = session.get(QueryAsset, task.query_asset_id)
    if qa is None:
        raise NotFound("query asset not found")
    if qa.dataset_id != dataset_id:
        raise NotFound("query asset does not belong to this dataset")
    if not Path(asset.local_path).is_file():
        raise EvidenceExpired("original video is unavailable")

    frame_search = task.scope_windows.get("frame_min_similarity") is not None
    frozen_run = _active_run(session, asset, w_start, w_end)
    if frame_search and (frozen_run is None or frozen_run.vision_profile != DENSE_PROFILE):
        raise VectorBackendUnavailable("该时间范围的逐帧人体索引尚未就绪")
    active_space = frozen_run.feature_space_id if frozen_run else None
    if active_space and qa.feature_space_id != active_space:
        raise FeatureSpaceMismatch(f"query space {qa.feature_space_id} != active {active_space}")
    verified = qa.origin_status == "verified"
    if verified:
        origin = session.get(Tracklet, qa.origin_tracklet_revision_id) if qa.origin_tracklet_revision_id else None
        obs = session.get(Observation, qa.origin_observation_id) if qa.origin_observation_id else None
        if (not origin or not obs or obs.tracklet_revision_id != origin.tracklet_revision_id
                or origin.source_asset_id != asset.source_asset_id or origin.status != "committed"
                or is_tombstoned(session, "tracklet", origin.tracklet_revision_id)
                or (frozen_run and origin.ingest_run_id != frozen_run.ingest_run_id)):
            raise OriginUnresolved("select an observation from the run used for this search window")

    watermark = -1
    if frozen_run is not None and active_space:
        wm = session.get(PublishWatermark,
                         (dataset_id, asset.source_asset_id, frozen_run.ingest_run_id, active_space))
        watermark = wm.watermark_seq if wm is not None else -1
    coverage = _coverage_for(session, dataset_id, asset.source_asset_id, frozen_run, w_start, w_end)
    if frozen_run and frozen_run.status != "ready":
        coverage["status"] = "missing" if watermark < 0 else "partial"
        coverage["publication_status"] = frozen_run.status

    snapshot = {
        "search_revision": task.search_revision,
        "frozen_at": _now().isoformat(),
        "dataset_id": dataset_id,
        "source_asset_id": str(asset.source_asset_id),
        "camera_code": asset.camera.camera_code if asset.camera else None,
        "time_axis": "asset_relative",
        "evaluation_scope": task.evaluation_scope,
        "scope_windows": windows,
        "frozen_ingest_run_id": str(frozen_run.ingest_run_id) if frozen_run else None,
        "frozen_feature_space_id": active_space,
        "frozen_watermark": watermark,
        "appearance_scoring": "max_frame_cosine" if frame_search else "mean_representative_top2_cosine",
        "coverage": coverage,
        "query_origin": {
            "origin_tracklet_revision_id": str(qa.origin_tracklet_revision_id) if qa.origin_tracklet_revision_id else None,
            "origin_continuation_group_id": str(qa.origin_continuation_group_id) if qa.origin_continuation_group_id else None,
            "origin_status": qa.origin_status,
            "origin_observation_id": str(qa.origin_observation_id) if qa.origin_observation_id else None,
            "origin_source_asset_id": str(qa.source_asset_id) if qa.source_asset_id else None,
            "origin_camera_code": qa.origin_camera_code,
            "origin_exclusion_verified": verified and frozen_run is not None,
        },
        "metadata_snapshot": {
            "recording_time_status": asset.recording_time_status,
            "location_status": asset.location_status,
            "asset_sha256": asset.sha256,
            "resolved_window_us": [resolved_start, resolved_end],
        },
        "exclusions_applied": {
            "exclude_source_camera": False,
            "exclude_origin_chunk": verified,
            "exclude_origin_group": verified and task.evaluation_scope == "intra_camera_holdout",
        },
    }
    task.frozen_ingest_run_id = frozen_run.ingest_run_id if frozen_run else None
    task.frozen_feature_space_id = active_space
    task.frozen_watermark = watermark
    task.frozen_coverage = coverage
    task.snapshot = snapshot
    task.status = "VALIDATING"
    session.commit()
    return snapshot


def run_search(session: Session, task_id: uuid.UUID, store: MilvusStore, storage: StorageLayout) -> dict:
    """执行检索：分桶召回 → 回查/排除 → 补取 → 重排 → 物化。"""
    s = get_settings()
    task = session.get(SearchTask, task_id)
    if task is None:
        raise NotFound("search task not found")
    snapshot = task.snapshot or {}
    qa = session.get(QueryAsset, task.query_asset_id)

    frozen_run_id = task.frozen_ingest_run_id
    space_id = task.frozen_feature_space_id
    watermark = task.frozen_watermark if task.frozen_watermark is not None else -1
    coverage = task.frozen_coverage or {}

    if coverage.get("status") == "missing":
        task.status = "WAITING_INDEX"
        task.error = None
        session.commit()
        return {"status": "WAITING_INDEX"}
    if frozen_run_id is None or space_id is None:
        task.status = "WAITING_INDEX"
        session.commit()
        return {"status": "WAITING_INDEX"}

    # 查询向量（L2 归一化，与图库一致）
    query_vec = _load_query_vector(qa, storage)
    if query_vec is None:
        task.status = "FAILED"
        task.error = "query vector artifact missing or corrupt"
        session.commit()
        return {"status": "FAILED", "error": task.error}

    task.status = "RETRIEVING"
    session.commit()

    windows = task.scope_windows["asset_windows"]
    w_start, w_end = int(windows[0]["t_start_local_us"]), int(windows[0]["t_end_local_us"])
    dataset_id = task.dataset_id
    asset_id = str(snapshot["source_asset_id"])
    run_id = str(frozen_run_id)

    if snapshot.get("appearance_scoring") == "max_frame_cosine":
        return _run_frame_search(session, task, store, query_vec, asset_id, w_start, w_end)

    bucket_s = s.bucket_seconds * 1_000_000
    buckets = []
    b = w_start
    while b < w_end:
        buckets.append((b, min(b + bucket_s, w_end)))
        b += bucket_s

    base_expr = (
        f'dataset_id == {json.dumps(dataset_id)} and source_asset_id == "{asset_id}" '
        f'and feature_space_id == "{space_id}" and ingest_run_id == "{run_id}" '
        f"and publish_seq <= {watermark}"
    )

    query_origin = snapshot.get("query_origin", {})
    origin_rid = query_origin.get("origin_tracklet_revision_id")
    origin_gid = query_origin.get("origin_continuation_group_id")
    origin_status = query_origin.get("origin_status", "verified")

    # 分桶召回 + 补取
    kept: dict[str, dict] = {}
    bucket_queues: list[list[str]] = []
    checked: dict[str, dict | None] = {}
    truncated = False
    raw_hits_scanned = 0
    valid_bucket_hits = 0
    origin_revision = uuid.UUID(origin_rid) if origin_rid else None
    origin_group = uuid.UUID(origin_gid) if origin_gid else None
    for bucket_idx, (b_start, b_end) in enumerate(buckets if watermark >= 0 else []):
        time_expr = f"t_start_local_us < {b_end} and t_end_local_us > {b_start}"
        accepted: list[str] = []
        seen: set[str] = set()
        for limit in s.backfill_limit_list:
            try:
                hits = store.search(space_id, query_vec.tolist(), base_expr, limit, time_expr)
            except Exception as e:
                task.status = "FAILED"
                task.error = f"VECTOR_BACKEND_UNAVAILABLE: {e}"
                session.commit()
                return {"status": "FAILED", "error": task.error}
            raw_hits_scanned += len(hits)
            for hit in hits:
                rid = hit["tracklet_revision_id"]
                if rid in seen:
                    continue
                seen.add(rid)
                if rid not in checked:
                    checked[rid] = _sql_recheck(session, hit, frozen_run_id, space_id, watermark)
                recheck = checked[rid]
                if recheck is None:
                    continue
                excl = evaluate_exclusion(
                    evaluation_scope=task.evaluation_scope,
                    origin_tracklet_revision_id=origin_revision,
                    origin_continuation_group_id=origin_group,
                    origin_status=origin_status,
                    candidate_revision_id=uuid.UUID(rid),
                    candidate_group_id=recheck["continuation_group_id"],
                )
                if not excl.keep:
                    continue
                entry = kept.setdefault(rid, {
                    "tracklet_revision_id": rid,
                    "vector_key": recheck.get("vector_key"),
                    "row_index_key": recheck.get("row_index_key"),
                    "prototype_similarity": float(hit["prototype_similarity"]),
                    "sample_count": int(hit.get("sample_count") or 0),
                    "matched_buckets": [],
                    "same_continuation_as_query": excl.same_continuation_as_query,
                })
                entry["matched_buckets"].append(bucket_idx)
                accepted.append(rid)
                if len(accepted) >= s.per_bucket_top_k:
                    break
            if len(accepted) >= s.per_bucket_top_k or len(hits) < limit:
                break
        else:
            truncated = True
        valid_bucket_hits += len(accepted)
        bucket_queues.append(accepted)

    total_recalled = len(kept)
    if total_recalled > s.max_candidate_tracklets:
        from itertools import zip_longest

        reserved = {}
        for round_entries in zip_longest(*bucket_queues):
            for rid in round_entries:
                if rid is not None:
                    reserved[rid] = kept[rid]
                if len(reserved) == s.max_candidate_tracklets:
                    break
            if len(reserved) == s.max_candidate_tracklets:
                break
        kept = reserved
        truncated = True
    kept_list = list(kept.values())

    task.status = "RERANKING"
    session.commit()

    # 重排
    rerank_map = rerank_many(query_vec, storage, kept_list)
    for v in kept_list:
        r = rerank_map[v["tracklet_revision_id"]]
        v["rerank_similarity"] = r["similarity"]
        v["rerank_status"] = r["status"]
        v["rerank_note"] = r["note"]

    def sort_key(v: dict):
        return (
            -(v["rerank_similarity"] if v["rerank_similarity"] is not None else -2.0),
            -v["prototype_similarity"],
            v["tracklet_revision_id"],
        )

    kept_list.sort(key=sort_key)
    for rank, v in enumerate(kept_list):
        v["rank_position"] = rank

    # 物化候选
    for v in kept_list:
        session.add(SearchCandidate(
            search_task_id=task.task_id,
            tracklet_revision_id=uuid.UUID(v["tracklet_revision_id"]),
            prototype_similarity=v["prototype_similarity"],
            rerank_similarity=v["rerank_similarity"],
            rerank_status=v["rerank_status"],
            sample_count=v["sample_count"],
            same_continuation_as_query=v["same_continuation_as_query"],
            matched_buckets=v["matched_buckets"],
            review_status="pending_review",
            review_revision=0,
            rank_position=v["rank_position"],
        ))

    task.recalled_count = total_recalled
    task.kept_count = len(kept_list)
    task.candidate_truncated = truncated
    rerank_failures = sum(v["rerank_status"] != "ok" for v in kept_list)
    task.status = "COMPLETED" if coverage.get("status") == "full" and not truncated and not rerank_failures else "PARTIAL"
    snapshot["result"] = {
        "recalled_count": total_recalled,
        "raw_hits_scanned": raw_hits_scanned,
        "valid_bucket_hits": valid_bucket_hits,
        "kept_count": len(kept_list),
        "candidate_truncated": truncated,
        "buckets": len(buckets),
        "rerank_applied": True,
        "rerank_failed_count": rerank_failures,
    }
    flag_modified(task, "snapshot")  # 原地修改后强制 JSONB 变更落库
    task.updated_at = _now()
    session.commit()
    return {"status": task.status, "kept": len(kept_list), "recalled": total_recalled, "truncated": truncated}


def _run_frame_search(
    session: Session, task: SearchTask, store: MilvusStore, query_vec: np.ndarray,
    asset_id: str, start: int, end: int,
) -> dict:
    """Threshold every frame inside the frozen publication scope and keep one result per chunk."""
    started = perf_counter()
    space_id = task.frozen_feature_space_id
    run_id = task.frozen_ingest_run_id
    watermark = task.frozen_watermark
    snapshot = task.snapshot
    origin = snapshot["query_origin"]
    origin_revision = origin["origin_tracklet_revision_id"]
    origin_group = origin["origin_continuation_group_id"]
    origin_uuid = uuid.UUID(origin_revision) if origin_revision else None
    group_uuid = uuid.UUID(origin_group) if origin_group else None
    floor = float(task.scope_windows["frame_min_similarity"])
    expr = (
        f'source_asset_id == "{asset_id}" and ingest_run_id == "{run_id}" '
        f'and feature_space_id == "{space_id}" and publish_seq <= {watermark}'
    )
    if origin["origin_status"] == "verified" and origin_revision:
        expr += f' and tracklet_revision_id != "{origin_revision}"'

    hits = {}
    count = 0
    if watermark >= 0:
        for row, score in search_frames(store, space_id, query_vec.tolist(), expr, start, end, floor):
            count += 1
            rid = row["tracklet_revision_id"]
            hit = hits.get(rid)
            if hit is None or (score, -row["local_us"]) > (hit[0], -hit[1]["local_us"]):
                hits[rid] = (score, row)
    vector_search_s = round(perf_counter() - started, 3)

    tombstoned = select(Tombstone.tombstone_id).where(
        Tombstone.entity_type == "tracklet", Tombstone.entity_id == Tracklet.tracklet_revision_id,
    ).exists()
    materialized = []
    ordered = sorted(hits.values(), key=lambda hit: (-hit[0], hit[1]["local_us"], hit[1]["tracklet_revision_id"]))
    for start_index in range(0, len(ordered), 1024):
        batch = ordered[start_index : start_index + 1024]
        by_id = {uuid.UUID(row["observation_id"]): (score, row) for score, row in batch}
        records = {
            observation_id: (track, local_us, publish_seq)
            for track, observation_id, local_us, publish_seq in session.execute(
                select(Tracklet, Observation.observation_id, Observation.local_us, VectorOutbox.publish_seq)
                .join(VectorOutbox, VectorOutbox.tracklet_revision_id == Tracklet.tracklet_revision_id)
                .join(Observation, Observation.tracklet_revision_id == Tracklet.tracklet_revision_id)
                .where(
                    Observation.observation_id.in_(by_id), Observation.crop_status == "valid",
                    Observation.source_asset_id == uuid.UUID(asset_id),
                    Tracklet.ingest_run_id == run_id, Tracklet.feature_space_id == space_id,
                    Tracklet.status == "committed", VectorOutbox.status == "ready",
                    VectorOutbox.publish_seq <= watermark, ~tombstoned,
                )
            )
        }
        prototypes = store.get_by_ids(space_id, [
            _prototype_id(track) for track, _, _ in records.values()
        ], include_embedding=True) if records else {}
        for obs_id, (score, row) in by_id.items():
            if obs_id not in records:
                continue
            track, local_us, publish_seq = records[obs_id]
            if (str(track.tracklet_revision_id) != row["tracklet_revision_id"]
                    or local_us != row["local_us"] or publish_seq != row["publish_seq"]):
                raise VectorBackendUnavailable("逐帧向量与索引观测记录不一致")
            exclusion = evaluate_exclusion(
                evaluation_scope=task.evaluation_scope, origin_tracklet_revision_id=origin_uuid,
                origin_continuation_group_id=group_uuid, origin_status=origin["origin_status"],
                candidate_revision_id=track.tracklet_revision_id, candidate_group_id=track.continuation_group_id,
            )
            if not exclusion.keep:
                continue
            proto = prototypes.get(_prototype_id(track), {}).get("embedding")
            if proto is None:
                raise VectorBackendUnavailable("缺少已发布片段的聚合向量")
            materialized.append(SearchCandidate(
                search_task_id=task.task_id, tracklet_revision_id=track.tracklet_revision_id,
                prototype_similarity=float(np.dot(query_vec, np.asarray(proto, dtype=np.float32))),
                rerank_similarity=score, rerank_status="ok", sample_count=track.sample_count,
                matched_local_us=local_us, same_continuation_as_query=exclusion.same_continuation_as_query,
                review_status="pending_review", review_revision=0, rank_position=len(materialized),
            ))
    session.add_all(materialized)
    task.recalled_count = len(hits)
    task.kept_count = len(materialized)
    task.candidate_truncated = False
    task.status = "COMPLETED" if task.frozen_coverage.get("status") == "full" else "PARTIAL"
    snapshot["result"] = {
        "recalled_count": len(hits), "kept_count": len(materialized), "raw_hits_scanned": count,
        "candidate_truncated": False, "min_similarity": floor,
        "vector_search_s": vector_search_s, "elapsed_s": round(perf_counter() - started, 3),
    }
    flag_modified(task, "snapshot")
    task.updated_at = _now()
    session.commit()
    return {"status": task.status, "kept": len(materialized), "recalled": len(hits), "truncated": False}


def _prototype_id(track: Tracklet) -> str:
    from person_demo.feature_space import embedding_id

    return embedding_id(
        dataset_id=track.dataset_id, ingest_run_id=str(track.ingest_run_id),
        tracklet_revision_id=str(track.tracklet_revision_id), space_id=track.feature_space_id,
    )


def _active_run(session: Session, asset: SourceAsset, w_start: int, w_end: int) -> IngestRun | None:
    """Prefer the published frame index, retaining the historical fallback for shorter imports."""
    runs = session.scalars(
        select(IngestRun)
        .where(
            IngestRun.source_asset_id == asset.source_asset_id,
            IngestRun.window_start_us <= w_start,
            IngestRun.window_end_us >= w_end,
        )
        .order_by(IngestRun.created_at.desc())
    ).all()
    frame_runs = [r for r in runs if r.vision_profile == DENSE_PROFILE and r.status == "ready"]
    if frame_runs:
        return max(frame_runs, key=lambda run: run.window_end_us - run.window_start_us)
    runs = [r for r in runs if r.vision_profile != DENSE_PROFILE]
    for status in ("ready", "publishing"):
        available = [run for run in runs if run.status == status]
        if available:
            return min(available, key=lambda run: run.window_end_us - run.window_start_us)
    for run in runs:
        if run.status in ("analyzing", "pending", "committing"):
            return run
    return None


def _coverage_for(session: Session, dataset_id: str, asset_id: uuid.UUID,
                  run: IngestRun | None, w_start: int, w_end: int) -> dict:
    if run is None:
        return {"status": "missing", "segments": []}
    segs = session.scalars(
        select(CoverageSegment).where(
            CoverageSegment.dataset_id == dataset_id,
            CoverageSegment.source_asset_id == asset_id,
            CoverageSegment.ingest_run_id == run.ingest_run_id,
            CoverageSegment.start_us < w_end,
            CoverageSegment.end_us > w_start,
        )
    ).all()
    covered_from = w_start
    for seg in sorted(segs, key=lambda x: x.start_us):
        if seg.start_us <= covered_from:
            covered_from = max(covered_from, seg.end_us)
    status = "full" if covered_from >= w_end else ("partial" if segs else "missing")
    return {
        "status": status,
        "covered_through_us": covered_from,
        "window": [w_start, w_end],
        "segment_count": len(segs),
    }


def _sql_recheck(session: Session, hit: dict, frozen_run_id: uuid.UUID,
                 space_id: str, watermark: int) -> dict | None:
    """SQL 回查：ready、墓碑、活动 run、授权、媒体、实际 observation。"""
    rid = hit["tracklet_revision_id"]
    tracklet = session.get(Tracklet, uuid.UUID(rid))
    if tracklet is None or tracklet.status != "committed":
        return None
    if tracklet.feature_space_id != space_id or tracklet.ingest_run_id != frozen_run_id:
        return None
    if is_tombstoned(session, "tracklet", tracklet.tracklet_revision_id):
        return None
    outbox = session.scalar(
        select(VectorOutbox).where(VectorOutbox.tracklet_revision_id == tracklet.tracklet_revision_id)
    )
    if outbox is None or outbox.status != "ready" or outbox.publish_seq > watermark:
        return None
    has_obs = session.scalar(
        select(func.count(Observation.observation_id)).where(
            Observation.tracklet_revision_id == tracklet.tracklet_revision_id
        )
    )
    if not has_obs:
        return None
    return {"continuation_group_id": tracklet.continuation_group_id, "vector_key": tracklet.vector_key,
            "row_index_key": tracklet.row_index_key}


def _load_query_vector(qa: QueryAsset, storage: StorageLayout) -> np.ndarray | None:
    if qa.embedding_key is None or not storage.exists(qa.embedding_key):
        return None
    try:
        v = np.load(storage.resolve(qa.embedding_key))
    except Exception:  # noqa: BLE001
        return None
    if v.shape != (512,) or not np.isfinite(v).all():
        return None
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return None
    return (v / norm).astype(np.float32)


def paginate_candidates(session: Session, task_id: uuid.UUID, page: int, page_size: int) -> dict:
    total = session.scalar(
        select(func.count(SearchCandidate.hit_id)).where(SearchCandidate.search_task_id == task_id)
    ) or 0
    rows = session.scalars(
        select(SearchCandidate).where(SearchCandidate.search_task_id == task_id)
        .order_by(SearchCandidate.rank_position.asc())
        .offset((page - 1) * page_size).limit(page_size)
    ).all()
    return {"total": total, "page": page, "page_size": page_size, "items": list(rows)}
