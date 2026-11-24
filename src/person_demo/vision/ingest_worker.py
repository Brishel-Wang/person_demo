"""D2/D3 入库流水线 worker（单 GPU 串行视觉路径）。

流程：窗口帧采样 → YOLO person → ByteTrack → 30s 分块（continuation）→
代表图 → OSNet 512 维特征 → 原型(mean_l2_v1) → 不可变产物写入 → SQL 提交（含 outbox）。
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from person_demo.config import get_settings
from person_demo.db import new_session
from person_demo.db.models import (
    CoverageSegment,
    IngestRun,
    Observation,
    SourceAsset,
    Tracklet,
    VectorOutbox,
)
from person_demo.errors import ModelUnavailable
from person_demo.feature_space import AGGREGATION_VERSION, embedding_id
from person_demo.media.reader import crop_person, encode_jpg, iter_window_frames
from person_demo.media.probe import ProbeResult
from person_demo.storage import ArtifactBundle, StorageLayout, file_sha256
from person_demo.vision.representatives import CropCandidate, select_representatives
from person_demo.vision.tracker import ByteTrack

log = logging.getLogger(__name__)

# 疑似混人检查：中心跳变 > 0.6*最大边长 且 面积比 >2.5 时标记可疑
_CENTER_JUMP_RATIO = 0.6
_AREA_JUMP_RATIO = 2.5
DENSE_PROFILE = "dense_appearance_v2"


def tracker_config(settings, vision_profile: str) -> dict:
    dense = vision_profile == DENSE_PROFILE
    return {
        "track_buffer": settings.tracker_lost_buffer_frames,
        "match_thresh_high": 0.45 if dense else settings.tracker_match_high,
        "match_thresh_low": 0.40 if dense else settings.tracker_match_low,
        "min_hits": 1 if dense else settings.tracker_min_hits,
        "include_unmatched": dense,
    }


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass
class TrackObservation:
    local_us: int
    frame_pts: int
    bbox: tuple[int, int, int, int]
    frame_w: int
    frame_h: int
    crop_bgr: np.ndarray | None
    crop_status: str
    crop_reason: str | None
    embedding: np.ndarray | None = None


@dataclass
class TrackChunk:
    local_track_id: int
    chunk_no: int
    observations: list[TrackObservation] = field(default_factory=list)
    suspicious: bool = False
    suspicious_reason: str | None = None


_model_cache: dict = {}


def build_encoder(settings=None, _use_cache=True):
    from person_demo.vision.reid.encoder import OSNetEncoder

    s = settings or get_settings()
    ckpt = s.model_dir / s.osnet_checkpoint
    key = ("osnet", str(ckpt), s.device)
    if _use_cache and key in _model_cache:
        return _model_cache[key]
    enc = OSNetEncoder(ckpt, device=s.device)
    if _use_cache:
        _model_cache[key] = enc
    return enc


def build_detector(settings=None, _use_cache=True, *, run: IngestRun | None = None):
    from person_demo.vision.detector import PersonDetector, recorded_detector_weights

    s = settings or get_settings()
    weights, imgsz = s.yolo_weights, s.yolo_imgsz
    expected_sha = None
    if run is not None:
        expected_sha = run.detector_sha256
        weights = recorded_detector_weights(run.tracker_config, expected_sha) or weights
        imgsz = run.detector_imgsz
    path = s.model_dir / weights
    key = ("yolo", str(path), imgsz, expected_sha, s.device)
    if _use_cache and key in _model_cache:
        return _model_cache[key]
    if expected_sha and file_sha256(path) != expected_sha:
        raise ValueError(f"detector checkpoint differs from the registered SHA-256: {path.name}")
    det = PersonDetector(path, imgsz=imgsz, device=s.device)
    if _use_cache:
        _model_cache[key] = det
    return det


def process_ingest_run(
    ingest_run_id: uuid.UUID,
    *,
    storage: StorageLayout,
    detector=None,
    encoder=None,
    probe: ProbeResult | None = None,
    staged_storage: StorageLayout | None = None,
) -> dict:
    """执行一个 ingest_run 的完整视觉处理与提交；幂等：同配置 run 已存在时复用。"""
    s = get_settings()
    session: Session = new_session()
    lock_connection = None
    got_lock = False
    bundle = None
    try:
        # Holding this session lock across commits lets a killed worker be retried as a fresh tracker epoch.
        engine = session.get_bind()
        lock_connection = engine.connect()
        got_lock = lock_connection.scalar(
            select(func.pg_try_advisory_lock(func.hashtextextended(str(ingest_run_id), 1))))
        if not got_lock:
            return {"status": "analyzing", "ingest_run_id": str(ingest_run_id), "note": "already_running"}
        run = session.get(IngestRun, ingest_run_id)
        if run is None:
            raise ValueError(f"ingest run {ingest_run_id} not found")
        if run.status in ("ready", "publishing"):
            return {"status": run.status, "ingest_run_id": str(ingest_run_id), "note": "reused"}
        asset = session.get(SourceAsset, run.source_asset_id)
        if asset is None:
            raise ValueError("source asset not found")

        if probe is None:
            from person_demo.media.probe import probe_file

            probe = probe_file(asset.local_path, (run.window_start_us, run.window_end_us))
        if (probe.timebase_status != "ok" or probe.sha256 != asset.sha256
                or probe.resolved_window_us != (run.window_start_us, run.window_end_us)):
            run.status = "failed"
            run.error = f"timebase invalid: {probe.error}"
            session.commit()
            return {"status": "failed", "error": run.error}

        dense = run.vision_profile == DENSE_PROFILE
        artifacts = staged_storage or storage
        run.error = None
        run.status = "indexing_dense" if dense else "analyzing"
        run.started_at = run.started_at or _now()
        session.commit()

        model_start = perf_counter()
        if detector is None:
            try:
                detector = build_detector(s, _use_cache=False, run=run)
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailable(f"detector load failed: {e}")
        if encoder is None:
            try:
                encoder = build_encoder(s, _use_cache=False)
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailable(f"osnet load failed: {e}")

        model_load_s = round(perf_counter() - model_start, 3)
        tracker = ByteTrack(**tracker_config(s, run.vision_profile))
        if dense:
            log.info("dense vision models loaded in %.3fs", model_load_s)

        # ---- 逐帧采样、检测、跟踪，聚合 chunk（30 秒）----
        chunk_boundary_us = s.tracklet_max_duration_s * 1_000_000
        lost_gap_us = (s.tracker_lost_buffer_frames + 3) * (1_000_000 // run.analysis_fps)
        chunks: list[TrackChunk] = []
        active: dict[int, TrackChunk] = {}
        group_ids: dict[int, uuid.UUID] = {}
        chunk_counters: dict[int, int] = {}  # 每 track_id 的下一个 chunk_no（失联重现→continuation）
        last_seen_frame: dict[int, int] = {}
        frames_done = 0
        detections_seen = 0

        samples = iter_window_frames(
            asset.local_path, probe, (run.window_start_us, run.window_end_us), run.analysis_fps
        )
        for batch in iter(lambda: list(islice(samples, 16)), []):
            previous_frames = frames_done
            crops_to_encode = []
            dets_all = detector.detect_persons([x.frame_bgr for x in batch])
            for sample, dets in zip(batch, dets_all, strict=True):
                frames_done += 1
                detections_seen += len(dets)
                tracked = tracker.update(dets)
                if dense and len(tracked) != len(dets):
                    raise ValueError(f"observation count differs from detector output at {sample.local_us}")
                now_us = sample.local_us
                tracked_ids = set()
                for tr in tracked:
                    tid = tr["track_id"]
                    tracked_ids.add(tid)
                    bbox = tuple(float(v) for v in tr["bbox"])
                    raw_box = tuple(int(v) for v in bbox)
                    crop = crop_person(sample.frame_bgr, raw_box)
                    if crop is not None:
                        crop = crop.copy()
                    obs = TrackObservation(
                        local_us=now_us, frame_pts=sample.pts,
                        bbox=tuple(int(v) for v in bbox),
                        frame_w=sample.width, frame_h=sample.height,
                        crop_bgr=crop,
                        crop_status="valid" if crop is not None else "empty_box",
                        crop_reason=None if crop is not None else "empty or degenerate box",
                    )
                    if dense and crop is not None:
                        crops_to_encode.append(obs)
                    chunk = active.get(tid)
                    if chunk is None:
                        # 首次出现或失联后重现：chunk_no 递增（同一 group → continuation）
                        chunk_no = chunk_counters.get(tid, 0)
                        chunk = TrackChunk(local_track_id=tid, chunk_no=chunk_no)
                        chunk_counters[tid] = chunk_no
                        active[tid] = chunk
                        if tid not in group_ids:
                            group_ids[tid] = uuid.uuid4()
                    # 30 秒分块边界 → 封闭当前 chunk，开 continuation chunk
                    if (
                        chunk.observations
                        and now_us - chunk.observations[0].local_us >= chunk_boundary_us
                    ):
                        chunks.append(chunk)
                        new_no = chunk_counters[tid] + 1
                        chunk_counters[tid] = new_no
                        new_chunk = TrackChunk(local_track_id=tid, chunk_no=new_no)
                        active[tid] = new_chunk
                        chunk = new_chunk
                    _check_suspicious(chunk, obs)
                    chunk.observations.append(obs)
                    last_seen_frame[tid] = now_us
                # 失联超过缓冲 → 明确的轨迹边界，封闭 chunk 并递增 chunk_no 计数
                for tid in list(active.keys()):
                    if tid not in tracked_ids:
                        last = last_seen_frame.get(tid, 0)
                        if now_us - last > lost_gap_us:
                            chunks.append(active.pop(tid))
                            chunk_counters[tid] = chunk_counters.get(tid, 0) + 1

            if dense and crops_to_encode:
                vectors = encoder.encode_batch([o.crop_bgr for o in crops_to_encode], batch_size=96)
                for obs, vector in zip(crops_to_encode, vectors, strict=True):
                    obs.embedding = vector
            if dense and frames_done // 1000 != previous_frames // 1000:
                log.info("dense ingest: %d frames, position %.2fs, elapsed %.1fs",
                         frames_done, batch[-1].local_us / 1e6, perf_counter() - model_start)

        if frames_done == 0:
            raise ValueError("no frames decoded for the requested window")
        for c in active.values():
            chunks.append(c)
        chunks.sort(key=lambda c: (c.local_track_id, c.chunk_no))

        log.info("decoded %d frames, staging %d chunks", frames_done, len(chunks))

        # ---- 写入（单 SQL 事务）：tracklet + observations + 代表图 + 向量 + outbox ----
        publish_seq = 0  # 本 run 内按 scope 递增
        prev_chunk_ids: dict[int, uuid.UUID] = {}
        frames_tracked = 0
        if dense and staged_storage is not None:
            bundle = ArtifactBundle(artifacts, str(run.ingest_run_id))
        artifact_writer = bundle or artifacts
        key_for = bundle.key if bundle else _plain_key

        for c in chunks:
            if bundle and publish_seq % 1024 == 0 and shutil.disk_usage(artifacts.root).free < 5 * 2**30:
                raise OSError("artifact staging reached the local disk reserve")
            if not c.observations:
                continue
            t_start = min(o.local_us for o in c.observations)
            t_end = max(o.local_us for o in c.observations) + 1  # 左闭右开
            candidates = [
                CropCandidate(
                    local_us=o.local_us, frame_pts=o.frame_pts, bbox=o.bbox,
                    frame_w=o.frame_w, frame_h=o.frame_h, crop=o.crop_bgr,
                    status=o.crop_status, reason=o.crop_reason,
                )
                for o in c.observations
            ]
            rep = select_representatives(candidates, t_start, t_end,
                                         max_count=s.representative_frames_max,
                                         min_gap_us=int(s.representative_min_gap_s * 1e6))
            if not rep.selected:
                log.warning("tracklet %d#%d no valid representative crops; skipped",
                            c.local_track_id, c.chunk_no)
                continue

            tracklet = Tracklet(
                ingest_run_id=run.ingest_run_id,
                dataset_id=run.dataset_id,
                camera_code=asset.camera.camera_code if asset.camera else s.camera_code,
                source_asset_id=asset.source_asset_id,
                local_track_id=c.local_track_id,
                chunk_no=c.chunk_no,
                t_start_local_us=t_start,
                t_end_local_us=t_end,
                feature_space_id=run.feature_space_id,
                aggregation_version=AGGREGATION_VERSION,
                suspicious=c.suspicious,
                suspicious_reason=c.suspicious_reason,
                status="committed",
            )
            # continuation：同组所有 chunk 都带 group_id；chunk_no>0 带 continuation_of
            tracklet.continuation_group_id = group_ids[c.local_track_id]
            if c.chunk_no > 0:
                tracklet.continuation_of = prev_chunk_ids.get(c.local_track_id)
            session.add(tracklet)
            session.flush()  # 取得 tracklet_revision_id
            prev_chunk_ids[c.local_track_id] = tracklet.tracklet_revision_id

            # 代表图产物 + 向量（先临时文件，核验后落位不可变 storage_key）
            crops_bgr = [x.crop for x in rep.selected]
            try:
                if dense:
                    vectors_by_time = {o.local_us: o.embedding for o in c.observations if o.embedding is not None}
                    vecs = np.stack([vectors_by_time[x.local_us] for x in rep.selected])
                    dense_key = key_for(artifacts.frame_vectors_key(str(tracklet.tracklet_revision_id)))
                    _save_array(artifact_writer, dense_key, np.stack(list(vectors_by_time.values())))
                else:
                    vecs = encoder.encode_batch(crops_bgr)
            except Exception as e:  # noqa: BLE001
                session.rollback()
                raise ModelUnavailable(f"osnet encode failed: {e}")
            if vecs.shape != (len(crops_bgr), 512):
                session.rollback()
                raise ValueError(f"unexpected embedding shape {vecs.shape}")

            vec_key = key_for(artifacts.vectors_key(str(tracklet.tracklet_revision_id)))
            idx_key = key_for(artifacts.row_index_key(str(tracklet.tracklet_revision_id)))
            _save_array(artifact_writer, vec_key, vecs.astype(np.float32))

            row_index = []
            for n, (sel, vec) in enumerate(zip(rep.selected, vecs)):
                crop_key = key_for(artifacts.crop_key(str(tracklet.tracklet_revision_id), n))
                jpg = encode_jpg(sel.crop)
                artifact_writer.write_bytes(crop_key, jpg)
                row_index.append({
                    "tracklet_revision_id": str(tracklet.tracklet_revision_id),
                    "order": n,
                    "local_us": sel.local_us,
                    "frame_pts": sel.frame_pts,
                    "bbox": list(sel.bbox),
                    "frame_w": sel.frame_w,
                    "frame_h": sel.frame_h,
                    "crop_key": crop_key,
                    "laplacian_var": sel.laplacian_var,
                    "l2_norm": 1.0,
                })
            index_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in row_index)
            artifact_writer.write_bytes(idx_key, index_text.encode("utf-8"))

            proto = vecs.mean(axis=0)
            pnorm = float(np.linalg.norm(proto))
            if not np.isfinite(proto).all() or pnorm <= 1e-12:
                session.rollback()
                raise ValueError("invalid prototype (non-finite or zero norm)")
            proto = proto / pnorm

            tracklet.vector_key = vec_key
            tracklet.row_index_key = idx_key
            tracklet.sample_count = len(rep.selected)
            tracklet.quality = rep.quality_summary
            tracklet.representative_local_us = rep.selected[0].local_us

            # observations（全部；代表图标记与 crop_key 一并记录）
            sel_by_us = {x.local_us: x for x in rep.selected}
            rep_crop_keys = {r["local_us"]: r["crop_key"] for r in row_index}
            for o in c.observations:
                is_rep = o.local_us in rep_crop_keys
                session.add(Observation(
                    tracklet_revision_id=tracklet.tracklet_revision_id,
                    source_asset_id=asset.source_asset_id,
                    local_us=o.local_us,
                    frame_pts=o.frame_pts,
                    time_base_num=probe.time_base_num,
                    time_base_den=probe.time_base_den,
                    bbox_x1=o.bbox[0], bbox_y1=o.bbox[1], bbox_x2=o.bbox[2], bbox_y2=o.bbox[3],
                    frame_w=o.frame_w, frame_h=o.frame_h,
                    crop_key=rep_crop_keys.get(o.local_us),
                    crop_status=o.crop_status,
                    laplacian_var=(sel_by_us[o.local_us].laplacian_var if is_rep else None),
                    is_representative=is_rep,
                ))
                frames_tracked += 1

            eid = embedding_id(
                dataset_id=run.dataset_id,
                ingest_run_id=str(run.ingest_run_id),
                tracklet_revision_id=str(tracklet.tracklet_revision_id),
                space_id=run.feature_space_id,
            )
            session.add(VectorOutbox(
                dataset_id=run.dataset_id,
                source_asset_id=asset.source_asset_id,
                ingest_run_id=run.ingest_run_id,
                tracklet_revision_id=tracklet.tracklet_revision_id,
                feature_space_id=run.feature_space_id,
                publish_seq=publish_seq,
                status="dense_pending" if dense else "pending",
            ))
            publish_seq += 1

        if bundle:
            bundle.finish()
            bundle.close()
            bundle = None
        if staged_storage is not None:
            session.flush()
            transfer_started = perf_counter()
            copied = _copy_staged_artifacts(staged_storage, storage)
            log.info("copied %d staged artifacts in %.1fs", copied, perf_counter() - transfer_started)

        session.add(CoverageSegment(
            dataset_id=run.dataset_id, source_asset_id=asset.source_asset_id,
            ingest_run_id=run.ingest_run_id,
            start_us=run.window_start_us, end_us=run.window_end_us, status="analyzed",
        ))
        run.status = "publishing" if publish_seq else "ready"
        run.finished_at = _now()
        session.commit()

        return {
            "status": run.status,
            "frames_done": frames_done,
            "vision_model_load_s": model_load_s,
            "elapsed_after_probe_s": round(perf_counter() - model_start, 3),
            "tracklets": publish_seq,
            "observations": frames_tracked,
            "detector_boxes": detections_seen,
            "ingest_run_id": str(run.ingest_run_id),
        }
    except Exception as exc:
        session.rollback()
        if got_lock:
            run = session.get(IngestRun, ingest_run_id)
            if run is not None and run.status not in ("publishing", "ready"):
                run.status = "failed"
                run.error = str(exc)
                session.commit()
        raise
    finally:
        if bundle:
            bundle.close()
        session.close()
        if lock_connection is not None:
            try:
                if got_lock:
                    lock_connection.scalar(
                        select(func.pg_advisory_unlock(func.hashtextextended(str(ingest_run_id), 1))))
            finally:
                lock_connection.close()


def _plain_key(key: str) -> str:
    return key


def _save_array(writer: ArtifactBundle | StorageLayout, key: str, values: np.ndarray) -> None:
    with io.BytesIO() as data:
        np.save(data, values, allow_pickle=False)
        writer.write_bytes(key, data.getvalue())


def _copy_staged_artifacts(staged: StorageLayout, destination: StorageLayout) -> int:
    paths = [
        path for folder in (staged.crops, staged.vectors, staged.frame_vectors, staged.row_indexes, staged.bundles)
        for path in folder.rglob("*") if path.is_file()
    ]

    def copy(path: Path) -> None:
        target = destination.root / path.relative_to(staged.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".staging")
        shutil.copyfile(path, temporary)
        temporary.replace(target)

    # WAL records become visible only when every immutable evidence file has landed.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(copy, paths):
            pass
    return len(paths)


def _check_suspicious(chunk: TrackChunk, obs: TrackObservation) -> None:
    if not chunk.observations:
        return
    prev = chunk.observations[-1]
    px1, py1, px2, py2 = prev.bbox
    x1, y1, x2, y2 = obs.bbox
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    diag = max(obs.frame_w, obs.frame_h)
    area_prev = max((px2 - px1) * (py2 - py1), 1)
    area_now = max((x2 - x1) * (y2 - y1), 1)
    ratio = max(area_now / area_prev, area_prev / area_now)
    jump = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
    if jump > _CENTER_JUMP_RATIO * diag and ratio > _AREA_JUMP_RATIO:
        chunk.suspicious = True
        chunk.suspicious_reason = f"possible identity switch: center jump {jump:.0f}px, area ratio {ratio:.1f}"
