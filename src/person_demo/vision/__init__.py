"""D2/D3 入库流水线 worker（单 GPU 串行视觉路径）。

流程：窗口帧采样 → YOLO11s person → ByteTrack → 30s 分块（continuation）→
代表图 → OSNet 512 维特征 → 原型(mean_l2_v1) → 不可变产物写入 → SQL 提交（含 outbox）。
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
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
from person_demo.db.repo import find_ingest_run_same_config
from person_demo.errors import ModelUnavailable
from person_demo.feature_space import AGGREGATION_VERSION, embedding_id
from person_demo.media.reader import crop_person, encode_jpg, iter_window_frames
from person_demo.media.probe import ProbeResult
from person_demo.storage import StorageLayout
from person_demo.vision.representatives import CropCandidate, select_representatives
from person_demo.vision.tracker import ByteTrack

log = logging.getLogger(__name__)

# 疑似混人检查：中心跳变 > 0.6*最大边长 且 面积比 >2.5 时标记可疑
_CENTER_JUMP_RATIO = 0.6
_AREA_JUMP_RATIO = 2.5


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


@dataclass
class TrackChunk:
    local_track_id: int
    chunk_no: int
    observations: list[TrackObservation] = field(default_factory=list)
    suspicious: bool = False
    suspicious_reason: str | None = None


def build_encoder(settings=None):
    from person_demo.vision.reid.encoder import OSNetEncoder

    s = settings or get_settings()
    ckpt = s.model_dir / s.osnet_checkpoint
    return OSNetEncoder(ckpt, device=s.device)


def build_detector(settings=None):
    from person_demo.vision.detector import PersonDetector

    s = settings or get_settings()
    return PersonDetector(s.model_dir / s.yolo_weights, imgsz=s.yolo_imgsz, device=s.device)


def process_ingest_run(
    ingest_run_id: uuid.UUID,
    *,
    storage: StorageLayout,
    detector=None,
    encoder=None,
    probe: ProbeResult | None = None,
) -> dict:
    """执行一个 ingest_run 的完整视觉处理与提交；幂等：同配置 run 已存在时复用。"""
    s = get_settings()
    session: Session = new_session()
    try:
        run = session.get(IngestRun, ingest_run_id)
        if run is None:
            raise ValueError(f"ingest run {ingest_run_id} not found")
        asset = session.get(SourceAsset, run.source_asset_id)
        if asset is None:
            raise ValueError("source asset not found")

        # 幂等：同文件、同范围、同模型与 pipeline 重复导入复用原 run
        existing = find_ingest_run_same_config(
            session, run.source_asset_id, run.window_start_us, run.window_end_us,
            run.analysis_fps, run.detector_sha256, run.feature_space_id,
        )
        if existing is not None and existing.ingest_run_id != run.ingest_run_id:
            run.status = "ready" if existing.status in ("ready", "publishing") else "pending"
            session.commit()
            return {"reused_existing_run": str(existing.ingest_run_id), "status": run.status}

        if probe is None:
            from person_demo.media.probe import probe_file

            probe = probe_file(asset.local_path, (run.window_start_us, run.window_end_us))
        if probe.timebase_status != "ok":
            run.status = "failed"
            run.error = f"timebase invalid: {probe.error}"
            session.commit()
            return {"status": "failed", "error": run.error}

        run.status = "analyzing"
        run.started_at = run.started_at or _now()
        session.commit()

        if detector is None:
            try:
                detector = build_detector(s)
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailable(f"detector load failed: {e}")
        if encoder is None:
            try:
                encoder = build_encoder(s)
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailable(f"osnet load failed: {e}")

        tracker = ByteTrack(
            track_thresh=0.25,
            high_thresh=0.5,
            match_thresh_high=s.tracker_match_high,
            match_thresh_low=s.tracker_match_low,
            track_buffer=s.tracker_lost_buffer_frames,
            min_hits=s.tracker_min_hits,
        )

        # ---- 逐帧采样、检测、跟踪，聚合 chunk（30 秒）----
        frame_batch: list[tuple] = []  # (sample, detections)
        batch_size = 16
        chunks: list[TrackChunk] = []
        active: dict[int, TrackChunk] = {}
        chunk_boundary_us = s.tracklet_max_duration_s * 1_000_000
        group_ids: dict[int, uuid.UUID] = {}
        last_seen_frame: dict[int, int] = {}
        frames_done = 0

        samples = list(iter_window_frames(asset.local_path, probe, (run.window_start_us, run.window_end_us), s.analysis_fps))
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            dets_all = detector.detect_persons([x.frame_bgr for x in batch])
            for sample, dets in zip(batch, dets_all):
                frames_done += 1
                tracked = tracker.update(dets)
                now_us = sample.local_us
                for tr in tracked:
                    tid = tr["track_id"]
                    bbox = tuple(float(v) for v in tr["bbox"])
                    crop = crop_person(sample.frame_bgr, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                    obs = TrackObservation(
                        local_us=now_us, frame_pts=sample.pts,
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        frame_w=sample.width, frame_h=sample.height,
                        crop_bgr=crop,
                        crop_status="valid" if crop is not None else "empty_box",
                        crop_reason=None if crop is not None else "empty or degenerate box",
                    )
                    chunk = active.get(tid)
                    if chunk is None:
                        chunk = TrackChunk(local_track_id=tid, chunk_no=0)
                        active[tid] = chunk
                        group_ids[tid] = uuid.uuid4()
                    # 30 秒分块边界 → 封闭当前 chunk，开 continuation chunk
                    if (
                        chunk.observations
                        and now_us - chunk.observations[0].local_us >= chunk_boundary_us
                    ):
                        chunks.append(chunk)
                        new_chunk = TrackChunk(local_track_id=tid, chunk_no=chunk.chunk_no + 1)
                        active[tid] = new_chunk
                        chunk = new_chunk
                    _check_suspicious(chunk, obs)
                    chunk.observations.append(obs)
                    last_seen_frame[tid] = now_us
                # 目标失联（超出 track_buffer）在 ByteTrack 内标记 lost；
                # 失联超过缓冲后从 active 移除 → 明确的轨迹边界
                for tid in list(active.keys()):
                    if tid not in {t["track_id"] for t in tracked}:
                        last = last_seen_frame.get(tid, 0)
                        if now_us - last > (s.tracker_lost_buffer_frames + 3) * (1_000_000 // s.analysis_fps):
                            chunks.append(active.pop(tid))

        for c in active.values():
            chunks.append(c)
        chunks.sort(key=lambda c: (c.local_track_id, c.chunk_no))

        # ---- 写入：tracklet + observations + 代表图 + 向量 + outbox（一个 SQL 事务）----
        # 代表图与向量产物先写临时目录，核验后落位 storage_key（不可变）
        publish_seq_base = session.scalar(select(VectorOutbox.publish_seq.max()) or 0) or 0
        publish_seq = publish_seq_base + 1

        for c in chunks:
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
                log.warning("tracklet %s:%d has no valid representative crops; skipped", c.local_track_id, c.chunk_no)
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
            # continuation 关联：同 local_track_id 的前一个 chunk
            prev = next((cc for cc in reversed(chunks) if cc.local_track_id == c.local_track_id and cc.chunk_no == c.chunk_no - 1), None)
            if c.chunk_no > 0:
                tracklet.continuation_group_id = group_ids[c.local_track_id]
                # continuation_of 在写入前一个 chunk 的 id 后回填
                tracklet.continuation_of = None  # 后面由 prev_map 回填
            session.add(tracklet)
            session.flush()  # 拿到 tracklet_revision_id

            # 回填 continuation_of：需要前一 chunk 的 revision id
            if c.chunk_no > 0:
                prev_tracklet = session.scalar(
                    select(Tracklet).where(
                        Tracklet.ingest_run_id == run.ingest_run_id,
                        Tracklet.local_track_id == c.local_track_id,
                        Tracklet.chunk_no == c.chunk_no - 1,
                    )
                )
                if prev_tracklet is not None:
                    tracklet.continuation_of = prev_tracklet.tracklet_revision_id

            # 代表图产物 + 向量
            crops_bgr = [x.crop for x in rep.selected]
            try:
                vecs = encoder.encode_batch(crops_bgr)
            except Exception as e:  # noqa: BLE001
                run.status = "failed"
                run.error = f"encode failed: {e}"
                session.rollback()
                session = new_session()
                run = session.get(IngestRun, ingest_run_id)
                run.status = "failed"
                run.error = f"encode failed: {e}"
                session.commit()
                return {"status": "failed", "error": run.error}
            if vecs.shape != (len(crops_bgr), 512):
                raise ValueError(f"unexpected embedding shape {vecs.shape}")

            row_index = []
            vec_key = storage.vectors_key(str(tracklet.tracklet_revision_id))
            idx_key = storage.row_index_key(str(tracklet.tracklet_revision_id))
            # 先写临时再核验
            tmp_vec = storage.tmp / (str(tracklet.tracklet_revision_id) + ".npy.tmp")
            np.save(tmp_vec, vecs.astype(np.float32))
            ok = np.load(tmp_vec).shape == vecs.shape
            if not ok:
                raise ValueError("vector artifact verification failed")
            tmp_vec.replace(storage.resolve(vec_key))

            for n, (sel, vec) in enumerate(zip(rep.selected, vecs)):
                crop_key = storage.crop_key(str(tracklet.tracklet_revision_id), n)
                jpg = encode_jpg(sel.crop)
                storage.write_bytes(crop_key, jpg)
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
            storage.write_text(idx_key, "\n".join(json.dumps(r, ensure_ascii=False) for r in row_index))

            proto = vecs.mean(axis=0)
            pnorm = float(np.linalg.norm(proto))
            if not np.isfinite(proto).all() or pnorm == 0.0:
                raise ValueError("invalid prototype (non-finite or zero norm)")
            proto = proto / pnorm

            tracklet.vector_key = vec_key
            tracklet.row_index_key = idx_key
            tracklet.sample_count = len(rep.selected)
            tracklet.quality = rep.quality_summary
            tracklet.representative_local_us = rep.selected[0].local_us

            # observations（全部，含非代表图）
            for o in c.observations:
                session.add(Observation(
                    tracklet_revision_id=tracklet.tracklet_revision_id,
                    source_asset_id=asset.source_asset_id,
                    local_us=o.local_us,
                    frame_pts=o.frame_pts,
                    time_base_num=probe.time_base_num,
                    time_base_den=probe.time_base_den,
                    bbox_x1=o.bbox[0], bbox_y1=o.bbox[1], bbox_x2=o.bbox[2], bbox_y2=o.bbox[3],
                    frame_w=o.frame_w, frame_h=o.frame_h,
                    crop_key=None, crop_status=o.crop_status,
                    is_representative=False,
                ))
            for sel in rep.selected:
                obs = session.scalar(
                    select(Observation).where(
                        Observation.tracklet_revision_id == tracklet.tracklet_revision_id,
                        Observation.local_us == sel.local_us,
                    )
                )
                if obs is not None:
                    obs.is_representative = True
                    obs.crop_key = storage.crop_key(str(tracklet.tracklet_revision_id), row_index.index(
                        next(r for r in row_index if r["local_us"] == sel.local_us)))
                    obs.laplacian_var = sel.laplacian_var

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
                status="pending",
            ))
            session.flush()
            publish_seq += 1

        # coverage 与 run 状态
        session.add(CoverageSegment(
            dataset_id=run.dataset_id, source_asset_id=asset.source_asset_id,
            ingest_run_id=run.ingest_run_id,
            start_us=run.window_start_us, end_us=run.window_end_us, status="analyzed",
        ))
        run.status = "publishing"
        run.finished_at = _now()
        session.commit()

        return {
            "status": run.status,
            "frames_done": frames_done,
            "tracklets": len(chunks),
            "ingest_run_id": str(run.ingest_run_id),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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


def _now():
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc)
