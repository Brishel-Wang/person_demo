"""Read-only source media and PTS-aligned target selection."""

from fractions import Fraction
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from person_demo.api.deps import get_db
from person_demo.config import get_settings
from person_demo.db.models import IngestRun, Observation, SourceAsset, Tracklet
from person_demo.errors import EvidenceExpired, InvalidRequest, NotFound
from person_demo.media.probe import probe_file
from person_demo.media.reader import decode_frame_at_local_us, encode_jpg
from person_demo.vision.detector import recorded_detector_weights
from person_demo.vision.ingest_worker import DENSE_PROFILE

router = APIRouter(prefix="/v1/video-sources")


def source_for(db: Session, asset_id: UUID) -> SourceAsset:
    asset = db.get(SourceAsset, asset_id)
    if asset is None or asset.dataset_id != get_settings().dataset_id:
        raise NotFound("找不到该视频")
    return asset


def original_path(asset: SourceAsset) -> Path:
    path = Path(asset.local_path).resolve()
    if (path != Path(get_settings().default_video_path).resolve() or not path.is_file()
            or path.stat().st_size != asset.size_bytes):
        raise EvidenceExpired("原片不可读或已变更")
    return path


def indexed_run(db: Session, asset: SourceAsset) -> IngestRun | None:
    return db.scalar(select(IngestRun).where(
        IngestRun.source_asset_id == asset.source_asset_id,
        IngestRun.window_start_us == 0, IngestRun.status == "ready",
    ).order_by(
        IngestRun.window_end_us.desc(),
        (IngestRun.vision_profile == DENSE_PROFILE).desc(),
        IngestRun.created_at.desc(),
    ).limit(1))


@router.get("/{asset_id}")
def source_metadata(asset_id: UUID, db: Session = Depends(get_db)):
    asset = source_for(db, asset_id)
    original_path(asset)
    run = indexed_run(db, asset)
    s = get_settings()
    weights = recorded_detector_weights(run.tracker_config, run.detector_sha256) if run else None
    return {
        "source_asset_id": str(asset_id), "filename": asset.filename,
        "camera_code": asset.camera.camera_code, "duration_us": asset.duration_us,
        "indexed_end_us": run.window_end_us if run else 0,
        "frame_indexed": bool(run and run.vision_profile == DENSE_PROFILE),
        "analysis_fps": run.analysis_fps if run else None,
        "indexed_detector": {"weights": Path(weights).name if weights else None,
                             "imgsz": run.detector_imgsz} if run else None,
        "configured_detector": {"weights": Path(s.yolo_weights).name, "imgsz": s.yolo_imgsz},
        "width": asset.width, "height": asset.height,
        "offset_us": round(Fraction(asset.pts_anchor * asset.time_base_num, asset.time_base_den) * 1_000_000),
        "video_url": f"/v1/video-sources/{asset_id}/original",
    }


@router.get("/{asset_id}/original")
def source_video(asset_id: UUID, db: Session = Depends(get_db)):
    return FileResponse(original_path(source_for(db, asset_id)), media_type="video/mp4")


@router.get("/{asset_id}/observations")
def select_frame(asset_id: UUID, local_us: int = Query(ge=0), db: Session = Depends(get_db)):
    asset = source_for(db, asset_id)
    run = indexed_run(db, asset)
    if run is None:
        raise InvalidRequest("视频索引尚未就绪")
    end = run.window_end_us
    if local_us >= end:
        raise InvalidRequest("请选择已建立索引的时间范围")
    terms = (Observation.source_asset_id == asset_id, Tracklet.ingest_run_id == run.ingest_run_id,
             Tracklet.status == "committed", Observation.crop_status == "valid", Observation.local_us < end)
    times = select(Observation.local_us).join(Tracklet).where(*terms)
    before = db.scalar(times.where(Observation.local_us <= local_us).order_by(Observation.local_us.desc()).limit(1))
    after = db.scalar(times.where(Observation.local_us > local_us).order_by(Observation.local_us).limit(1))
    nearby = [t for t in (before, after) if t is not None and abs(t - local_us) <= 2_000_000]
    if not nearby:
        return {"local_us": local_us, "items": [], "frame_url": None}
    moment = min(nearby, key=lambda t: abs(t - local_us))
    entries = db.scalars(select(Observation).join(Tracklet).where(
        *terms, Observation.local_us == moment).order_by(Observation.observation_id)).all()
    return {
        "local_us": moment, "frame_url": f"/v1/video-sources/{asset_id}/frames/{entries[0].observation_id}",
        "items": [{"observation_id": str(o.observation_id), "bbox": [o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2],
                   "frame_w": o.frame_w, "frame_h": o.frame_h} for o in entries],
    }


@router.get("/{asset_id}/frames/{observation_id}")
def observation_image(asset_id: UUID, observation_id: UUID, db: Session = Depends(get_db)):
    asset = source_for(db, asset_id)
    obs = db.get(Observation, observation_id)
    if obs is None or obs.source_asset_id != asset_id:
        raise NotFound("找不到该原片观测帧")
    track = db.get(Tracklet, obs.tracklet_revision_id)
    if track is None or track.status != "committed":
        raise EvidenceExpired("该观测帧已失效")
    path = str(original_path(asset))
    probe = probe_file(path, hash_file=False)
    frame = decode_frame_at_local_us(path, probe, obs.local_us)
    if frame is None or frame.pts != obs.frame_pts:
        raise EvidenceExpired("原片观测帧无法对应")
    return Response(encode_jpg(frame.frame_bgr), media_type="image/jpeg")
