"""FastAPI 应用（spec 6.1 最小接口 + demo 辅助端点）。"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from person_demo.api.deps import get_app_state, get_db, get_storage, get_store
from person_demo.api.video_chat import chat_worker_loop, router as chat_router
from person_demo.api.video_sources import router as video_sources_router
from person_demo.api.schemas import (
    ImportManifestIn,
    MetadataRevisionIn,
    QueryFromFrameIn,
    ReportRequestIn,
    ReviewIn,
    SearchTaskIn,
)
from person_demo.config import get_settings
from person_demo.db.repo import is_tombstoned
from person_demo.db.models import (
    AssetProxy,
    Camera,
    DatasetStatus,
    EvidenceBundle,
    IngestRun,
    MetadataRevision,
    Observation,
    PublishWatermark,
    QueryAsset,
    Report,
    ReportJob,
    Review,
    SearchCandidate,
    SearchTask,
    SourceAsset,
    Tracklet,
    VectorOutbox,
)
from person_demo.errors import (
    DemoError,
    EvidenceExpired,
    InvalidManifest,
    InvalidRequest,
    ModelUnavailable,
    OriginUnresolved,
    NotFound,
    RevisionConflict,
    SingleSourceRequired,
    TargetAmbiguous,
    TimebaseInvalid,
)
from person_demo.evidence.bundles import build_bundle
from person_demo.evidence.media import build_evidence_payload
from person_demo.feature_space import AGGREGATION_VERSION
from person_demo.retrieval import _active_run, _coverage_for, paginate_candidates, run_search, validate_and_freeze
from person_demo.storage import StorageLayout
from person_demo.vectors import MilvusStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_app_state()
    state.video_worker_task = asyncio.create_task(_video_worker_loop(state))
    state.publisher_task = asyncio.create_task(_publisher_loop())
    state.search_worker_task = asyncio.create_task(_search_worker_loop(state))
    state.report_worker_task = asyncio.create_task(_report_worker_loop(state))
    chat_task = asyncio.create_task(chat_worker_loop(state))
    log.info("workers started")
    yield
    for t in (state.video_worker_task, state.publisher_task, state.search_worker_task, state.report_worker_task, chat_task):
        if t:
            t.cancel()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="单路人体检索与大模型报告 Demo", version="1.6.1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(DemoError)
    async def demo_error_handler(request: Request, exc: DemoError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    # ------------------------------------------------------------ datasets
    @app.post("/v1/datasets/import", status_code=202)
    def import_dataset(manifest: ImportManifestIn, db: Session = Depends(get_db)):
        from person_demo.media.probe import probe_file

        if len(manifest.cameras) != 1:
            raise SingleSourceRequired("exactly 1 camera required")
        cam_in = manifest.cameras[0]
        path = Path(cam_in.local_path).resolve()
        if (manifest.dataset_id != s.dataset_id or cam_in.camera_code != s.camera_code
                or cam_in.source_kind != "file" or manifest.time_axis != "asset_relative"
                or path != Path(s.default_video_path).resolve()):
            raise InvalidManifest("import must use this demo's configured local video")
        if not path.is_file():
            raise InvalidManifest("configured video is unavailable")

        window = manifest.requested_window_us or {"start": 0, "end": 200_000_000}
        if type(window.get("start")) is not int or type(window.get("end")) is not int:
            raise InvalidManifest("window start and end must be integer microseconds")
        probe = probe_file(str(path), (window["start"], window["end"]))
        if probe.timebase_status != "ok" or not probe.sha256 or not probe.resolved_window_us:
            raise TimebaseInvalid(probe.error or "probe failed")
        if ((cam_in.sha256 and cam_in.sha256 != probe.sha256)
                or (cam_in.metadata_expected_bytes is not None and cam_in.metadata_expected_bytes != probe.size_bytes)):
            raise InvalidManifest("video checksum or size differs from manifest")

        # One registered source per dataset. Serialize the check and run creation in PostgreSQL.
        db.execute(select(func.pg_advisory_xact_lock(func.hashtext(manifest.dataset_id))))
        registered = db.scalar(select(SourceAsset).where(SourceAsset.dataset_id == manifest.dataset_id))
        if registered is not None and registered.sha256 != probe.sha256:
            raise InvalidManifest("registered video checksum differs from the current file")
        camera = db.scalar(select(Camera).where(
            Camera.dataset_id == manifest.dataset_id, Camera.camera_code == cam_in.camera_code))
        if camera is None:
            camera = Camera(dataset_id=manifest.dataset_id, camera_code=cam_in.camera_code)
            db.add(camera)
            db.flush()

        asset = db.scalar(select(SourceAsset).where(
            SourceAsset.dataset_id == manifest.dataset_id, SourceAsset.sha256 == probe.sha256))
        if asset is None:
            asset = SourceAsset(
                dataset_id=manifest.dataset_id, camera_id=camera.camera_id,
                filename=cam_in.filename, local_path=str(path),
                sha256=probe.sha256, size_bytes=probe.size_bytes,
                codec_name=probe.codec_name, width=probe.width, height=probe.height,
                avg_fps=probe.avg_fps, is_vfr=probe.is_vfr,
                duration_us=probe.duration_us, nb_frames=probe.nb_frames,
                pts_anchor=probe.pts_anchor, time_base_num=probe.time_base_num,
                time_base_den=probe.time_base_den, timebase_status=probe.timebase_status,
                resolved_window_start_us=probe.resolved_window_us[0] if probe.resolved_window_us else None,
                resolved_window_end_us=probe.resolved_window_us[1] if probe.resolved_window_us else None,
            )
            db.add(asset)
            db.flush()
        else:
            # 同文件重复导入：复用 asset；窗口以 resolved 为准
            pass

        # 幂等 ingest run
        from person_demo.db.repo import find_ingest_run_same_config

        checkpoint = s.model_dir / s.osnet_checkpoint
        weights = s.model_dir / s.yolo_weights
        if not checkpoint.is_file() or not weights.is_file():
            raise ModelUnavailable("configured vision checkpoints are unavailable")
        from person_demo.feature_space import canonical_space_json, feature_space_id

        space_id = feature_space_id(canonical_space_json(
            architecture="osnet_ain_x1_0", checkpoint_sha256=_file_sha(checkpoint),
            source_revision=None))
        det_sha = _file_sha(weights)
        w_start = probe.resolved_window_us[0] if probe.resolved_window_us else 0
        w_end = probe.resolved_window_us[1] if probe.resolved_window_us else 200_000_000
        # 显式扩展处理窗口时同步扩展 asset 的 resolved 窗口（以 probe 时长为限）
        if asset.duration_us is not None and w_end > (asset.resolved_window_end_us or 0) and w_end <= asset.duration_us:
            asset.resolved_window_end_us = w_end
        run = find_ingest_run_same_config(
            db, asset.source_asset_id, w_start, w_end, s.analysis_fps, det_sha, space_id,
            detector_imgsz=s.yolo_imgsz,
        )
        reused = run is not None
        if run is None:
            run = IngestRun(
                dataset_id=manifest.dataset_id, source_asset_id=asset.source_asset_id,
                window_start_us=w_start, window_end_us=w_end,
                analysis_fps=s.analysis_fps, detector_sha256=det_sha, detector_imgsz=s.yolo_imgsz,
                feature_space_id=space_id, aggregation_version=AGGREGATION_VERSION,
                tracker_config={
                    "implementation": "bytetrack_local", "track_buffer": s.tracker_lost_buffer_frames,
                    "match_thresh_high": s.tracker_match_high, "match_thresh_low": s.tracker_match_low,
                    "min_hits": s.tracker_min_hits, "analysis_fps": s.analysis_fps,
                    "detector_weights": s.yolo_weights,
                },
                status="pending",
            )
            db.add(run)
            db.flush()
        ds = db.get(DatasetStatus, manifest.dataset_id)
        if ds is None:
            ds = DatasetStatus(dataset_id=manifest.dataset_id)
            db.add(ds)
        if run.status == "failed":
            run.status, run.error = "pending", None
        ds.status = "importing" if run.status == "pending" else run.status
        ds.last_ingest_run_id = run.ingest_run_id
        db.commit()
        return {"import_job_id": str(run.ingest_run_id), "status": "accepted",
                "source_asset_id": str(asset.source_asset_id), "note": "reused" if reused else None}

    @app.get("/v1/datasets/{dataset_id}/status")
    def dataset_status(dataset_id: str, db: Session = Depends(get_db)):
        assets = db.scalars(select(SourceAsset).where(SourceAsset.dataset_id == dataset_id)).all()
        if not assets:
            return {"dataset_id": dataset_id, "status": "empty", "detail": None}
        asset = assets[0]
        # 状态优先展示可检索的 run（ready/publishing），其次最新（处理中）
        runs = db.scalars(select(IngestRun).where(IngestRun.source_asset_id == asset.source_asset_id)
                         .order_by(IngestRun.created_at.desc())).all()
        run = next((r for r in runs if r.status in ("ready", "publishing")), runs[0] if runs else None)
        watermark = None
        if run is not None and run.feature_space_id:
            wm = db.get(PublishWatermark, (dataset_id, asset.source_asset_id, run.ingest_run_id, run.feature_space_id))
            watermark = wm.watermark_seq if wm else None
        coverage = _coverage_for(
            db, dataset_id, asset.source_asset_id, run, run.window_start_us, run.window_end_us
        ) if run else None
        tracklet_count = db.scalar(select(func.count(Tracklet.tracklet_revision_id)).where(
            Tracklet.ingest_run_id == run.ingest_run_id)) if run else 0
        obs_count = db.scalar(select(func.count(Observation.observation_id)).join(Tracklet).where(
            Tracklet.ingest_run_id == run.ingest_run_id)) if run else 0
        demo_end = min(asset.resolved_window_end_us or 200_000_000, 200_000_000)
        demo_run = _active_run(db, asset, 0, demo_end)
        latest_run = runs[0] if runs else None
        return {
            "dataset_id": dataset_id,
            "status": (run.status if run else "empty"),
            "latest_run": {
                "ingest_run_id": str(latest_run.ingest_run_id) if latest_run else None,
                "status": latest_run.status if latest_run else None,
                "window_us": [latest_run.window_start_us, latest_run.window_end_us] if latest_run else None,
                "started_at": latest_run.started_at.isoformat() if latest_run and latest_run.started_at else None,
            },
            "camera_code": asset.camera.camera_code if asset.camera else None,
            "source_asset_id": str(asset.source_asset_id),
            "resolved_window_us": [asset.resolved_window_start_us, asset.resolved_window_end_us],
            "ingest_run_id": str(run.ingest_run_id) if run else None,
            "ingest_status": run.status if run else None,
            "watermark": watermark,
            "coverage": coverage,
            "demo_window_us": [0, demo_end],
            "demo_ingest_run_id": str(demo_run.ingest_run_id) if demo_run else None,
            "tracklet_count": tracklet_count,
            "observation_count": obs_count,
            "detail": {
                "sha256": asset.sha256, "size_bytes": asset.size_bytes,
                "codec": asset.codec_name, "width": asset.width, "height": asset.height,
                "avg_fps": asset.avg_fps, "duration_us": asset.duration_us,
                "timebase_status": asset.timebase_status,
                "recording_time_status": asset.recording_time_status,
                "location_status": asset.location_status,
            },
        }

    # ------------------------------------------------------------ query assets
    @app.post("/v1/query-assets/from-frame")
    def create_query_from_frame(req: QueryFromFrameIn, db: Session = Depends(get_db),
                                storage: StorageLayout = Depends(get_storage),
                                state=Depends(get_app_state)):
        from person_demo.media.probe import probe_file
        from person_demo.media.reader import crop_person, decode_frame_at_local_us, encode_jpg

        asset = db.get(SourceAsset, req.source_asset_id)
        if asset is None:
            raise NotFound("source asset not found")
        if req.observation_id is None:
            raise TargetAmbiguous("select one analyzed observation by observation_id")
        obs = db.get(Observation, req.observation_id)
        if obs is None or obs.source_asset_id != req.source_asset_id:
            raise NotFound("observation not found")
        tracklet = db.get(Tracklet, obs.tracklet_revision_id)
        if not tracklet or tracklet.status != "committed" or is_tombstoned(db, "tracklet", tracklet.tracklet_revision_id):
            raise OriginUnresolved("source tracklet is unavailable")
        bbox = [obs.bbox_x1, obs.bbox_y1, obs.bbox_x2, obs.bbox_y2]
        if req.target_bbox is not None and req.target_bbox != [round(value) for value in bbox]:
            raise TargetAmbiguous("target_bbox must match the selected observation")
        if req.local_us is not None and req.local_us != obs.local_us:
            raise TargetAmbiguous("timestamp does not match the selected observation")

        probe = probe_file(asset.local_path, hash_file=False)
        if probe.timebase_status != "ok" or probe.size_bytes != asset.size_bytes or probe.pts_anchor != asset.pts_anchor:
            raise EvidenceExpired("original video or its time base is unavailable")
        frame = decode_frame_at_local_us(asset.local_path, probe, obs.local_us)
        if frame is None or (frame.pts, frame.time_base_num, frame.time_base_den) != (
                obs.frame_pts, obs.time_base_num, obs.time_base_den):
            raise EvidenceExpired("the selected observation does not resolve to an original frame")
        crop = crop_person(frame.frame_bgr, tuple(bbox))
        if crop is None:
            raise TargetAmbiguous("empty crop")

        space_id, vec = _encode_query(crop, state)
        qa = QueryAsset(
            dataset_id=asset.dataset_id, kind="from_frame",
            source_asset_id=asset.source_asset_id,
            origin_observation_id=obs.observation_id,
            origin_tracklet_revision_id=tracklet.tracklet_revision_id if tracklet else None,
            origin_continuation_group_id=tracklet.continuation_group_id if tracklet else None,
            origin_camera_code=asset.camera.camera_code if asset.camera else None,
            origin_status="verified",
            bbox=bbox, local_us=obs.local_us,
        )
        db.add(qa)
        db.flush()
        img_key = storage.query_image_key(str(qa.asset_id))
        storage.write_bytes(img_key, encode_jpg(crop))

        qa.feature_space_id = space_id
        emb_key = f"query_images/{qa.asset_id}.npy"
        np.save(storage.resolve(emb_key), vec.astype(np.float32))
        qa.embedding_key = emb_key
        qa.image_key = img_key
        db.commit()
        return _query_asset_out(qa)

    @app.post("/v1/query-assets/upload")
    async def create_query_upload(
        file: UploadFile = File(...),
        bbox: str = Form(...),
        dataset_id: str = Form("single_camera_demo_v1"),
        db: Session = Depends(get_db), storage: StorageLayout = Depends(get_storage),
        state=Depends(get_app_state),
    ):
        import cv2

        try:
            bb = json.loads(bbox)
        except Exception:  # noqa: BLE001
            raise TargetAmbiguous("bbox must be JSON [x1,y1,x2,y2]")
        if (not isinstance(bb, list) or len(bb) != 4 or any(type(value) is not int for value in bb)
                or bb[2] <= bb[0] or bb[3] <= bb[1]):
            raise TargetAmbiguous("bbox must be four integers [x1,y1,x2,y2] with positive area")
        if dataset_id != s.dataset_id:
            raise InvalidRequest("unknown demo dataset")
        data = await file.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024:
            raise InvalidRequest("query image exceeds 10 MiB")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise TargetAmbiguous("image decode failed")
        from person_demo.media.reader import crop_person

        crop = crop_person(img, tuple(bb))
        if crop is None:
            raise TargetAmbiguous("empty crop")
        qa = QueryAsset(
            dataset_id=dataset_id, kind="upload",
            origin_status="unknown", bbox=bb,
        )
        db.add(qa)
        db.flush()
        img_key = storage.query_image_key(str(qa.asset_id))
        storage.write_bytes(img_key, _encode_bgr(crop))
        space_id, vec = await asyncio.to_thread(_encode_query, crop, state)
        qa.feature_space_id = space_id
        emb_key = f"query_images/{qa.asset_id}.npy"
        np.save(storage.resolve(emb_key), vec.astype(np.float32))
        qa.embedding_key = emb_key
        qa.image_key = img_key
        db.commit()
        return _query_asset_out(qa)

    @app.get("/v1/query-assets/{asset_id}")
    def get_query_asset(asset_id: uuid.UUID, db: Session = Depends(get_db)):
        qa = db.get(QueryAsset, asset_id)
        if qa is None:
            raise NotFound("query asset not found")
        return _query_asset_out(qa)

    # ------------------------------------------------------------ search
    @app.post("/v1/search-tasks", status_code=202)
    def create_search_task(req: SearchTaskIn, db: Session = Depends(get_db)):
        qa = db.get(QueryAsset, req.query_asset_id)
        if qa is None:
            raise NotFound("query asset not found")
        task = SearchTask(
            dataset_id=req.dataset_id,
            query_asset_id=req.query_asset_id,
            mode=req.mode,
            evaluation_scope=req.evaluation_scope,
            scope_windows=req.scope.model_dump(mode="json"),
            status="QUEUED",
        )
        db.add(task)
        db.commit()
        return {"task_id": str(task.task_id), "status": "QUEUED"}

    @app.get("/v1/search-tasks/{task_id}")
    def get_search_task(task_id: uuid.UUID, db: Session = Depends(get_db)):
        task = db.get(SearchTask, task_id)
        if task is None:
            raise NotFound("search task not found")
        return {
            "task_id": str(task.task_id), "status": task.status,
            "search_revision": task.search_revision,
            "evaluation_scope": task.evaluation_scope,
            "snapshot": task.snapshot,
            "candidate_truncated": task.candidate_truncated,
            "error": task.error,
        }

    @app.get("/v1/search-tasks/{task_id}/results")
    def search_results(task_id: uuid.UUID, page: int = Query(1, ge=1),
                       page_size: int = Query(10, ge=1, le=200), db: Session = Depends(get_db),
                       storage: StorageLayout = Depends(get_storage)):
        task = db.get(SearchTask, task_id)
        if task is None:
            raise NotFound("search task not found")
        page_out = paginate_candidates(db, task_id, page, page_size)
        items = []
        for c in page_out["items"]:
            items.append(_candidate_out(db, c, storage))
        report_job = db.scalar(select(ReportJob).where(ReportJob.search_task_id == task_id)
                               .order_by(ReportJob.created_at.desc()).limit(1))
        report_row = db.scalar(select(Report).where(Report.search_task_id == task_id)
                               .order_by(Report.created_at.desc()).limit(1))
        return {
            "task_id": str(task_id), "status": task.status, "total": page_out["total"],
            "page": page, "page_size": page_size, "items": items,
            "snapshot": task.snapshot,
            "report_job_id": str(report_job.report_job_id) if report_job else None,
            "report_status": report_job.status if report_job else None,
            "report_id": str(report_row.report_id) if report_row else None,
            "report_provider": report_row.provider if report_row else None,
        }

    @app.post("/v1/search-tasks/{task_id}/reviews")
    def submit_review(task_id: uuid.UUID, req: ReviewIn, db: Session = Depends(get_db)):
        task = db.get(SearchTask, task_id)
        if task is None:
            raise NotFound("search task not found")
        cand = db.scalar(select(SearchCandidate).where(
            SearchCandidate.search_task_id == task_id, SearchCandidate.hit_id == req.hit_id).with_for_update())
        if cand is None:
            raise NotFound("candidate not found")
        if req.action not in ("confirm", "reject"):
            raise InvalidRequest("action must be confirm or reject")
        if req.expected_revision != cand.review_revision:
            raise RevisionConflict(
                f"expected_revision {req.expected_revision} != current {cand.review_revision}")
        cand.review_status = "confirmed" if req.action == "confirm" else "rejected"
        cand.review_revision += 1
        db.add(Review(hit_id=cand.hit_id, action=req.action, revision=cand.review_revision))
        db.commit()
        return {"hit_id": str(req.hit_id), "action": req.action,
                "revision": cand.review_revision, "status": cand.review_status}

    # ------------------------------------------------------------ reports
    @app.post("/v1/search-tasks/{task_id}/reports", status_code=202)
    def create_report(task_id: uuid.UUID, req: ReportRequestIn, db: Session = Depends(get_db)):
        task = db.get(SearchTask, task_id)
        if task is None:
            raise NotFound("search task not found")
        if req.expected_search_revision != task.search_revision:
            raise RevisionConflict(
                f"expected_search_revision {req.expected_search_revision} != {task.search_revision}")
        if task.status not in ("COMPLETED", "PARTIAL"):
            raise InvalidRequest("search results are not available for reporting")
        job = ReportJob(search_task_id=task_id, search_revision=task.search_revision, status="QUEUED",
                        timing={"force_new_version": req.force_new_version})
        db.add(job)
        db.commit()
        return {"report_job_id": str(job.report_job_id), "status": "QUEUED"}

    @app.get("/v1/report-jobs/{job_id}")
    def get_report_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
        job = db.get(ReportJob, job_id)
        if job is None:
            raise NotFound("report job not found")
        report_id = (job.timing or {}).get("reused_existing_report") or db.scalar(
            select(Report.report_id).where(Report.report_job_id == job_id))
        return {"report_job_id": str(job.report_job_id), "status": job.status,
                "report_id": str(report_id) if report_id else None,
                "provider": job.provider, "error": job.error,
                "timing": job.timing, "resource": job.resource}

    @app.get("/v1/reports/{report_id}")
    def get_report(report_id: uuid.UUID, db: Session = Depends(get_db)):
        report = db.get(Report, report_id)
        if report is None:
            raise NotFound("report not found")
        bundle = db.scalar(select(EvidenceBundle).where(
            EvidenceBundle.search_task_id == report.search_task_id,
            EvidenceBundle.search_revision == report.search_revision,
            EvidenceBundle.evidence_hash == report.evidence_hash,
        )) if report.evidence_hash else None
        for evidence_id in {eid for c in report.content.get("claims", []) for eid in c["evidence_ids"]}:
            evidence = build_evidence_payload(db, uuid.UUID(evidence_id))
            if evidence["search_task_id"] != str(report.search_task_id):
                raise EvidenceExpired("report citation is outside its task")
        return {
            "report_id": str(report.report_id), "search_task_id": str(report.search_task_id),
            "search_revision": report.search_revision, "provider": report.provider,
            "status": report.status, "content": report.content,
            "validation": report.validation, "created_at": report.created_at.isoformat(),
            "facts": bundle.payload if bundle else None,
            "prompt_version": (report.raw_output or {}).get("prompt_version"),
        }

    @app.get("/v1/search-tasks/{task_id}/export")
    def export_results(task_id: uuid.UUID, fmt: str = Query("json", pattern="^(json|csv)$"),
                       search_revision: int | None = None, report_id: uuid.UUID | None = None,
                       db: Session = Depends(get_db), storage: StorageLayout = Depends(get_storage)):
        task = db.get(SearchTask, task_id)
        if task is None:
            raise NotFound("search task not found")
        cands = db.scalars(select(SearchCandidate).where(SearchCandidate.search_task_id == task_id)
                           .order_by(SearchCandidate.rank_position.asc())).all()
        if search_revision is not None and search_revision != task.search_revision:
            raise RevisionConflict("requested search revision is unavailable")
        if report_id is not None:
            report = db.get(Report, report_id)
            if not report or report.search_task_id != task_id or report.search_revision != task.search_revision:
                raise NotFound("report does not belong to this search revision")
        else:
            report = db.scalar(select(Report).where(Report.search_task_id == task_id)
                               .order_by(Report.created_at.desc()).limit(1))
        rows = []
        for c in cands:
            t = db.get(Tracklet, c.tracklet_revision_id)
            rows.append({
                "rank": c.rank_position,
                "hit_id": str(c.hit_id),
                "evidence_id": str(c.evidence_id),
                "tracklet_revision_id": str(c.tracklet_revision_id),
                "t_start_local_us": t.t_start_local_us if t else None,
                "t_end_local_us": t.t_end_local_us if t else None,
                "prototype_similarity": c.prototype_similarity,
                "rerank_similarity": c.rerank_similarity,
                "matched_local_us": c.matched_local_us,
                "same_continuation_as_query": c.same_continuation_as_query,
                "review_status": c.review_status,
                "matched_buckets": c.matched_buckets,
            })
        payload = {
            "export_revision": task.search_revision,
            "task_id": str(task_id),
            "report_id": str(report.report_id) if report else None,
            "report": report.content if report else None,
            "status": task.status,
            "evaluation_scope": task.evaluation_scope,
            "appearance_scoring": (task.snapshot or {}).get("appearance_scoring"),
            "min_similarity": task.scope_windows.get("frame_min_similarity"),
            "candidates": rows,
        }
        if fmt == "json":
            return JSONResponse(payload)
        buf = io.StringIO()
        fieldnames = ["rank", "hit_id", "evidence_id", "tracklet_revision_id", "t_start_local_us",
                      "t_end_local_us", "prototype_similarity", "rerank_similarity", "matched_local_us",
                      "same_continuation_as_query", "review_status", "matched_buckets"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=search_export.csv"})

    # ------------------------------------------------------------ evidence / media
    @app.get("/v1/evidence/{evidence_id}")
    def get_evidence(evidence_id: uuid.UUID, db: Session = Depends(get_db)):
        return build_evidence_payload(db, evidence_id)

    @app.get("/v1/evidence/{evidence_id}/frames/{local_us}")
    def evidence_frame(evidence_id: uuid.UUID, local_us: int, db: Session = Depends(get_db)):
        from person_demo.media.probe import probe_file
        from person_demo.media.reader import decode_frame_at_local_us

        payload = build_evidence_payload(db, evidence_id)
        obs = next((o for o in payload["bbox_sequence"] if o["local_us"] == local_us), None)
        if obs is None:
            raise NotFound("no evidence observation at this timestamp")
        asset = db.get(SourceAsset, uuid.UUID(payload["source_asset_id"]))
        probe = probe_file(asset.local_path, hash_file=False)
        frame = decode_frame_at_local_us(asset.local_path, probe, local_us)
        if frame is None or frame.pts != obs["frame_pts"]:
            raise EvidenceExpired("evidence observation does not resolve to an original frame")
        return Response(_encode_bgr(frame.frame_bgr), media_type="image/jpeg")

    @app.get("/v1/evidence/{evidence_id}/video")
    def evidence_video(evidence_id: uuid.UUID, db: Session = Depends(get_db)):
        payload = build_evidence_payload(db, evidence_id)
        asset = db.get(SourceAsset, uuid.UUID(payload["source_asset_id"]))
        path = Path(asset.local_path)
        if path.resolve() != Path(s.default_video_path).resolve() or not path.is_file():
            raise EvidenceExpired("original video is unavailable")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/v1/media/{storage_key:path}")
    def serve_media(storage_key: str, storage: StorageLayout = Depends(get_storage)):
        try:
            if storage_key.startswith("bundles/"):
                media_type = "image/jpeg" if storage_key.endswith(".jpg") else "application/octet-stream"
                return Response(storage.read_bytes(storage_key), media_type=media_type)
            path = storage.resolve(storage_key)
        except FileNotFoundError:
            raise NotFound("media not found")
        except ValueError as e:
            raise NotFound(f"invalid storage_key: {e}")
        if not path.is_file():
            raise NotFound("media not found")
        return FileResponse(path)

    @app.get("/v1/observations/frames")
    def analyzed_frames(source_asset_id: uuid.UUID,
                        representative_only: bool = Query(True),
                        ingest_run_id: uuid.UUID | None = None,
                        start_us: int = Query(0, ge=0),
                        end_us: int | None = Query(None, gt=0),
                        limit: int = Query(200, ge=1, le=2000),
                        db: Session = Depends(get_db), storage: StorageLayout = Depends(get_storage)):
        asset = db.get(SourceAsset, source_asset_id)
        if asset is None:
            raise NotFound("source asset not found")
        if ingest_run_id is None:
            run = _active_run(db, asset, 0, min(asset.resolved_window_end_us or 200_000_000, 200_000_000))
            ingest_run_id = run.ingest_run_id if run else None
        if ingest_run_id is None:
            return {"items": [], "count": 0}
        q = select(Observation).join(Tracklet).where(
            Observation.source_asset_id == source_asset_id, Tracklet.ingest_run_id == ingest_run_id,
            Tracklet.status == "committed", Observation.local_us >= start_us)
        if end_us is not None:
            q = q.where(Observation.local_us < end_us)
        if representative_only:
            q = q.where(Observation.is_representative.is_(True))
        q = q.order_by(Observation.local_us.asc(), Observation.observation_id).limit(limit)
        obs = db.scalars(q).all()
        out = []
        for o in obs:
            out.append({
                "observation_id": str(o.observation_id),
                "local_us": o.local_us,
                "bbox": [o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2],
                "frame_w": o.frame_w, "frame_h": o.frame_h,
                "crop_url": f"/v1/media/{o.crop_key}" if o.crop_key else None,
                "tracklet_revision_id": str(o.tracklet_revision_id),
            })
        return {"items": out, "count": len(out)}

    @app.get("/v1/proxy/{source_asset_id}")
    def get_proxy(source_asset_id: uuid.UUID, db: Session = Depends(get_db)):
        proxy = db.scalar(select(AssetProxy).where(AssetProxy.source_asset_id == source_asset_id)
                          .order_by(AssetProxy.created_at.desc()).limit(1))
        if proxy is None:
            raise NotFound("proxy video not built yet")
        return {
            "storage_key": proxy.storage_key,
            "fps": proxy.fps,
            "window_start_us": proxy.window_start_us,
            "window_end_us": proxy.window_end_us,
            "time_mapping": proxy.time_mapping,
        }

    # ------------------------------------------------------------ metadata
    @app.get("/v1/cameras/{camera_id}/metadata")
    def camera_metadata(camera_id: uuid.UUID, db: Session = Depends(get_db)):
        revs = db.scalars(select(MetadataRevision).where(
            MetadataRevision.entity_type == "camera_location",
            MetadataRevision.entity_id == camera_id).order_by(MetadataRevision.revision.desc())).all()
        current = revs[0].payload if revs else {"status": "unknown"}
        return {"camera_id": str(camera_id), "revision": revs[0].revision if revs else 0,
                "location": current, "verified": revs[0].verified if revs else False}

    @app.post("/v1/cameras/{camera_id}/location-revisions")
    def post_camera_location(camera_id: uuid.UUID, req: MetadataRevisionIn, db: Session = Depends(get_db)):
        if db.scalar(select(Camera).where(Camera.camera_id == camera_id).with_for_update()) is None:
            raise NotFound("metadata source does not exist")
        current = db.scalar(select(func.max(MetadataRevision.revision)).where(
            MetadataRevision.entity_type == "camera_location",
            MetadataRevision.entity_id == camera_id))
        current = current if current is not None else 0
        if req.expected_revision != current:
            raise RevisionConflict(f"expected_revision {req.expected_revision} != {current}")
        revision = current + 1
        db.add(MetadataRevision(entity_type="camera_location", entity_id=camera_id,
                                revision=revision, payload=req.payload, verified=False))
        db.commit()
        return {"camera_id": str(camera_id), "revision": revision, "verified": False}

    @app.get("/v1/assets/{asset_id}/metadata")
    def asset_metadata(asset_id: uuid.UUID, db: Session = Depends(get_db)):
        revs = db.scalars(select(MetadataRevision).where(
            MetadataRevision.entity_type == "asset_metadata",
            MetadataRevision.entity_id == asset_id).order_by(MetadataRevision.revision.desc())).all()
        current = revs[0].payload if revs else {"recording_time": {"status": "unknown"},
                                                "dataset_alignment": {"status": "unknown"},
                                                "location_binding": {"status": "unknown"}}
        return {"source_asset_id": str(asset_id), "revision": revs[0].revision if revs else 0,
                "metadata": current, "verified": revs[0].verified if revs else False}

    @app.post("/v1/assets/{asset_id}/metadata-revisions")
    def post_asset_metadata(asset_id: uuid.UUID, req: MetadataRevisionIn, db: Session = Depends(get_db)):
        if db.scalar(select(SourceAsset).where(SourceAsset.source_asset_id == asset_id).with_for_update()) is None:
            raise NotFound("metadata source does not exist")
        current = db.scalar(select(func.max(MetadataRevision.revision)).where(
            MetadataRevision.entity_type == "asset_metadata",
            MetadataRevision.entity_id == asset_id))
        current = current if current is not None else 0
        if req.expected_revision != current:
            raise RevisionConflict(f"expected_revision {req.expected_revision} != {current}")
        revision = current + 1
        db.add(MetadataRevision(entity_type="asset_metadata", entity_id=asset_id,
                                revision=revision, payload=req.payload, verified=False))
        db.commit()
        return {"source_asset_id": str(asset_id), "revision": revision, "verified": False}

    app.include_router(chat_router)
    app.include_router(video_sources_router)

    # 静态前端（若已构建）
    frontend_dist = s.frontend_dist
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


# ------------------------------------------------------------ helpers
def _file_sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _encode_bgr(img) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()


def _encode_query(crop, state):
    """用当前活动 feature space 编码查询图（与图库相同预处理）。"""
    import time

    import torch

    from person_demo.feature_space import canonical_space_json, feature_space_id
    from person_demo.vision.reid.encoder import OSNetEncoder

    s = get_settings()
    with state.gpu_lock:
        start = time.monotonic()
        try:
            if s.device.startswith("cuda"):
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(s.device)
            encoder = OSNetEncoder(s.model_dir / s.osnet_checkpoint, device=s.device)
            load_s = time.monotonic() - start
            space_id = feature_space_id(canonical_space_json(
                architecture="osnet_ain_x1_0", checkpoint_sha256=encoder.checkpoint_sha256,
                source_revision=None))
            vec = encoder.encode_single(crop)
            peak_mb = (torch.cuda.max_memory_allocated(s.device) / 2**20
                       if s.device.startswith("cuda") else None)
            log.info("OSNet query model load %.3fs, total %.3fs, peak MiB %s, probe %s",
                     load_s, time.monotonic() - start, peak_mb, encoder.probe_info)
            del encoder
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelUnavailable(f"OSNet failed: {exc}") from exc
        finally:
            if s.device.startswith("cuda") and torch.cuda.is_available():
                with torch.cuda.device(s.device):
                    torch.cuda.empty_cache()
    return space_id, vec


def _query_asset_out(qa: QueryAsset) -> dict:
    return {
        "asset_id": str(qa.asset_id), "kind": qa.kind, "origin_status": qa.origin_status,
        "origin_observation_id": str(qa.origin_observation_id) if qa.origin_observation_id else None,
        "origin_tracklet_revision_id": str(qa.origin_tracklet_revision_id) if qa.origin_tracklet_revision_id else None,
        "origin_continuation_group_id": str(qa.origin_continuation_group_id) if qa.origin_continuation_group_id else None,
        "origin_camera_code": qa.origin_camera_code,
        "source_asset_id": str(qa.source_asset_id) if qa.source_asset_id else None,
        "local_us": qa.local_us, "bbox": qa.bbox,
        "feature_space_id": qa.feature_space_id,
        "image_url": f"/v1/media/{qa.image_key}" if qa.image_key else None,
    }


def _candidate_out(db: Session, c: SearchCandidate, storage: StorageLayout) -> dict:
    t = db.get(Tracklet, c.tracklet_revision_id)
    reps = db.scalars(select(Observation).where(
        Observation.tracklet_revision_id == c.tracklet_revision_id,
        Observation.is_representative.is_(True)).order_by(Observation.local_us.asc())).all()
    return {
        "hit_id": str(c.hit_id),
        "camera_id": None,
        "camera_code": t.camera_code if t else None,
        "source_asset_id": str(t.source_asset_id) if t else None,
        "tracklet_revision_id": str(c.tracklet_revision_id),
        "t_start_local_us": t.t_start_local_us if t else None,
        "t_end_local_us": t.t_end_local_us if t else None,
        "prototype_similarity": c.prototype_similarity,
        "rerank_similarity": c.rerank_similarity,
        "matched_local_us": c.matched_local_us,
        "rerank_status": c.rerank_status,
        "sample_count": c.sample_count,
        "same_continuation_as_query": c.same_continuation_as_query,
        "representative_images": [{"local_us": o.local_us, "url": f"/v1/media/{o.crop_key}"}
                                  for o in reps if o.crop_key],
        "evidence_id": str(c.evidence_id),
        "matched_buckets": c.matched_buckets,
        "review_status": c.review_status,
        "review_revision": c.review_revision,
        "rank_position": c.rank_position,
        "suspicious": t.suspicious if t else False,
        "suspicious_reason": t.suspicious_reason if t else None,
    }


# ------------------------------------------------------------ workers
async def _video_worker_loop(state):
    from person_demo.db import new_session
    from person_demo.errors import DemoError
    from person_demo.vision.ingest_worker import process_ingest_run

    while True:
        try:
            await asyncio.sleep(2)
            db = new_session()
            try:
                run = db.scalar(select(IngestRun).where(IngestRun.status.in_(["pending", "analyzing"]))
                                .order_by(IngestRun.created_at.asc()).limit(1))
                if run is not None:
                    try:
                        def ingest():
                            with state.gpu_lock:
                                import torch

                                try:
                                    return process_ingest_run(run.ingest_run_id, storage=get_storage())
                                finally:
                                    device = get_settings().device
                                    if device.startswith("cuda") and torch.cuda.is_available():
                                        with torch.cuda.device(device):
                                            torch.cuda.empty_cache()

                        await asyncio.to_thread(ingest)
                    except DemoError as e:
                        run.status = "failed"
                        run.error = f"{e.code}: {e.message}"
                        db.commit()
                    except Exception as e:  # noqa: BLE001
                        run.status = "failed"
                        run.error = str(e)
                        db.commit()
                    else:
                        db = new_session()
                        ds = db.get(DatasetStatus, run.dataset_id)
                        if ds is not None:
                            ds.status = "indexing"
                            db.commit()
            finally:
                db.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("video worker error: %s", e)


async def _publisher_loop():
    from person_demo.db import new_session
    from person_demo.vectors.publisher import publish_loop_once

    while True:
        try:
            await asyncio.sleep(1)
            db = new_session()
            try:
                await asyncio.to_thread(publish_loop_once, get_store(), db)
                _mark_runs_ready(db)
            finally:
                db.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("publisher error (will retry): %s", e)
            await asyncio.sleep(3)


def _mark_runs_ready(db):
    """publishing → ready：该 run 的全部 outbox 均 ready 时。"""
    runs = db.scalars(select(IngestRun).where(IngestRun.status == "publishing")).all()
    for run in runs:
        if run.feature_space_id is None:
            continue
        pending = db.scalar(
            select(func.count(VectorOutbox.id)).where(
                VectorOutbox.ingest_run_id == run.ingest_run_id,
                VectorOutbox.feature_space_id == run.feature_space_id,
                VectorOutbox.status.not_in(["ready", "skipped"]),
            )
        ) or 0
        total = db.scalar(
            select(func.count(VectorOutbox.id)).where(
                VectorOutbox.ingest_run_id == run.ingest_run_id,
                VectorOutbox.feature_space_id == run.feature_space_id,
            )
        ) or 0
        if total > 0 and pending == 0:
            run.status = "ready"
            ds = db.get(DatasetStatus, run.dataset_id)
            if ds is not None:
                ds.status = "ready"
    db.commit()


async def _search_worker_loop(state):
    from person_demo.db import new_session

    while True:
        try:
            await asyncio.sleep(1)
            db = new_session()
            try:
                tasks = db.scalars(select(SearchTask).where(
                    SearchTask.status.in_(["QUEUED", "VALIDATING", "WAITING_INDEX"]))
                    .order_by(SearchTask.created_at.asc()).limit(5)).all()
                for task in tasks:
                    try:
                        if task.status in ("QUEUED", "WAITING_INDEX"):
                            if (db.execute(select(func.now())).scalar_one() - task.created_at).total_seconds() > get_settings().wait_index_timeout_s:
                                task.status = "FAILED"
                                task.error = "INDEX_WAIT_TIMEOUT: index did not become ready"
                                db.commit()
                                continue
                            validate_and_freeze(db, task)
                        result = await asyncio.to_thread(run_search, db, task.task_id, get_store(), get_storage())
                        if (result.get("status") in ("COMPLETED", "PARTIAL") and get_settings().auto_report
                                and task.scope_windows.get("generate_legacy_report", True)):
                            existing_job = db.scalar(select(ReportJob).where(
                                ReportJob.search_task_id == task.task_id).limit(1))
                            if existing_job is None:
                                db.add(ReportJob(search_task_id=task.task_id,
                                                 search_revision=task.search_revision, status="QUEUED"))
                                db.commit()
                    except DemoError as e:
                        db.rollback()
                        task.status = "FAILED"
                        task.error = f"{e.code}: {e.message}"
                        db.commit()
                    except Exception as e:  # noqa: BLE001
                        db.rollback()
                        task.status = "FAILED"
                        task.error = str(e)
                        db.commit()
            finally:
                db.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("search worker error: %s", e)


async def _report_worker_loop(state):
    from person_demo.db import new_session
    from person_demo.reports import ReportJobProcessor

    processor = ReportJobProcessor(gpu_lock=state.gpu_lock)
    while True:
        try:
            await asyncio.sleep(1)
            db = new_session()
            try:
                job = db.scalar(select(ReportJob).where(ReportJob.status == "QUEUED")
                                .order_by(ReportJob.created_at.asc()).limit(1))
                if job is not None:
                    def report():
                        with state.gpu_lock:
                            return processor.process(job.report_job_id)

                    await asyncio.to_thread(report)
            finally:
                db.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("report worker error: %s", e)
