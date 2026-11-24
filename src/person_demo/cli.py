"""`project` CLI（spec 8.2 命令契约）。"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from person_demo.config import PROJECT_ROOT, get_settings


def _storage():
    from person_demo.storage import StorageLayout

    return StorageLayout(get_settings().storage_root)


def cmd_doctor(args) -> int:
    """环境、依赖、数据库、模型、特征空间与可写目录检查。"""
    import platform
    import subprocess

    out: dict = {"profile": args.profile, "checks": {}}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        out["gpus"] = [dict(zip(["index", "name", "total_mb", "used_mb", "free_mb"], line.split(", ")))
                       for line in gpu.stdout.strip().splitlines() if line]
        out["cuda_driver"] = gpu.stdout.strip().splitlines() or "nvidia-smi failed"
    except Exception as e:  # noqa: BLE001
        out["gpus"] = f"nvidia-smi error: {e}"

    import psutil

    out["host"] = {
        "platform": platform.platform(), "python": sys.version.split()[0],
        "cpu_count": psutil.cpu_count(), "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }

    def _pkg(name):
        try:
            m = __import__(name)
            return getattr(m, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            return f"MISSING ({e})"

    out["packages"] = {n: _pkg(n) for n in [
        "torch", "torchvision", "ultralytics", "cv2", "av", "numpy",
        "fastapi", "sqlalchemy", "alembic", "pymilvus", "transformers", "jsonschema"]}

    s = get_settings()
    try:
        from person_demo.db import new_session

        db = new_session()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
        out["checks"]["postgres"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["checks"]["postgres"] = f"FAIL: {e}"

    try:
        from person_demo.vectors import MilvusStore

        store = MilvusStore(s.milvus_uri, s.milvus_database, s.milvus_token)
        out["checks"]["milvus"] = "ok" if store.health() else "FAIL: unreachable"
    except Exception as e:  # noqa: BLE001
        out["checks"]["milvus"] = f"FAIL: {e}"

    # 模型文件
    out["models"] = {}
    for name, p in [
        (s.yolo_weights, s.model_dir / s.yolo_weights),
        ("osnet_checkpoint", s.model_dir / s.osnet_checkpoint),
        ("qwen_dir", s.model_dir / s.qwen_dir),
    ]:
        if p.exists():
            out["models"][name] = {"path": str(p), "status": "present"}
        else:
            out["models"][name] = {"path": str(p), "status": "MISSING"}

    # 特征空间（若权重在）
    try:
        from person_demo.feature_space import canonical_space_json, feature_space_id
        from person_demo.vision.reid.encoder import OSNetEncoder

        enc = OSNetEncoder(s.model_dir / s.osnet_checkpoint, device="cpu")
        space_id = feature_space_id(canonical_space_json(
            architecture="osnet_ain_x1_0", checkpoint_sha256=enc.checkpoint_sha256, source_revision=None))
        out["feature_space"] = {"space_id": space_id, "dim": enc.dim, "probe": enc.probe_info}
    except Exception as e:  # noqa: BLE001
        out["feature_space"] = f"not available yet: {e}"

    out["storage"] = {"root": str(s.storage_root), "writable": _check_writable(s.storage_root)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _check_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        t = path / ".wtest"
        t.write_text("x")
        t.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


def cmd_probe(args) -> int:
    """probe 指定 manifest 的本地文件并写 resolved manifest。"""
    from person_demo.media.probe import probe_file

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cameras = manifest.get("cameras", [])
    if len(cameras) != 1:
        print("SINGLE_SOURCE_REQUIRED: manifest must contain exactly one camera", file=sys.stderr)
        return 1
    cam = cameras[0]
    if Path(cam["local_path"]).resolve() != Path(get_settings().default_video_path).resolve():
        print("INVALID_MANIFEST: manifest must use the configured local video", file=sys.stderr)
        return 1
    win = manifest.get("requested_window_us") or {"start": 0, "end": 200_000_000}
    r = probe_file(cam["local_path"], (win["start"], win["end"]))
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    if r.timebase_status != "ok" or not r.sha256 or not r.resolved_window_us:
        print(f"PROBE_FAILED: {r.error}", file=sys.stderr)
        return 1
    if ((cam.get("sha256") and cam["sha256"] != r.sha256)
            or (cam.get("metadata_expected_bytes") is not None and cam["metadata_expected_bytes"] != r.size_bytes)):
        print("PROBE_FAILED: video checksum or size differs from manifest", file=sys.stderr)
        return 1

    cam["sha256"] = r.sha256
    cam["probe"] = {
        "codec": r.codec_name, "width": r.width, "height": r.height,
        "avg_fps": r.avg_fps, "is_vfr": r.is_vfr, "duration_us": r.duration_us,
        "nb_frames": r.nb_frames, "timebase_status": r.timebase_status,
    }
    cam["timebase"] = {
        "pts_anchor": r.pts_anchor, "time_base_num": r.time_base_num,
        "time_base_den": r.time_base_den, "status": "ok" if r.timebase_status == "ok" else r.timebase_status,
    }
    cam["resolved_window_us"] = {"start": r.resolved_window_us[0], "end": r.resolved_window_us[1],
                                 "note": r.window_note} if r.resolved_window_us else None
    if r.resolved_window_us and r.timebase_status == "ok":
        manifest["status"] = "ingest_ready"
    else:
        manifest["status"] = "draft"
    manifest["verification_level"] = "server_local_probe_verified"
    out_path = Path(args.out) if getattr(args, "out", None) else Path(args.manifest).with_name(
        Path(args.manifest).stem + ".resolved.json")
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"resolved manifest written: {out_path}")
    return 0


def cmd_ingest(args) -> int:
    """提交/执行入库。--window-us 0:200000000。"""
    from person_demo.db import new_session
    from person_demo.db.models import IngestRun, SourceAsset
    from person_demo.db.repo import find_ingest_run_same_config
    from person_demo.feature_space import canonical_space_json, feature_space_id
    from person_demo.storage import StorageLayout
    from person_demo.vision.ingest_worker import DENSE_PROFILE, process_ingest_run, tracker_config
    from sqlalchemy import select

    if args.vision_profile == DENSE_PROFILE:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    start, end = (int(x) for x in args.window_us.split(":"))
    db = new_session()
    asset = db.scalar(select(SourceAsset).where(SourceAsset.dataset_id == args.dataset)
                      .order_by(SourceAsset.created_at.desc()).limit(1))
    if asset is None:
        # 未注册素材时按 manifest 默认路径 probe 并注册（跳过 fetch）
        from pathlib import Path as _Path

        from person_demo.db.models import Camera
        from person_demo.media.probe import probe_file

        path = _Path(s.default_video_path)
        probe = probe_file(str(path), (start, end))
        if probe.timebase_status != "ok" or not probe.resolved_window_us or not probe.sha256:
            print(f"probe failed: {probe.error}", file=sys.stderr)
            return 1
        camera = db.scalar(select(Camera).where(
            Camera.dataset_id == args.dataset, Camera.camera_code == s.camera_code))
        if camera is None:
            camera = Camera(dataset_id=args.dataset, camera_code=s.camera_code)
            db.add(camera)
            db.flush()
        asset = SourceAsset(
            dataset_id=args.dataset, camera_id=camera.camera_id,
            filename=path.name, local_path=str(path),
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
        db.commit()
    space_id = feature_space_id(canonical_space_json(
        architecture="osnet_ain_x1_0", checkpoint_sha256=_sha(s.model_dir / s.osnet_checkpoint), source_revision=None))
    det_sha = _sha(s.model_dir / s.yolo_weights)
    if asset.duration_us is None or start < 0 or start >= asset.duration_us or end <= start:
        print("invalid ingest window", file=sys.stderr)
        return 1
    end = min(end, asset.duration_us)
    # 处理窗口超出已 resolved 窗口时扩展（以 probe 时长为限）
    if asset.duration_us is not None and end > (asset.resolved_window_end_us or 0) and end <= asset.duration_us:
        asset.resolved_window_end_us = end
        db.commit()
    # 幂等：同配置已有 run 则复用（失败可重试，不重复创建）
    run = find_ingest_run_same_config(
        db, asset.source_asset_id, start, end, s.analysis_fps, det_sha, space_id, args.vision_profile,
        detector_imgsz=s.yolo_imgsz,
    )
    initial_status = "indexing_dense" if args.vision_profile == DENSE_PROFILE else "pending"
    if run is None:
        run = IngestRun(
            dataset_id=args.dataset, source_asset_id=asset.source_asset_id,
            window_start_us=start, window_end_us=end, analysis_fps=s.analysis_fps,
            detector_sha256=det_sha, detector_imgsz=s.yolo_imgsz, feature_space_id=space_id,
            tracker_config={"implementation": "bytetrack_local", "detector_weights": s.yolo_weights,
                            **tracker_config(s, args.vision_profile)},
            vision_profile=args.vision_profile,
            status=initial_status,
        )
        db.add(run)
        db.commit()
    elif run.status in ("failed",):
        run.status = initial_status
        run.error = None
        db.commit()
    run_id = run.ingest_run_id
    db.close()
    storage = _storage()
    print(f"ingest run {run_id} ({args.vision_profile}, {start}:{end})", flush=True)
    if args.artifact_staging_dir:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="person-demo-ingest-", dir=args.artifact_staging_dir) as tmp:
            result = process_ingest_run(run_id, storage=storage, staged_storage=StorageLayout(Path(tmp)))
    else:
        result = process_ingest_run(run_id, storage=storage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result["status"] == "failed" else 0


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def cmd_reconcile(args) -> int:
    """index reconcile：发布 pending outbox 并推进水位。"""
    from person_demo.db import new_session
    from person_demo.vectors import MilvusStore
    from person_demo.vectors.publisher import publish_loop_once

    s = get_settings()
    store = MilvusStore(s.milvus_uri, s.milvus_database, s.milvus_token)
    import time

    total = {"published": 0, "ready": 0}
    consecutive_failures = 0
    for _ in range(2000):
        db = new_session()
        try:
            res = publish_loop_once(store, db)
            if res.get("scopes", 0) == 0 and "error" not in res:
                from person_demo.api.app import _mark_runs_ready

                _mark_runs_ready(db)
                print("reconcile: no pending outbox; final:", json.dumps(total, default=str))
                return 0
            for k in ("published", "ready"):
                total[k] += res.get(k, 0)
            if res.get("error") or res.get("busy"):
                consecutive_failures += 1
                print("reconcile batch not completed:", res.get("error", "publisher busy"))
                if consecutive_failures >= 3:
                    return 1
                time.sleep(1)
            else:
                consecutive_failures = 0
                print(json.dumps(res, default=str))
        finally:
            db.close()
    return 1


def cmd_evaluate(args) -> int:
    """独立评测（intra_camera_holdout）套件；无独立正样本时如实报告。"""
    from sqlalchemy import select

    from person_demo.db import new_session
    from person_demo.db.models import SearchTask

    with new_session() as db:
        tasks = db.scalars(select(SearchTask).where(SearchTask.evaluation_scope == "intra_camera_holdout")
                           .order_by(SearchTask.created_at.desc())).all()
    report = {"suite": args.suite, "queries": []}
    for task in tasks:
        origin = (task.snapshot or {}).get("query_origin", {})
        report["queries"].append({
            "task_id": str(task.task_id),
            "origin_status": origin.get("origin_status"),
            "positive_count": None,
            "recall": None,
            "evaluation_status": "INSUFFICIENT_POSITIVE_GALLERY",
            "note": "No independently verified gallery labels supplied",
        })
    if not tasks:
        report["note"] = "no holdout search tasks run yet"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_report_doctor(args) -> int:
    """报告模型探针：权重清单、tokenizer、chat template、thinking 开关与生成冒烟。"""
    import torch

    from person_demo.reports.llm import QwenReportModel

    s = get_settings()
    model = QwenReportModel(s.model_dir / s.qwen_dir, device=s.device, precision=s.qwen_precision)
    out: dict = {"profile": args.profile, "model_id": s.qwen_model_id}
    try:
        model.load()
        out["loaded"] = True
        out["load_seconds"] = model.load_seconds
        out["peak_memory_mb"] = model.peak_memory_mb
        out["thinking_disabled_via_template"] = model.thinking_disabled_via_template
        out["chat_template_len"] = len(model.tokenizer.chat_template or "")
        text, timing = model.generate(
            [{"role": "user", "content": '{"probe": true} 请只输出 JSON: {"ok": true}'}],
            max_new_tokens=64)
        out["smoke_output"] = text[:200]
        out["smoke_timing"] = timing
        out["device"] = s.device
        out["torch_cuda_available"] = torch.cuda.is_available()
        if s.qwen_unload_after_job:
            out["unload"] = model.unload()
    except Exception as e:  # noqa: BLE001
        out["loaded"] = False
        out["error"] = str(e)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("loaded") else 1


def cmd_serve(args) -> int:
    import uvicorn

    s = get_settings()
    uvicorn.run("person_demo.api.app:create_app", factory=True,
                host=s.host, port=s.port, log_level="info")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="project", description="单路人体检索与大模型报告 Demo CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="环境与模型检查")
    d.add_argument("--profile", default="single_body_rag_demo_linux_milvus")
    d.set_defaults(fn=cmd_doctor)

    pr = sub.add_parser("dataset", help="dataset 子命令")
    pr_sub = pr.add_subparsers(dest="sub", required=True)
    pr_probe = pr_sub.add_parser("probe", help="本地文件 probe")
    pr_probe.add_argument("--manifest", required=True)
    pr_probe.add_argument("--out", default=None)
    pr_probe.set_defaults(fn=cmd_probe)

    ing = sub.add_parser("ingest", help="入库")
    ing.add_argument("--dataset", default="single_camera_demo_v1")
    ing.add_argument("--window-us", default="0:200000000")
    ing.add_argument("--vision-profile", choices=("legacy", "dense_appearance_v2"), default="legacy")
    ing.add_argument("--artifact-staging-dir", help="在本地目录暂存本次运行的证据文件，再批量写入配置的存储目录")
    ing.set_defaults(fn=cmd_ingest)

    rec = sub.add_parser("index", help="index 子命令")
    rec_sub = rec.add_subparsers(dest="sub", required=True)
    rec_reconcile = rec_sub.add_parser("reconcile", help="发布 outbox 并推进水位")
    rec_reconcile.add_argument("--dataset", default="single_camera_demo_v1")
    rec_reconcile.set_defaults(fn=cmd_reconcile)

    ev = sub.add_parser("evaluate", help="独立评测")
    ev.add_argument("--suite", default="single_camera_demo")
    ev.set_defaults(fn=cmd_evaluate)

    rd = sub.add_parser("report", help="report 子命令")
    rd_sub = rd.add_subparsers(dest="sub", required=True)
    rd_doctor = rd_sub.add_parser("doctor", help="报告模型探针")
    rd_doctor.add_argument("--profile", default="qwen_single_report_local_v1")
    rd_doctor.set_defaults(fn=cmd_report_doctor)

    sv = sub.add_parser("serve", help="启动 API + workers")
    sv.add_argument("--profile", default="single_body_rag_demo_linux_milvus")
    sv.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
