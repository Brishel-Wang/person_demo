"""集成测试：导入幂等、outbox/水位、检索排除与重排、复核 409、报告校验。"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from sqlalchemy import select

from person_demo.db.models import (
    IngestRun,
    Observation,
    PublishWatermark,
    SearchCandidate,
    SearchTask,
    SourceAsset,
    Tracklet,
    VectorOutbox,
)
from person_demo.db.repo import advance_watermark, find_ingest_run_same_config
from person_demo.retrieval.rerank import rerank_one
from person_demo.reports.validate import parse_and_validate_model_output


def _make_asset_and_run(db_session, window=(0, 200_000_000)):
    from person_demo.db.models import Camera

    cam = Camera(dataset_id="single_camera_demo_v1", camera_code="cam_01")
    db_session.add(cam)
    db_session.flush()
    asset = SourceAsset(
        dataset_id="single_camera_demo_v1", camera_id=cam.camera_id,
        filename="cam1.mp4", local_path="/nonexistent/cam1.mp4",
        sha256="f" * 64, size_bytes=1, pts_anchor=0, time_base_num=1, time_base_den=60,
        timebase_status="ok", resolved_window_start_us=window[0], resolved_window_end_us=window[1],
    )
    db_session.add(asset)
    db_session.flush()
    run = IngestRun(
        dataset_id="single_camera_demo_v1", source_asset_id=asset.source_asset_id,
        window_start_us=window[0], window_end_us=window[1], analysis_fps=10,
        detector_sha256="d" * 64, feature_space_id="s" * 64, status="ready",
    )
    db_session.add(run)
    db_session.commit()
    return asset, run


def _make_tracklet(db_session, run, tid, chunk_no, start, end, group=None, status="committed"):
    t = Tracklet(
        ingest_run_id=run.ingest_run_id, dataset_id=run.dataset_id, camera_code="cam_01",
        source_asset_id=run.source_asset_id, local_track_id=tid, chunk_no=chunk_no,
        t_start_local_us=start, t_end_local_us=end, feature_space_id=run.feature_space_id,
        status=status, continuation_group_id=group,
    )
    db_session.add(t)
    db_session.flush()
    return t


def test_idempotent_ingest_run_reuse(db_session):
    asset, run = _make_asset_and_run(db_session)
    found = find_ingest_run_same_config(
        db_session, asset.source_asset_id, 0, 200_000_000, 10, "d" * 64, "s" * 64)
    assert found is not None and found.ingest_run_id == run.ingest_run_id


def test_ingest_run_reuse_distinguishes_detector_input_size(db_session):
    asset, baseline = _make_asset_and_run(db_session)
    args = (db_session, asset.source_asset_id, 0, 200_000_000, 10, "d" * 64, "s" * 64)
    assert baseline.detector_imgsz == 640
    assert find_ingest_run_same_config(*args, detector_imgsz=1920) is None

    upgraded = IngestRun(
        dataset_id=baseline.dataset_id, source_asset_id=asset.source_asset_id,
        window_start_us=0, window_end_us=200_000_000, analysis_fps=10,
        detector_sha256=baseline.detector_sha256, detector_imgsz=1920,
        feature_space_id=baseline.feature_space_id, status="ready",
    )
    db_session.add(upgraded)
    db_session.commit()
    assert find_ingest_run_same_config(*args).ingest_run_id == baseline.ingest_run_id
    assert find_ingest_run_same_config(*args, detector_imgsz=1920).ingest_run_id == upgraded.ingest_run_id


def test_outbox_watermark_contiguous(db_session):
    asset, run = _make_asset_and_run(db_session)
    t1 = _make_tracklet(db_session, run, 1, 0, 0, 10_000_000)
    t2 = _make_tracklet(db_session, run, 2, 0, 10_000_000, 20_000_000)
    db_session.add_all([
        VectorOutbox(dataset_id="single_camera_demo_v1", source_asset_id=asset.source_asset_id,
                     ingest_run_id=run.ingest_run_id, tracklet_revision_id=t1.tracklet_revision_id,
                     feature_space_id="s" * 64, publish_seq=0, status="ready"),
        VectorOutbox(dataset_id="single_camera_demo_v1", source_asset_id=asset.source_asset_id,
                     ingest_run_id=run.ingest_run_id, tracklet_revision_id=t2.tracklet_revision_id,
                     feature_space_id="s" * 64, publish_seq=1, status="pending"),
    ])
    db_session.commit()
    wm = advance_watermark(db_session, "single_camera_demo_v1", asset.source_asset_id,
                           run.ingest_run_id, "s" * 64)
    assert wm == 0  # 不跨空洞
    row = db_session.scalar(select(VectorOutbox).where(
        VectorOutbox.tracklet_revision_id == t2.tracklet_revision_id))
    row.status = "ready"
    db_session.commit()
    wm = advance_watermark(db_session, "single_camera_demo_v1", asset.source_asset_id,
                           run.ingest_run_id, "s" * 64)
    assert wm == 1


def test_rerank_from_artifacts(storage):
    rid = str(uuid.uuid4())
    vecs = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32)
    # 512 维
    vecs = np.zeros((2, 512), dtype=np.float32)
    vecs[0, 0] = 1.0
    vecs[1, 1] = 1.0
    np.save(storage.resolve(storage.vectors_key(rid)), vecs)
    rows = [
        {"tracklet_revision_id": rid, "order": 0, "local_us": 0, "frame_pts": 0,
         "bbox": [0, 0, 10, 10], "frame_w": 100, "frame_h": 100, "crop_key": "x", "laplacian_var": 1.0},
        {"tracklet_revision_id": rid, "order": 1, "local_us": 1_000_000, "frame_pts": 60,
         "bbox": [0, 0, 10, 10], "frame_w": 100, "frame_h": 100, "crop_key": "y", "laplacian_var": 2.0},
    ]
    storage.write_text(storage.row_index_key(rid), "\n".join(json.dumps(r) for r in rows))
    q = np.zeros(512, dtype=np.float32)
    q[0] = 1.0
    score, status, note = rerank_one(q, storage, rid)
    assert status == "ok"
    assert abs(score - 0.5) < 1e-6  # 两张 (1.0, 0.0) → top2 均值 0.5

    # 单张
    vecs1 = np.zeros((1, 512), dtype=np.float32)
    vecs1[0, 0] = 1.0
    np.save(storage.resolve(storage.vectors_key(rid)), vecs1)
    rows1 = [rows[0]]
    storage.write_text(storage.row_index_key(rid), "\n".join(json.dumps(r) for r in rows1))
    score, status, note = rerank_one(q, storage, rid)
    assert status == "ok" and note == "single_representative_used" and abs(score - 1.0) < 1e-6

    # 损坏
    (storage.resolve(storage.vectors_key(rid))).unlink()
    score, status, note = rerank_one(q, storage, rid)
    assert status == "rerank_failed" and score is None


def test_review_conflict_409(client, db_session):
    """通过 API：expected_revision 冲突返回 409。"""
    asset, run = _make_asset_and_run(db_session)
    t = _make_tracklet(db_session, run, 1, 0, 0, 10_000_000)
    db_session.add(Observation(tracklet_revision_id=t.tracklet_revision_id,
                               source_asset_id=asset.source_asset_id, local_us=0, frame_pts=0,
                               time_base_num=1, time_base_den=60, bbox_x1=0, bbox_y1=0,
                               bbox_x2=10, bbox_y2=20, frame_w=100, frame_h=100))
    from person_demo.db.models import QueryAsset

    qa = QueryAsset(dataset_id="single_camera_demo_v1", kind="from_frame",
                    origin_status="verified")
    db_session.add(qa)
    db_session.flush()
    task = SearchTask(dataset_id="single_camera_demo_v1", query_asset_id=qa.asset_id,
                      scope_windows={"asset_windows": []}, status="COMPLETED")
    db_session.add(task)
    db_session.flush()
    cand = SearchCandidate(search_task_id=task.task_id, tracklet_revision_id=t.tracklet_revision_id,
                           prototype_similarity=0.5, review_status="pending_review", review_revision=0)
    db_session.add(cand)
    db_session.commit()

    r = client.post(f"/v1/search-tasks/{task.task_id}/reviews",
                    json={"hit_id": str(cand.hit_id), "action": "confirm", "expected_revision": 1})
    assert r.status_code == 409
    assert r.json()["code"] == "REVISION_CONFLICT"
    r = client.post(f"/v1/search-tasks/{task.task_id}/reviews",
                    json={"hit_id": str(cand.hit_id), "action": "confirm", "expected_revision": 0})
    assert r.status_code == 200
    assert r.json()["revision"] == 1
    # 再次用旧 revision → 409
    r = client.post(f"/v1/search-tasks/{task.task_id}/reviews",
                    json={"hit_id": str(cand.hit_id), "action": "reject", "expected_revision": 0})
    assert r.status_code == 409


def test_report_validation_whitelist_and_kinds():
    bundle = {"evidences": [
        {"evidence_id": "E1", "review_status": "confirmed", "t_start_local_us": 0,
         "t_end_local_us": 1000, "rerank_similarity": 0.8, "same_continuation_as_query": False},
        {"evidence_id": "E2", "review_status": "pending_review", "t_start_local_us": 2000,
         "t_end_local_us": 3000, "rerank_similarity": 0.6, "same_continuation_as_query": True},
    ]}
    good = {
        "schema_version": "1.6.1", "mode": "single_video_summary",
        "narrative": "x", "claims": [
            {"kind": "reviewed_observation", "text": "确认", "evidence_ids": ["E1"], "hypothesis_id": None},
            {"kind": "candidate_observation", "text": "候选", "evidence_ids": ["E2"], "hypothesis_id": None},
        ], "gaps": ["g"], "hypotheses": [], "next_information_needed": ["n"],
    }
    content, errors = parse_and_validate_model_output(json.dumps(good), bundle)
    assert content is not None and not errors

    bad_kind = dict(good)
    bad_kind["claims"] = [{"kind": "reviewed_observation", "text": "x",
                           "evidence_ids": ["E2"], "hypothesis_id": None}]  # pending 被写成确认
    content, errors = parse_and_validate_model_output(json.dumps(bad_kind), bundle)
    assert content is None and any("non-confirmed" in e for e in errors)

    bad_ref = dict(good)
    bad_ref["claims"] = [{"kind": "candidate_observation", "text": "x",
                          "evidence_ids": ["E999"], "hypothesis_id": None}]
    content, errors = parse_and_validate_model_output(json.dumps(bad_ref), bundle)
    assert content is None and any("unknown evidence_id" in e for e in errors)

    bad_json = '{"schema_version": "1.6.1", 不完整'
    content, errors = parse_and_validate_model_output(bad_json, bundle)
    assert content is None and any("invalid_json" in e for e in errors)


@pytest.fixture()
def client(db_session):
    from fastapi.testclient import TestClient

    from person_demo.api.app import create_app

    app = create_app()
    # 覆盖依赖以使用测试会话
    from person_demo.api.deps import get_db as api_get_db

    async def _ovr():
        yield db_session

    app.dependency_overrides[api_get_db] = _ovr
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_recall_excludes_sql_rejected_rows_and_counts_shared_bucket_hits(db_session, storage, monkeypatch):
    """A SQL-rejected high-scoring hit must never occupy a bucket slot or become a candidate."""
    import person_demo.retrieval as retrieval
    from person_demo.db.models import QueryAsset

    asset, run = _make_asset_and_run(db_session)
    tracks = [_make_tracklet(db_session, run, i, 0, 0, 61_000_000) for i in range(3)]
    rejected, shared, tail = [str(t.tracklet_revision_id) for t in tracks]
    qa = QueryAsset(dataset_id=asset.dataset_id, kind="upload", origin_status="unknown")
    db_session.add(qa)
    db_session.flush()
    task = SearchTask(dataset_id=asset.dataset_id, query_asset_id=qa.asset_id,
                      evaluation_scope="intra_camera_demo",
                      scope_windows={"asset_windows": [{"source_asset_id": str(asset.source_asset_id),
                                                       "t_start_local_us": 0, "t_end_local_us": 120_000_000}]},
                      frozen_ingest_run_id=run.ingest_run_id, frozen_feature_space_id=run.feature_space_id,
                      frozen_watermark=0, frozen_coverage={"status": "full"},
                      snapshot={"source_asset_id": str(asset.source_asset_id),
                                "query_origin": {"origin_status": "unknown"}})
    db_session.add(task)
    db_session.commit()
    monkeypatch.setattr(retrieval.get_settings(), "per_bucket_top_k", 1)
    monkeypatch.setattr(retrieval.get_settings(), "backfill_limits", "1,2,4")
    monkeypatch.setattr(retrieval, "_load_query_vector", lambda *_: np.ones(512, dtype=np.float32))
    monkeypatch.setattr(retrieval, "_sql_recheck",
                        lambda _, hit, *_args: None if hit["tracklet_revision_id"] == rejected
                        else {"continuation_group_id": None})
    monkeypatch.setattr(retrieval, "rerank_many",
                        lambda query, files, candidates: {c["tracklet_revision_id"]:
                            {"similarity": 0.8, "status": "ok", "note": None} for c in candidates})

    class Store:
        def search(self, space, vec, expression, limit, time_expression):
            assert "publish_seq <= 0" in expression
            return [{"tracklet_revision_id": rid, "prototype_similarity": 0.9 - i * 0.1, "sample_count": 1}
                    for i, rid in enumerate([rejected, shared, tail][:limit])]

    assert retrieval.run_search(db_session, task.task_id, Store(), storage)["status"] == "COMPLETED"
    saved = db_session.scalars(select(SearchCandidate).where(SearchCandidate.search_task_id == task.task_id)).all()
    assert [(str(c.tracklet_revision_id), c.matched_buckets) for c in saved] == [(shared, [0, 1])]
    assert db_session.get(SearchTask, task.task_id).snapshot["result"]["valid_bucket_hits"] == 2


def test_live_report_cannot_claim_a_pending_candidate_is_the_target():
    from person_demo.reports.validate import report_wording

    bundle = {"data_origin": "real_video_processing_result", "total_candidates": 1, "confirmed_count": 0,
              "coverage_status": "full", "evidences": [{"evidence_id": "ev-pending",
              "review_status": "pending_review", "same_continuation_as_query": True,
              "t_start_local_us": 1}]}
    narrative, gaps = report_wording(bundle)
    claim = {"kind": "candidate_observation", "text": "检测到目标，已完成独立重识别。",
             "evidence_ids": ["ev-pending"], "hypothesis_id": None}
    obj = {"schema_version": "1.6.1", "mode": "single_video_summary", "narrative": narrative,
           "claims": [claim], "gaps": gaps, "hypotheses": [], "next_information_needed": []}
    assert parse_and_validate_model_output(json.dumps(obj), bundle)[0] is None
    claim["text"] = "这是相似候选，待人工复核。该片段属于同一跟踪段的后续片段。"
    assert parse_and_validate_model_output(json.dumps(obj), bundle)[1] == []
