"""单元测试：feature_space_id、embedding_id、来源排除、代表图。"""
import uuid

import numpy as np

from person_demo.feature_space import (
    canonical_space_json,
    collection_name,
    embedding_id,
    feature_space_id,
    verify_space_prefix,
)
from person_demo.retrieval.exclude import evaluate_exclusion
from person_demo.vision.representatives import CropCandidate, select_representatives


def test_space_id_stable_and_prefix():
    c = canonical_space_json(
        architecture="osnet_ain_x1_0", checkpoint_sha256="abc123", source_revision=None
    )
    sid = feature_space_id(c)
    assert len(sid) == 64
    assert collection_name(sid) == f"body_{sid[:12]}_ar1"
    assert verify_space_prefix(sid, sid[:12])


def test_space_id_changes_with_checkpoint():
    c1 = canonical_space_json(architecture="osnet_ain_x1_0", checkpoint_sha256="aaa", source_revision=None)
    c2 = canonical_space_json(architecture="osnet_ain_x1_0", checkpoint_sha256="bbb", source_revision=None)
    assert feature_space_id(c1) != feature_space_id(c2)


def test_embedding_id_contract():
    eid = embedding_id(
        dataset_id="single_camera_demo_v1", ingest_run_id="run1",
        tracklet_revision_id="t1", space_id="s" * 64,
    )
    assert len(eid) == 64
    # 顺序敏感
    eid2 = embedding_id(
        dataset_id="single_camera_demo_v1", ingest_run_id="t1",
        tracklet_revision_id="run1", space_id="s" * 64,
    )
    assert eid != eid2


# ---------------- 来源排除 ----------------
def test_demo_excludes_origin_chunk_keeps_continuation():
    origin_rid = uuid.uuid4()
    group = uuid.uuid4()
    other = uuid.uuid4()
    r1 = evaluate_exclusion(
        evaluation_scope="intra_camera_demo", origin_tracklet_revision_id=origin_rid,
        origin_continuation_group_id=group, origin_status="verified",
        candidate_revision_id=origin_rid, candidate_group_id=group)
    assert not r1.keep and r1.exclusion_verified
    r2 = evaluate_exclusion(
        evaluation_scope="intra_camera_demo", origin_tracklet_revision_id=origin_rid,
        origin_continuation_group_id=group, origin_status="verified",
        candidate_revision_id=other, candidate_group_id=group)
    assert r2.keep and r2.same_continuation_as_query
    r3 = evaluate_exclusion(
        evaluation_scope="intra_camera_demo", origin_tracklet_revision_id=origin_rid,
        origin_continuation_group_id=group, origin_status="verified",
        candidate_revision_id=uuid.uuid4(), candidate_group_id=uuid.uuid4())
    assert r3.keep and not r3.same_continuation_as_query


def test_holdout_excludes_whole_group():
    origin_rid = uuid.uuid4()
    group = uuid.uuid4()
    r = evaluate_exclusion(
        evaluation_scope="intra_camera_holdout", origin_tracklet_revision_id=origin_rid,
        origin_continuation_group_id=group, origin_status="verified",
        candidate_revision_id=uuid.uuid4(), candidate_group_id=group)
    assert not r.keep


def test_origin_unknown_not_verified():
    r = evaluate_exclusion(
        evaluation_scope="intra_camera_demo", origin_tracklet_revision_id=None,
        origin_continuation_group_id=None, origin_status="unknown",
        candidate_revision_id=uuid.uuid4(), candidate_group_id=None)
    assert r.keep and not r.exclusion_verified


# ---------------- 代表图 ----------------
def _mk_crop(local_us, var):
    img = np.full((300, 160, 3), 60, dtype=np.uint8)
    if var:
        img[::8, ::8] = 255
    c = CropCandidate(local_us=local_us, frame_pts=local_us // 100, bbox=(0, 0, 160, 300),
                      frame_w=1920, frame_h=1080, crop=img, status="valid")
    c.laplacian_var = float(var)
    return c


def test_select_representatives_slots_and_gap():
    cands = [_mk_crop(us, var=us % 7) for us in range(0, 30_000_000, 500_000)]  # 0..30s 每 0.5s
    res = select_representatives(cands, 0, 30_000_000, max_count=5, min_gap_us=500_000)
    assert len(res.selected) == 5
    us = [c.local_us for c in res.selected]
    assert all(b - a >= 500_000 for a, b in zip(us, us[1:]))
    # 时间分桶：每 6 秒槽恰一张
    bins = sorted(set(u // 6_000_000 for u in us))
    assert len(bins) == 5


def test_select_representatives_fewer_than_five():
    cands = [_mk_crop(us, var=3) for us in (1_000_000, 5_000_000)]
    res = select_representatives(cands, 0, 10_000_000, max_count=5)
    assert len(res.selected) == 2  # 不复制凑数


def test_select_representatives_rejects_invalid():
    bad = CropCandidate(local_us=1_000_000, frame_pts=1, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop=None, status="empty_box",
                        reason="empty or degenerate box")
    good = _mk_crop(2_000_000, var=5)
    res = select_representatives([bad, good], 0, 10_000_000, max_count=5)
    assert len(res.selected) == 1
    assert res.rejected and res.rejected[0]["reason"] == "empty or degenerate box"
