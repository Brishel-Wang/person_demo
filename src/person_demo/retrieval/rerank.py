"""代表向量重排（spec 5.3）。

从不可变特征产物读每候选最多 5 个归一化代表向量，计算 q 与每个向量的余弦，
取最高 min(2,n) 个的均值作为 rerank_similarity。只有一张时使用单样本分并提示。
向量产物损坏标 rerank_failed，不静默拿原型分冒充重排分。
"""
from __future__ import annotations

import json
import logging
from io import BytesIO

import numpy as np

from person_demo.storage import StorageLayout, bundle_id

log = logging.getLogger(__name__)


def rerank_one(
    query_vec: np.ndarray,
    storage: StorageLayout,
    tracklet_revision_id: str,
    *,
    vector_key: str | None = None,
    row_index_key: str | None = None,
) -> tuple[float | None, str, str | None]:
    """返回 (rerank_similarity, status, note)。status: ok / rerank_failed。"""
    vec_key = vector_key or StorageLayout.vectors_key(tracklet_revision_id)
    idx_key = row_index_key or StorageLayout.row_index_key(tracklet_revision_id)
    try:
        group = bundle_id(vec_key)
        if group is not None and bundle_id(idx_key) == group:
            packed = storage.read_bundle(group, [vec_key, idx_key])
            raw_vecs, raw_rows = packed[vec_key], packed[idx_key]
        else:
            raw_vecs, raw_rows = storage.read_bytes(vec_key), storage.read_bytes(idx_key)
        vecs = np.load(BytesIO(raw_vecs), allow_pickle=False)
        rows = [json.loads(line) for line in raw_rows.decode("utf-8").splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001
        return None, "rerank_failed", f"artifact read error: {e}"

    if vecs.ndim != 2 or vecs.shape[1] != 512 or len(vecs) == 0:
        return None, "rerank_failed", f"bad vector shape {getattr(vecs, 'shape', None)}"
    if len(rows) != len(vecs):
        return None, "rerank_failed", f"row index mismatch: {len(rows)} rows vs {len(vecs)} vectors"
    if not np.isfinite(vecs).all():
        return None, "rerank_failed", "non-finite vectors"

    q = query_vec.astype(np.float32)
    qn = np.linalg.norm(q)
    if qn == 0 or not np.isfinite(q).all():
        return None, "rerank_failed", "invalid query vector"
    q = q / qn
    sims = vecs @ q  # 图库向量入库时已 L2 归一化
    sims = np.clip(sims, -1.0, 1.0)
    n = len(sims)
    k = min(2, n)
    top = np.sort(sims)[::-1][:k]
    score = float(top.mean())
    note = None
    if n == 1:
        note = "single_representative_used"
    return score, "ok", note


def rerank_many(
    query_vec: np.ndarray,
    storage: StorageLayout,
    candidates: list[dict],
) -> dict[str, dict]:
    """按候选登记的向量和行索引重排。"""
    out: dict[str, dict] = {}
    for c in candidates:
        rid = c["tracklet_revision_id"]
        score, status, note = rerank_one(
            query_vec, storage, rid, vector_key=c.get("vector_key"), row_index_key=c.get("row_index_key"))
        out[rid] = {"similarity": score, "status": status, "note": note}
    return out
