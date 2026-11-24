"""独立评测套件（spec 9.2）：15 个查询、holdout 口径、候选裁剪图导出供人工复核。

流程：选 15 个不同出现事件的查询 → intra_camera_holdout 检索 → 导出查询图与
Top5 候选代表图到 docs/validation/eval_review/ → 人工（或代理视觉复核）确认/排除
→ 计算 Recall@1/5/20；排除后无正样本的查询如实记录 INSUFFICIENT_POSITIVE_GALLERY。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

API = "http://127.0.0.1:8010"
ASSET = "6efee9f8-71db-439e-adb5-f649a0db054c"
DATASET = "single_camera_demo_v1"
DB = "postgresql://person_demo:person_demo@127.0.0.1:5433/person_demo"
WINDOW_END_US = 200_000_000


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path) as r:
        return json.load(r)


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def psql(sql: str) -> str:
    out = subprocess.run(
        ["psql", DB, "-t", "-A", "-c", sql], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def pick_queries(n: int = 15) -> list[dict]:
    """选 n 个查询 tracklet：chunk_no=0、代表图≥2、按时间均匀分布、不同 group。"""
    rows = psql(
        f"""SELECT DISTINCT ON (bucket) t.tracklet_revision_id, t.continuation_group_id,
                t.local_track_id, t.t_start_local_us, t.t_end_local_us, t.sample_count,
                (t.t_start_local_us / 13000000)::int AS bucket
            FROM tracklets t
            WHERE t.chunk_no = 0 AND t.sample_count >= 2 AND t.status = 'committed'
              AND t.t_end_local_us < {WINDOW_END_US}
            ORDER BY bucket, t.sample_count DESC, t.quality DESC NULLS LAST
            LIMIT {n};"""
    )
    out = []
    for line in rows.splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        out.append({
            "tracklet_revision_id": parts[0],
            "continuation_group_id": parts[1],
            "local_track_id": int(parts[2]),
            "t_start_local_us": int(parts[3]),
            "t_end_local_us": int(parts[4]),
            "sample_count": int(parts[5]),
        })
    return out


def run_one(idx: int, tracklet: dict, review_dir: Path) -> dict:
    obs = psql(
        f"""SELECT o.observation_id FROM observations o
            WHERE o.tracklet_revision_id = '{tracklet['tracklet_revision_id']}'
              AND o.is_representative ORDER BY o.local_us LIMIT 1;"""
    )
    if not obs:
        return {"error": "no representative observation"}
    qa = post("/v1/query-assets/from-frame", {"source_asset_id": ASSET, "observation_id": obs})
    task = post("/v1/search-tasks", {
        "dataset_id": DATASET, "mode": "body_only", "query_asset_id": qa["asset_id"],
        "evaluation_scope": "intra_camera_holdout",
        "scope": {"asset_windows": [{"source_asset_id": ASSET,
                                     "t_start_local_us": 0, "t_end_local_us": WINDOW_END_US}]},
    })
    tid = task["task_id"]
    for _ in range(60):
        st = get(f"/v1/search-tasks/{tid}")["status"]
        if st in ("COMPLETED", "PARTIAL", "FAILED"):
            break
        time.sleep(2)
    if st == "FAILED":
        return {"error": "search failed", "task_id": tid}
    res = get(f"/v1/search-tasks/{tid}/results?page=1&page_size=10")

    # 导出复核图
    qdir = review_dir / f"q{idx:02d}"
    qdir.mkdir(parents=True, exist_ok=True)
    _save_img(f"{qa['image_url']}", qdir / "query.jpg")
    items = res["items"]
    for rank, c in enumerate(items[:5]):
        imgs = c["representative_images"]
        if imgs:
            _save_img(imgs[0]["url"], qdir / f"rank{rank}.jpg")
    meta = {
        "index": idx,
        "query_asset_id": qa["asset_id"],
        "origin_tracklet_revision_id": qa["origin_tracklet_revision_id"],
        "origin_continuation_group_id": qa["origin_continuation_group_id"],
        "query_local_us": qa["local_us"],
        "task_id": tid,
        "search_status": st,
        "candidates": [
            {
                "rank": c["rank_position"],
                "tracklet_revision_id": c["tracklet_revision_id"],
                "t_start_local_us": c["t_start_local_us"],
                "t_end_local_us": c["t_end_local_us"],
                "prototype_similarity": c["prototype_similarity"],
                "rerank_similarity": c["rerank_similarity"],
                "same_continuation_as_query": c["same_continuation_as_query"],
                "hit_id": c["hit_id"],
                "review_revision": c["review_revision"],
            }
            for c in items
        ],
    }
    (qdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _save_img(url: str, path: Path) -> None:
    try:
        with urllib.request.urlopen(API + url) as r:
            path.write_bytes(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"  image fetch failed {url}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="docs/validation/eval_review")
    args = ap.parse_args()

    review_dir = Path(args.out)
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)

    tracklets = pick_queries(args.n)
    print(f"picked {len(tracklets)} query tracklets")
    metas = []
    for i, t in enumerate(tracklets):
        print(f"[{i:02d}] track {t['local_track_id']} t=[{t['t_start_local_us']/1e6:.1f}s,"
              f"{t['t_end_local_us']/1e6:.1f}s) reps={t['sample_count']}")
        m = run_one(i, t, review_dir)
        if "error" in m:
            print("   error:", m["error"])
        else:
            print(f"   status={m['search_status']} top1 sim={m['candidates'][0]['rerank_similarity'] if m['candidates'] else None}")
        metas.append(m)
    (review_dir / "queries.json").write_text(
        json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done. review crops at", review_dir)


if __name__ == "__main__":
    main()
