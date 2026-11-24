"""人工复核标签 → 提交 review → 计算独立评测 Recall。

用法：
1. 查看 docs/validation/eval_review/qNN/ 下的 montage.jpg（QUERY + R0..R4 候选）
2. 编辑 docs/validation/eval_review/labels.json（模板自动生成），把与查询同一人的
   候选 rank 填入 "same_person_ranks"；无同人则空数组
3. 运行: python scripts/apply_reviews.py
   脚本对 same_person_ranks 执行 confirm、对其余 Top5 执行 reject（expected_revision 校验），
   并输出 Recall@1/5/20 与 INSUFFICIENT_POSITIVE_GALLERY 报告。
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

API = "http://127.0.0.1:8010"
BASE = Path("docs/validation/eval_review")


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path) as r:
        return json.load(r)


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def load_meta(idx: int) -> dict:
    return json.loads((BASE / f"q{idx:02d}" / "meta.json").read_text(encoding="utf-8"))


def main() -> None:
    labels_path = BASE / "labels.json"
    metas = []
    for qdir in sorted(BASE.glob("q*")):
        if qdir.is_dir():
            metas.append(json.loads((qdir / "meta.json").read_text(encoding="utf-8")))
    metas.sort(key=lambda m: m["index"])

    if not labels_path.exists():
        template = {
            m["index"]: {
                "note": f"track {m.get('candidates') and '' or ''}",
                "same_person_ranks": [],
                "reviewed": False,
            }
            for m in metas
        }
        labels_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"标签模板已生成: {labels_path}（人工填写 same_person_ranks 后重跑本脚本）")
        return

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    stats = []
    for m in metas:
        idx = m["index"]
        cands = m["candidates"]
        entry = labels.get(str(idx), {})
        task_id = m["task_id"]
        if entry.get("reviewed") is not True:
            stats.append({"query": idx, "positive_count": None,
                          "evaluation_status": "AWAITING_HUMAN_REVIEW", "recall": None})
            continue
        same = set(entry.get("same_person_ranks", []))
        task = get(f"/v1/search-tasks/{task_id}")
        snapshot = task.get("snapshot") or {}
        if (task.get("evaluation_scope") != "intra_camera_holdout"
                or not (snapshot.get("query_origin") or {}).get("origin_exclusion_verified")):
            stats.append({"query": idx, "positive_count": None,
                          "evaluation_status": "ORIGIN_UNRESOLVED", "recall": None})
            continue
        gallery_count = entry.get("independent_gallery_positive_count")
        for candidate in cands:
            if candidate["rank"] >= 5:
                continue
            want = "confirm" if candidate["rank"] in same else "reject"
            try:
                post(f"/v1/search-tasks/{task_id}/reviews",
                     {"hit_id": candidate["hit_id"], "action": want,
                      "expected_revision": candidate["review_revision"]})
            except urllib.error.HTTPError as exc:
                body = json.loads(exc.read() or b"{}")
                print(f"q{idx:02d} rank{candidate['rank']} review failed: {exc.code} {body.get('code')}")
        # Only five candidate images were provisioned. Full-gallery ground truth and top-20
        # labels are needed before interpreting a missing match as an algorithm miss.
        stats.append({"query": idx, "positive_count": gallery_count,
                      "positive_in_reviewed_top5": len(same),
                      "evaluation_status": "INSUFFICIENT_POSITIVE_GALLERY",
                      "recall": None})
        time.sleep(0.2)

    evaluated = [s for s in stats if s["recall"] is not None]
    insufficient = [s for s in stats if s["evaluation_status"] == "INSUFFICIENT_POSITIVE_GALLERY"]
    summary = {
        "total_queries": len(stats),
        "evaluated_with_positive": len(evaluated),
        "insufficient_positive_gallery": len(insufficient),
        "recall_at_1": (sum(s["recall"]["recall@1"] for s in evaluated) / len(evaluated)) if evaluated else None,
        "recall_at_5": (sum(s["recall"]["recall@5"] for s in evaluated) / len(evaluated)) if evaluated else None,
        "recall_at_20": (sum(s["recall"]["recall@20"] for s in evaluated) / len(evaluated)) if evaluated else None,
        "queries": stats,
        "note": "正样本仅由人工视觉确认（labels.json）；无正样本查询不计入 Recall 分母。",
    }
    (BASE / "evaluation_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
