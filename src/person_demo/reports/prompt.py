"""报告输入构建（spec 7.1）：结构化文字证据 + 系统提示 + token 预算控制。

模型完整输入含 system 提示与 template 最多 6144 token；超限按确定性排名
移除证据并记 evidence_truncated/omitted_count，不裁断 JSON 或 ID。
"""
from __future__ import annotations

import json

from person_demo.config import get_settings
from person_demo.reports.validate import report_wording


def load_system_prompt() -> str:
    s = get_settings()
    path = s.report_system_prompt_file
    return path.read_text(encoding="utf-8").strip()


def evidence_to_prompt_item(ev: dict) -> dict:
    """把 bundle 内 evidence 转成模型输入字段（只给结构化文字，紧凑字段与数值）。

    片内时间用秒（保留 1 位小数）减小 token 消耗；模型只需引用 evidence_id。
    """
    def _s(us):
        return round(us / 1_000_000, 1) if isinstance(us, (int, float)) else None

    def _r(x):
        return round(x, 4) if isinstance(x, (int, float)) else None

    return {
        "evidence_id": ev["evidence_id"],
        "t_start_s": _s(ev.get("t_start_local_us")),
        "t_end_s": _s(ev.get("t_end_local_us")),
        "review_status": ev.get("review_status"),
        "proto_sim": _r(ev.get("prototype_similarity")),
        "rerank_sim": _r(ev.get("rerank_similarity")),
        "same_continuation_as_query": ev.get("same_continuation_as_query"),
    }


def build_bundle_prompt_payload(bundle: dict) -> dict:
    payload = {
        "schema_version": bundle.get("schema_version", "1.6.1"),
        "mode": "single_video_summary",
        "task_id": bundle.get("task_id"),
        "search_revision": bundle.get("search_revision"),
        "coverage_status": bundle.get("coverage_status"),
        "total_candidates": bundle.get("total_candidates"),
        "confirmed_count": bundle.get("confirmed_count"),
        "rejected_count": bundle.get("rejected_count"),
        "candidate_truncated": bundle.get("candidate_truncated"),
        "included_evidence_count": bundle.get("included_evidence_count"),
        "evidence_truncated": bundle.get("evidence_truncated", False),
        "omitted_count": bundle.get("omitted_count", 0),
        "evaluation_scope": bundle.get("evaluation_scope"),
        "dataset_id": bundle.get("dataset_id"),
        "query_origin": bundle.get("query_origin"),
        "capability_status": bundle.get("capability_status"),
        "evidences": [evidence_to_prompt_item(e) for e in bundle.get("evidences", [])],
    }
    return payload


def build_messages(bundle: dict, max_input_tokens: int | None = None,
                   token_counter=None) -> tuple[list[dict], dict]:
    """构建 messages；按 token 预算确定性裁剪证据（先裁排名最低的 pending 项）。"""
    s = get_settings()
    budget = max_input_tokens or s.qwen_max_input_tokens
    system = load_system_prompt()
    payload = build_bundle_prompt_payload(bundle)
    originally_omitted = payload["omitted_count"]
    evidence = payload["evidences"]
    drop_order = sorted(evidence, key=lambda item: (
        item["review_status"] == "confirmed",
        item["rerank_sim"] if item["rerank_sim"] is not None else -2,
        item["evidence_id"],
    ))

    def make_messages() -> list[dict]:
        narrative, gaps = report_wording(payload)
        payload.update(required_narrative=narrative, required_gaps=gaps)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    messages = make_messages()
    if token_counter is not None:
        n = token_counter(messages)
        for item in drop_order:
            if n <= budget:
                break
            evidence.remove(item)
            payload["evidence_truncated"] = True
            payload["omitted_count"] += 1
            payload["included_evidence_count"] = len(evidence)
            messages = make_messages()
            n = token_counter(messages)
        if n > budget:
            raise ValueError("report prompt exceeds input token budget without evidence")
    else:
        n = None
    return messages, {
        "evidence_truncated": payload["evidence_truncated"],
        "omitted_count": payload["omitted_count"],
        "token_budget_omitted_count": payload["omitted_count"] - originally_omitted,
        "supplied_evidence_ids": [item["evidence_id"] for item in evidence],
        "final_tokens": n,
    }
