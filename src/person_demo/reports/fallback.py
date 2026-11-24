"""模板降级报告（spec 7.2）：provider=template 且 status=DEGRADED，不计真实模型验收通过。"""
from __future__ import annotations


def build_template_report(bundle: dict, reason: str) -> dict:
    evidences = bundle.get("evidences", [])
    confirmed = [e for e in evidences if e.get("review_status") == "confirmed"]
    pending = [e for e in evidences if e.get("review_status") == "pending_review"]
    rejected_count = bundle.get("rejected_count", 0)
    truncated = bundle.get("evidence_truncated", False)
    omitted = bundle.get("omitted_count", 0)

    claims = []
    for ev in evidences[:14]:
        state = ev.get("review_status")
        if state == "confirmed":
            kind, wording = "reviewed_observation", "该片段已获人工确认。"
        else:
            kind, wording = "candidate_observation", "这是相似候选，待人工复核。"
        if ev.get("same_continuation_as_query"):
            wording += "该片段属于同一跟踪段的后续片段。"
        claims.append({
            "kind": kind, "text": wording,
            "evidence_ids": [ev["evidence_id"]], "hypothesis_id": None,
        })

    gaps = [
        f"报告生成降级为模板（原因：{reason}），未经真实模型整理，不计入 D6 真实模型验收。",
        "真实日期与机位地点尚未补录，候选身份仍待人工复核。",
    ]
    if bundle.get("coverage_status") != "full":
        gaps.append("查询窗口的索引覆盖不足。")
    if bundle.get("candidate_truncated") or truncated:
        gaps.append(f"证据预算截断：omitted_count={omitted}，本报告不宣称全量覆盖。")

    return {
        "schema_version": "1.6.1",
        "mode": "single_video_summary",
        "narrative": (
            f"模板降级报告（provider=template, status=DEGRADED）。共展示 {len(claims)} 条证据记录"
            f"（人工确认 {len(confirmed)} 条，待复核 {len(pending)} 条，排除 {rejected_count} 条）。"
            "此内容由程序模板生成，未经大模型整理，不构成真实模型验收。"
        ),
        "claims": claims,
        "gaps": gaps,
        "hypotheses": [],
        "next_information_needed": [
            "人工确认或排除待复核候选后，重新生成报告。",
            "使用真实 Qwen 模型生成报告以完成 D6 验收。",
        ],
    }
