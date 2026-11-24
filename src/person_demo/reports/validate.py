"""报告输出校验（spec 7.2）：JSON Schema、枚举、长度、引用白名单、人工状态规则。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "model_response.schema.json"


def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def extract_json(text: str) -> str:
    """从模型输出提取 JSON（容忍代码围栏、前后缀文本与尾随内容）。

    扫描首个 '{'，按括号配对找到完整平衡对象，忽略其后的多余文本。
    """
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        t = m.group(1).strip()
    i = t.find("{")
    if i < 0:
        return t
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(t)):
        ch = t[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[i : j + 1]
    return t[i:]


def report_wording(bundle: dict) -> tuple[str, list[str]]:
    if bundle.get("confirmed_count"):
        narrative = "检索结果中有经人工确认的片段。其余证据请按复核状态查看。"
    elif bundle.get("total_candidates", 0) > bundle.get("rejected_count", 0):
        narrative = "检索结果中有相似候选，目前没有人工确认的片段。"
    else:
        narrative = "目前没有可用于正向结论的证据。"
    gaps = ["时间仅指原片内位置。"]
    if bundle.get("coverage_status") != "full":
        gaps.append("查询窗口的索引覆盖不足。")
    if bundle.get("candidate_truncated") or bundle.get("evidence_truncated"):
        gaps.append("部分候选或证据因预算省略。")
    if bundle.get("query_origin", {}).get("origin_status") == "unknown":
        gaps.append("上传图来源未知。")
    return narrative, gaps


def parse_and_validate_model_output(
    raw_text: str,
    bundle: dict,
) -> tuple[dict | None, list[str]]:
    """返回 (content, errors)。content 为校验通过的报告内容。"""
    errors: list[str] = []
    try:
        obj = json.loads(extract_json(raw_text))
    except Exception as e:  # noqa: BLE001
        return None, [f"invalid_json: {e}"]

    try:
        jsonschema.validate(obj, _schema())
    except jsonschema.ValidationError as e:
        return None, [f"schema: {e.message}"]

    evidence_ids = {ev["evidence_id"] for ev in bundle.get("evidences", [])}
    review_by_id = {ev["evidence_id"]: ev.get("review_status") for ev in bundle.get("evidences", [])}
    for claim in obj.get("claims", []):
        if not isinstance(claim, dict):
            errors.append("claim must be an object, got non-dict entry")
            continue
        for eid in claim.get("evidence_ids", []):
            if eid not in evidence_ids:
                errors.append(f"claim references unknown evidence_id {eid}")
                break
        if claim.get("kind") == "reviewed_observation":
            for eid in claim.get("evidence_ids", []):
                if review_by_id.get(eid) != "confirmed":
                    errors.append(f"reviewed_observation uses non-confirmed evidence {eid}")
                    break
    if bundle.get("data_origin") == "real_video_processing_result":
        narrative, gaps = report_wording(bundle)
        if obj["narrative"] != narrative or obj["gaps"] != gaps:
            errors.append("copy required_narrative and required_gaps exactly")
        if obj["next_information_needed"] not in ([], ["请人工复核待复核候选。"]):
            errors.append("next_information_needed must use approved wording")
        by_id = {item["evidence_id"]: item for item in bundle["evidences"]}
        cited = set()
        confirmed = set()
        pending_count = 0
        for claim in obj["claims"]:
            if len(claim["evidence_ids"]) != 1:
                errors.append("cite exactly one evidence item in each claim")
                continue
            eid = claim["evidence_ids"][0]
            if eid not in by_id:
                continue
            if eid in cited:
                errors.append("cite each selected evidence only once")
            cited.add(eid)
            item = by_id[eid]
            if item["review_status"] == "confirmed":
                confirmed.add(eid)
                kind, wording = "reviewed_observation", "该片段已获人工确认。"
            else:
                pending_count += 1
                kind, wording = "candidate_observation", "这是相似候选，待人工复核。"
            if item["same_continuation_as_query"]:
                wording += "该片段属于同一跟踪段的后续片段。"
            if claim["kind"] != kind or claim["text"] != wording:
                errors.append(f"claim status or wording is invalid for {eid}")
        required = {eid for eid, status in review_by_id.items() if status == "confirmed"}
        if confirmed != required or pending_count > 3:
            errors.append("include every supplied confirmed item and at most three pending items")
        if bundle.get("evidences") and not cited:
            errors.append("select at least one eligible item")
    if errors:
        return None, errors
    if bundle.get("data_origin") == "real_video_processing_result":
        times = {item["evidence_id"]: item["t_start_local_us"] for item in bundle["evidences"]}
        obj["claims"].sort(key=lambda claim: times[claim["evidence_ids"][0]])
    return obj, []
