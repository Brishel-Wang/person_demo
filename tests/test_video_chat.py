import json

import pytest

from person_demo.api.video_chat import _parse_window, _validated_reply
from person_demo.errors import InvalidRequest


def test_uses_explicit_video_relative_times():
    assert _parse_window("查 00:10 到 00:15 的片段", 200_000_000) == (10_000_000, 15_000_001)
    assert _parse_window("查 10 到 15 秒", 200_000_000) == (10_000_000, 15_000_001)
    assert _parse_window("00:15以后还有吗", 200_000_000) == (15_000_000, 200_000_000)
    assert _parse_window("轨迹 13877 给我看第 2 个结果", 200_000_000) is None
    with pytest.raises(InvalidRequest):
        _parse_window("04:00 的片段", 200_000_000)


def test_qwen_references_resolve_only_to_current_passages():
    references = {"E1": {"cue_id": "clip-1"}, "E2": {"cue_id": "clip-2"}}
    reply = _validated_reply('{"answer":"也可回看 E2","citations":["E1"]}', references)
    assert reply["citation_ids"] == ["clip-1", "clip-2"]
    assert reply["citation_labels"] == {"E1": "clip-1", "E2": "clip-2"}
    with pytest.raises(ValueError):
        _validated_reply('{"answer":"见 E99","citations":["E1"]}', references)
    with pytest.raises(ValueError):
        _validated_reply('{"answer":"看这段","citations":["E3"]}', references)


def test_timeline_answers_cannot_claim_appearance_scores():
    with pytest.raises(ValueError):
        _validated_reply('{"answer":"有外观相似度分数","citations":[]}', {}, "timeline")


def test_limited_video_answers_cannot_claim_full_video_scope():
    raw = '{"answer":"全片检索到两段","citations":[]}'
    with pytest.raises(ValueError):
        _validated_reply(raw, {}, whole_video=False)
    assert _validated_reply(raw, {}, whole_video=True)["answer"] == "全片检索到两段"
    bounded = '{"answer":"已索引的前十二秒内检索到两段","citations":[]}'
    assert _validated_reply(bounded, {}, whole_video=False)["answer"].startswith("已索引")


def test_video_answer_uses_retrieval_count_and_accepts_every_supplied_reference():
    refs = {f"E{i}": {"cue_id": f"clip-{i}"} for i in range(1, 13)}
    content = {"answer": "全片共找到16个候选，可在结果区查看。", "citations": list(refs)}
    reply = _validated_reply(json.dumps(content), refs, total_candidates=16)
    assert len(reply["citation_ids"]) == 12
    content["answer"] = "全片共找到12个候选，可参考 E1。"
    with pytest.raises(ValueError, match="total number"):
        _validated_reply(json.dumps(content), refs, total_candidates=16)
