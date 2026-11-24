"""单路检索来源排除（spec 第 5 章）。

- 不排除整个源相机（cross_camera 规则不适用单路）
- intra_camera_demo：排除查询图所在来源 chunk（origin_tracklet_revision_id），
  同 continuation 的其他 chunk 保留但标 same_continuation_as_query=true
- intra_camera_holdout：额外排除完整 continuation_group（独立评测）
- origin_unknown：不声称已完成来源排除或独立评测
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class ExclusionResult:
    keep: bool
    reason: str
    same_continuation_as_query: bool = False
    exclusion_verified: bool = True


def evaluate_exclusion(
    *,
    evaluation_scope: str,
    origin_tracklet_revision_id: uuid.UUID | None,
    origin_continuation_group_id: uuid.UUID | None,
    origin_status: str,
    candidate_revision_id: uuid.UUID,
    candidate_group_id: uuid.UUID | None,
) -> ExclusionResult:
    """判定候选是否保留及其来源关系标记。"""
    if origin_status == "unknown":
        return ExclusionResult(keep=True, reason="origin_unknown_exclusion_not_verified",
                               exclusion_verified=False)
    if origin_tracklet_revision_id is None:
        return ExclusionResult(keep=True, reason="origin_missing_exclusion_not_verified",
                               exclusion_verified=False)

    same_chunk = candidate_revision_id == origin_tracklet_revision_id
    same_group = (
        origin_continuation_group_id is not None
        and candidate_group_id is not None
        and candidate_group_id == origin_continuation_group_id
    )

    if evaluation_scope == "intra_camera_demo":
        if same_chunk:
            return ExclusionResult(keep=False, reason="origin_chunk_excluded", exclusion_verified=True)
        if same_group:
            return ExclusionResult(keep=True, reason="same_continuation_as_query",
                                   same_continuation_as_query=True, exclusion_verified=True)
        return ExclusionResult(keep=True, reason="independent_candidate", exclusion_verified=True)

    if evaluation_scope == "intra_camera_holdout":
        if same_chunk:
            return ExclusionResult(keep=False, reason="origin_chunk_excluded", exclusion_verified=True)
        if same_group:
            return ExclusionResult(keep=False, reason="origin_continuation_group_excluded", exclusion_verified=True)
        return ExclusionResult(keep=True, reason="holdout_candidate", exclusion_verified=True)

    raise ValueError(f"unsupported evaluation_scope: {evaluation_scope}")
