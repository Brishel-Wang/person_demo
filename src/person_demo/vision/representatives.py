"""代表图筛选（spec 3.2）。

每个 chunk 最多 5 张有效人体代表图：
- 先去掉空框、非法裁剪、解码失败图并记录原因
- 按片段时间等分最多 5 槽，各槽选择统一缩放至 128×256 后的灰度拉普拉斯
  方差最高者；并列按 PTS 排序，最终图间隔至少 0.5 秒
- 实际不足 5 张按实际数量保存，不复制凑数；质量分只用于选图
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CropCandidate:
    local_us: int
    frame_pts: int
    bbox: tuple[int, int, int, int]
    frame_w: int
    frame_h: int
    crop: np.ndarray | None = None
    status: str = "valid"  # valid/empty_box/clip_fail/decode_fail
    reason: str | None = None
    laplacian_var: float | None = None


@dataclass
class RepresentativeResult:
    selected: list[CropCandidate] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {local_us, reason}
    quality_summary: float | None = None  # 均值质量分（仅选图用，非身份概率）


def laplacian_variance(img_bgr: np.ndarray, resize_hw: tuple[int, int] = (256, 128)) -> float:
    """统一缩放至 128×256 后的灰度拉普拉斯方差。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def select_representatives(
    candidates: list[CropCandidate],
    t_start_us: int,
    t_end_us: int,
    max_count: int = 5,
    min_gap_us: int = 500_000,
) -> RepresentativeResult:
    """等分时间槽 + 每槽最高拉普拉斯方差；最终间隔 >= min_gap_us。"""
    result = RepresentativeResult()
    valid = [c for c in candidates if c.status == "valid" and c.crop is not None]
    for c in candidates:
        if c.status != "valid" or c.crop is None:
            result.rejected.append({"local_us": c.local_us, "reason": c.reason or "invalid"})
    if not valid:
        return result

    valid.sort(key=lambda c: (c.local_us, c.frame_pts))
    k = min(max_count, len(valid))
    duration = max(t_end_us - t_start_us, 1)
    bins: list[list[CropCandidate]] = [[] for _ in range(k)]
    for c in valid:
        idx = int((c.local_us - t_start_us) / duration * k)
        idx = min(max(idx, 0), k - 1)
        bins[idx].append(c)

    for bucket in bins:
        if not bucket:
            continue
        for c in bucket:
            if c.laplacian_var is None:
                c.laplacian_var = laplacian_variance(c.crop)
        bucket.sort(key=lambda c: (-(c.laplacian_var or 0.0), c.frame_pts))
        # 与已选图保持最小间隔；冲突时顺延取本槽次优
        chosen = False
        for c in bucket:
            if all(abs(c.local_us - s.local_us) >= min_gap_us for s in result.selected):
                result.selected.append(c)
                chosen = True
                break
        if not chosen:
            # 本槽全部与已选过近：跳过（不复制凑数）
            pass

    result.selected.sort(key=lambda c: (c.local_us, c.frame_pts))
    if result.selected:
        result.quality_summary = float(np.mean([c.laplacian_var or 0.0 for c in result.selected]))
    return result
