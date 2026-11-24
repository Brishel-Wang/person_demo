"""片内时间（asset_relative local_us）与 PTS 的换算。

local_us = round((frame_pts - pts_anchor) * (time_base_num / time_base_den) * 1_000_000)
第一帧 local_us=0 是片内原点，不代表真实午夜（spec 2.2 / R05）。
"""
from __future__ import annotations

from fractions import Fraction


def pts_to_local_us(frame_pts: int, pts_anchor: int, time_base_num: int, time_base_den: int) -> int:
    """PTS → 片内微秒（有理数精确计算后取整）。"""
    if time_base_den <= 0 or time_base_num <= 0:
        raise ValueError("invalid time_base")
    tb = Fraction(time_base_num, time_base_den)
    us = (Fraction(frame_pts) - Fraction(pts_anchor)) * tb * 1_000_000
    return int(round(us))


def local_us_to_pts(local_us: int, pts_anchor: int, time_base_num: int, time_base_den: int) -> int:
    """片内微秒 → 最近 PTS（用于 seek）。"""
    if time_base_den <= 0 or time_base_num <= 0:
        raise ValueError("invalid time_base")
    tb = Fraction(time_base_num, time_base_den)
    pts = Fraction(pts_anchor) + Fraction(local_us, 1_000_000) / tb
    return int(round(pts))


def format_local_us(local_us: int) -> str:
    """片内微秒 → 可读 'MM:SS.mmm' 字符串（仅片内时间，非真实时刻）。"""
    total_ms = local_us // 1000
    minutes, ms = divmod(total_ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def bucket_index_for(local_us: int, bucket_seconds: int = 60) -> int:
    """60 秒分桶下标，[0,200s) 为 4 桶。"""
    return local_us // (bucket_seconds * 1_000_000)
