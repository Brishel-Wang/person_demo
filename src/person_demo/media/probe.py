"""视频 probe（D1）：PyAV 探测可解码性、时长、PTS 锚点、time_base、VFR 与 resolved 窗口。

local_us=0 为第一张可定位解码帧，不代表真实午夜（spec 2.2 / R05）。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av

from person_demo.errors import InvalidTimeRange, TimebaseInvalid
from person_demo.timebase import pts_to_local_us


@dataclass
class ProbeResult:
    exists: bool
    readable: bool
    path: str
    size_bytes: int | None = None
    sha256: str | None = None
    error: str | None = None
    # 媒体
    container: str | None = None
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    avg_fps: float | None = None
    is_vfr: bool | None = None
    duration_s: float | None = None
    duration_us: int | None = None
    nb_frames: int | None = None
    # 时间基准
    pts_anchor: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    timebase_status: str = "unprobed"
    first_frame_pts: int | None = None
    # resolved 窗口
    requested_window_us: tuple[int, int] = (0, 0)
    resolved_window_us: tuple[int, int] | None = None
    window_note: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def probe_file(path: str, requested_window_us: tuple[int, int] | None = None, *,
               hash_file: bool = True) -> ProbeResult:
    """探测本地文件。不下载、不换源；失败给出具体原因。"""
    r = ProbeResult(exists=False, readable=False, path=path)
    if requested_window_us is not None:
        r.requested_window_us = requested_window_us
    p = Path(path)
    if not p.exists():
        r.error = f"file_not_found: {path}"
        return r
    r.exists = True
    if not os.access(path, os.R_OK):
        r.error = f"file_not_readable: {path}"
        return r
    r.readable = True
    r.size_bytes = p.stat().st_size

    if hash_file:
        try:
            r.sha256 = _sha256_stream(path)
        except OSError as exc:
            r.error = f"sha256 failed: {exc}"
            return r

    try:
        with av.open(path) as c:
            r.container = c.format.name
            v = next((s for s in c.streams if s.type == "video"), None)
            if v is None:
                r.error = "no_video_stream"
                return r
            r.codec_name = v.codec_context.name
            r.width = v.codec_context.width
            r.height = v.codec_context.height
            r.nb_frames = v.frames or None
            tb = v.time_base
            r.time_base_num = tb.numerator
            r.time_base_den = tb.denominator
            rate = v.average_rate
            if rate and rate > 0:
                r.avg_fps = float(rate)
            r.is_vfr = bool(v.duration and v.frames and rate and v.frames / float(rate) > v.duration * 1.001 + 1)
            r.duration_s = float(v.duration * tb) if v.duration else None
            r.duration_us = int(round(r.duration_s * 1e6)) if r.duration_s is not None else None
            if r.duration_s is None and c.duration:
                r.duration_s = float(c.duration / av.time_base)
                r.duration_us = int(round(r.duration_s * 1e6))

            frames = c.decode(v)
            first = next(frames, None)
            second = next(frames, None)
            if (first is None or first.pts is None or
                    (second is not None and (second.pts is None or second.pts <= first.pts))):
                r.timebase_status = "TIMEBASE_INVALID"
                r.error = "missing or non-increasing initial frame PTS"
                return r
            r.first_frame_pts = r.pts_anchor = first.pts
            r.timebase_status = "ok"
    except TimebaseInvalid:
        raise
    except Exception as e:  # noqa: BLE001
        r.timebase_status = "TIMEBASE_INVALID" if r.pts_anchor is None else r.timebase_status
        r.error = f"probe_failed: {e}"
        return r

    # resolved 窗口：不超过时长，不足 200s 明确缩小
    if requested_window_us is not None and r.duration_us is not None:
        start, end = requested_window_us
        if start < 0 or end <= start:
            r.error = f"invalid_requested_window {start}:{end}"
            r.resolved_window_us = None
            r.window_note = "invalid_requested_window"
        elif start >= r.duration_us:
            r.error = "window_start_beyond_duration"
            r.resolved_window_us = None
            r.window_note = "window_start_beyond_duration"
        else:
            resolved_end = min(end, r.duration_us)
            r.resolved_window_us = (start, resolved_end)
            if resolved_end < end:
                r.window_note = (
                    f"window_shrunk_to_duration: requested_end={end}us duration={r.duration_us}us"
                )
            else:
                r.window_note = "window_within_duration"
    return r


def _sha256_stream(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require_valid_timebase(r: ProbeResult) -> None:
    if r.timebase_status != "ok" or r.pts_anchor is None:
        raise TimebaseInvalid(f"timebase status={r.timebase_status}, error={r.error}")
    if r.time_base_num is None or r.time_base_den is None:
        raise TimebaseInvalid("missing time_base")


def window_pts_bounds(r: ProbeResult, window_us: tuple[int, int]) -> tuple[int, int]:
    """窗口微秒 → [pts_start, pts_end)（按锚点与 time_base 换算）。"""
    from person_demo.timebase import local_us_to_pts

    require_valid_timebase(r)
    start_pts = local_us_to_pts(window_us[0], r.pts_anchor, r.time_base_num, r.time_base_den)
    end_pts = local_us_to_pts(window_us[1], r.pts_anchor, r.time_base_num, r.time_base_den)
    return start_pts, end_pts


def frame_interval_pts(r: ProbeResult, analysis_fps: int) -> int:
    """10FPS 采样间隔（PTS 单位），四舍五入到 time_base 网格。"""
    require_valid_timebase(r)
    tb = Fraction(r.time_base_num, r.time_base_den)
    step = Fraction(1, analysis_fps) / tb
    return max(1, int(round(step)))
