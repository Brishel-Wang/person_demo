"""帧读取：按 PTS 顺序解码窗口内分析帧，保留原图坐标；裁剪与代理视频。"""
from __future__ import annotations

import io
import math
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

from person_demo.errors import InvalidTimeRange, TimebaseInvalid
from person_demo.media.probe import ProbeResult, frame_interval_pts, require_valid_timebase
from person_demo.timebase import pts_to_local_us


class FrameSample:
    __slots__ = ("pts", "local_us", "time_base_num", "time_base_den", "frame_bgr", "width", "height")

    def __init__(self, pts: int, local_us: int, tb_num: int, tb_den: int, frame_bgr: np.ndarray):
        self.pts = pts
        self.local_us = local_us
        self.time_base_num = tb_num
        self.time_base_den = tb_den
        self.frame_bgr = frame_bgr
        self.height, self.width = frame_bgr.shape[:2]


def iter_window_frames(
    path: str,
    probe: ProbeResult,
    window_us: tuple[int, int],
    analysis_fps: int,
    max_frames: int | None = None,
) -> Iterator[FrameSample]:
    """顺序解码窗口内视频帧并按 analysis_fps 均匀网格采样（10FPS）。

    按真实 PTS 采样：帧 PTS 落在网格步长上才保留。PTS 缺失时抛错（不伪造定位）。
    """
    require_valid_timebase(probe)
    if window_us[0] < 0 or window_us[1] <= window_us[0]:
        raise InvalidTimeRange(f"invalid window {window_us}")
    start_pts, end_pts = _window_pts(probe, window_us)
    # 按片内时间 100ms 桶采样（59.94fps 的 PTS 步长 3003 不能整除 10FPS 网格，
    # 因此按 local_us 网格取每桶首帧，VFR 同样适用）
    step_us = 1_000_000 // analysis_fps
    next_target_us = window_us[0]
    yielded = 0
    with av.open(path) as c:
        v = c.streams.video[0]
        v.codec_context.thread_count = 8
        v.thread_type = "AUTO"
        previous_pts = None
        for frame in c.decode(v):
            pts = frame.pts
            if pts is None or (previous_pts is not None and pts <= previous_pts):
                raise TimebaseInvalid("missing, duplicate or non-increasing frame PTS")
            previous_pts = pts
            if pts < start_pts:
                continue
            if pts >= end_pts:
                break
            local_us = pts_to_local_us(pts, probe.pts_anchor, probe.time_base_num, probe.time_base_den)
            if local_us < next_target_us:
                continue
            img = frame.to_ndarray(format="bgr24")
            yield FrameSample(pts, local_us, probe.time_base_num, probe.time_base_den, img)
            yielded += 1
            next_target_us = window_us[0] + ((local_us - window_us[0]) // step_us + 1) * step_us
            if max_frames is not None and yielded >= max_frames:
                return


def _window_pts(probe: ProbeResult, window_us: tuple[int, int]) -> tuple[int, int]:
    from person_demo.timebase import local_us_to_pts

    a = local_us_to_pts(window_us[0], probe.pts_anchor, probe.time_base_num, probe.time_base_den)
    b = local_us_to_pts(window_us[1], probe.pts_anchor, probe.time_base_num, probe.time_base_den)
    return a, b


def decode_frame_at_local_us(path: str, probe: ProbeResult, local_us: int) -> FrameSample | None:
    """按片内时间取最近的可解码帧（用于查询图创建与证据回放）。"""
    require_valid_timebase(probe)
    from person_demo.timebase import local_us_to_pts

    target = local_us_to_pts(local_us, probe.pts_anchor, probe.time_base_num, probe.time_base_den)
    best: FrameSample | None = None
    best_d = 1 << 62
    with av.open(path) as c:
        v = c.streams.video[0]
        c.seek(max(0, target - 3 * frame_interval_pts(probe, 10)), stream=v)  # noqa
        for frame in c.decode(v):
            if frame.pts is None:
                continue
            d = abs(frame.pts - target)
            if d < best_d:
                best_d = d
                img = frame.to_ndarray(format="bgr24")
                us = pts_to_local_us(frame.pts, probe.pts_anchor, probe.time_base_num, probe.time_base_den)
                best = FrameSample(frame.pts, us, probe.time_base_num, probe.time_base_den, img)
            else:
                break
            if best_d == 0:
                break
    return best


def crop_person(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """检测框逆变换后的原图矩形裁剪，越界裁至图像边界；空框返回 None。"""
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1c, y1c = max(0, int(round(x1))), max(0, int(round(y1)))
    x2c, y2c = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2c - x1c < 2 or y2c - y1c < 2:
        return None
    return frame_bgr[y1c:y2c, x1c:x2c]


def encode_jpg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def decode_jpg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def build_proxy_video(
    path: str,
    probe: ProbeResult,
    window_us: tuple[int, int],
    out_path: Path,
    analysis_fps: int,
    draw_boxes: dict[int, list[dict]] | None = None,
    scale: float = 0.5,
) -> dict:
    """生成窗口代理视频（带框叠加），保存回到原视频的 local_us 时间映射。

    draw_boxes: {local_us: [{bbox:[x1,y1,x2,y2], color:(r,g,b), label:str}]}
    返回 {frame_index: local_us}。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = frame_interval_pts(probe, analysis_fps)
    mapping: dict[int, int] = {}
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=analysis_fps)
    stream.width = max(2, int(probe.width * scale) // 2 * 2)
    stream.height = max(2, int(probe.height * scale) // 2 * 2)
    stream.pix_fmt = "yuv420p"
    idx = 0
    for sample in iter_window_frames(path, probe, window_us, analysis_fps):
        img = sample.frame_bgr
        if draw_boxes:
            for box in draw_boxes.get(sample.local_us, []):
                x1, y1, x2, y2 = box["bbox"]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box["color"], 3)
                if box.get("label"):
                    cv2.putText(img, box["label"], (int(x1), max(16, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box["color"], 2)
        if scale != 1.0:
            img = cv2.resize(img, (stream.width, stream.height), interpolation=cv2.INTER_AREA)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = idx
        for pkt in stream.encode(frame):
            container.mux(pkt)
        mapping[idx] = sample.local_us
        idx += 1
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return mapping


def frame_index_for_local_us(mapping: dict[int, int], local_us: int) -> int | None:
    """local_us → 代理视频帧序号（最近邻）。"""
    if not mapping:
        return None
    items = sorted(mapping.items(), key=lambda kv: kv[1])
    best_i, best_d = None, 1 << 62
    for i, us in items:
        d = abs(us - local_us)
        if d < best_d:
            best_i, best_d = i, d
    return best_i
