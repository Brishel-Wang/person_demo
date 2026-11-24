"""ByteTrack 单镜头跟踪（本地实现，spec 2.1 显式实现与配置，10FPS 适配）。

实现原始 ByteTrack 算法核心（Kalman 滤波 + 两阶段 IoU 关联）：
- 高置信度检测先与所有活跃/丢失轨道关联（match_thresh=0.8）
- 剩余低置信度检测与未匹配轨道二次关联（match_thresh=0.5）
- min_hits 连续命中后轨道确认输出；丢失轨道保留 track_buffer 帧
- 每帧输出 {track_id, bbox, status}

分析频率 10FPS：track_buffer=30 帧 ≈ 3 秒；30 秒分块在 ingest 层处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanBox:
    """8 维状态 [cx, cy, a, h, vx, vy, va, vh]。"""

    def __init__(self, bbox: list[float]):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a = max((x2 - x1) * (y2 - y1), 1.0)
        h = max(y2 - y1, 1.0)
        self.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8)
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4])
        self.R = np.diag([1e-1, 1e-1, 1e-1, 1e-1])
        self.Q = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4])

    def predict(self) -> list[float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.xywh()

    def update(self, bbox: list[float]) -> None:
        x1, y1, x2, y2 = bbox
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max((x2 - x1) * (y2 - y1), 1.0), max(y2 - y1, 1.0)])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def xywh(self) -> list[float]:
        cx, cy, a, h = self.x[:4]
        h = float(np.clip(h, 1.0, 1e5))
        w = float(np.clip(a / h, 1.0, 1e5))
        return [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-12 else 0.0


@dataclass
class Track:
    track_id: int
    kalman: KalmanBox
    bbox: list[float]
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    lost: bool = False

    def predict(self) -> list[float]:
        self.bbox = self.kalman.predict()
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: list[float]) -> None:
        self.kalman.update(bbox)
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.lost = False


class ByteTrack:
    """简单明确的两阶段 ByteTrack。"""

    def __init__(
        self,
        track_thresh: float = 0.25,
        high_thresh: float = 0.5,
        match_thresh_high: float = 0.8,
        match_thresh_low: float = 0.5,
        track_buffer: int = 30,
        min_hits: int = 3,
        max_lost_keep: int = 60,
        include_unmatched: bool = False,
    ):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.max_lost_keep = max_lost_keep
        self.include_unmatched = include_unmatched
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """返回真实检测框。高召回模式保留未关联低置信度框作为单帧片段。"""
        dets = [d for d in detections if d["conf"] >= self.track_thresh]
        high = [d for d in dets if d["conf"] >= self.high_thresh]
        low = [d for d in dets if d["conf"] < self.high_thresh]

        for t in self.tracks:
            t.predict()

        # 第一阶段：高置信度检测关联
        matched: set[int] = set()
        matched_high: set[int] = set()
        matched_low: set[int] = set()
        if high:
            ious = np.array([[ _iou(t.bbox, d["bbox"]) for d in high] for t in self.tracks])
            for _ in range(min(len(self.tracks), len(high))):
                if ious.size == 0:
                    break
                i, j = np.unravel_index(np.argmax(ious), ious.shape)
                if ious[i, j] < self.match_thresh_high:
                    break
                self.tracks[i].update(high[j]["bbox"])
                ious[i, :] = -1
                ious[:, j] = -1
                matched.add(i)
                matched_high.add(j)

        # 第二阶段：低置信度检测与未匹配轨道关联
        if low:
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched]
            if unmatched_tracks:
                ious = np.array([[ _iou(self.tracks[i].bbox, d["bbox"]) for d in low] for i in unmatched_tracks])
                for _ in range(min(len(unmatched_tracks), len(low))):
                    if ious.size == 0:
                        break
                    i, j = np.unravel_index(np.argmax(ious), ious.shape)
                    if ious[i, j] < self.match_thresh_low:
                        break
                    self.tracks[unmatched_tracks[i]].update(low[j]["bbox"])
                    matched_low.add(j)
                    ious[i, :] = -1
                    ious[:, j] = -1

        # 新轨道：未匹配的高置信度检测
        for j, d in enumerate(high):
            if j not in matched_high:
                t = Track(self._next_id, KalmanBox(d["bbox"]), list(d["bbox"]))
                self._next_id += 1
                self.tracks.append(t)

        # 确认与清理
        out: list[dict] = []
        for t in self.tracks:
            if t.time_since_update == 0:
                t.lost = False
                if t.hits >= self.min_hits:
                    t.confirmed = True
            elif t.time_since_update > self.track_buffer:
                t.lost = True
        self.tracks = [t for t in self.tracks if not (t.lost and t.time_since_update > self.max_lost_keep)]

        for t in self.tracks:
            if t.confirmed and t.time_since_update == 0:
                out.append({"track_id": t.track_id, "bbox": list(t.bbox), "status": "tracked"})
        if self.include_unmatched:
            for j, det in enumerate(low):
                if j not in matched_low:
                    out.append({"track_id": self._next_id, "bbox": list(det["bbox"]), "status": "unassociated"})
                    self._next_id += 1
        return out
