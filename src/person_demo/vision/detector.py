"""只输出人体的 YOLO 检测。权重从本机显式加载。"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

LEGACY_YOLO11S_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"


def recorded_detector_weights(tracker_config: dict | None, detector_sha256: str | None) -> str | None:
    recorded = (tracker_config or {}).get("detector_weights")
    if recorded:
        return recorded
    if detector_sha256 == LEGACY_YOLO11S_SHA256:
        return "yolo11s.pt"
    return None


class PersonDetector:
    def __init__(self, weights_path: Path, imgsz: int = 640, device: str = "cuda:0"):
        from ultralytics import YOLO

        if not weights_path.exists():
            raise FileNotFoundError(f"yolo weights not found: {weights_path}")
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.device = device
        self._lock = threading.Lock()
        self.model = YOLO(str(weights_path))
        if self.model.names.get(0) != "person":
            raise ValueError("the configured YOLO checkpoint must map class 0 to person")
        self.model.to(device)
        self.model.eval()

    def detect_persons(self, frames_bgr: list[np.ndarray]) -> list[list[dict]]:
        """批量检测；只保留 person 类。返回每帧 [{bbox:[x1,y1,x2,y2], conf:float}]。"""
        if not frames_bgr:
            return []
        with self._lock:
            results = self.model.predict(
                frames_bgr, imgsz=self.imgsz, conf=0.25, iou=0.7,
                classes=[0], verbose=False, device=self.device,
            )
        out: list[list[dict]] = []
        for r in results:
            boxes = []
            for bbox, confidence in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=True):
                boxes.append({"bbox": bbox, "conf": confidence})
            out.append(boxes)
        return out
