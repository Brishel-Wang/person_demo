"""OSNet-AIN x1.0 人体编码器（spec 3.2）。

- 显式 checkpoint 加载（vendored osnet_ain 架构 + 官方 msmt17 权重）
- 启动探针确认 512 维、有限、L2 归一化可用；零范数/NaN/Inf/维度不符直接失败
- eval 模式输出嵌入（无分类 logits）；查询与图库使用相同裁剪与预处理
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

from person_demo.feature_space import OSNET_INPUT_HW, OSNET_MEAN, OSNET_STD, OSNET_NORM_EPS
from person_demo.vision.reid.osnet_ain import osnet_ain_x1_0


class OSNetEncoder:
    def __init__(self, checkpoint_path: Path, device: str = "cuda:0"):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"osnet checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

        # num_classes 由检查点分类头形状决定（官方 msmt17 检查点为 4101）
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        num_classes = None
        for k, v in ckpt.items():
            k2 = k[7:] if k.startswith("module.") else k
            if k2 == "classifier.weight":
                num_classes = v.shape[0]
                break
        if num_classes is None:
            raise ValueError("checkpoint has no classifier.weight; cannot infer num_classes")
        self.num_classes = num_classes

        self.model: nn.Module = osnet_ain_x1_0(num_classes=num_classes, pretrained=False, loss="softmax")
        state_dict = {k.removeprefix("module."): v for k, v in ckpt.items()}
        self.model.load_state_dict(state_dict, strict=True)
        n_matched, n_discarded = len(state_dict), 0
        self.model.to(device)
        self.model.eval()
        self._lock = threading.Lock()
        self.dim = self.model.feature_dim
        if self.dim != 512:
            raise ValueError(f"expected embedding dim 512, got {self.dim}")

        # 启动探针
        probe = self.encode_batch([np.zeros((OSNET_INPUT_HW[0], OSNET_INPUT_HW[1], 3), dtype=np.uint8)])
        self.probe_ok = True
        self.probe_info = {
            "num_classes": num_classes,
            "matched_layers": n_matched,
            "discarded_layers": n_discarded,
            "embedding_dim": self.dim,
            "probe_norm": float(np.linalg.norm(probe[0])),
        }

    def _preprocess(self, crops_bgr: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for crop in crops_bgr:
            img = _resize_pad(crop, OSNET_INPUT_HW)  # RGB 256x128
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - np.array(OSNET_MEAN, dtype=np.float32)) / np.array(OSNET_STD, dtype=np.float32)
            tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)))
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def encode_batch(self, crops_bgr: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """返回 L2 归一化 float32 向量 (n, 512)。"""
        out: list[np.ndarray] = []
        with self._lock:
            for i in range(0, len(crops_bgr), batch_size):
                batch = crops_bgr[i : i + batch_size]
                x = self._preprocess(batch)
                v = self.model(x)
                v = v.float().cpu().numpy()
                if v.shape != (len(batch), self.dim):
                    raise ValueError(f"invalid embedding shape: {v.shape}")
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                if np.any(~np.isfinite(v)):
                    raise ValueError("non-finite embedding (NaN/Inf)")
                if np.any(~np.isfinite(norms)) or np.any(norms <= OSNET_NORM_EPS):
                    raise ValueError("invalid embedding norm")
                v = v / norms
                out.append(v.astype(np.float32))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def encode_single(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.encode_batch([crop_bgr])[0]


def _resize_pad(img_bgr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """按 OSNet 约定 resize：等比缩放后按均值填充到 256x128。"""
    import cv2

    h, w = hw
    ih, iw = img_bgr.shape[:2]
    scale = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * scale))), max(1, int(round(iw * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 127, dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas
