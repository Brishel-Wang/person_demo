"""feature_space_id 与 Milvus collection 命名（spec 3.3 / 4.2）。

feature_space_id 由模型权重、RGB/resize、mean/std、输出层、数值精度、L2 规范
与接口版本生成；aggregation_version 独立登记。完整 64 hex；collection 用前 12 hex。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

AGGREGATION_VERSION = "mean_l2_v1"
VECTOR_ROLE = "tracklet_prototype"
INTERFACE_VERSION = "body_encoder_v1"
COLLECTION_NAME_TEMPLATE = "body_<space_hash12>_ar1"
EMBEDDING_DIM = 512

# OSNet-AIN x1_0（torchreid 约定）：RGB 输入 256x128，ImageNet mean/std。
OSNET_INPUT_HW = (256, 128)
OSNET_MEAN = [0.485, 0.456, 0.406]
OSNET_STD = [0.229, 0.224, 0.225]
OSNET_NORM_EPS = 1e-12


def canonical_space_json(
    *,
    architecture: str,
    checkpoint_sha256: str,
    source_revision: str | None,
    input_color: str = "RGB",
    resize_hw: tuple[int, int] = (256, 128),
    mean: list[float] | None = None,
    std: list[float] | None = None,
    output_layer: str = "feature_map_global_pool",
    output_dim: int = 512,
    precision: str = "float32",
    l2_normalize: bool = True,
    l2_epsilon: float = 1e-12,
    interface_version: str = INTERFACE_VERSION,
) -> str:
    obj: dict[str, Any] = {
        "model": {
            "architecture": architecture,
            "checkpoint_sha256": checkpoint_sha256,
            "source_revision": source_revision,
        },
        "input": {"color": input_color, "resize": list(resize_hw)},
        "preprocess": {"mean": mean, "std": std},
        "output": {"layer": output_layer, "dim": output_dim},
        "numeric": {"precision": precision},
        "postprocess": {"l2_normalize": l2_normalize, "epsilon": l2_epsilon},
        "interface_version": interface_version,
    }
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def feature_space_id(canonical_json: str) -> str:
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def collection_name(space_id: str) -> str:
    return COLLECTION_NAME_TEMPLATE.replace("<space_hash12>", space_id[:12])


def embedding_id(
    *,
    dataset_id: str,
    ingest_run_id: str,
    tracklet_revision_id: str,
    space_id: str,
    role: str = VECTOR_ROLE,
    aggregation: str = AGGREGATION_VERSION,
) -> str:
    """UTF-8 紧凑 JSON 数组的 SHA256（spec 4.2 主键推导）。"""
    parts = [dataset_id, ingest_run_id, tracklet_revision_id, space_id, role, aggregation]
    s = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify_space_prefix(full: str, short12: str | None = None) -> bool:
    """完整空间 hash 校验；短 hash 只用于定位 collection。"""
    if len(full) != 64:
        return False
    if short12 is not None and not full.startswith(short12):
        return False
    return True
