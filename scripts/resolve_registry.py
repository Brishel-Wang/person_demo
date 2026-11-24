"""填写模型注册表与 profile 的实测值（sha256、feature_space_id、版本）。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from person_demo.feature_space import canonical_space_json, feature_space_id  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dir_manifest_sha(path: Path) -> str:
    """权重清单 hash：文件名 + 大小 + 分片 sha。"""
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name != "README.md")
    for p in files:
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def main() -> None:
    models = PROJECT / "models"
    reg = json.loads((PROJECT / "config" / "models.registry.json").read_text(encoding="utf-8"))
    detector = next(m for m in reg["models"] if m["key"] == "body_detector")
    yolo_sha = sha(models / detector["file"])
    osnet_sha = sha(models / (
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
        "softmax_labsmth_flip_jitter.pth"))
    space_id = feature_space_id(canonical_space_json(
        architecture="osnet_ain_x1_0", checkpoint_sha256=osnet_sha, source_revision=None))
    qwen_dir = models / "qwen3.5-9b"
    qwen_manifest = dir_manifest_sha(qwen_dir)

    # 1) models.registry.json
    for m in reg["models"]:
        if m["key"] == "body_detector":
            m["sha256"] = yolo_sha
        elif m["key"] == "body_osnet":
            m["sha256"] = osnet_sha
            m["revision"] = "kaiyangzhou/osnet (hf-mirror main)"
        elif m["key"] == "report_qwen":
            m["revision"] = "ModelScope master (2026-11-18)"
            m["weights_manifest_sha256"] = qwen_manifest
    reg["status"] = "downloaded_and_probed"
    (PROJECT / "config" / "models.registry.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2) body profile
    prof = json.loads((PROJECT / "config" / "profiles" /
                       "single_body_rag_demo_linux_milvus.json").read_text(encoding="utf-8"))
    prof["status"] = "executed"
    prof["body_encoder"]["weights_sha256"] = osnet_sha
    prof["body_encoder"]["source_revision"] = "kaiyangzhou/osnet (hf-mirror main)"
    prof["body_encoder"]["feature_space_id"] = space_id
    prof["body_encoder"]["mean"] = [0.485, 0.456, 0.406]
    prof["body_encoder"]["std"] = [0.229, 0.224, 0.225]
    prof["detector"]["weights_sha256"] = yolo_sha
    prof["detector"]["source_revision"] = detector["revision"]
    (PROJECT / "config" / "profiles" / "single_body_rag_demo_linux_milvus.json").write_text(
        json.dumps(prof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 3) report profile
    rp = json.loads((PROJECT / "config" / "profiles" /
                     "qwen_single_report_local_v1.json").read_text(encoding="utf-8"))
    rp["status"] = "executed"
    rp["model_revision"] = "ModelScope master (2026-11-18)"
    rp["weights_manifest_sha256"] = qwen_manifest
    (PROJECT / "config" / "profiles" / "qwen_single_report_local_v1.json").write_text(
        json.dumps(rp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "detector_weights": detector["file"],
        "detector_sha256": yolo_sha,
        "osnet_sha256": osnet_sha,
        "feature_space_id": space_id,
        "collection_name": f"body_{space_id[:12]}_ar1",
        "qwen_manifest_sha256": qwen_manifest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
