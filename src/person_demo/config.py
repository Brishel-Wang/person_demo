"""应用配置：从环境变量与配置文件读取，D0 锁定运行参数。

环境变量前缀：PERSON_DEMO_（pydantic-settings 自动大写映射）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERSON_DEMO_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # 存储与数据库
    storage_root: Path = Field(default=PROJECT_ROOT / "storage")
    database_url: str = "postgresql+psycopg2://person_demo:person_demo@127.0.0.1:5433/person_demo"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_database: str = "person_demo"
    milvus_token: str = ""

    # 计算
    device: str = "cuda:0"
    workers: int = 0  # 数据加载线程数，0=自动

    # 数据集
    dataset_id: str = "single_camera_demo_v1"
    camera_code: str = "cam_01"

    # 模型
    model_dir: Path = Field(default=PROJECT_ROOT / "models")
    yolo_weights: str = "yolo26x.pt"
    yolo_imgsz: int = Field(default=1920, gt=0)
    osnet_checkpoint: str = (
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
        "softmax_labsmth_flip_jitter.pth"
    )
    qwen_model_id: str = "Qwen/Qwen3.5-9B"
    qwen_dir: str = "qwen3.5-9b"  # model_dir 下的目录名
    qwen_precision: str = "bfloat16"
    qwen_max_input_tokens: int = 6144
    qwen_max_new_tokens: int = 1536
    qwen_total_timeout_s: int = 120
    qwen_unload_after_job: bool = True

    # 流水线
    analysis_fps: int = 10
    tracklet_max_duration_s: int = 30
    representative_frames_max: int = 5
    representative_min_gap_s: float = 0.5
    tracker_lost_buffer_frames: int = 30  # 10FPS 下约 3 秒
    tracker_match_high: float = 0.8
    tracker_match_low: float = 0.5
    tracker_min_hits: int = 3
    upsert_batch_size: int = 256
    upsert_max_wait_ms: int = 1000
    bucket_seconds: int = 60
    per_bucket_top_k: int = 20
    backfill_limits: str = "20,40,80,160,200"
    max_candidate_tracklets: int = 100
    display_top_k: int = 5
    wait_index_timeout_s: int = 120
    evidence_max_items: int = 14

    # 报告
    report_system_prompt_file: Path = Field(
        default=PROJECT_ROOT / "config" / "report_system_prompt.txt"
    )
    auto_report: bool = True

    # API
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"
    host: str = "127.0.0.1"
    port: int = 8000

    # 输入视频（manifest 主来源）
    default_video_path: str = "/root/ppt-agent-wildtrack/Wildtrack/Wildtrack/cam1.mp4"

    @property
    def backfill_limit_list(self) -> list[int]:
        return [int(x) for x in self.backfill_limits.split(",") if x.strip()]


def load_profile(path: Path) -> dict[str, Any]:
    """加载 JSON profile（body_demo_profile / report_model_profile 等）。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def profile_path(name: str) -> Path:
    return PROJECT_ROOT / "config" / "profiles" / name


def load_body_profile() -> dict[str, Any]:
    return load_profile(profile_path("single_body_rag_demo_linux_milvus.json"))


def load_report_profile() -> dict[str, Any]:
    return load_profile(profile_path("qwen_single_report_local_v1.json"))


def load_model_registry() -> dict[str, Any]:
    return load_profile(PROJECT_ROOT / "config" / "models.registry.json")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


class ResourceTracker(BaseModel):
    """资源测量记录（D6 需要）。"""

    gpu_name: str | None = None
    gpu_total_mb: int | None = None
    gpu_peak_mb: int | None = None
    qwen_load_s: float | None = None
    qwen_generate_s: float | None = None
    qwen_peak_mb: int | None = None
    qwen_released: bool | None = None
    vision_peak_mb: int | None = None
    notes: list[str] = []
