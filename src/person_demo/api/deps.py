"""API 依赖注入：数据库会话、Milvus store、storage 布局、workers 引用。"""
from __future__ import annotations

import asyncio
from threading import Lock

from person_demo.config import get_settings
from person_demo.db import new_session
from person_demo.storage import StorageLayout
from person_demo.vectors import MilvusStore

_settings = get_settings()
_storage = StorageLayout(_settings.storage_root)
_store = MilvusStore(_settings.milvus_uri, _settings.milvus_database, _settings.milvus_token)


def get_db():
    session = new_session()
    try:
        yield session
    finally:
        session.close()


def get_storage() -> StorageLayout:
    return _storage


def get_store() -> MilvusStore:
    return _store


class AppState:
    """workers 状态（lifespan 注入）。"""

    def __init__(self):
        self.gpu_lock = Lock()
        self.video_worker_task: asyncio.Task | None = None
        self.publisher_task: asyncio.Task | None = None
        self.report_worker_task: asyncio.Task | None = None
        self.llm_holder = None  # 惰性加载的报告模型


_app_state: AppState | None = None


def get_app_state() -> AppState:
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state
