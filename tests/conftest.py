"""pytest fixtures：测试数据库与存储。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("PERSON_DEMO_STORAGE_ROOT", "/tmp/person_demo_test_storage")
os.environ.setdefault(
    "PERSON_DEMO_DATABASE_URL",
    "postgresql+psycopg2://person_demo_test:person_demo_test@127.0.0.1:5433/person_demo_test",
)
os.environ.setdefault("PERSON_DEMO_MILVUS_URI", "http://127.0.0.1:19530")
os.environ.setdefault("PERSON_DEMO_MILVUS_DATABASE", "person_demo_test")
os.environ.setdefault("PERSON_DEMO_DEVICE", "cpu")


@pytest.fixture(scope="session")
def db_engine():
    from sqlalchemy import create_engine, text

    from person_demo.config import get_settings
    from person_demo.db import Base

    s = get_settings()
    engine = create_engine(s.database_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = factory()
    # 每用例清表，避免用例间数据污染
    with db_engine.begin() as conn:
        conn.execute(text(
            "TRUNCATE cameras, source_assets, ingest_runs, tracklets, observations, "
            "vector_outbox, publish_watermarks, coverage_segments, query_assets, "
            "search_tasks, search_candidates, reviews, evidence_bundles, report_jobs, "
            "reports, report_claims, metadata_revisions, tombstones, asset_proxies, "
            "dataset_status CASCADE"
        ))
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def storage(tmp_path):
    from person_demo.storage import StorageLayout

    return StorageLayout(tmp_path / "storage")
