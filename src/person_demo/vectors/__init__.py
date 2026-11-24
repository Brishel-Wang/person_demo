"""Milvus 客户端（spec 4.2/4.3）。

- 单个人体 collection：body_<space_hash12>_ar1，schema 1.4 字段契约
- FLAT / COSINE / CPU / Strong 一致性；auto_id=false、dynamic_fields=false
- 稳定主键完整 upsert；Strong 读取校验主键、空间与发布数据
"""
from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from person_demo.errors import VectorBackendUnavailable
from person_demo.feature_space import EMBEDDING_DIM, collection_name

log = logging.getLogger(__name__)

_FIELD_CONTRACT: list[dict[str, Any]] = [
    {"name": "embedding_id", "dtype": DataType.VARCHAR, "max_length": 64, "is_primary": True},
    {"name": "dataset_id", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "camera_code", "dtype": DataType.VARCHAR, "max_length": 32},
    {"name": "source_asset_id", "dtype": DataType.VARCHAR, "max_length": 36},
    {"name": "ingest_run_id", "dtype": DataType.VARCHAR, "max_length": 36},
    {"name": "tracklet_revision_id", "dtype": DataType.VARCHAR, "max_length": 36},
    {"name": "feature_space_id", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "publish_seq", "dtype": DataType.INT64},
    {"name": "t_start_local_us", "dtype": DataType.INT64},
    {"name": "t_end_local_us", "dtype": DataType.INT64},
    {"name": "representative_local_us", "dtype": DataType.INT64},
    {"name": "quality", "dtype": DataType.FLOAT},
    {"name": "sample_count", "dtype": DataType.INT32},
    {"name": "embedding", "dtype": DataType.FLOAT_VECTOR, "dim": EMBEDDING_DIM},
]

_OUTPUT_FIELDS = [
    "embedding_id", "dataset_id", "camera_code", "source_asset_id", "ingest_run_id",
    "tracklet_revision_id", "feature_space_id", "publish_seq",
    "t_start_local_us", "t_end_local_us", "representative_local_us",
    "quality", "sample_count",
]


class MilvusStore:
    def __init__(self, uri: str, database: str = "person_demo", token: str = ""):
        self.uri = uri
        self.database = database
        self.token = token
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        try:
            kw: dict[str, Any] = {"alias": "default", "uri": self.uri}
            if self.token:
                kw["token"] = self.token
            connections.connect(**kw)
            if self.database:
                from pymilvus import db as milvus_db

                if not any(d == self.database for d in milvus_db.list_database()):
                    milvus_db.create_database(self.database)
                milvus_db.using_database(self.database)
            self._connected = True
        except Exception as e:  # noqa: BLE001
            raise VectorBackendUnavailable(f"milvus connect failed: {e}")

    def health(self) -> bool:
        try:
            self.connect()
            return True
        except VectorBackendUnavailable:
            return False

    def ensure_collection(self, space_id: str, *, create: bool = True) -> Collection:
        self.connect()
        name = collection_name(space_id)
        if utility.has_collection(name):
            col = Collection(name)
            col.load()
            return col
        if not create:
            raise VectorBackendUnavailable(f"published collection {name} is absent")
        fields = [FieldSchema(**f) for f in _FIELD_CONTRACT]
        schema = CollectionSchema(fields=fields, description="body tracklet prototype (contract 1.4)",
                                  enable_dynamic_field=False)
        col = Collection(name=name, schema=schema, consistency_level="Strong")
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "FLAT", "metric_type": "COSINE", "params": {}},
        )
        col.load()
        return col

    def upsert_rows(self, rows: list[dict]) -> None:
        """按 embedding_id 完整 upsert（幂等）。rows 含 embedding。"""
        if not rows:
            return
        self.connect()
        space_id = rows[0]["feature_space_id"]
        col = self.ensure_collection(space_id)
        for i in range(0, len(rows), 256):
            batch = rows[i : i + 256]
            col.upsert(batch)
        col.flush()

    def verify_ids(self, space_id: str, embedding_ids: list[str]) -> set[str]:
        """Strong 读取校验：返回实际存在且空间匹配的 id 集合。"""
        self.connect()
        col = self.ensure_collection(space_id)
        found: set[str] = set()
        for i in range(0, len(embedding_ids), 256):
            batch = embedding_ids[i : i + 256]
            res = col.query(
                expr=f"embedding_id in {_in_expr(batch)}",
                output_fields=["embedding_id", "feature_space_id"],
                consistency_level="Strong",
            )
            for row in res:
                if row.get("feature_space_id") == space_id:
                    found.add(row["embedding_id"])
        return found

    def search(
        self,
        space_id: str,
        query_vec: list[float],
        expr: str,
        top_k: int,
        bucket_time_expr: str | None = None,
    ) -> list[dict]:
        self.connect()
        col = self.ensure_collection(space_id, create=False)
        full_expr = expr
        if bucket_time_expr:
            full_expr = f"({expr}) and ({bucket_time_expr})" if expr else bucket_time_expr
        res = col.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            expr=full_expr,
            output_fields=_OUTPUT_FIELDS,
            consistency_level="Strong",
        )
        out = []
        for hit in res[0]:
            ent = dict(hit.entity)
            inner = ent.get("entity", ent)  # pymilvus 2.6: 输出字段在 entity.entity 内
            row = dict(inner)
            row["prototype_similarity"] = float(hit.distance)
            out.append(row)
        return out

    def get_by_ids(self, space_id: str, embedding_ids: list[str], *, include_embedding: bool = False) -> dict[str, dict]:
        self.connect()
        col = self.ensure_collection(space_id)
        out: dict[str, dict] = {}
        fields = [*_OUTPUT_FIELDS, "embedding"] if include_embedding else _OUTPUT_FIELDS
        for i in range(0, len(embedding_ids), 256):
            batch = embedding_ids[i : i + 256]
            res = col.query(expr=f"embedding_id in {_in_expr(batch)}",
                            output_fields=fields, consistency_level="Strong")
            for row in res:
                out[row["embedding_id"]] = row
        return out


def _in_expr(ids: list[str]) -> str:
    quoted = ", ".join(f'"{i}"' for i in ids)
    return f"[{quoted}]"
