"""Searchable OSNet vectors keyed by actual SQL observations."""

from collections.abc import Iterator

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from person_demo.errors import VectorBackendUnavailable
from person_demo.feature_space import EMBEDDING_DIM
from person_demo.vectors import MilvusStore, _in_expr

_FIELDS = ("observation_id", "source_asset_id", "ingest_run_id", "tracklet_revision_id",
           "feature_space_id", "publish_seq", "local_us")


def frame_collection(store: MilvusStore, space_id: str, *, create: bool = False) -> Collection:
    store.connect()
    name = f"body_frames_{space_id[:12]}_v2"
    if utility.has_collection(name):
        return Collection(name)
    if not create:
        raise VectorBackendUnavailable(f"published observation collection {name} is absent")
    fields = [
        FieldSchema("observation_id", DataType.VARCHAR, max_length=36, is_primary=True),
        *(FieldSchema(key, DataType.VARCHAR, max_length=36) for key in _FIELDS[1:4]),
        FieldSchema("feature_space_id", DataType.VARCHAR, max_length=64),
        FieldSchema("publish_seq", DataType.INT64),
        FieldSchema("local_us", DataType.INT64),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    col = Collection(name, CollectionSchema(fields, enable_dynamic_field=False), consistency_level="Strong")
    col.create_index("embedding", {"index_type": "FLAT", "metric_type": "COSINE", "params": {}})
    col.load()
    return col


def publish_frames(store: MilvusStore, space_id: str, frames: list[dict]) -> set[str]:
    """Upsert by observation ID, then Strong-read the exact SQL mapping before marking ready."""
    if not frames:
        return set()
    col = frame_collection(store, space_id, create=True)
    for i in range(0, len(frames), 1024):
        col.upsert(frames[i : i + 1024], timeout=45)
    verified = set()
    for i in range(0, len(frames), 1024):
        batch = {row["observation_id"]: row for row in frames[i : i + 1024]}
        for got in col.query(
            f"observation_id in {_in_expr(list(batch))}", output_fields=list(_FIELDS),
            consistency_level="Strong", timeout=45,
        ):
            expected = batch[got["observation_id"]]
            if all(got[key] == expected[key] for key in _FIELDS):
                verified.add(got["observation_id"])
    return verified


_RANGE_LIMIT = 4096


def search_frames(
    store: MilvusStore, space_id: str, embedding: list[float], expr: str,
    start: int, end: int, min_score: float,
) -> Iterator[tuple[dict, float]]:
    """Return every FLAT range match, splitting any saturated request by timestamp."""
    col = frame_collection(store, space_id)
    windows = [(start, end)]
    while windows:
        low, high = windows.pop()
        result = col.search(
            data=[embedding], anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"radius": min_score}},
            expr=f"({expr}) and local_us >= {low} and local_us < {high}",
            limit=_RANGE_LIMIT, output_fields=list(_FIELDS), consistency_level="Strong", timeout=45,
        )[0]
        if len(result) == _RANGE_LIMIT:
            if high - low <= 1:
                raise VectorBackendUnavailable("too many observations for an exact search at one timestamp")
            mid = (low + high) // 2
            windows.extend([(mid, high), (low, mid)])
            continue
        for hit in result:
            row = dict(hit.entity)
            yield row.get("entity", row), float(hit.distance)
