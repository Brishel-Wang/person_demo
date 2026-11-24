from types import SimpleNamespace

from person_demo.vectors import observations
from person_demo.vision.tracker import ByteTrack, KalmanBox


def test_dense_tracking_emits_all_observed_boxes_without_prediction():
    tall = [50.0, 20.0, 90.0, 190.0]
    moved = [52.0, 22.0, 92.0, 192.0]
    low = [110.0, 45.0, 136.0, 100.0]
    assert KalmanBox(tall).predict() == tall
    tracker = ByteTrack(min_hits=1, include_unmatched=True)
    first = tracker.update([{"bbox": tall, "conf": 0.90}, {"bbox": low, "conf": 0.35}])
    assert [item["bbox"] for item in first] == [tall, low]
    assert first[1]["status"] == "unassociated"
    second = tracker.update([{"bbox": moved, "conf": 0.90}])
    assert second == [{"track_id": first[0]["track_id"], "bbox": moved, "status": "tracked"}]
    assert tracker.update([]) == []


def test_saturated_milvus_time_ranges_are_subdivided_without_duplicates(monkeypatch):
    records = [(4, 0.82), (20, 0.81), (33, 0.91), (50, 0.7), (63, 0.99), (75, 0.92)]
    queries = []

    class Collection:
        def search(self, *, expr, limit, param, **kwargs):
            low = int(expr.split("local_us >= ")[-1].split()[0])
            high = int(expr.split("local_us < ")[-1])
            queries.append((low, high))
            radius = param["params"]["radius"]
            hits = [SimpleNamespace(entity={"local_us": t}, distance=s)
                    for t, s in records if low <= t < high and s > radius]
            return [hits[:limit]]

    monkeypatch.setattr(observations, "_RANGE_LIMIT", 2)
    monkeypatch.setattr(observations, "frame_collection", lambda store, space: Collection())
    hits = list(observations.search_frames(None, "space", [1.0], "publish_seq <= 10", 0, 75, 0.7))
    assert queries[0] == (0, 75) and len(queries) > 2
    assert [row["local_us"] for row, _ in hits] == [4, 20, 33, 63]


def test_packed_artifacts_use_logical_keys_and_keep_loose_files_readable(tmp_path):
    from io import BytesIO
    from uuid import uuid4

    import numpy as np
    import pytest

    from person_demo.retrieval.rerank import rerank_many
    from person_demo.storage import ArtifactBundle, StorageLayout, related_key

    storage = StorageLayout(tmp_path)
    group = str(uuid4())
    writer = ArtifactBundle(storage, group)
    cover = writer.key("crops/sample/00.jpg")
    vector = related_key(cover, "vectors/sample.npy")
    rows = related_key(cover, "row_indexes/sample.jsonl")
    sample = np.eye(2, 512, dtype=np.float32)
    encoded = BytesIO()
    np.save(encoded, sample, allow_pickle=False)
    try:
        writer.write_bytes(cover, b"crop")
        writer.write_bytes(vector, encoded.getvalue())
        writer.write_bytes(rows, b'{"order":0}\n{"order":1}\n')
        writer.finish()
    finally:
        writer.close()
    assert not storage.resolve(cover).exists()
    assert storage.read_bundle(group, [cover, vector]) == {cover: b"crop", vector: encoded.getvalue()}
    scores = rerank_many(sample[0], storage, [
        {"tracklet_revision_id": "sample", "vector_key": vector, "row_index_key": rows},
    ])
    assert scores["sample"] == {"similarity": 0.5, "status": "ok", "note": None}
    storage.write_bytes("row_indexes/history.jsonl", b"old")
    assert storage.read_bytes("row_indexes/history.jsonl") == b"old"
    with pytest.raises(FileNotFoundError):
        storage.read_bytes(related_key(cover, "vectors/missing.npy"))
    with pytest.raises(ValueError):
        storage.read_bytes(f"bundles/{group}/../data")
