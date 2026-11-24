"""storage 目录布局与受控 storage_key 解析（spec 4.1）。

媒体路径只能由受控 storage_key 解析，不接受任意磁盘路径。
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from pathlib import Path

_KEY_RE = re.compile(r"^[a-z0-9_\-./]+$")


_BUNDLE_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def bundle_id(key: str) -> str | None:
    if not key.startswith("bundles/"):
        return None
    parts = key.split("/", 2)
    if (len(parts) != 3 or not _BUNDLE_ID.fullmatch(parts[1]) or not parts[2]
            or not _KEY_RE.fullmatch(key) or ".." in key.split("/")):
        raise ValueError("invalid stored bundle member")
    return parts[1]


def related_key(anchor: str, relative: str) -> str:
    group = bundle_id(anchor)
    return f"bundles/{group}/{relative}" if group else relative


class StorageLayout:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.proxies = self.root / "proxies"
        self.crops = self.root / "crops"
        self.vectors = self.root / "vectors"
        self.frame_vectors = self.root / "frame_vectors"
        self.row_indexes = self.root / "row_indexes"
        self.bundles = self.root / "bundles"
        self.frames = self.root / "frames"
        self.query_images = self.root / "query_images"
        self.reports = self.root / "reports"
        self.exports = self.root / "exports"
        self.tmp = self.root / "tmp"
        for p in (self.proxies, self.crops, self.vectors, self.row_indexes,
                  self.frames, self.query_images, self.reports, self.exports, self.tmp, self.frame_vectors, self.bundles):
            p.mkdir(parents=True, exist_ok=True)

    # ---- 受控 key 构造 ----
    @staticmethod
    def crop_key(tracklet_revision_id: str, n: int) -> str:
        return f"crops/{tracklet_revision_id}/{n:02d}.jpg"

    @staticmethod
    def vectors_key(tracklet_revision_id: str) -> str:
        return f"vectors/{tracklet_revision_id}.npy"

    @staticmethod
    def frame_vectors_key(tracklet_revision_id: str) -> str:
        return f"frame_vectors/{tracklet_revision_id}.npy"

    @staticmethod
    def row_index_key(tracklet_revision_id: str) -> str:
        return f"row_indexes/{tracklet_revision_id}.jsonl"

    @staticmethod
    def frame_key(source_asset_id: str, local_us: int) -> str:
        return f"frames/{source_asset_id}/{local_us}.jpg"

    @staticmethod
    def proxy_key(source_asset_id: str, window_start_us: int, window_end_us: int) -> str:
        return f"proxies/{source_asset_id}_{window_start_us}_{window_end_us}.mp4"

    @staticmethod
    def query_image_key(asset_id: str) -> str:
        return f"query_images/{asset_id}.jpg"

    def resolve(self, storage_key: str) -> Path:
        """受控解析：key 必须是登记过的相对路径形态，禁止 .. 与绝对路径。"""
        if not storage_key or not _KEY_RE.match(storage_key):
            raise ValueError(f"invalid storage_key: {storage_key!r}")
        if ".." in storage_key.split("/") or storage_key.startswith("/"):
            raise ValueError(f"invalid storage_key: {storage_key!r}")
        p = (self.root / storage_key).resolve()
        if not str(p).startswith(str(self.root.resolve()) + "/"):
            raise ValueError("storage_key escapes storage root")
        return p

    # ---- 文件工具 ----
    def write_bytes(self, storage_key: str, data: bytes) -> Path:
        p = self.resolve(storage_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def write_text(self, storage_key: str, text: str) -> Path:
        p = self.resolve(storage_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def bundle_path(self, group: str) -> Path:
        if not _BUNDLE_ID.fullmatch(group):
            raise ValueError("invalid bundle id")
        return self.resolve(f"bundles/{group}.sqlite3")

    def read_bytes(self, key: str) -> bytes:
        group = bundle_id(key)
        if group is None:
            return self.resolve(key).read_bytes()
        return self.read_bundle(group, [key])[key]

    def read_bundle(self, group: str, keys: list[str]) -> dict[str, bytes]:
        if any(bundle_id(key) != group for key in keys):
            raise ValueError("bundle read mixes artifact groups")
        path = self.bundle_path(group)
        if not path.is_file():
            raise FileNotFoundError(path)
        result = {}
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            for start in range(0, len(keys), 500):
                subset = keys[start : start + 500]
                markers = ", ".join("?" for _ in subset)
                result.update(db.execute(
                    f"SELECT key, data FROM artifacts WHERE key IN ({markers})", subset,
                ))
        for key in keys:
            if key not in result:
                raise FileNotFoundError(key)
        return result

    def exists(self, storage_key: str) -> bool:
        if bundle_id(storage_key) is None:
            return self.resolve(storage_key).exists()
        try:
            self.read_bytes(storage_key)
            return True
        except FileNotFoundError:
            return False


class ArtifactBundle:
    """Build one immutable file for an ingest run on staging storage."""

    def __init__(self, storage: StorageLayout, group: str):
        self.group = group
        self.db = sqlite3.connect(storage.bundle_path(group))
        self.db.execute("CREATE TABLE artifacts (key TEXT PRIMARY KEY, data BLOB NOT NULL)")

    def key(self, relative: str) -> str:
        return f"bundles/{self.group}/{relative}"

    def write_bytes(self, key: str, data: bytes) -> None:
        self.db.execute("INSERT INTO artifacts (key, data) VALUES (?, ?)", (key, data))

    def finish(self) -> None:
        self.db.commit()
        if self.db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("artifact bundle integrity check failed")

    def close(self) -> None:
        self.db.close()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
