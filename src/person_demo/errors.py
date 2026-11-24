"""统一错误码与异常（spec 6.3）。

错误结构统一为 {code, message, details, request_id}。
"""
from __future__ import annotations

from typing import Any


class DemoError(Exception):
    code = "INTERNAL_ERROR"
    http_status = 500

    def __init__(self, message: str, details: Any = None, request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.request_id = request_id

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "request_id": self.request_id,
        }


class SingleSourceRequired(DemoError):
    code = "SINGLE_SOURCE_REQUIRED"
    http_status = 400


class TargetAmbiguous(DemoError):
    code = "TARGET_AMBIGUOUS"
    http_status = 400


class InvalidTimeRange(DemoError):
    code = "INVALID_TIME_RANGE"
    http_status = 400


class TimebaseInvalid(DemoError):
    code = "TIMEBASE_INVALID"
    http_status = 400


class OriginUnresolved(DemoError):
    code = "ORIGIN_UNRESOLVED"
    http_status = 422


class ModelUnavailable(DemoError):
    code = "MODEL_UNAVAILABLE"
    http_status = 503


class FeatureSpaceMismatch(DemoError):
    code = "FEATURE_SPACE_MISMATCH"
    http_status = 409


class CapabilityDisabled(DemoError):
    code = "CAPABILITY_DISABLED"
    http_status = 400


class VectorBackendUnavailable(DemoError):
    code = "VECTOR_BACKEND_UNAVAILABLE"
    http_status = 503


class RerankArtifactCorrupt(DemoError):
    code = "RERANK_ARTIFACT_CORRUPT"
    http_status = 500


class EvidenceExpired(DemoError):
    code = "EVIDENCE_EXPIRED"
    http_status = 410


class RevisionConflict(DemoError):
    code = "REVISION_CONFLICT"
    http_status = 409


class NotFound(DemoError):
    code = "NOT_FOUND"
    http_status = 404


class InvalidManifest(DemoError):
    code = "INVALID_MANIFEST"
    http_status = 400


class DuplicateImport(DemoError):
    code = "DUPLICATE_IMPORT"
    http_status = 409


class InvalidRequest(DemoError):
    code = "INVALID_REQUEST"
    http_status = 400


class InsufficientPositiveGallery(DemoError):
    """独立评测正样本不足是评测状态，不作为软件错误抛出；保留类用于 evaluate CLI。"""

    code = "INSUFFICIENT_POSITIVE_GALLERY"
    http_status = 200
