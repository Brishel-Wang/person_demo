"""报告 worker（D6）：证据包 → 真实 Qwen 生成 → 校验 → 持久化；失败降级模板。

- report_job 状态机：QUEUED → BUILDING_EVIDENCE → GENERATING → VALIDATING →
  SUCCEEDED / DEGRADED / FAILED；并发 1、总时限 120 秒
- 非法 JSON 或引用至多修复重试 1 次（仍在总时限内）
- 同 search revision、metadata snapshot、evidence hash、模型与 prompt 版本幂等复用
- 模型失败使用模板 provider=template 且 status=DEGRADED，不计真实模型成功
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from person_demo.config import get_settings
from person_demo.db import new_session
from person_demo.db.models import EvidenceBundle, Report, ReportClaim, ReportJob, SearchTask
from person_demo.evidence.bundles import build_bundle
from person_demo.reports.fallback import build_template_report
from person_demo.reports.llm import QwenReportModel
from person_demo.reports.prompt import build_messages
from person_demo.reports.validate import parse_and_validate_model_output

log = logging.getLogger(__name__)

PROMPT_VERSION = "single_video_report_v2"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class ReportJobProcessor:
    """可注入 LLM 工厂，便于测试；gpu_lock 保证单卡串行。"""

    def __init__(self, llm: QwenReportModel | None = None, gpu_lock=None):
        self.llm = llm
        self.gpu_lock = gpu_lock

    def _get_llm(self) -> QwenReportModel:
        if self.llm is None:
            s = get_settings()
            self.llm = QwenReportModel(
                s.model_dir / s.qwen_dir, device=s.device, precision=s.qwen_precision
            )
        return self.llm

    def process(self, report_job_id: uuid.UUID) -> dict:
        session: Session = new_session()
        try:
            job = session.get(ReportJob, report_job_id)
            if job is None:
                return {"status": "FAILED", "error": "job not found"}
            if job.status in ("SUCCEEDED", "DEGRADED"):
                return {"status": job.status, "note": "already_done"}
            task = session.get(SearchTask, job.search_task_id)
            if task is None:
                job.status = "FAILED"
                job.error = "search task not found"
                session.commit()
                return {"status": "FAILED", "error": job.error}

            t_start = time.monotonic()
            force_new = bool((job.timing or {}).get("force_new_version"))
            job.prompt_version = PROMPT_VERSION
            job.status = "BUILDING_EVIDENCE"
            session.commit()
            bundle = build_bundle(session, job.search_task_id, job.search_revision)

            # 幂等复用：同 search revision 且报告记录的 evidence_hash 与当前一致
            # （复核/元数据变化 → 新 evidence hash → 新版本，不改旧报告）
            s = get_settings()
            existing = session.scalar(
                select(Report).where(
                    Report.search_task_id == job.search_task_id,
                    Report.search_revision == job.search_revision,
                    Report.evidence_hash == bundle.evidence_hash,
                    Report.raw_output["prompt_version"].astext == PROMPT_VERSION,
                    Report.report_job_id != job.report_job_id,
                ).order_by(Report.created_at.desc()).limit(1)
            )
            if not force_new and existing is not None and existing.provider == "qwen":  # 模板降级不阻断真实模型重试
                job.status = "SUCCEEDED"
                job.provider = existing.provider
                job.finished_at = _now()
                job.timing = {"reused_existing_report": str(existing.report_id)}
                session.commit()
                return {"status": "SUCCEEDED", "note": "reused_existing_report",
                        "report_id": str(existing.report_id)}

            llm = self._get_llm()
            job.status = "GENERATING"
            job.attempts += 1
            session.commit()

            content = None
            raw_output = None
            validation: dict = {}
            provider = "qwen"
            resource: dict = {}
            try:
                bundle_payload = dict(bundle.payload)
                deadline = t_start + s.qwen_total_timeout_s
                messages, trim = build_messages(bundle_payload, token_counter=llm.count_tokens)
                if not llm.thinking_disabled_via_template:
                    raise ValueError("Qwen chat template cannot disable thinking")
                supplied = set(trim["supplied_evidence_ids"])
                bundle_payload["evidences"] = [
                    item for item in bundle_payload["evidences"] if item["evidence_id"] in supplied
                ]
                bundle_payload["evidence_truncated"] = trim["evidence_truncated"]
                bundle_payload["omitted_count"] = trim["omitted_count"]
                raw_output = {"prompt_version": PROMPT_VERSION, **trim, "attempts": []}
                first_errors = []
                for attempt in range(2):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Qwen report exceeded the total time budget")
                    raw_text, timing = llm.generate(
                        messages, max_new_tokens=s.qwen_max_new_tokens, max_time=remaining,
                    )
                    raw_output["attempts"].append({"text": raw_text, "timing": timing})
                    if time.monotonic() > deadline:
                        raise TimeoutError("Qwen report exceeded the total time budget")
                    content, errors = parse_and_validate_model_output(raw_text, bundle_payload)
                    validation = {"errors": errors, "repairs": attempt, "first_errors": first_errors}
                    if content is not None:
                        break
                    first_errors = errors
                    correction = "重新输出完整 JSON。逐字复制规定的句子和引用，不要加词。"
                    amended = [messages[0], {**messages[1], "content": messages[1]["content"] + "\n" + correction}]
                    if llm.count_tokens(amended) <= s.qwen_max_input_tokens:
                        messages = amended
                if content is None:
                    raise ValueError(f"model output validation failed: {validation.get('errors')}")
                resource = {
                    "model_load_s": llm.load_seconds,
                    "model_peak_mb": llm.peak_memory_mb,
                    "generate_s": timing.get("generate_s"),
                    "device": s.device,
                    "precision": s.qwen_precision,
                    "thinking_disabled_via_template": llm.thinking_disabled_via_template,
                }
            except Exception as e:  # noqa: BLE001
                log.exception("qwen generation failed")
                provider = "template"
                content = build_template_report(bundle.payload, reason=f"{type(e).__name__}: {e}")
                validation = {"errors": [str(e)], "degraded_to_template": True}
                job.status = "DEGRADED"
                resource = {"error": str(e), "device": s.device}
            finally:
                if s.qwen_unload_after_job:
                    resource.update(llm.unload())

            job.status = "VALIDATING"
            session.commit()
            job.status = "SUCCEEDED" if provider == "qwen" else "DEGRADED"
            job.provider = provider
            job.finished_at = _now()
            job.timing = {"total_s": time.monotonic() - t_start}
            job.resource = resource
            session.flush()

            report = Report(
                search_task_id=job.search_task_id,
                search_revision=job.search_revision,
                report_job_id=job.report_job_id,
                provider=provider,
                status=job.status,
                evidence_hash=bundle.evidence_hash,
                content=content,
                raw_output=raw_output,
                validation=validation,
            )
            session.add(report)
            session.flush()
            for claim in content.get("claims", []):
                session.add(ReportClaim(
                    report_id=report.report_id,
                    kind=claim["kind"],
                    text=claim["text"],
                    evidence_ids=claim["evidence_ids"],
                    hypothesis_id=claim.get("hypothesis_id"),
                ))
            session.commit()

            return {"status": job.status, "provider": provider, "report_id": str(report.report_id)}
        except Exception as exc:
            session.rollback()
            job = session.get(ReportJob, report_job_id)
            if job is not None:
                job.status = "FAILED"
                job.error = str(exc)
                job.finished_at = _now()
                session.commit()
            raise
        finally:
            session.close()
