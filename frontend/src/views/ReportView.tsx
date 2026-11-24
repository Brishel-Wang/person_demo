import { useEffect, useState } from "react";
import { api, Report, fmtLocalUs } from "../api";
import { EvidencePanel } from "./SearchView";

export default function ReportView({ taskId }: { taskId: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [evidenceId, setEvidenceId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const latestId = jobId ?? (await api.searchResults(taskId, 1, 1)).report_job_id;
        if (!active) return;
        if (!latestId) {
          timer = setTimeout(refresh, 2500);
          return;
        }
        const job = await api.reportJob(latestId);
        if (!active) return;
        setJobStatus(job.status);
        if (job.report_id) {
          const next = await api.report(job.report_id);
          if (active) setReport(next);
        } else if (job.status === "FAILED" || job.status === "CANCELLED") {
          setErr(job.error ?? job.status);
        } else {
          timer = setTimeout(refresh, 2500);
        }
      } catch (e) {
        if (active) setErr(String(e));
      }
    };
    refresh();
    return () => { active = false; clearTimeout(timer); };
  }, [taskId, jobId]);

  const generate = async () => {
    setBusy(true);
    setErr(null);
    try {
      const task = await api.searchTask(taskId);
      const job = await api.createReport(taskId, task.search_revision);
      setJobStatus("QUEUED");
      setReport(null);
      setJobId(job.report_job_id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h1 style={{ margin: 0 }}>检索报告</h1>
          <button className="btn primary" disabled={busy || ["QUEUED", "BUILDING_EVIDENCE", "GENERATING", "VALIDATING"].includes(jobStatus ?? "")} onClick={generate}>
            生成 / 重新生成报告
          </button>
        </div>
        {err && <div className="error">{err}</div>}
        {!report && (
          <p className="small">
            报告任务状态：{jobStatus ?? "未创建"}（自动报告需检索完成后触发；模型失败不影响回放与复核）
          </p>
        )}
        {report && report.provider === "template" && (
          <div className="notice">
            本报告由模板降级生成（provider=template, status=DEGRADED），
            <b>不计入 D6 真实模型验收通过</b>。
          </div>
        )}
        {report && report.provider === "qwen" && report.prompt_version !== "single_video_report_v2" && (
          <p className="notice">这是历史报告。重新生成可应用当前的候选与人工确认措辞校验。</p>
        )}
        {report && report.provider === "qwen" && (
          <p className="small">
            由 Qwen/Qwen3.5-9B 真实生成（provider=qwen, status={report.status}）· 生成说明待复核 ·
            报告 revision {report.search_revision}
          </p>
        )}
        {report && (
          <>
            <p style={{ whiteSpace: "pre-wrap" }}>{report.content.narrative}</p>
            {report.content.claims.map((c, i) => (
              <div key={i} className={"claim" + (c.kind === "reviewed_observation" ? " reviewed" : "")}>
                <div>
                  [{c.kind === "reviewed_observation" ? "人工确认" : "相似候选"}] {c.text}
                </div>
                <div className="small">
                  引用：{" "}
                  {c.evidence_ids.map((eid) => (
                    <button key={eid} className="btn" onClick={() => setEvidenceId(eid)}>
                      查看证据 {eid.slice(0, 8)}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {report.facts && (
              <>
                <p className="small">事实记录 · 覆盖 {report.facts.coverage_status} · 共 {report.facts.total_candidates} 条 · 省略 {report.facts.omitted_count} 条</p>
                <table>
                  <thead><tr><th>片内时间</th><th>复核</th><th>重排分</th><th>证据</th></tr></thead>
                  <tbody>{report.facts.evidences.map((e) => (
                    <tr key={e.evidence_id}>
                      <td>{fmtLocalUs(e.t_start_local_us)} – {fmtLocalUs(e.t_end_local_us)}</td>
                      <td>{e.review_status === "confirmed" ? "已确认" : "待复核"}
                        {e.same_continuation_as_query && " · 同跟踪段后续片段"}</td>
                      <td>{e.rerank_similarity?.toFixed(4) ?? "—"}</td>
                      <td><button className="btn" onClick={() => setEvidenceId(e.evidence_id)}>打开</button></td>
                    </tr>
                  ))}</tbody>
                </table>
              </>
            )}
            <a className="btn" href={api.exportUrl(taskId, "json", report.search_revision, report.report_id)} download>导出当前报告与候选</a>
            <h1 style={{ fontSize: 15, marginTop: 16 }}>缺口说明</h1>
            <ul>
              {report.content.gaps.map((g, i) => (
                <li key={i}>{g}</li>
              ))}
            </ul>
            <h1 style={{ fontSize: 15 }}>下一步信息需求</h1>
            <ul>
              {report.content.next_information_needed.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          </>
        )}
      </div>
      {evidenceId && <EvidencePanel key={evidenceId} evidenceId={evidenceId} />}
    </div>
  );
}
