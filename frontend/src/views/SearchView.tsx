import { useEffect, useRef, useState } from "react";
import {
  api,
  Candidate,
  Evidence,
  QueryAsset,
  SearchResults,
  fmtLocalUs,
} from "../api";

export function EvidencePanel({ evidenceId }: { evidenceId: string }) {
  const [ev, setEv] = useState<Evidence | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selectedUs, setSelectedUs] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let active = true;
    api.evidence(evidenceId).then((e) => {
      if (active) {
        setEv(e);
        setSelectedUs(e.t_start_local_us);
      }
    }).catch((e) => {
      if (active) setErr(String(e));
    });
    return () => { active = false; };
  }, [evidenceId]);

  const seekTo = (localUs: number) => {
    setSelectedUs(localUs);
    if (videoRef.current) videoRef.current.currentTime = Math.max(0, (localUs + (ev?.video_timeline_offset_us ?? 0)) / 1_000_000);
  };

  if (err) return <div className="error">{err}</div>;
  if (!ev) return <div className="small">证据加载中…</div>;

  const obs = ev.bbox_sequence.find((item) => item.local_us === selectedUs);
  return (
    <div className="card">
      <h1>证据回放（{fmtLocalUs(ev.t_start_local_us)} → {fmtLocalUs(ev.t_end_local_us)}，片内时间）</h1>
      <div className="row">
        <div style={{ flex: "1 1 460px" }}>
          <video ref={videoRef} controls preload="metadata" src={ev.original_video_url}
                 onLoadedMetadata={() => seekTo(selectedUs ?? ev.t_start_local_us)} />
          {obs && (
            <div>
              <p className="small">原片观测帧 · {fmtLocalUs(obs.local_us)}</p>
              <div style={{ position: "relative" }}>
                <img style={{ display: "block", width: "100%" }} alt="原片观测帧"
                     src={`/v1/evidence/${ev.evidence_id}/frames/${obs.local_us}`} />
                <div style={{
                  position: "absolute", pointerEvents: "none", border: "2px solid #47fa81",
                  left: `${obs.bbox[0] / obs.frame_w * 100}%`,
                  top: `${obs.bbox[1] / obs.frame_h * 100}%`,
                  width: `${(obs.bbox[2] - obs.bbox[0]) / obs.frame_w * 100}%`,
                  height: `${(obs.bbox[3] - obs.bbox[1]) / obs.frame_h * 100}%`,
                }} />
              </div>
            </div>
          )}
        </div>
        <div style={{ flex: "1 1 320px" }}>
          <dl className="kv">
            <dt>tracklet_revision_id</dt>
            <dd>{ev.tracklet_revision_id}</dd>
            <dt>local_track_id / chunk_no</dt>
            <dd>
              {ev.local_track_id} / {ev.chunk_no}
            </dd>
            <dt>continuation</dt>
            <dd>
              {ev.same_continuation_as_query ? (
                <span className="badge continuation">同一跟踪段后续片段（不计独立重识别）</span>
              ) : (
                "独立候选"
              )}
            </dd>
            <dt>原型分 / 重排分</dt>
            <dd>
              {ev.prototype_similarity.toFixed(4)} / {ev.rerank_similarity?.toFixed(4) ?? "—"}
            </dd>
            <dt>观测数（bbox 序列）</dt>
            <dd>{ev.bbox_sequence.length}</dd>
            <dt>复核状态</dt>
            <dd>{ev.review_status}</dd>
            {ev.suspicious && (
              <>
                <dt>可疑标记</dt>
                <dd className="badge warn">{ev.suspicious_reason}</dd>
              </>
            )}
          </dl>
          <div className="crops-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(84px,1fr))" }}>
            {ev.representative_images.map((r) => (
              <div key={r.local_us} className="crop-item" onClick={() => seekTo(r.local_us)}>
                <img src={`/v1/media/${r.storage_key}`} alt="" />
                <div className="t">{fmtLocalUs(r.local_us)}</div>
              </div>
            ))}
          </div>
          <p className="small">点击代表图可定位原片，目标框显示在对应的实际观测帧。</p>
        </div>
      </div>
    </div>
  );
}

export default function SearchView(props: {
  query: QueryAsset;
  onReportReady: (taskId: string) => void;
}) {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResults | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [evidenceId, setEvidenceId] = useState<string | null>(null);
  const [timeSort, setTimeSort] = useState(false);
  const [evaluationScope, setEvaluationScope] = useState<"intra_camera_demo" | "intra_camera_holdout">(
    "intra_camera_demo"
  );

  const startSearch = async () => {
    setBusy(true);
    setErr(null);
    setResults(null);
    setEvidenceId(null);
    try {
      const dataset = await api.datasetStatus();
      const sourceId = props.query.source_asset_id ?? dataset.source_asset_id;
      if (!sourceId || !dataset.demo_window_us) throw new Error("当前视频尚未导入");
      const r = await api.createSearch(
        props.query.asset_id, sourceId, evaluationScope, dataset.demo_window_us[1]
      );
      setTaskId(r.task_id);
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const t = await api.searchTask(taskId);
        if (cancelled) return;
        if (t.status === "COMPLETED" || t.status === "PARTIAL" || t.status === "FAILED" || t.status === "CANCELLED") {
          if (t.status === "FAILED" || t.status === "CANCELLED") {
            setErr(t.error ?? t.status);
            setBusy(false);
            return;
          }
          const res = await api.searchResults(taskId, 1, 5);
          if (!cancelled) {
            setResults(res);
            setBusy(false);
            props.onReportReady(taskId);
          }
          return;
        }
      } catch (e) {
        if (!cancelled) { setErr(String(e)); setBusy(false); }
        return;
      }
      setTimeout(poll, 2000);
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const doReview = async (c: Candidate, action: "confirm" | "reject") => {
    if (!taskId) return;
    try {
      await api.review(taskId, c.hit_id, action, c.review_revision);
      const res = await api.searchResults(taskId, results?.page ?? 1, 5);
      setResults(res);
    } catch (e) {
      setErr(String(e));
    }
  };

  const goToPage = async (page: number) => {
    if (!taskId) return;
    try {
      setResults(await api.searchResults(taskId, page, 5));
    } catch (e) {
      setErr(String(e));
    }
  };
  const items = results?.items ?? [];
  const display = timeSort
    ? [...items].sort((a, b) => (a.t_start_local_us ?? 0) - (b.t_start_local_us ?? 0))
    : items;

  return (
    <div>
      <div className="card">
        <h1>提交检索</h1>
        <div className="row">
          <img
            src={props.query.image_url ?? ""}
            alt="query"
            style={{ width: 96, height: 150, objectFit: "cover", borderRadius: 6 }}
          />
          <dl className="kv">
            <dt>查询图</dt>
            <dd>
              {props.query.asset_id}（{props.query.kind}）
            </dd>
            <dt>来源</dt>
            <dd>
              {props.query.origin_status === "verified"
                ? `已核验（tracklet ${props.query.origin_tracklet_revision_id?.slice(0, 8)}…）`
                : "origin_unknown（不声称已完成来源排除）"}
            </dd>
            <dt>片内时间</dt>
            <dd>{fmtLocalUs(props.query.local_us)}</dd>
            <dt>评测口径</dt>
            <dd>
              <select
                value={evaluationScope}
                onChange={(e) => setEvaluationScope(e.target.value as never)}
              >
                <option value="intra_camera_demo">intra_camera_demo（功能演示：排除来源 chunk）</option>
                <option value="intra_camera_holdout">intra_camera_holdout（独立评测：排除完整来源组）</option>
              </select>
            </dd>
          </dl>
        </div>
        <button className="btn primary" style={{ marginTop: 10 }} disabled={busy} onClick={startSearch}>
          {busy ? "检索中…" : "开始检索（0–200s，60 秒桶 × Top20）"}
        </button>
        {err && <div className="error">{err}</div>}
        {results && (
          <p className="small">
            状态 {results.status} · 保留候选 {results.total} ·{" "}
            {results.snapshot && (
              <>快照：水位 {(results.snapshot as never)["frozen_watermark"] ?? "-"} · 覆盖{" "}
                {String((results.snapshot.coverage as { status?: string } | undefined)?.status ?? "-")} · {props.query.origin_status === "verified" ? "已排除来源片段" : "上传图来源未知"}</>
            )}
          </p>
        )}
      </div>

      {results && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h1 style={{ margin: 0 }}>检索候选</h1>
            <div>
              <button className="btn" onClick={() => setTimeSort((v) => !v)}>
                {timeSort ? "本页按重排分排序" : "本页按片内时间排序"}
              </button>
              <a className="btn" href={api.exportUrl(taskId!, "json")} download style={{ marginLeft: 8 }}>
                导出 JSON
              </a>
              <a className="btn" href={api.exportUrl(taskId!, "csv")} download style={{ marginLeft: 8 }}>
                导出 CSV
              </a>
            </div>
          </div>
          <p className="small">
            相似候选默认 pending_review，需要人工确认或排除；TopK 在图库没有目标时也可能返回相似人，
            不自动称为身份命中。同一跟踪段的后续片段已明确标记，不计独立重识别成果。
          </p>
          {display.map((c) => (
            <div className="candidate" key={c.hit_id}>
              <div>
                <div className="imgs">
                  {c.representative_images.map((r) => (
                    <img key={r.local_us} src={r.url} alt="" />
                  ))}
                </div>
                <div className="small" style={{ marginTop: 6 }}>
                  #{c.rank_position} · 片内 {fmtLocalUs(c.t_start_local_us)} → {fmtLocalUs(c.t_end_local_us)}
                </div>
              </div>
              <div>
                <div>
                  {c.same_continuation_as_query && (
                    <span className="badge continuation">同一跟踪段后续片段</span>
                  )}
                  {c.review_status === "confirmed" && <span className="badge ok">已确认</span>}
                  {c.review_status === "rejected" && <span className="badge err">已排除</span>}
                  {c.review_status === "pending_review" && <span className="badge warn">待复核</span>}
                  {c.suspicious && <span className="badge warn">可疑混人标记</span>}
                </div>
                <div>
                  原型分 {c.prototype_similarity.toFixed(4)}
                  <div className="meter">
                    <div style={{ width: `${Math.max(0, Math.min(100, c.prototype_similarity * 100))}%` }} />
                  </div>
                  重排分 {c.rerank_similarity?.toFixed(4) ?? "—"}
                  <div className="meter">
                    <div
                      style={{
                        width: `${Math.max(0, Math.min(100, (c.rerank_similarity ?? 0) * 100))}%`,
                        background: "#157347",
                      }}
                    />
                  </div>
                </div>
                <div className="small">
                  tracklet {c.tracklet_revision_id.slice(0, 8)}… · 代表图 {c.sample_count} 张 ·{" "}
                  <a onClick={() => setEvidenceId(c.evidence_id)} style={{ cursor: "pointer", color: "#1264a3" }}>
                    打开证据回放
                  </a>
                </div>
                <div style={{ marginTop: 8 }}>
                  <button
                    className="btn"
                    disabled={c.review_status === "confirmed"}
                    onClick={() => doReview(c, "confirm")}
                  >
                    确认
                  </button>
                  <button
                    className="btn"
                    style={{ marginLeft: 6 }}
                    disabled={c.review_status === "rejected"}
                    onClick={() => doReview(c, "reject")}
                  >
                    排除
                  </button>
                </div>
              </div>
            </div>
          ))}
          {results.total > 5 && (
            <div className="row">
              <button className="btn" disabled={results.page <= 1} onClick={() => goToPage(results.page - 1)}>上一页</button>
              <span>第 {results.page} / {Math.ceil(results.total / 5)} 页</span>
              <button className="btn" disabled={results.page * 5 >= results.total} onClick={() => goToPage(results.page + 1)}>下一页</button>
            </div>
          )}
          {display.length === 0 && (
            <div className="notice">
              没有候选（或因来源排除、覆盖不足）。零候选不等于证明目标未出现；请检查任务状态与覆盖。
            </div>
          )}
        </div>
      )}

      {evidenceId && <EvidencePanel key={evidenceId} evidenceId={evidenceId} />}
    </div>
  );
}
