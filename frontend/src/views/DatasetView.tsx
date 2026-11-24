import { useEffect, useState } from "react";
import { api, DatasetStatus, fmtLocalUs } from "../api";
import type { VideoSource } from "../videoRagApi";

export default function DatasetView({ source }: { source?: VideoSource }) {
  const [status, setStatus] = useState<DatasetStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTick((x) => x + 1), 4000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    api
      .datasetStatus()
      .then(setStatus)
      .catch((e) => setErr(String(e)));
  }, [tick]);

  const doImport = async () => {
    setBusy(true);
    setErr(null);
    try {
      await api.importDataset();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  if (err && !status) return <div className="error">{err}</div>;

  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h1 style={{ margin: 0 }}>数据集状态（单路 · cam_01 · 片内时间）</h1>
          <button className="btn primary" disabled={busy} onClick={doImport}>
            {busy ? "导入中…" : "导入基础 200 秒"}
          </button>
        </div>
        {err && <div className="error">{err}</div>}
        {status && (
          <dl className="kv" style={{ marginTop: 12 }}>
            <dt>状态</dt>
            <dd>
              {status.status}{" "}
              {status.status === "ready" ? (
                <span className="badge ok">可检索</span>
              ) : ["importing", "analyzing", "indexing_dense", "publishing"].includes(status.status) ? (
                <span className="badge info">处理中</span>
              ) : (
                <span className="badge warn">未就绪</span>
              )}
            </dd>
            <dt>source_asset_id</dt>
            <dd>{status.source_asset_id ?? "-"}</dd>
            <dt>素材时间范围</dt>
            <dd>
              {status.resolved_window_us
                ? `${fmtLocalUs(status.resolved_window_us[0])} → ${fmtLocalUs(status.resolved_window_us[1])}（片内时间）`
                : "-"}
            </dd>
            {source?.indexed_detector && <>
              <dt>当前索引检测器</dt>
              <dd>{source.indexed_detector.weights ?? "权重未登记"} · 输入 {source.indexed_detector.imgsz}</dd>
            </>}
            {source?.configured_detector && <>
              <dt>新分析检测器</dt>
              <dd>{source.configured_detector.weights} · 输入 {source.configured_detector.imgsz}</dd>
            </>}
            <dt>跟踪片段数 / 观测数</dt>
            <dd>
              {status.tracklet_count} / {status.observation_count}
            </dd>
            <dt>发布水位</dt>
            <dd>{status.watermark ?? "-"}</dd>
            <dt>覆盖</dt>
            <dd>
              {status.coverage
                ? `${status.coverage.status}（已覆盖到 ${fmtLocalUs(status.coverage.covered_through_us)}）`
                : "-"}
            </dd>
            <dt>最新处理任务</dt>
            <dd>
              {status.latest_run
                ? `${status.latest_run.status} · 窗口 0 → ${fmtLocalUs(status.latest_run.window_us?.[1] ?? null)}${["analyzing", "indexing_dense", "publishing"].includes(status.latest_run.status ?? "") ? "（索引正在处理）" : ""}`
                : "-"}
            </dd>
            <dt>编码/尺寸/帧率</dt>
            <dd>
              {status.detail
                ? `${status.detail["codec"]} ${status.detail["width"]}x${status.detail["height"]} ${status.detail["avg_fps"]}fps`
                : "-"}
            </dd>
            <dt>真实采集时间 / 地点</dt>
            <dd>
              {String((status.detail as Record<string, unknown>)?.["recording_time_status"] ?? "unknown")} /{" "}
              {String((status.detail as Record<string, unknown>)?.["location_status"] ?? "unknown")}（未补录，不阻塞演示）
            </dd>
          </dl>
        )}
      </div>
      <div className="notice">
        知识库按 10 FPS 分析 cam1.mp4，工作台显示当前已发布的检索范围。片内时间从原片第一帧起算。
      </div>
    </div>
  );
}
