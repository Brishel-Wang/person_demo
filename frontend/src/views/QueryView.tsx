import { useEffect, useRef, useState } from "react";
import { api, DatasetStatus, FrameItem, QueryAsset, fmtLocalUs } from "../api";

export default function QueryView(props: {
  onQueryCreated: (qa: QueryAsset) => void;
}) {
  const [status, setStatus] = useState<DatasetStatus | null>(null);
  const [startUs, setStartUs] = useState(0);
  const [frames, setFrames] = useState<FrameItem[]>([]);
  const [selected, setSelected] = useState<FrameItem | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [bboxInput, setBboxInput] = useState("[]");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.datasetStatus().then(setStatus).catch((e) => setErr(String(e)));
  }, []);

  const demoEnd = status?.demo_window_us?.[1] ?? 0;
  useEffect(() => {
    if (!status?.source_asset_id || !demoEnd) return;
    setSelected(null);
    setFrames([]);
    let active = true;
    api.frames(status.source_asset_id, 2000, status.demo_ingest_run_id ?? undefined,
      Math.min(demoEnd, startUs + 30_000_000), startUs)
      .then((r) => { if (active) setFrames(r.items); })
      .catch((e) => { if (active) setErr(String(e)); });
    return () => { active = false; };
  }, [status?.source_asset_id, status?.demo_ingest_run_id, startUs, demoEnd]);

  const createFromFrame = async () => {
    if (!selected || !status?.source_asset_id) return;
    setBusy(true);
    setErr(null);
    try {
      const qa = await api.createQueryFromFrame(status.source_asset_id, selected.observation_id);
      props.onQueryCreated(qa);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const createFromUpload = async () => {
    const f = fileRef.current?.files?.[0];
    if (!f) return;
    let bbox: number[];
    try {
      bbox = JSON.parse(bboxInput || "[]");
    } catch {
      setErr("bbox 必须是 [x1,y1,x2,y2] JSON 数组");
      return;
    }
    if (bbox.length !== 4) {
      setErr("bbox 必须是 [x1,y1,x2,y2] JSON 数组");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const qa = await api.createQueryUpload(f, bbox);
      props.onQueryCreated(qa);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h1>选择查询目标</h1>
        <p className="small">
          方式一：从已分析画面选择一个人体（来源由服务端确定，参与来源排除）。共{" "}
          {frames.length} 张代表图。
        </p>
        {err && <div className="error">{err}</div>}
        {demoEnd > 0 && (
          <label>片内时间段{" "}
            <select value={startUs} onChange={(event) => setStartUs(Number(event.target.value))}>
              {Array.from({ length: Math.ceil(demoEnd / 30_000_000) }, (_, i) => {
                const us = i * 30_000_000;
                return <option key={us} value={us}>{fmtLocalUs(us)} – {fmtLocalUs(Math.min(demoEnd, us + 30_000_000))}</option>;
              })}
            </select>
          </label>
        )}
        <div className="crops-grid" style={{ maxHeight: 420, overflowY: "auto" }}>
          {frames.map((f) => (
            <div
              key={f.observation_id}
              data-observation-id={f.observation_id}
              className={"crop-item" + (selected?.observation_id === f.observation_id ? " selected" : "")}
              onClick={() => setSelected(f)}
            >
              {f.crop_url ? <img src={f.crop_url} alt="" loading="lazy" /> : <div className="t">无图</div>}
              <div className="t">{fmtLocalUs(f.local_us)}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12 }}>
          <button className="btn primary" disabled={!selected || busy} onClick={createFromFrame}>
            以选中人体创建查询图（来源可信）
          </button>
        </div>
      </div>

      <div className="card">
        <h1>上传人体照片（来源未知）</h1>
        <div className="upload-box">
          <input ref={fileRef} type="file" accept="image/*" />
          <div style={{ marginTop: 8 }}>
            目标框 [x1,y1,x2,y2]：
            <input
              value={bboxInput}
              onChange={(e) => setBboxInput(e.target.value)}
              placeholder="例如 [100,50,400,700]"
              style={{ marginLeft: 8, width: 220 }}
            />
          </div>
          <button className="btn" style={{ marginTop: 10 }} disabled={busy} onClick={createFromUpload}>
            上传并编码（origin_unknown，不声称已完成来源排除）
          </button>
        </div>
      </div>
    </div>
  );
}
