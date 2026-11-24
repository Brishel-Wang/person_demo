import { PointerEvent, ReactNode, useEffect, useRef, useState } from "react";
import { api, QueryAsset } from "../api";
import { Box, clock, PickFrame, rag, VideoSource } from "../videoRagApi";
import { Icon } from "./RagIcons";

export function RagDialog({ title, onClose, children, label }: { title: string; onClose: () => void; children: ReactNode; label?: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  return <dialog ref={dialog} className="rag-dialog" aria-label={label ?? title}
    onCancel={event => { event.preventDefault(); onClose(); }}>
    <header><h2>{title}</h2><button type="button" aria-label="关闭" className="icon-button" onClick={onClose}><Icon name="close"/></button></header>
    {children}
  </dialog>;
}

export function FrameTargetPicker({ source, initialTime, onClose, onTarget }: {
  source: VideoSource; initialTime: number; onClose: () => void; onTarget: (qa: QueryAsset) => void;
}) {
  const [timeUs, setTimeUs] = useState(initialTime);
  const [result, setResult] = useState<PickFrame>();
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    setSelected("");
    const timer = setTimeout(async () => {
      try {
        const data = await rag.frame(source.source_asset_id, Math.max(0, Math.min(source.indexed_end_us - 1, timeUs)));
        if (live) setResult(data);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      } finally { if (live) setLoading(false); }
    }, 120);
    return () => { live = false; clearTimeout(timer); };
  }, [timeUs, source.source_asset_id, source.indexed_end_us]);
  return <RagDialog title="从原片选人" onClose={onClose}>
    <div className="picker-range">
      <Icon name="clock" size={16}/><input aria-label="选人时间" type="range" min={0} max={Math.max(0, source.indexed_end_us - 1)}
        step={100_000} value={timeUs} onChange={e => setTimeUs(Number(e.target.value))} />
      <span>{clock(result?.local_us ?? timeUs, true)}</span>
    </div>
    <div className="picker-preview">
      {result?.frame_url && <img src={result.frame_url} alt="原片分析帧" />}
      {!loading && !result?.frame_url && <p>这个时间附近没有可选的人体。移动时间轴查看更多画面。</p>}
      {!loading && result?.items.map((item, index) => <button key={item.observation_id} type="button"
        data-observation-id={item.observation_id} aria-label={`选择目标 ${index + 1}`} title={`目标 ${index + 1}`}
        className={`pick-box ${selected === item.observation_id ? "chosen" : ""}`}
        onClick={() => setSelected(item.observation_id)}
        style={{ left: `${item.bbox[0] / item.frame_w * 100}%`, top: `${item.bbox[1] / item.frame_h * 100}%`,
          width: `${(item.bbox[2] - item.bbox[0]) / item.frame_w * 100}%`,
          height: `${(item.bbox[3] - item.bbox[1]) / item.frame_h * 100}%`,
        }} />)}
      {loading && <span className="picker-loading">正在读取观测帧…</span>}
    </div>
    {error && <p className="rag-error" role="alert">{error}</p>}
    <footer>
      <span className="muted">点选画面中的人体框</span>
      <button className="rag-button solid" disabled={!selected || busy || loading} type="button" onClick={async () => {
        setBusy(true);
        setError("");
        try { onTarget(await api.createQueryFromFrame(source.source_asset_id, selected)); }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setBusy(false); }
      }}>{busy ? "正在编码…" : "使用该目标"}<Icon name="arrow" size={16}/></button>
    </footer>
  </RagDialog>;
}

export function ImageTargetPicker({ file, onClose, onTarget }: { file: File; onClose: () => void; onTarget: (qa: QueryAsset) => void }) {
  const [url, setUrl] = useState("");
  const [box, setBox] = useState<Box>();
  const [size, setSize] = useState<[number, number]>([0, 0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const drag = useRef<[number, number]>();
  const image = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const previewUrl = URL.createObjectURL(file);
    setUrl(previewUrl);
    return () => URL.revokeObjectURL(previewUrl);
  }, [file]);
  const point = (event: PointerEvent<HTMLDivElement>): [number, number] => {
    const rect = image.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    return [Math.round(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * size[0]),
      Math.round(Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) * size[1])];
  };
  const valid = box && box[2] - box[0] >= 2 && box[3] - box[1] >= 2;
  return <RagDialog title="在图片中选择目标" onClose={onClose}>
    <div className="upload-crop"><div className="crop-surface"
      onPointerDown={event => {
        if (busy || !size[0]) return;
        drag.current = point(event); event.currentTarget.setPointerCapture(event.pointerId); setBox(undefined);
      }}
      onPointerMove={event => {
        if (!drag.current) return;
        const [x, y] = point(event); const [x0, y0] = drag.current;
        setBox([Math.min(x, x0), Math.min(y, y0), Math.max(x, x0), Math.max(y, y0)]);
      }}
      onPointerUp={() => { drag.current = undefined; }} onPointerCancel={() => { drag.current = undefined; }}>
      {url && <img ref={image} src={url} alt="待框选的查询图片" draggable={false}
        onLoad={e => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
        onError={() => setError("无法读取这张图片")} />}
      {box && <span className="crop-outline" style={{ left: `${box[0] / size[0] * 100}%`, top: `${box[1] / size[1] * 100}%`,
        width: `${(box[2] - box[0]) / size[0] * 100}%`, height: `${(box[3] - box[1]) / size[1] * 100}%` }} />}
    </div></div>
    {error && <p className="rag-error" role="alert">{error}</p>}
    <footer>
      <button className="rag-button quiet" type="button" disabled={!size[0] || busy} onClick={() => setBox([0, 0, ...size])}>使用整张图</button>
      <button className="rag-button solid" type="button" disabled={!valid || busy} onClick={async () => {
        if (!box) return;
        setBusy(true);
        setError("");
        try { onTarget(await api.createQueryUpload(file, box)); }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setBusy(false); }
      }}>{busy ? "正在编码…" : "使用选中区域"}<Icon name="arrow" size={16}/></button>
    </footer>
  </RagDialog>;
}
