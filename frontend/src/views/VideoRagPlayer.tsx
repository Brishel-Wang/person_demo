import { useEffect, useRef, useState } from "react";
import { clock, isSingleFrameCue, PlaybackObservation, VideoCue, VideoSource } from "../videoRagApi";
import { Icon } from "./RagIcons";

const SINGLE_FRAME_PREVIEW_US = 2_000_000;

export interface VideoJump { cue: VideoCue; key: number }

interface Props {
  source: VideoSource;
  cues: VideoCue[];
  active: VideoJump | null;
  observations: PlaybackObservation[];
  onCue: (cue: VideoCue) => void;
  onPick: (localUs: number) => void;
  onPosition: (localUs: number) => void;
}

function observationAt(rows: PlaybackObservation[], localUs: number) {
  let low = 0, high = rows.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (rows[mid].local_us < localUs) low = mid + 1;
    else high = mid;
  }
  return [rows[low], rows[low - 1]].find(o => o && Math.abs(o.local_us - localUs) <= 8_000);
}

export default function VideoRagPlayer({ source, cues, active, observations, onCue, onPick, onPosition }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const stopAtUs = useRef<number | null>(null);
  const [localUs, setLocalUs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [message, setMessage] = useState("");
  const [overlay, setOverlay] = useState<PlaybackObservation>();
  const [rate, setRate] = useState(1);
  const [surface, setSurface] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!frame.current) return;
    const observer = new ResizeObserver(([item]) =>
      setSurface({ width: item.contentRect.width, height: item.contentRect.height }));
    observer.observe(frame.current);
    return () => observer.disconnect();
  }, []);

  const play = () => {
    const element = video.current;
    if (!element) return;
    setMessage("");
    element.play().catch((error) => {
      if (error.name === "AbortError") return;
      setMessage("点击播放可继续查看原片。");
    });
  };

  const seek = (timeUs: number, endUs: number | null = null) => {
    if (!video.current || video.current.readyState === 0) return;
    setMessage("");
    stopAtUs.current = endUs;
    video.current.currentTime = (timeUs + source.offset_us) / 1e6;
    setLocalUs(timeUs);
    onPosition(timeUs);
  };

  const jump = () => {
    if (!active || !video.current || video.current.readyState === 0) return;
    const cue = active.cue;
    const matchUs = cue.matched_us ?? cue.start_us;
    if (isSingleFrameCue(cue)) {
      const endUs = Math.min(source.duration_us, matchUs + SINGLE_FRAME_PREVIEW_US);
      const startUs = Math.max(0, endUs - SINGLE_FRAME_PREVIEW_US);
      seek(startUs, endUs);
    } else {
      seek(matchUs, cue.end_us);
    }
    play();
  };

  useEffect(() => {
    if (active) jump();
    else { stopAtUs.current = null; video.current?.pause(); }
    setOverlay(undefined);
  }, [active?.key, source.source_asset_id]);

  const endClip = (element: HTMLVideoElement) => {
    stopAtUs.current = null;
    element.pause();
    const last = observations[observations.length - 1];
    if (last) {
      element.currentTime = (last.local_us + source.offset_us) / 1e6;
      setOverlay(last);
    }
  };

  useEffect(() => {
    const element = video.current;
    if (!element || !("requestVideoFrameCallback" in element)) return;
    let frameId: number;
    const rendered = (_now: number, info: VideoFrameCallbackMetadata) => {
      const us = Math.round(info.mediaTime * 1e6) - source.offset_us;
      setOverlay(previous => {
        const next = observationAt(observations, us);
        return previous?.local_us === next?.local_us ? previous : next;
      });
      if (stopAtUs.current !== null && us >= stopAtUs.current && !element.paused && !element.seeking) {
        endClip(element);
      }
      frameId = element.requestVideoFrameCallback(rendered);
    };
    frameId = element.requestVideoFrameCallback(rendered);
    // The newly fetched observation sequence may arrive while the player is paused.
    if (element.paused) setOverlay(observationAt(observations, Math.round(element.currentTime * 1e6) - source.offset_us));
    return () => element.cancelVideoFrameCallback(frameId);
  }, [observations, source.offset_us]);

  const viewportWidth = source.width || 1920;
  const viewportHeight = source.height || 1080;
  const scale = Math.min(surface.width / viewportWidth, surface.height / viewportHeight);
  const videoTime = () => {
    const value = Math.round((video.current?.currentTime ?? 0) * 1e6) - source.offset_us;
    return Math.min(source.indexed_end_us - 1, Math.max(0, value));
  };

  return <section className="watch" aria-label="原片播放器">
    <div className="watch-frame" ref={frame} style={{ aspectRatio: `${viewportWidth} / ${viewportHeight}` }}>
      <video ref={video} src={source.video_url} preload="auto" playsInline muted={muted}
        aria-label="原片播放"
        onLoadedMetadata={() => { if (active) jump(); else seek(0); }}
        onTimeUpdate={() => {
          const us = Math.max(0, Math.round((video.current?.currentTime ?? 0) * 1e6) - source.offset_us);
          setLocalUs(us);
          onPosition(us);
          const element = video.current;
          if (element && stopAtUs.current !== null && us >= stopAtUs.current && !element.seeking) endClip(element);
        }}
        onEnded={() => { if (stopAtUs.current !== null && video.current) endClip(video.current); }}
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onError={() => setMessage("原片当前无法播放，请检查视频连接。")} />
      <div className="watch-source"><span />{source.camera_code} <i /> {source.filename}</div>
      <span className="watch-time">{clock(localUs, true)}</span>
      {overlay && scale > 0 && <span className="watch-box" style={{
        left: (surface.width - viewportWidth * scale) / 2 + overlay.bbox[0] / overlay.frame_w * viewportWidth * scale,
        top: (surface.height - viewportHeight * scale) / 2 + overlay.bbox[1] / overlay.frame_h * viewportHeight * scale,
        width: (overlay.bbox[2] - overlay.bbox[0]) / overlay.frame_w * viewportWidth * scale,
        height: (overlay.bbox[3] - overlay.bbox[1]) / overlay.frame_h * viewportHeight * scale,
      }} />}
      {message && <span className="watch-message" role="status">{message}</span>}
    </div>
    <div className="watch-controls">
      <button className="icon-button transport" type="button" title={playing ? "暂停" : "播放"} aria-label={playing ? "暂停" : "播放"}
        onClick={() => playing ? video.current?.pause() : play()}><Icon name={playing ? "pause" : "play"} size={16}/></button>
      <button className="icon-button transport sound-control" type="button" aria-label={muted ? "取消静音" : "静音"}
        onClick={() => setMuted(!muted)}><Icon name={muted ? "mute" : "volume"} size={16}/></button>
      <div className="playhead-label">{clock(localUs)} <span>/ {clock(source.indexed_end_us)}</span></div>
      <div className="watch-timeline">
        <input aria-label="片内时间轴" type="range" min={0} max={source.indexed_end_us} step={100_000}
          value={Math.min(localUs, source.indexed_end_us)}
          onChange={event => { video.current?.pause(); seek(Number(event.target.value)); }} />
        <div className="watch-markers">
          {cues.slice(0, 24).map(cue => <button key={cue.cue_id} type="button"
            aria-label={`播放 ${clock(cue.matched_us ?? cue.start_us, true)}`} title={clock(cue.matched_us ?? cue.start_us, true)}
            className={active?.cue.cue_id === cue.cue_id ? "current" : ""}
            style={{ left: `${(cue.matched_us ?? cue.start_us) / source.indexed_end_us * 100}%` }}
            onClick={() => onCue(cue)} />)}
        </div>
      </div>
      <button className="icon-button transport pick-control" type="button" aria-label="从当前画面选人" title="从当前画面选人"
        onClick={() => { video.current?.pause(); onPick(videoTime()); }}><Icon name="target" size={17}/></button>
      <label className="rate-control"><span className="sr-only">播放速度</span>
        <select value={rate} onChange={e => { const r = Number(e.target.value); setRate(r); if (video.current) video.current.playbackRate = r; }}>
          <option value={0.5}>0.5×</option><option value={1}>1×</option><option value={1.5}>1.5×</option><option value={2}>2×</option>
        </select>
      </label>
      <button className="icon-button transport" type="button" aria-label="全屏" title="全屏" onClick={() => void frame.current?.requestFullscreen().catch(() => null)}><Icon name="expand" size={15}/></button>
    </div>
  </section>;
}
