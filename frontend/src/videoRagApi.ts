export type Box = [number, number, number, number];

export interface VideoSource {
  source_asset_id: string;
  filename: string;
  camera_code: string;
  duration_us: number;
  indexed_end_us: number;
  frame_indexed: boolean;
  analysis_fps: number | null;
  indexed_detector?: { weights: string | null; imgsz: number } | null;
  configured_detector?: { weights: string; imgsz: number };
  width: number;
  height: number;
  offset_us: number;
  video_url: string;
}

export interface VideoCue {
  cue_id: string;
  tracklet_id: string;
  track_number: number;
  start_us: number;
  end_us: number;
  matched_us?: number | null;
  same_continuation: boolean;
  review_status: "confirmed" | "rejected" | "pending_review" | null;
  review_revision: number | null;
  hit_id: string | null;
  score: number | null;
  image_url: string | null;
}

export function isSingleFrameCue(cue: VideoCue): boolean {
  return cue.end_us === cue.start_us + 1;
}

export interface VideoTurn {
  turn_id: string;
  parent_id: string | null;
  question: string;
  search_task_id: string | null;
  source_asset_id: string;
  query_asset_id: string | null;
  status: "QUEUED" | "RETRIEVING" | "GENERATING" | "COMPLETED" | "PARTIAL" | "FAILED" | "CANCELLED";
  provider: "qwen" | null;
  error: string | null;
  review_updated?: boolean;
  snapshot: {
    mode: "appearance" | "timeline";
    window_us: [number, number];
    min_similarity?: number | null;
    coverage: string;
    partial: boolean;
    origin_exclusion_verified: boolean;
    cues: VideoCue[];
    autoplay_cue_id: string | null;
  } | null;
  reply: { answer?: string; citation_ids?: string[]; citation_labels?: Record<string, string>; timing?: Record<string, unknown> } | null;
}

export interface ObservationBox {
  bbox: Box;
  frame_w: number;
  frame_h: number;
}

export interface PlaybackObservation extends ObservationBox {
  local_us: number;
}

export interface PickFrame {
  local_us: number;
  frame_url: string | null;
  items: (ObservationBox & { observation_id: string })[];
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.message ?? (typeof body?.detail === "string" ? body.detail : null);
    throw new Error(message ?? `请求失败（${response.status}）`);
  }
  return response.json();
}

export const rag = {
  source: (id: string) => request<VideoSource>(`/v1/video-sources/${id}`),
  frame: (id: string, localUs: number) => request<PickFrame>(`/v1/video-sources/${id}/observations?local_us=${localUs}`),
  turn: (id: string) => request<VideoTurn>(`/v1/video-chat/turns/${id}`),
  history: (id: string) => request<{ turns: VideoTurn[] }>(`/v1/video-chat/turns/${id}/history`),
  cue: (turn: string, cue: string) => request<{ items: PlaybackObservation[] }>(`/v1/video-chat/turns/${turn}/cues/${cue}`),
  send: (data: { source_asset_id: string; question: string; query_asset_id?: string; parent_id?: string; focused_cue_id?: string; min_similarity?: number }) =>
    request<VideoTurn>("/v1/video-chat/turns", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }),
};

export function clock(us: number, exact = false): string {
  const ms = Math.floor(Math.max(0, us) / 1000);
  const pad = (x: number, n = 2) => String(x).padStart(n, "0");
  return `${pad(Math.floor(ms / 60000))}:${pad(Math.floor(ms / 1000) % 60)}${exact ? `.${pad(ms % 1000, 3)}` : ""}`;
}

export function running(turn?: VideoTurn): boolean {
  return !!turn && ["QUEUED", "RETRIEVING", "GENERATING"].includes(turn.status);
}
