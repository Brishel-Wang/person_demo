export interface DatasetStatus {
  dataset_id: string;
  latest_run?: {
    ingest_run_id: string | null;
    status: string | null;
    window_us: [number, number] | null;
    started_at: string | null;
  } | null;
  status: string;
  camera_code: string | null;
  source_asset_id: string | null;
  resolved_window_us: [number, number] | null;
  ingest_run_id: string | null;
  ingest_status: string | null;
  demo_window_us: [number, number] | null;
  demo_ingest_run_id: string | null;
  watermark: number | null;
  coverage: { covered_through_us: number; window: [number, number]; status: string } | null;
  tracklet_count: number;
  observation_count: number;
  detail: Record<string, unknown> | null;
}

export interface QueryAsset {
  asset_id: string;
  kind: string;
  origin_status: string;
  origin_observation_id: string | null;
  origin_tracklet_revision_id: string | null;
  origin_continuation_group_id: string | null;
  origin_camera_code: string | null;
  source_asset_id: string | null;
  local_us: number | null;
  bbox: number[] | null;
  feature_space_id: string | null;
  image_url: string | null;
}

export interface Candidate {
  hit_id: string;
  camera_code: string | null;
  source_asset_id: string | null;
  tracklet_revision_id: string;
  t_start_local_us: number | null;
  t_end_local_us: number | null;
  prototype_similarity: number;
  rerank_similarity: number | null;
  rerank_status: string | null;
  sample_count: number;
  same_continuation_as_query: boolean;
  representative_images: { local_us: number; url: string }[];
  evidence_id: string;
  matched_buckets: number[] | null;
  review_status: string;
  review_revision: number;
  rank_position: number;
  suspicious: boolean;
  suspicious_reason: string | null;
}

export interface SearchResults {
  task_id: string;
  status: string;
  total: number;
  page: number;
  page_size: number;
  items: Candidate[];
  snapshot: Record<string, unknown> | null;
  report_job_id: string | null;
  report_status: string | null;
  report_id: string | null;
  report_provider: string | null;
}

export interface Evidence {
  evidence_id: string;
  hit_id: string;
  camera_code: string;
  source_asset_id: string;
  tracklet_revision_id: string;
  local_track_id: number;
  chunk_no: number;
  continuation_of: string | null;
  continuation_group_id: string | null;
  t_start_local_us: number;
  t_end_local_us: number;
  review_status: string;
  review_revision: number;
  prototype_similarity: number;
  rerank_similarity: number | null;
  same_continuation_as_query: boolean;
  rank_position: number;
  suspicious: boolean;
  suspicious_reason: string | null;
  original_video_url: string;
  video_timeline_offset_us: number;
  bbox_sequence: {
    frame_pts: number;
    local_us: number;
    bbox: number[];
    frame_w: number;
    frame_h: number;
    is_representative: boolean;
  }[];
  representative_images: { local_us: number; storage_key: string }[];
  proxy: {
    proxy_storage_key: string;
    fps: number;
    window_start_us: number;
    window_end_us: number;
    start_frame_index: number | null;
    time_mapping: Record<string, number>;
  } | null;
}

export interface ReportContent {
  schema_version: string;
  mode: string;
  narrative: string;
  claims: {
    kind: "candidate_observation" | "reviewed_observation";
    text: string;
    evidence_ids: string[];
    hypothesis_id: null;
  }[];
  gaps: string[];
  hypotheses: unknown[];
  next_information_needed: string[];
}

export interface Report {
  report_id: string;
  search_task_id: string;
  search_revision: number;
  provider: string;
  status: string;
  content: ReportContent;
  validation: Record<string, unknown> | null;
  prompt_version: string | null;
  facts: {
    coverage_status: string;
    total_candidates: number;
    omitted_count: number;
    evidences: Pick<Evidence, "evidence_id" | "t_start_local_us" | "t_end_local_us" | "review_status"
      | "rerank_similarity" | "same_continuation_as_query">[];
  } | null;
}

export interface ReportJob {
  report_job_id: string;
  report_id: string | null;
  status: string;
  provider: string | null;
  error: string | null;
  timing: Record<string, unknown> | null;
  resource: Record<string, unknown> | null;
}

export interface FrameItem {
  observation_id: string;
  local_us: number;
  bbox: number[];
  frame_w: number;
  frame_h: number;
  crop_url: string | null;
  tracklet_revision_id: string;
}

export function fmtLocalUs(us: number | null): string {
  if (us === null || us === undefined) return "-";
  const ms = Math.floor(us / 1000);
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const rem = ms % 1000;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(rem).padStart(3, "0")}`;
}

const BASE = "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let body: { code?: string; message?: string } = {};
    try {
      body = await res.json();
    } catch {
      /* ignore */
    }
    throw new Error(`${body.code || res.status}: ${body.message || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  datasetStatus: () => req<DatasetStatus>("/v1/datasets/single_camera_demo_v1/status"),
  importDataset: () =>
    req<{ import_job_id: string; status: string; source_asset_id: string | null }>(
      "/v1/datasets/import",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: "single_camera_demo_v1",
          cameras: [
            {
              camera_code: "cam_01",
              filename: "cam1.mp4",
              local_path: "/root/ppt-agent-wildtrack/Wildtrack/Wildtrack/cam1.mp4",
            },
          ],
          requested_window_us: { start: 0, end: 200000000 },
        }),
      }
    ),
  frames: (sourceAssetId: string, limit = 300, ingestRunId?: string, endUs?: number, startUs = 0) =>
    req<{ items: FrameItem[]; count: number }>(
      `/v1/observations/frames?source_asset_id=${sourceAssetId}&representative_only=true&limit=${limit}&start_us=${startUs}${ingestRunId ? `&ingest_run_id=${ingestRunId}` : ""}${endUs ? `&end_us=${endUs}` : ""}`
    ),
  createQueryFromFrame: (sourceAssetId: string, observationId: string) =>
    req<QueryAsset>("/v1/query-assets/from-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_asset_id: sourceAssetId, observation_id: observationId }),
    }),
  createQueryUpload: async (file: File, bbox: number[]) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("bbox", JSON.stringify(bbox));
    return req<QueryAsset>("/v1/query-assets/upload", { method: "POST", body: fd });
  },
  createSearch: (queryAssetId: string, sourceAssetId: string, evaluationScope: string, tEndUs: number) =>
    req<{ task_id: string; status: string }>("/v1/search-tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: "single_camera_demo_v1",
        mode: "body_only",
        query_asset_id: queryAssetId,
        evaluation_scope: evaluationScope,
        scope: {
          asset_windows: [
            { source_asset_id: sourceAssetId, t_start_local_us: 0, t_end_local_us: tEndUs },
          ],
        },
        coverage_policy: "wait",
        retrieval_profile: "single_body_mean_l2_v1",
      }),
    }),
  searchTask: (taskId: string) =>
    req<{ task_id: string; status: string; search_revision: number; snapshot: Record<string, unknown> | null; error: string | null }>(
      `/v1/search-tasks/${taskId}`
    ),
  searchResults: (taskId: string, page = 1, pageSize = 50) =>
    req<SearchResults>(`/v1/search-tasks/${taskId}/results?page=${page}&page_size=${pageSize}`),
  evidence: (evidenceId: string) => req<Evidence>(`/v1/evidence/${evidenceId}`),
  review: (taskId: string, hitId: string, action: "confirm" | "reject", expectedRevision: number) =>
    req<{ hit_id: string; action: string; revision: number; status: string }>(
      `/v1/search-tasks/${taskId}/reviews`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hit_id: hitId, action, expected_revision: expectedRevision }),
      }
    ),
  createReport: (taskId: string, searchRevision: number) =>
    req<{ report_job_id: string; status: string }>(`/v1/search-tasks/${taskId}/reports`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_search_revision: searchRevision, report_profile: "qwen_single_report_local_v1" }),
    }),
  reportJob: (jobId: string) => req<ReportJob>(`/v1/report-jobs/${jobId}`),
  report: (reportId: string) => req<Report>(`/v1/reports/${reportId}`),
  exportUrl: (taskId: string, fmt: "json" | "csv", revision?: number, reportId?: string) => {
    const params = new URLSearchParams({ fmt });
    if (revision !== undefined) params.set("search_revision", String(revision));
    if (reportId) params.set("report_id", reportId);
    return `${BASE}/v1/search-tasks/${taskId}/export?${params}`;
  },
  proxy: (sourceAssetId: string) =>
    req<{ storage_key: string; fps: number; time_mapping: Record<string, number> } | null>(
      `/v1/proxy/${sourceAssetId}`
    ).catch(() => null),
};
