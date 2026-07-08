export type BackendStatus = {
  running: boolean;
  port: number;
  url: string;
};

export type JobFile = {
  url: string;
  size: number;
};

export type EvidenceQuality = {
  score: number;
  min_frame_score?: number;
  label: string;
  warnings?: string[];
};

export type EvidenceEvent = {
  event_type: string;
  start_frame: number;
  end_frame: number;
  start_time_s: number;
  end_time_s: number;
  peak_frame: number;
  metrics: Record<string, number | null>;
  notes: string[];
  quality?: EvidenceQuality;
  review_frames?: number[];
};

export type EvidenceSummary = {
  schema: string;
  session: Record<string, unknown>;
  quality: EvidenceQuality;
  key_metrics: Record<string, number | null>;
  events: EvidenceEvent[];
};

export type AthleteJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | string;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  return_code?: number | null;
  error?: string | null;
  video_name: string;
  video_url?: string | null;
  output_dir: string;
  command: string[];
  logs?: string[];
  report_md?: string;
  metrics_csv?: string;
  events_csv?: string;
  summary?: Record<string, number | string | null>;
  evidence?: EvidenceSummary;
  events?: EvidenceEvent[];
  files: Record<string, JobFile>;
  rerun_requested: boolean;
  rerun_status: string;
  rerun_error?: string | null;
};

export type SplatlineApi = {
  startBackend: () => Promise<BackendStatus>;
  stopBackend: () => Promise<BackendStatus>;
  getBackendStatus: () => Promise<BackendStatus>;
  revealPath: (path: string) => Promise<{ path: string }>;
};

declare global {
  interface Window {
    splatline?: SplatlineApi;
  }
}
