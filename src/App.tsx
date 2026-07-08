import {
  Activity,
  AlertTriangle,
  BarChart3,
  Box,
  CheckCircle2,
  Download,
  ExternalLink,
  FileText,
  FolderOpen,
  Gauge,
  Play,
  RotateCcw,
  Upload,
  Video
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AthleteJob, BackendStatus, EvidenceEvent } from "./types";

type RunConfig = {
  device: string;
  poseModel: string;
  athleteHeight: string;
  maxFrames: string;
  extractSkip: string;
  analysisSkip: string;
  openRerun: boolean;
  splatBackend: string;
  humanTier: string;
};

type BackendInfo = {
  key: string;
  name: string;
  status: string;
  license: string;
  commercial_ok: boolean;
  recommended_for: string;
  notes: string;
};

type HumanTier = {
  key: string;
  name: string;
  description: string;
};

const defaultBackend: BackendStatus = {
  running: false,
  port: 8787,
  url: "http://127.0.0.1:8787"
};

const defaultConfig: RunConfig = {
  device: "default",
  poseModel: "yolo26x-pose.pt",
  athleteHeight: "1.75",
  maxFrames: "",
  extractSkip: "1",
  analysisSkip: "1",
  openRerun: false,
  splatBackend: "sharp",
  humanTier: "skeleton"
};

const qualityText: Record<string, string> = {
  high: "High",
  moderate: "Moderate",
  low: "Low",
  review_only: "Review only"
};

const formatNumber = (value: unknown, digits = 2, suffix = "") => {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) return "n/a";
  return `${numeric.toFixed(digits)}${suffix}`;
};

const parseCsv = (text?: string) => {
  if (!text) return [];
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
};

const toBackendUrl = (backendUrl: string, path?: string | null) => {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${backendUrl}${path}`;
};

function MetricChart({ csv }: { csv?: string }) {
  const rows = useMemo(() => parseCsv(csv), [csv]);
  const series = useMemo(() => {
    const defs = [
      ["left_knee_flexion_deg", "#34c759"],
      ["right_knee_flexion_deg", "#0a84ff"],
      ["trunk_lean_deg", "#ffb340"],
      ["pelvis_speed_mps", "#ff453a"]
    ] as const;
    return defs
      .map(([key, color]) => ({
        key,
        color,
        values: rows.map((row) => Number.parseFloat(row[key])).map((value) => (Number.isFinite(value) ? value : null))
      }))
      .filter((item) => item.values.some((value) => value !== null));
  }, [rows]);

  const allValues = series.flatMap((item) => item.values.filter((value): value is number => value !== null));
  const min = allValues.length ? Math.min(...allValues) : 0;
  const max = allValues.length ? Math.max(...allValues) : 1;
  const span = Math.max(max - min, 1);
  const width = 900;
  const height = 300;
  const left = 46;
  const right = width - 24;
  const top = 24;
  const bottom = height - 52;

  const makePath = (values: Array<number | null>) => {
    let path = "";
    values.forEach((value, index) => {
      if (value === null) return;
      const x = left + ((right - left) * index) / Math.max(values.length - 1, 1);
      const y = bottom - ((value - min) / span) * (bottom - top);
      path += path ? ` L ${x} ${y}` : `M ${x} ${y}`;
    });
    return path;
  };

  return (
    <div className="chart-shell">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Movement metrics chart">
        {[0, 1, 2, 3, 4].map((line) => {
          const y = top + ((bottom - top) * line) / 4;
          return <line key={line} x1={left} x2={right} y1={y} y2={y} className="grid-line" />;
        })}
        {series.map((item) => (
          <path key={item.key} d={makePath(item.values)} fill="none" stroke={item.color} strokeWidth="3" />
        ))}
        {!series.length && <text x="48" y="80" className="empty-chart">Metrics appear after analysis</text>}
      </svg>
      <div className="legend">
        {series.map((item) => (
          <span key={item.key}>
            <i style={{ background: item.color }} />
            {item.key.replaceAll("_", " ")}
          </span>
        ))}
      </div>
    </div>
  );
}

function EventTimeline({ events, selected, onSelect }: { events: EvidenceEvent[]; selected?: EvidenceEvent; onSelect: (event: EvidenceEvent) => void }) {
  if (!events.length) {
    return <div className="empty-state">No movement events detected yet.</div>;
  }
  const start = Math.min(...events.map((event) => event.start_frame));
  const end = Math.max(...events.map((event) => event.end_frame), start + 1);

  return (
    <div className="timeline">
      {events.map((event, index) => {
        const left = ((event.start_frame - start) / Math.max(end - start, 1)) * 100;
        const width = Math.max(((event.end_frame - event.start_frame + 1) / Math.max(end - start + 1, 1)) * 100, 4);
        const active = selected === event;
        return (
          <button
            key={`${event.event_type}-${event.peak_frame}-${index}`}
            className={`event-pill ${active ? "active" : ""}`}
            style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%` }}
            onClick={() => onSelect(event)}
            title={`${event.event_type} peak ${event.peak_frame}`}
          >
            {event.event_type.replaceAll("_", " ")}
          </button>
        );
      })}
    </div>
  );
}

function MarkdownPreview({ text }: { text?: string }) {
  const html = useMemo(() => {
    if (!text) return "<p>Run analysis to generate the sport-science report.</p>";
    return text
      .split("\n")
      .map((line) => {
        const escaped = line.replace(/[&<>"']/g, (char) => ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#039;"
        }[char] ?? char));
        if (line.startsWith("# ")) return `<h2>${escaped.slice(2)}</h2>`;
        if (line.startsWith("## ")) return `<h3>${escaped.slice(3)}</h3>`;
        if (line.startsWith("- ")) return `<p class="bullet">${escaped.slice(2)}</p>`;
        if (!line.trim()) return "<br />";
        return `<p>${escaped}</p>`;
      })
      .join("");
  }, [text]);
  return <div className="report" dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function App() {
  const [backend, setBackend] = useState<BackendStatus>(defaultBackend);
  const [backendError, setBackendError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<RunConfig>(defaultConfig);
  const [job, setJob] = useState<AthleteJob | null>(null);
  const [jobs, setJobs] = useState<AthleteJob[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<EvidenceEvent | undefined>();
  const [dragging, setDragging] = useState(false);
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [tiers, setTiers] = useState<HumanTier[]>([]);
  const [sseLogs, setSseLogs] = useState<string[]>([]);
  const pollRef = useRef<number | undefined>(undefined);
  const sseRef = useRef<EventSource | undefined>(undefined);

  const backendUrl = backend.url || defaultBackend.url;
  const evidence = job?.evidence;
  const events = evidence?.events ?? job?.events ?? [];
  const selected = selectedEvent && events.includes(selectedEvent) ? selectedEvent : events[0];

  const refreshJobs = useCallback(async () => {
    const response = await fetch(`${backendUrl}/api/jobs`);
    if (!response.ok) throw new Error(`Backend HTTP ${response.status}`);
    const payload = await response.json() as AthleteJob[];
    setJobs(payload.sort((a, b) => b.created_at - a.created_at));
  }, [backendUrl]);

  const refreshJob = useCallback(async (id: string) => {
    const response = await fetch(`${backendUrl}/api/jobs/${id}`);
    if (!response.ok) throw new Error(`Job HTTP ${response.status}`);
    const payload = await response.json() as AthleteJob;
    setJob(payload);
    if (payload.evidence?.events?.length && !selectedEvent) {
      setSelectedEvent(payload.evidence.events[0]);
    }
    if (payload.status === "completed" || payload.status === "failed") {
      window.clearInterval(pollRef.current);
      pollRef.current = undefined;
      void refreshJobs();
    }
  }, [backendUrl, refreshJobs, selectedEvent]);

  useEffect(() => {
    const boot = async () => {
      try {
        const status = window.splatline ? await window.splatline.startBackend() : defaultBackend;
        setBackend(status);
        const url = status.url || defaultBackend.url;
        if (status.running || !window.splatline) {
          await refreshJobs();
          // Load v2 backend and tier options
          try {
            const [beRes, tierRes, cfgRes] = await Promise.all([
              fetch(`${url}/api/backends`),
              fetch(`${url}/api/tiers`),
              fetch(`${url}/api/config`)
            ]);
            if (beRes.ok) setBackends(await beRes.json());
            if (tierRes.ok) setTiers(await tierRes.json());
            if (cfgRes.ok) {
              const cfg = await cfgRes.json();
              setConfig((prev) => ({
                ...prev,
                splatBackend: cfg.splat_backend ?? prev.splatBackend,
                humanTier: cfg.human_tier ?? prev.humanTier,
                device: cfg.device ?? prev.device,
                poseModel: cfg.pose_model ?? prev.poseModel,
                athleteHeight: String(cfg.athlete_height_m ?? prev.athleteHeight)
              }));
            }
          } catch { /* v2 endpoints may not be ready yet */ }
        }
      } catch (error) {
        setBackendError(error instanceof Error ? error.message : "Backend did not start");
      }
    };
    void boot();
    return () => {
      window.clearInterval(pollRef.current);
      sseRef.current?.close();
    };
  }, [refreshJobs]);

  const updateConfig = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };

  const runAnalysis = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("video", file);
    form.append("device", config.device);
    form.append("pose_model", config.poseModel);
    form.append("athlete_height_m", config.athleteHeight);
    form.append("extract_skip", config.extractSkip);
    form.append("analysis_skip", config.analysisSkip);
    form.append("open_rerun", String(config.openRerun));
    form.append("splat_backend", config.splatBackend);
    form.append("human_tier", config.humanTier);
    if (config.maxFrames.trim()) form.append("max_frames", config.maxFrames.trim());

    const response = await fetch(`${backendUrl}/api/jobs`, { method: "POST", body: form });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Run failed with HTTP ${response.status}`);
    }
    const payload = await response.json() as AthleteJob;
    setJob(payload);
    setSelectedEvent(undefined);
    setSseLogs([]);
    await refreshJobs();

    // v2: SSE live log streaming (replaces polling for logs)
    sseRef.current?.close();
    const sse = new EventSource(`${backendUrl}/api/jobs/${payload.id}/stream`);
    sseRef.current = sse;
    sse.addEventListener("log", (e) => {
      setSseLogs((prev) => [...prev, e.data]);
    });
    sse.addEventListener("done", (e) => {
      sse.close();
      sseRef.current = undefined;
      void refreshJob(payload.id);
      void refreshJobs();
    });
    sse.onerror = () => {
      sse.close();
      sseRef.current = undefined;
      void refreshJob(payload.id);
    };

    // Still poll for job status (SSE handles logs)
    window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(() => void refreshJob(payload.id), 2000);
  };

  const openRerun = async () => {
    if (!job) return;
    await fetch(`${backendUrl}/api/jobs/${job.id}/rerun`, { method: "POST" });
    await refreshJob(job.id);
  };

  const revealRun = async () => {
    if (job?.output_dir && window.splatline) {
      await window.splatline.revealPath(job.output_dir);
    }
  };

  const runState = backendError ? "Backend error" : job?.status ?? (backend.running ? "Ready" : "Starting");
  const quality = evidence?.quality;
  const files = job?.files ?? {};
  const videoUrl = toBackendUrl(backendUrl, job?.video_url) || (file ? URL.createObjectURL(file) : "");

  return (
    <main className="viewer-shell">
      <div className="mac-workspace">
        <aside className="mac-sidebar">
          <div className="mac-window-controls" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div className="brand-block">
            <div className="brand-icon"><Activity size={20} /></div>
            <div>
              <h1>Splatline</h1>
              <p>Field Movement Lab</p>
            </div>
          </div>

          <nav className="sidebar-nav" aria-label="Runs">
            <button className="nav-item active"><Video size={16} /> Current Review</button>
            {jobs.slice(0, 6).map((item) => (
              <button key={item.id} className="nav-item" onClick={() => void refreshJob(item.id)}>
                <FileText size={16} />
                <span>{item.video_name}</span>
              </button>
            ))}
          </nav>

          <div className="sidebar-status">
            <span className={backend.running ? "status-dot live" : "status-dot"} />
            <span>{backend.running ? backendUrl : backendError || "Backend starting"}</span>
          </div>
        </aside>

        <section className="app-screen">
          <header className="mac-toolbar">
            <div>
              <p className="eyebrow">Movement evidence from ordinary video</p>
              <h2>Field Movement Lab</h2>
            </div>
            <div className="toolbar-actions">
              <span className={`run-badge ${job?.status ?? ""}`}>{runState}</span>
              <button className="icon-button" onClick={() => void refreshJobs()} title="Refresh runs"><RotateCcw size={17} /></button>
              <button className="icon-button" onClick={revealRun} disabled={!job?.output_dir || !window.splatline} title="Reveal run"><FolderOpen size={17} /></button>
            </div>
          </header>

          <section className="work-grid">
            <div className="left-stack">
              <section className="panel import-panel">
                <div className="panel-head">
                  <div><Upload size={17} /><h3>Capture</h3></div>
                  <span>{file ? `${(file.size / 1048576).toFixed(1)} MB` : "No file"}</span>
                </div>
                <label
                  className={`drop-target ${dragging ? "dragging" : ""}`}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(event) => {
                    event.preventDefault();
                    setDragging(false);
                    const upload = event.dataTransfer.files[0];
                    if (upload) setFile(upload);
                  }}
                >
                  <input type="file" accept="video/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
                  <Video size={22} />
                  <span>{file ? file.name : "Drop field/court video or choose file"}</span>
                </label>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div><Gauge size={17} /><h3>Analysis Setup</h3></div>
                  <span>Local</span>
                </div>
                <div className="field-grid">
                  <label>3D Backend
                    <select value={config.splatBackend} onChange={(event) => updateConfig("splatBackend", event.target.value)}>
                      {backends.length ? backends.filter(b => b.status === "implemented").map((b) => (
                        <option key={b.key} value={b.key}>{b.name}</option>
                      )) : (
                        <>
                          <option value="sharp">Apple SHARP (per-frame)</option>
                          <option value="triposplat">TripoSplat (per-frame, MIT)</option>
                          <option value="vggt">VGGT (geometry bootstrap)</option>
                          <option value="depthsplat">DepthSplat (multi-view, MIT)</option>
                          <option value="longsplat">LongSplat (video-native coherent)</option>
                        </>
                      )}
                    </select>
                  </label>
                  <label>Human tier
                    <select value={config.humanTier} onChange={(event) => updateConfig("humanTier", event.target.value)}>
                      {tiers.length ? tiers.map((t) => (
                        <option key={t.key} value={t.key}>{t.name}</option>
                      )) : (
                        <>
                          <option value="skeleton">Skeleton (fast)</option>
                          <option value="mesh">SMPL mesh (detailed)</option>
                          <option value="both">Both (tiered)</option>
                        </>
                      )}
                    </select>
                  </label>
                  <label>Device
                    <select value={config.device} onChange={(event) => updateConfig("device", event.target.value)}>
                      <option value="default">Auto</option>
                      <option value="mps">Apple MPS</option>
                      <option value="cuda">CUDA</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </label>
                  <label>Pose model
                    <select value={config.poseModel} onChange={(event) => updateConfig("poseModel", event.target.value)}>
                      <option value="yolo26x-pose.pt">YOLO26x Pose</option>
                      <option value="yolo26l-pose.pt">YOLO26l Pose</option>
                      <option value="yolo26m-pose.pt">YOLO26m Pose</option>
                      <option value="yolo26s-pose.pt">YOLO26s Pose</option>
                      <option value="yolo26n-pose.pt">YOLO26n Pose</option>
                    </select>
                  </label>
                  <label>Height m
                    <input value={config.athleteHeight} onChange={(event) => updateConfig("athleteHeight", event.target.value)} />
                  </label>
                  <label>Max frames
                    <input value={config.maxFrames} placeholder="All" onChange={(event) => updateConfig("maxFrames", event.target.value)} />
                  </label>
                  <label>Extract skip
                    <input value={config.extractSkip} onChange={(event) => updateConfig("extractSkip", event.target.value)} />
                  </label>
                  <label>Analyze skip
                    <input value={config.analysisSkip} onChange={(event) => updateConfig("analysisSkip", event.target.value)} />
                  </label>
                </div>
                {backends.find(b => b.key === config.splatBackend) && (
                  <p className="license-note">
                    {backends.find(b => b.key === config.splatBackend)?.license}
                  </p>
                )}
                <label className="check-row">
                  <input type="checkbox" checked={config.openRerun} onChange={(event) => updateConfig("openRerun", event.target.checked)} />
                  <span>Open 3D Rerun after processing</span>
                </label>
                <button className="primary-button" disabled={!file || job?.status === "running"} onClick={() => void runAnalysis()}>
                  <Play size={17} /> Run analysis
                </button>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div><CheckCircle2 size={17} /><h3>Evidence Quality</h3></div>
                  <span>{quality ? qualityText[quality.label] ?? quality.label : "Pending"}</span>
                </div>
                <div className="quality-meter">
                  <div style={{ width: `${Math.round((quality?.score ?? 0) * 100)}%` }} />
                </div>
                <div className="quality-list">
                  {(quality?.warnings?.length ? quality.warnings : ["No quality warnings yet"]).map((warning) => (
                    <span key={warning}><AlertTriangle size={14} /> {warning.replaceAll("_", " ")}</span>
                  ))}
                </div>
              </section>
            </div>

            <div className="center-stack">
              <section className="video-stage">
                {videoUrl ? <video src={videoUrl} controls playsInline /> : <div className="video-empty"><Box size={28} />Video preview</div>}
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div><Activity size={17} /><h3>Event Timeline</h3></div>
                  <span>{events.length} events</span>
                </div>
                <EventTimeline events={events} selected={selected} onSelect={setSelectedEvent} />
                <div className="event-detail">
                  {selected ? (
                    <>
                      <div>
                        <strong>{selected.event_type.replaceAll("_", " ")}</strong>
                        <span>frames {selected.start_frame}-{selected.end_frame}, peak {selected.peak_frame}</span>
                      </div>
                      <p>{selected.notes?.join(", ") || "No review notes."}</p>
                      <div className="metric-row">
                        <span>Quality {qualityText[selected.quality?.label ?? ""] ?? selected.quality?.label ?? "n/a"}</span>
                        <span>Review frames {(selected.review_frames ?? []).join(", ") || "n/a"}</span>
                      </div>
                    </>
                  ) : (
                    <span>Select an event after analysis.</span>
                  )}
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div><BarChart3 size={17} /><h3>Metrics</h3></div>
                  <span>CSV traces</span>
                </div>
                <MetricChart csv={job?.metrics_csv} />
              </section>
            </div>

            <div className="right-stack">
              <section className="panel metric-cards">
                <div className="panel-head">
                  <div><Gauge size={17} /><h3>Key Measures</h3></div>
                  <span>Session</span>
                </div>
                <div className="stat"><span>Peak speed</span><strong>{formatNumber(evidence?.key_metrics.peak_pelvis_speed_mps, 2, " m/s")}</strong></div>
                <div className="stat"><span>Knee asymmetry</span><strong>{formatNumber(evidence?.key_metrics.knee_flexion_asymmetry_pct, 1, "%")}</strong></div>
                <div className="stat"><span>Max trunk lean</span><strong>{formatNumber(evidence?.key_metrics.max_trunk_lean_deg, 1, " deg")}</strong></div>
                <div className="stat"><span>COD events</span><strong>{formatNumber(evidence?.key_metrics.change_of_direction_count, 0)}</strong></div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div><Download size={17} /><h3>Outputs</h3></div>
                  <span>{Object.keys(files).length} files</span>
                </div>
                <div className="downloads">
                  {Object.entries(files).length ? Object.entries(files).map(([name, fileInfo]) => (
                    <a key={name} href={toBackendUrl(backendUrl, fileInfo.url)} target="_blank" rel="noreferrer">
                      <span>{name}</span>
                      <small>{Math.max(1, Math.round(fileInfo.size / 1024))} KB</small>
                    </a>
                  )) : <span className="muted">Outputs appear after a completed run.</span>}
                </div>
                <button className="secondary-button" disabled={!job || job.status !== "completed"} onClick={() => void openRerun()}>
                  <ExternalLink size={16} /> Open 3D Rerun
                </button>
              </section>

              <section className="panel report-panel">
                <div className="panel-head">
                  <div><FileText size={17} /><h3>Report</h3></div>
                  <span>Coach/scientist</span>
                </div>
                <MarkdownPreview text={job?.report_md} />
              </section>

              <section className="panel logs-panel">
                <div className="panel-head">
                  <div><Activity size={17} /><h3>Logs</h3></div>
                  <span>{sseLogs.length ? "Live (SSE)" : "Live"}</span>
                </div>
                <pre>{backendError || (sseLogs.length ? sseLogs.join("\n") : job?.logs?.join("\n")) || "Waiting for a run."}</pre>
              </section>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}
