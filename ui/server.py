#!/usr/bin/env python3
"""
Splatline v2 FastAPI backend server.

Replaces the v1 stdlib HTTP server with FastAPI + SSE for:
- Real-time progress streaming via Server-Sent Events
- Pydantic request/response validation
- Auto OpenAPI docs at /docs
- Clean async architecture

Endpoints:
  GET  /                     — Serve the web UI (static files)
  GET  /api/config           — Get current Splatline config
  POST /api/config           — Update config (backend choice, tier, etc.)
  GET  /api/backends         — List available reconstruction backends
  GET  /api/tiers            — List available human pipeline tiers
  GET  /api/jobs             — List all jobs
  POST /api/jobs             — Create a new analysis job (multipart upload)
  GET  /api/jobs/{id}        — Get job status + outputs
  GET  /api/jobs/{id}/stream — SSE stream of live logs
  GET  /api/jobs/{id}/video  — Download source video
  GET  /api/jobs/{id}/files/{name} — Download output file
  POST /api/jobs/{id}/rerun  — Launch Rerun viewer for a completed job
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

UI_ROOT = PROJECT_ROOT / "ui"
STATIC_ROOT = UI_ROOT / "static"
RUNS_ROOT = PROJECT_ROOT / "ui_runs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

from utils.splatline_config import (
    SplatlineConfig,
    load_config,
    save_config,
    BACKEND_CHOICES,
    HUMAN_TIER_CHOICES,
    TIER_SKELETON,
    TIER_MESH,
    TIER_BOTH,
)
from utils.splat_models import list_splat_backends


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    status: str
    created_at: float
    video_path: Path
    output_dir: Path
    command: List[str]
    logs: List[str] = field(default_factory=list)
    return_code: Optional[int] = None
    rerun_requested: bool = False
    rerun_status: str = "not_requested"
    rerun_error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    _process: Optional[subprocess.Popen] = None
    _log_subscribers: List = field(default_factory=list)

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))
        # Notify SSE subscribers
        for queue in self._log_subscribers:
            try:
                queue.put_nowait(line.rstrip("\n"))
            except Exception:
                pass


JOBS: Dict[str, Job] = {}


def _safe_filename(name: str) -> str:
    name = Path(name).name.replace("\x00", "")
    cleaned = []
    for char in name:
        if char.isalnum() or char in "._- ":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip() or "upload.mp4"


def _read_file_preview(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    if not path.exists() or path.stat().st_size > max_bytes:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _job_payload(job: Job, include_logs: bool = True) -> dict:
    files = {}
    for name in ("report.md", "metrics.csv", "events.csv", "athlete_twin.json", "evidence_summary.json"):
        path = job.output_dir / name
        if path.exists():
            files[name] = {"url": f"/api/jobs/{job.id}/files/{name}", "size": path.stat().st_size}

    video_url = f"/api/jobs/{job.id}/video" if job.video_path.exists() else None
    payload = {
        "id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "return_code": job.return_code,
        "error": job.error,
        "rerun_requested": job.rerun_requested,
        "rerun_status": job.rerun_status,
        "rerun_error": job.rerun_error,
        "video_name": job.video_path.name,
        "video_url": video_url,
        "output_dir": str(job.output_dir),
        "command": job.command,
        "files": files,
    }
    if include_logs:
        payload["logs"] = job.logs[-800:]

    report = _read_file_preview(job.output_dir / "report.md")
    if report is not None:
        payload["report_md"] = report

    metrics_csv = _read_file_preview(job.output_dir / "metrics.csv")
    if metrics_csv is not None:
        payload["metrics_csv"] = metrics_csv

    events_csv = _read_file_preview(job.output_dir / "events.csv")
    if events_csv is not None:
        payload["events_csv"] = events_csv

    evidence_path = job.output_dir / "evidence_summary.json"
    if evidence_path.exists() and evidence_path.stat().st_size <= 5_000_000:
        try:
            payload["evidence"] = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    summary_path = job.output_dir / "athlete_twin.json"
    if summary_path.exists() and summary_path.stat().st_size <= 20_000_000:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["summary"] = data.get("summary", {})
            payload["events"] = data.get("events", [])[:100]
        except Exception:
            pass

    return payload


def _run_job(job: Job) -> None:
    """Run analysis job in a background thread."""
    import threading

    def _worker():
        job.status = "running"
        job.started_at = time.time()
        job.append_log("$ " + " ".join(job.command))

        try:
            job._process = subprocess.Popen(
                job.command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert job._process.stdout is not None
            for line in job._process.stdout:
                job.append_log(line)
            return_code = job._process.wait()
            job.return_code = return_code
            job.finished_at = time.time()
            job.status = "completed" if return_code == 0 else "failed"
            if return_code != 0:
                job.error = f"Process exited with code {return_code}"
        except Exception as exc:
            job.status = "failed"
            job.finished_at = time.time()
            job.error = str(exc)
            job.append_log(f"ERROR: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _launch_rerun(job: Job) -> None:
    """Launch Rerun viewer for a completed job."""
    analysis_json = job.output_dir / "athlete_twin.json"
    try:
        analysis = json.loads(analysis_json.read_text(encoding="utf-8"))
        metadata = analysis.get("metadata", {})
        frames_dir = Path(metadata.get("frames_dir") or (job.output_dir / "frames"))
        gaussians_dir = Path(metadata.get("gaussians_dir") or (job.output_dir / "gaussians"))
        fps = str(metadata.get("fps") or "30")
    except Exception:
        frames_dir = job.output_dir / "frames"
        gaussians_dir = job.output_dir / "gaussians"
        fps = "30"

    command = [
        sys.executable,
        "scripts/sports/analyze_athlete_twin.py",
        "--view-only",
        "--analysis-json",
        str(analysis_json),
        "--frames-dir",
        str(frames_dir),
        "--gaussians-dir",
        str(gaussians_dir),
        "--fps",
        fps,
    ]
    job.rerun_status = "launching"
    job.append_log("Launching Rerun after processing...")

    try:
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        job.rerun_status = "launched"
        job.append_log("Rerun launch requested.")
    except Exception as exc:
        job.rerun_status = "failed"
        job.rerun_error = str(exc)
        job.append_log(f"Rerun launch failed: {exc}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Splatline v2",
    version="2.0.0",
    description="Video-to-3D reconstruction with temporal human motion recovery",
)


@app.get("/api/config")
async def get_config():
    """Get current Splatline configuration."""
    config = load_config()
    return config.to_dict()


@app.post("/api/config")
async def update_config(payload: dict):
    """Update Splatline configuration."""
    config = load_config()
    if "splat_backend" in payload:
        config.splat_backend = payload["splat_backend"]
    if "human_tier" in payload:
        config.human_tier = payload["human_tier"]
    if "device" in payload:
        config.device = payload["device"]
    if "pose_model" in payload:
        config.pose_model = payload["pose_model"]
    if "athlete_height_m" in payload:
        config.athlete_height_m = float(payload["athlete_height_m"])
    config.first_run_complete = True
    save_config(config)
    return config.to_dict()


@app.get("/api/backends")
async def get_backends():
    """List available reconstruction backends with license info."""
    backends = list_splat_backends()
    return [
        {
            "key": b.key,
            "name": b.name,
            "status": b.status,
            "license": b.license,
            "commercial_ok": b.commercial_ok,
            "input_mode": b.input_mode,
            "recommended_for": b.recommended_for,
            "notes": b.notes,
            "source_url": b.source_url,
        }
        for b in backends
    ]


@app.get("/api/tiers")
async def get_tiers():
    """List available human pipeline tiers."""
    return HUMAN_TIER_CHOICES


@app.get("/api/jobs")
async def list_jobs():
    """List all analysis jobs."""
    return [_job_payload(job, include_logs=False) for job in JOBS.values()]


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    device: str = Form("default"),
    pose_model: str = Form("yolo26x-pose.pt"),
    pose_imgsz: str = Form("960"),
    athlete_height_m: str = Form("1.75"),
    max_frames: str = Form(""),
    extract_skip: str = Form("1"),
    analysis_skip: str = Form("1"),
    sharp_internal_size: str = Form("1536"),
    splat_backend: str = Form("sharp"),
    human_tier: str = Form("skeleton"),
    open_rerun: str = Form("false"),
):
    """Create a new analysis job."""
    filename = _safe_filename(video.filename or "upload.mp4")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported video extension '{ext}'")

    job_id = uuid.uuid4().hex[:12]
    job_dir = RUNS_ROOT / job_id
    upload_dir = job_dir / "uploads"
    output_dir = job_dir / "athlete_twin"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / filename
    video_path.write_bytes(await video.read())

    open_rerun_flag = open_rerun.lower() in {"1", "true", "yes", "on"}

    command = [
        sys.executable,
        "scripts/sports/analyze_athlete_twin.py",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--pose-model",
        pose_model,
        "--pose-imgsz",
        pose_imgsz,
        "--athlete-height-m",
        athlete_height_m,
        "--extract-skip",
        extract_skip,
        "--analysis-skip",
        analysis_skip,
        "--sharp-internal-size",
        sharp_internal_size,
        "--splat-backend",
        splat_backend,
        "--human-tier",
        human_tier,
    ]
    if max_frames.strip():
        command.extend(["--max-frames", max_frames.strip()])
    if open_rerun_flag:
        command.append("--view")

    job = Job(
        id=job_id,
        status="queued",
        created_at=time.time(),
        video_path=video_path,
        output_dir=output_dir,
        command=command,
        rerun_requested=open_rerun_flag,
        rerun_status="pending" if open_rerun_flag else "not_requested",
    )
    JOBS[job_id] = job
    _run_job(job)

    return JSONResponse(_job_payload(job), status_code=201)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and outputs."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_payload(job)


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_logs(job_id: str):
    """SSE stream of live job logs."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    import queue
    log_queue: queue.Queue = queue.Queue()
    job._log_subscribers.append(log_queue)

    # Send existing logs first
    for line in job.logs:
        log_queue.put(line)

    async def event_generator():
        try:
            while True:
                try:
                    line = log_queue.get_nowait()
                    yield {"event": "log", "data": line}
                except Exception:
                    pass

                if job.status in ("completed", "failed"):
                    yield {"event": "done", "data": json.dumps({"status": job.status, "return_code": job.return_code})}
                    break

                await asyncio.sleep(0.1)
        finally:
            if log_queue in job._log_subscribers:
                job._log_subscribers.remove(log_queue)

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}/video")
async def get_job_video(job_id: str):
    """Download the source video for a job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.video_path.exists():
        raise HTTPException(404, "Video file not found")
    return FileResponse(job.video_path, media_type="video/mp4", filename=job.video_path.name)


@app.get("/api/jobs/{job_id}/files/{name}")
async def get_job_file(job_id: str, name: str):
    """Download an output file for a job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    allowed = {"report.md", "metrics.csv", "events.csv", "athlete_twin.json", "evidence_summary.json"}
    if name not in allowed:
        raise HTTPException(403, "File not allowed")

    path = job.output_dir / name
    if not path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(path, filename=name)


@app.post("/api/jobs/{job_id}/rerun")
async def launch_rerun(job_id: str):
    """Launch Rerun viewer for a completed job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not (job.output_dir / "athlete_twin.json").exists():
        raise HTTPException(400, "Analysis output is not ready")
    _launch_rerun(job)
    return _job_payload(job)


# Serve static UI files
if STATIC_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
async def index():
    """Serve the web UI."""
    index_path = STATIC_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Splatline v2 backend running. UI not found at ui/static/index.html"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Splatline v2 FastAPI backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    import uvicorn
    print(f"Splatline v2 backend running at http://{args.host}:{args.port}")
    print(f"API docs at http://{args.host}:{args.port}/docs")
    print(f"Runs directory: {RUNS_ROOT}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
