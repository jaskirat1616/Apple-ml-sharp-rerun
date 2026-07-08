#!/usr/bin/env python3
"""
Splatline Athlete Twin UI
=========================

A zero-build local web UI for uploading sport video, running Athlete Twin, and
reviewing outputs.  It intentionally uses only the Python standard library so
the project does not need a Node/npm toolchain just to open the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "ui"
STATIC_ROOT = UI_ROOT / "static"
RUNS_ROOT = PROJECT_ROOT / "ui_runs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


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
    rerun_command: List[str] = field(default_factory=list)
    rerun_error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _safe_filename(name: str) -> str:
    name = Path(name).name.replace("\x00", "")
    cleaned = []
    for char in name:
        if char.isalnum() or char in "._- ":
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip()
    return result or "upload.mp4"


def _parse_multipart(body: bytes, content_type: str) -> Dict[str, Tuple[Optional[str], bytes]]:
    """
    Minimal multipart/form-data parser for local uploads.

    Returns {field_name: (filename, content_bytes)}.
    """
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary")
    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
    delimiter = ("--" + boundary).encode("utf-8")
    fields: Dict[str, Tuple[Optional[str], bytes]] = {}

    for part in body.split(delimiter):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        header_blob, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = ""
        for header in headers:
            if header.lower().startswith("content-disposition:"):
                disposition = header
                break
        if not disposition:
            continue

        attrs = {}
        for item in disposition.split(";")[1:]:
            if "=" in item:
                key, value = item.strip().split("=", 1)
                attrs[key] = value.strip().strip('"')
        field_name = attrs.get("name")
        if not field_name:
            continue
        filename = attrs.get("filename")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        fields[field_name] = (filename, data)
    return fields


def _read_file_preview(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    if not path.exists() or path.stat().st_size > max_bytes:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _job_payload(job: Job, include_logs: bool = True) -> Dict[str, object]:
    files = {}
    for name in ("report.md", "metrics.csv", "events.csv", "athlete_twin.json", "evidence_summary.json"):
        path = job.output_dir / name
        if path.exists():
            files[name] = {
                "url": f"/api/jobs/{job.id}/files/{name}",
                "size": path.stat().st_size,
            }

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


def _launch_rerun_after_processing(job: Job) -> None:
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
    with JOBS_LOCK:
        job.rerun_command = command
        job.rerun_status = "launching"
    _append_log(job, "Launching Rerun after processing...")
    _append_log(job, "$ " + " ".join(command))

    try:
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        with JOBS_LOCK:
            job.rerun_status = "launched"
        _append_log(job, "Rerun launch requested.")
    except Exception as exc:
        with JOBS_LOCK:
            job.rerun_status = "failed"
            job.rerun_error = str(exc)
        _append_log(job, f"Rerun launch failed: {exc}")


def _append_log(job: Job, line: str) -> None:
    with JOBS_LOCK:
        job.logs.append(line.rstrip("\n"))


def _run_job(job: Job) -> None:
    with JOBS_LOCK:
        job.status = "running"
        job.started_at = time.time()

    _append_log(job, "$ " + " ".join(job.command))
    try:
        process = subprocess.Popen(
            job.command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _append_log(job, line)
        return_code = process.wait()
        with JOBS_LOCK:
            job.return_code = return_code
            job.finished_at = time.time()
            job.status = "completed" if return_code == 0 else "failed"
            if return_code != 0:
                job.error = f"Process exited with code {return_code}"
        if return_code == 0 and job.rerun_requested:
            _launch_rerun_after_processing(job)
    except Exception as exc:
        with JOBS_LOCK:
            job.status = "failed"
            job.finished_at = time.time()
            job.error = str(exc)
        _append_log(job, f"ERROR: {exc}")


def _create_job(fields: Dict[str, Tuple[Optional[str], bytes]]) -> Job:
    video_file = fields.get("video")
    if video_file is None or not video_file[0] or not video_file[1]:
        raise ValueError("Upload a video file")

    filename = _safe_filename(video_file[0])
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported video extension '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = RUNS_ROOT / job_id
    upload_dir = job_dir / "uploads"
    output_dir = job_dir / "athlete_twin"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / filename
    video_path.write_bytes(video_file[1])

    def field_text(name: str, default: str) -> str:
        item = fields.get(name)
        if item is None:
            return default
        return item[1].decode("utf-8", errors="replace").strip() or default

    def field_flag(name: str) -> bool:
        return field_text(name, "false").lower() in {"1", "true", "yes", "on"}

    device = field_text("device", "default")
    pose_model = field_text("pose_model", "yolo26x-pose.pt")
    pose_imgsz = field_text("pose_imgsz", "960")
    athlete_height = field_text("athlete_height_m", "1.75")
    max_frames = field_text("max_frames", "")
    extract_skip = field_text("extract_skip", "1")
    analysis_skip = field_text("analysis_skip", "1")
    sharp_internal_size = field_text("sharp_internal_size", "1536")
    open_rerun = field_flag("open_rerun")

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
        athlete_height,
        "--extract-skip",
        extract_skip,
        "--analysis-skip",
        analysis_skip,
        "--sharp-internal-size",
        sharp_internal_size,
    ]
    if max_frames:
        command.extend(["--max-frames", max_frames])
    if field_flag("no_height_scale"):
        command.append("--no-height-scale")

    job = Job(
        id=job_id,
        status="queued",
        created_at=time.time(),
        video_path=video_path,
        output_dir=output_dir,
        command=command,
        rerun_requested=open_rerun,
        rerun_status="pending" if open_rerun else "not_requested",
    )
    with JOBS_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return job


class AthleteTwinHandler(BaseHTTPRequestHandler):
    server_version = "SplatlineAthleteTwinUI/1.0"

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_static("index.html", head_only=True)
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :], head_only=True)
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return
        if path == "/api/jobs":
            with JOBS_LOCK:
                payload = [_job_payload(job, include_logs=False) for job in JOBS.values()]
            _json_response(self, payload)
            return
        if path.startswith("/api/jobs/"):
            self._serve_job_api(path)
            return
        _text_response(self, "Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/rerun"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4:
                job_id = parts[2]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if job is None:
                    _text_response(self, "Job not found", HTTPStatus.NOT_FOUND)
                    return
                if not (job.output_dir / "athlete_twin.json").exists():
                    _json_response(self, {"error": "Analysis output is not ready"}, HTTPStatus.BAD_REQUEST)
                    return
                _launch_rerun_after_processing(job)
                _json_response(self, _job_payload(job))
                return

        if parsed.path != "/api/jobs":
            _text_response(self, "Not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(length)
            fields = _parse_multipart(body, content_type)
            job = _create_job(fields)
            _json_response(self, _job_payload(job), HTTPStatus.CREATED)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _serve_job_api(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            _text_response(self, "Not found", HTTPStatus.NOT_FOUND)
            return
        job_id = parts[2]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            _text_response(self, "Job not found", HTTPStatus.NOT_FOUND)
            return

        if len(parts) == 3:
            _json_response(self, _job_payload(job))
            return
        if len(parts) == 4 and parts[3] == "video":
            self._serve_file(job.video_path, download_name=job.video_path.name)
            return
        if len(parts) == 5 and parts[3] == "files":
            name = unquote(parts[4])
            allowed = {"report.md", "metrics.csv", "events.csv", "athlete_twin.json", "evidence_summary.json"}
            if name not in allowed:
                _text_response(self, "File not allowed", HTTPStatus.FORBIDDEN)
                return
            self._serve_file(job.output_dir / name, download_name=name)
            return
        _text_response(self, "Not found", HTTPStatus.NOT_FOUND)

    def _serve_static(self, relative: str, head_only: bool = False) -> None:
        target = (STATIC_ROOT / relative).resolve()
        if not str(target).startswith(str(STATIC_ROOT.resolve())) or not target.exists() or target.is_dir():
            _text_response(self, "Not found", HTTPStatus.NOT_FOUND)
            return
        self._serve_file(target, head_only=head_only)

    def _serve_file(self, path: Path, download_name: Optional[str] = None, head_only: bool = False) -> None:
        if not path.exists() or not path.is_file():
            _text_response(self, "File not found", HTTPStatus.NOT_FOUND)
            return
        suffix = path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".md": "text/markdown; charset=utf-8",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
        }.get(suffix, "application/octet-stream")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        if download_name and suffix not in {".mp4", ".webm", ".mov"}:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Splatline Athlete Twin web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), AthleteTwinHandler)
    print(f"Splatline Athlete Twin UI running at http://{args.host}:{args.port}")
    print(f"Runs directory: {RUNS_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
