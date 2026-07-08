# Field Movement Lab Desktop UI

Splatline includes a local-first Electron app for running Athlete Twin as **Field Movement Lab**. The desktop app starts the Python analysis backend, keeps runs local, and presents movement evidence in a coach/scientist review workflow.

Start the Electron app:

```bash
npm install
npm run dev
```

For a packaged renderer build:

```bash
npm run build
```

The Python backend can still be run directly:

```bash
python ui/athlete_twin_ui.py --host 127.0.0.1 --port 8787
```

Open:

```text
http://127.0.0.1:8787
```

## What It Does

- Upload a sport video
- Configure device, YOLO26 pose model, athlete height, and frame limits
- Run `scripts/sports/analyze_athlete_twin.py`
- Show live logs
- Preview the uploaded video
- Render synchronized review panels for video, movement events, metrics, report, logs, and downloads
- Chart useful metrics from `metrics.csv`
- Show movement-evidence quality warnings from `evidence_summary.json`
- Open the 3D Rerun dashboard after processing finishes
- Download `athlete_twin.json`, `evidence_summary.json`, `metrics.csv`, `events.csv`, and `report.md`

Runs are written to:

```text
ui_runs/<job_id>/
```

## Practical First Run

For a fast smoke test, set **Max frames** to `3` or `5` before running. Full videos require SHARP and YOLO model downloads on first use and can take time.

The **Open 3D Rerun after processing** checkbox launches Rerun only after the main exports have completed successfully. You can also launch Rerun from the desktop review after a run completes.

## Dependencies

The desktop UI uses Electron, React, and Vite. The backend itself still uses only Python's standard library, while analysis depends on the Splatline pipeline dependencies, including SHARP, PyTorch, OpenCV, Rerun, and Ultralytics.
