const state = {
  file: null,
  jobId: null,
  pollTimer: null,
  progressTick: 0,
};

const els = {
  runState: document.getElementById("runState"),
  dropZone: document.getElementById("dropZone"),
  dropText: document.getElementById("dropText"),
  videoInput: document.getElementById("videoInput"),
  fileMeta: document.getElementById("fileMeta"),
  runButton: document.getElementById("runButton"),
  progressBar: document.getElementById("progressBar"),
  processText: document.getElementById("processText"),
  rerunText: document.getElementById("rerunText"),
  downloads: document.getElementById("downloads"),
  videoPreview: document.getElementById("videoPreview"),
  report: document.getElementById("report"),
  logs: document.getElementById("logs"),
  metricChart: document.getElementById("metricChart"),
  nodeUpload: document.getElementById("nodeUpload"),
  nodeConfig: document.getElementById("nodeConfig"),
  nodeProcess: document.getElementById("nodeProcess"),
  nodeOutputs: document.getElementById("nodeOutputs"),
  device: document.getElementById("device"),
  poseModel: document.getElementById("poseModel"),
  athleteHeight: document.getElementById("athleteHeight"),
  maxFrames: document.getElementById("maxFrames"),
  extractSkip: document.getElementById("extractSkip"),
  analysisSkip: document.getElementById("analysisSkip"),
  openRerun: document.getElementById("openRerun"),
};

function setNodeStatus(node, status) {
  node.classList.remove("active", "running", "done", "failed");
  if (status) node.classList.add(status);
}

function setRunState(text, tone = "") {
  els.runState.textContent = text;
  els.runState.style.color = tone || "";
}

function setFile(file) {
  state.file = file;
  if (!file) {
    els.fileMeta.textContent = "No video selected";
    els.dropText.textContent = "Drop video or choose file";
    els.runButton.disabled = true;
    return;
  }
  const mb = (file.size / (1024 * 1024)).toFixed(1);
  els.fileMeta.textContent = `${file.name} · ${mb} MB`;
  els.dropText.textContent = "Video ready";
  els.runButton.disabled = false;
  els.videoPreview.src = URL.createObjectURL(file);
  setNodeStatus(els.nodeUpload, "done");
  setNodeStatus(els.nodeConfig, "active");
}

function statusToProgress(status) {
  if (status === "queued") return 12;
  if (status === "running") {
    state.progressTick = (state.progressTick + 7) % 55;
    return 28 + state.progressTick;
  }
  if (status === "completed") return 100;
  if (status === "failed") return 100;
  return 0;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}

function renderMarkdownLite(markdown) {
  if (!markdown) return "Run analysis to generate the sport-science report.";
  return markdown
    .split("\n")
    .map((line) => {
      if (line.startsWith("# ")) return `<h2>${escapeHtml(line.slice(2))}</h2>`;
      if (line.startsWith("## ")) return `<h3>${escapeHtml(line.slice(3))}</h3>`;
      if (line.startsWith("- ")) return `<div class="md-bullet">• ${escapeHtml(line.slice(2))}</div>`;
      if (!line.trim()) return "<br />";
      return `<div>${escapeHtml(line)}</div>`;
    })
    .join("");
}

function parseCsv(text) {
  if (!text) return [];
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted && ch === '"' && next === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (!quoted && ch === ",") {
      row.push(cell);
      cell = "";
    } else if (!quoted && (ch === "\n" || ch === "\r")) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.length)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  if (rows.length < 2) return [];
  const header = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(header.map((key, idx) => [key, values[idx] || ""])));
}

function drawMetricChart(csvText) {
  const canvas = els.metricChart;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#101418";
  ctx.fillRect(0, 0, w, h);

  const rows = parseCsv(csvText);
  const seriesDefs = [
    ["left_knee_flexion_deg", "#35d399"],
    ["right_knee_flexion_deg", "#7cc7ff"],
    ["trunk_lean_deg", "#f7b955"],
    ["pelvis_speed_mps", "#ff6b6b"],
  ];
  const series = seriesDefs
    .map(([key, color]) => ({
      key,
      color,
      values: rows.map((row) => Number.parseFloat(row[key])).map((v) => (Number.isFinite(v) ? v : null)),
    }))
    .filter((item) => item.values.some((v) => v !== null));

  ctx.strokeStyle = "#2d353d";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = 28 + ((h - 64) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(44, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  if (!series.length) {
    ctx.fillStyle = "#98a6a0";
    ctx.font = "18px system-ui";
    ctx.fillText("Metrics chart appears after analysis", 44, 70);
    return;
  }

  const allValues = series.flatMap((item) => item.values.filter((v) => v !== null));
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const span = Math.max(max - min, 1);
  const left = 44;
  const right = w - 20;
  const top = 24;
  const bottom = h - 42;

  series.forEach((item) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    item.values.forEach((value, idx) => {
      if (value === null) return;
      const x = left + ((right - left) * idx) / Math.max(item.values.length - 1, 1);
      const y = bottom - ((value - min) / span) * (bottom - top);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  ctx.font = "13px system-ui";
  let legendX = 44;
  series.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, h - 24, 10, 10);
    ctx.fillStyle = "#dbe6e1";
    ctx.fillText(item.key.replaceAll("_", " "), legendX + 16, h - 15);
    legendX += Math.min(230, item.key.length * 8 + 52);
  });
}

function renderDownloads(files) {
  const names = Object.keys(files || {});
  if (!names.length) {
    els.downloads.innerHTML = "<span>No outputs yet</span>";
    return;
  }
  els.downloads.innerHTML = names
    .map((name) => {
      const sizeKb = Math.max(1, Math.round(files[name].size / 1024));
      return `<a href="${files[name].url}"><span>${name}</span><span>${sizeKb} KB</span></a>`;
    })
    .join("");
}

function renderJob(job) {
  setRunState(job.status, job.status === "failed" ? "#ff6b6b" : job.status === "completed" ? "#35d399" : "");
  els.progressBar.style.width = `${statusToProgress(job.status)}%`;
  els.processText.textContent = job.error || `${job.status} · ${job.video_name}`;
  if (job.rerun_requested) {
    const suffix = job.rerun_error ? `: ${job.rerun_error}` : "";
    els.rerunText.textContent = `Rerun: ${job.rerun_status}${suffix}`;
  } else {
    els.rerunText.textContent = "Rerun: off for this run";
  }
  els.logs.textContent = (job.logs || []).join("\n");
  els.logs.scrollTop = els.logs.scrollHeight;
  renderDownloads(job.files);
  drawMetricChart(job.metrics_csv || "");
  els.report.innerHTML = renderMarkdownLite(job.report_md || "");

  if (job.video_url && !els.videoPreview.src.includes(job.video_url)) {
    els.videoPreview.src = job.video_url;
  }

  if (job.status === "queued") {
    setNodeStatus(els.nodeProcess, "active");
  } else if (job.status === "running") {
    setNodeStatus(els.nodeProcess, "running");
  } else if (job.status === "completed") {
    setNodeStatus(els.nodeProcess, "done");
    setNodeStatus(els.nodeOutputs, "done");
    els.runButton.disabled = false;
    clearInterval(state.pollTimer);
  } else if (job.status === "failed") {
    setNodeStatus(els.nodeProcess, "failed");
    setNodeStatus(els.nodeOutputs, "failed");
    els.runButton.disabled = false;
    clearInterval(state.pollTimer);
  }
}

async function pollJob() {
  if (!state.jobId) return;
  const response = await fetch(`/api/jobs/${state.jobId}`);
  if (!response.ok) return;
  const job = await response.json();
  renderJob(job);
}

async function startRun() {
  if (!state.file) return;
  els.runButton.disabled = true;
  setNodeStatus(els.nodeProcess, "running");
  setNodeStatus(els.nodeOutputs, "");
  setRunState("Uploading");
  els.processText.textContent = "Uploading video";
  els.logs.textContent = "";
  els.report.textContent = "Analysis running...";
  drawMetricChart("");

  const form = new FormData();
  form.append("video", state.file);
  form.append("device", els.device.value);
  form.append("pose_model", els.poseModel.value);
  form.append("pose_imgsz", "960");
  form.append("athlete_height_m", els.athleteHeight.value || "1.75");
  form.append("extract_skip", els.extractSkip.value || "1");
  form.append("analysis_skip", els.analysisSkip.value || "1");
  form.append("sharp_internal_size", "1536");
  form.append("open_rerun", els.openRerun.checked ? "true" : "false");
  if (els.maxFrames.value) form.append("max_frames", els.maxFrames.value);

  const response = await fetch("/api/jobs", {
    method: "POST",
    body: form,
  });
  const payload = await response.json();
  if (!response.ok) {
    setRunState("Failed", "#ff6b6b");
    els.processText.textContent = payload.error || "Upload failed";
    els.runButton.disabled = false;
    setNodeStatus(els.nodeProcess, "failed");
    return;
  }
  state.jobId = payload.id;
  renderJob(payload);
  state.pollTimer = setInterval(pollJob, 1500);
}

els.videoInput.addEventListener("change", (event) => {
  setFile(event.target.files[0] || null);
});

["dragenter", "dragover"].forEach((name) => {
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
  });
});

els.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});

els.runButton.addEventListener("click", startRun);
drawMetricChart("");
