import { app, BrowserWindow, ipcMain, shell } from "electron";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isDev = !app.isPackaged;
const isVideoMode = process.env.SPLATLINE_VIDEO === "1";
const backendPort = Number(process.env.SPLATLINE_BACKEND_PORT ?? 8787);
const backendHost = "127.0.0.1";
const backendUrl = `http://${backendHost}:${backendPort}`;

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;

const getProjectRoot = () => {
  if (isDev) {
    return path.resolve(__dirname, "..");
  }
  return process.resourcesPath;
};

const getPythonPath = () => {
  if (process.env.SPLATLINE_PYTHON && existsSync(process.env.SPLATLINE_PYTHON)) {
    return process.env.SPLATLINE_PYTHON;
  }
  const homebrewPython = "/opt/homebrew/bin/python3.11";
  if (existsSync(homebrewPython)) return homebrewPython;
  return process.platform === "win32" ? "python" : "python3";
};

const backendStatus = async () => {
  try {
    const response = await fetch(`${backendUrl}/api/jobs`, { signal: AbortSignal.timeout(1200) });
    return { running: response.ok, port: backendPort, url: backendUrl };
  } catch {
    return { running: false, port: backendPort, url: backendUrl };
  }
};

const startBackend = async () => {
  const current = await backendStatus();
  if (current.running) return current;
  if (backendProcess) return { running: false, port: backendPort, url: backendUrl };

  const projectRoot = getProjectRoot();
  const scriptPath = path.join(projectRoot, "ui", "server.py");
  if (!existsSync(scriptPath)) {
    throw new Error(`Backend script not found: ${scriptPath}`);
  }

  backendProcess = spawn(getPythonPath(), [scriptPath, "--host", backendHost, "--port", String(backendPort)], {
    cwd: projectRoot,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: ["ignore", "pipe", "pipe"]
  });
  backendProcess.once("exit", () => {
    backendProcess = null;
  });

  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    const status = await backendStatus();
    if (status.running) return status;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Splatline backend did not start on ${backendUrl}`);
};

const stopBackend = async () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  return backendStatus();
};

const revealPath = async (_event: Electron.IpcMainInvokeEvent, targetPath: string) => {
  if (!targetPath) throw new Error("Missing path");
  shell.showItemInFolder(targetPath);
  return { path: targetPath };
};

const createWindow = async () => {
  mainWindow = new BrowserWindow({
    width: 1380,
    height: 900,
    minWidth: 1040,
    minHeight: 720,
    backgroundColor: "#050505",
    title: "Splatline Field Movement Lab",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  if (isDev) {
    await mainWindow.loadURL("http://localhost:5173");
  } else {
    await mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }
};

app.whenReady().then(async () => {
  if (isVideoMode) {
    // Video player mode — load the video entry point
    const { startVideoPlayer } = await import("./video-main.js");
    await startVideoPlayer();
    return;
  }

  ipcMain.handle("backend:status", backendStatus);
  ipcMain.handle("backend:start", startBackend);
  ipcMain.handle("backend:stop", stopBackend);
  ipcMain.handle("path:reveal", revealPath);
  await createWindow();
  void startBackend();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  if (isVideoMode) {
    import("./video-main.js").then(({ stopVideoPlayer }) => stopVideoPlayer());
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});
