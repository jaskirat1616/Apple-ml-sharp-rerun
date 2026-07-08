import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("splatline", {
  startBackend: () => ipcRenderer.invoke("backend:start") as Promise<{ running: boolean; port: number; url: string }>,
  stopBackend: () => ipcRenderer.invoke("backend:stop") as Promise<{ running: boolean; port: number; url: string }>,
  getBackendStatus: () => ipcRenderer.invoke("backend:status") as Promise<{ running: boolean; port: number; url: string }>,
  revealPath: (path: string) => ipcRenderer.invoke("path:reveal", path) as Promise<{ path: string }>
});
