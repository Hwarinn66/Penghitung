"use strict";
const byId = id => document.getElementById(id);
let requesting = false, latestStatus = null, renderedId = null, backendAvailable = false, actionEpoch = 0;
const countButton = byId("countButton"), resetButton = byId("resetButton");
function notice(id, message) { const el = byId(id); el.textContent = message || ""; el.hidden = !message; }
function buttons() {
  const busy = requesting || Boolean(latestStatus?.processing);
  countButton.disabled = busy || !backendAvailable || !latestStatus?.camera_connected || !latestStatus?.detector_loaded;
  resetButton.disabled = busy || !backendAvailable;
  byId("countLabel").textContent = busy ? "MEMPROSES..." : "HITUNG";
}
function clearResult() {
  renderedId = null; byId("countValue").textContent = "—"; byId("countCaption").textContent = "Siapkan benda di area counting.";
  byId("resultState").textContent = "BELUM DIHITUNG";
  for (const id of ["processingTime", "snapshotResolution", "tileInfo"]) byId(id).textContent = "—";
  byId("resultEmpty").hidden = false;
  for (const id of ["resultContainer", "fullImageLink", "timingDetails"]) byId(id).hidden = true;
  byId("resultImage").removeAttribute("src"); byId("fullImageLink").removeAttribute("href"); notice("resultWarning", "");
}
function showResult(result) {
  if (!result || result.result_id === renderedId) return;
  renderedId = result.result_id; byId("countValue").textContent = String(result.count);
  byId("countCaption").textContent = `Detected Objects: ${result.count} · Verifikasi hasil di bawah.`;
  byId("resultState").textContent = "SNAPSHOT SELESAI";
  byId("processingTime").textContent = `${result.processing_time.toFixed(2)} sec`;
  byId("snapshotResolution").textContent = `${result.image_resolution.width} × ${result.image_resolution.height}`;
  byId("tileInfo").textContent = `${result.tile_count} / ${result.snapshot_source === "still_endpoint" ? "Still image" : "Video frame"}`;
  byId("resultImage").src = result.image_url; byId("fullImageLink").href = result.image_url; byId("resultEmpty").hidden = true;
  for (const id of ["resultContainer", "fullImageLink", "timingDetails"]) byId(id).hidden = false;
  byId("timingText").textContent = Object.entries(result.timings).map(([key, value]) => `${key}: ${value.toFixed(3)} sec`).join("\n") +
    `\nROI (xyxy): ${result.roi.join(", ")}\nRaw detections: ${result.raw_detection_count}\nFinal detections: ${result.count}`;
  notice("resultWarning", result.warnings.join("\n"));
}
async function api(url, method = "GET") {
  const controller = new AbortController(); const timer = method === "GET" ? setTimeout(() => controller.abort(), 5000) : null;
  let response; try { response = await fetch(url, {method, cache: "no-store", signal: controller.signal}); } finally { if (timer) clearTimeout(timer); }
  let data; try { data = await response.json(); } catch { throw new Error(`Respons server tidak valid (HTTP ${response.status})`); }
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`); return data;
}
async function pollStatus() {
  const epoch = actionEpoch;
  try {
    const status = await api("/api/status"); if (epoch !== actionEpoch) return;
    latestStatus = status; backendAvailable = true;
    byId("cameraStatus").textContent = status.camera_state.toUpperCase();
    byId("statusDot").className = `dot ${status.camera_state}`;
    const r = status.camera_resolution; byId("liveResolution").textContent = r ? `${r.width} × ${r.height}` : "— × —";
    byId("detectorName").textContent = status.detector;
    const messages = [];
    if (!status.camera_connected) messages.push("Menunggu kamera. Periksa IP Webcam dan koneksi jaringan; reconnect berjalan otomatis.");
    if (!status.detector_loaded) messages.push(status.detector_error || "Detector belum tersedia.");
    if (!status.roi_calibrated) messages.push("Kalibrasikan ROI ke tanda fisik 90 × 90 cm sebelum memakai hasil untuk operasional.");
    if (status.detector.toLowerCase().includes("opencv")) messages.push("Mode baseline OpenCV: benda yang bersentuhan dapat menjadi satu blob.");
    notice("systemNotice", messages.join("\n"));
    if (!requesting) { if (status.last_result) showResult(status.last_result); else if (!status.processing) clearResult(); }
  } catch (error) {
    if (epoch !== actionEpoch) return; backendAvailable = false; byId("cameraStatus").textContent = "DISCONNECTED";
    byId("statusDot").className = "dot disconnected"; notice("systemNotice", "Backend tidak dapat dihubungi. Periksa proses Python; dashboard akan mencoba kembali.");
  }
  buttons();
}
countButton.addEventListener("click", async () => { if (countButton.disabled) return; actionEpoch++; requesting = true; buttons(); notice("connectionError", "");
  try { showResult(await api("/api/count", "POST")); } catch (error) { notice("connectionError", error.message); }
  finally { actionEpoch++; requesting = false; await pollStatus(); }
});
resetButton.addEventListener("click", async () => { if (resetButton.disabled) return; actionEpoch++; requesting = true; buttons();
  try { await api("/api/reset", "POST"); clearResult(); notice("connectionError", ""); } catch (error) { notice("connectionError", error.message); }
  finally { actionEpoch++; requesting = false; await pollStatus(); }
});
byId("liveCamera").addEventListener("error", () => { setTimeout(() => { byId("liveCamera").src = `/video_feed?t=${Date.now()}`; }, 2000); });
byId("resultImage").addEventListener("error", () => notice("connectionError", "Gambar hasil gagal dimuat. Coba buka resolusi penuh atau HITUNG ulang."));
(async function pollingLoop() { await pollStatus(); setTimeout(pollingLoop, Number(document.body.dataset.pollMs) || 2000); })();
