"""Only this module opens VideoCapture. Its worker owns open/read/release."""
import logging
import threading
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraUnavailable(RuntimeError):
    pass


@dataclass
class Snapshot:
    image: np.ndarray
    source: str
    warnings: list[str]


class CameraManager:
    def __init__(self, source, cfg, preview_transform=None, capture_factory=None):
        self.source, self.cfg = source, cfg
        self.preview_transform = preview_transform
        self._capture_factory = capture_factory
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.latest_frame = None
        self._last_frame_at = None
        self._resolution = None
        self._state = "disconnected"
        self._jpeg = None
        self._jpeg_seq = 0
        self._error = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            timeout = (self.cfg.CAMERA_OPEN_TIMEOUT_MS + self.cfg.CAMERA_READ_TIMEOUT_MS) / 1000 + 2
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.error("Camera driver did not return in time; worker releases on return")
        self._invalidate("disconnected", "Camera stopped")

    def _open_capture(self):
        if self._capture_factory:
            return self._capture_factory(self.source)
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, self.cfg.CAMERA_USB_BACKEND)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, self.cfg.CAMERA_FPS)
        else:
            cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.cfg.CAMERA_OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.cfg.CAMERA_READ_TIMEOUT_MS,
            ])
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _invalidate(self, state, error):
        with self.lock:
            self._state, self._error = state, error
            self.latest_frame = self._jpeg = None
            self._last_frame_at = None
            self._jpeg_seq += 1

    def _run(self):
        while not self._stop.is_set():
            self._invalidate("reconnecting", "Opening camera")
            cap = None
            try:
                cap = self._open_capture()
                if not cap.isOpened():
                    raise CameraUnavailable("Cannot open camera; check source and network")
                last_preview = 0.0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        raise CameraUnavailable("Camera read failed")
                    now = time.monotonic()
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self._last_frame_at = now
                        self._resolution = {"width": frame.shape[1], "height": frame.shape[0]}
                        self._state, self._error = "connected", None
                    if now - last_preview >= 1 / self.cfg.PREVIEW_FPS:
                        try:
                            preview = frame.copy()
                            if preview.shape[1] > self.cfg.PREVIEW_MAX_WIDTH:
                                scale = self.cfg.PREVIEW_MAX_WIDTH / preview.shape[1]
                                preview = cv2.resize(preview, (self.cfg.PREVIEW_MAX_WIDTH,
                                    max(1, round(preview.shape[0] * scale))))
                            if self.preview_transform:
                                preview = self.preview_transform(preview)
                            ok, encoded = cv2.imencode(".jpg", preview,
                                [cv2.IMWRITE_JPEG_QUALITY, self.cfg.PREVIEW_JPEG_QUALITY])
                            if ok:
                                with self.lock:
                                    self._jpeg = encoded.tobytes()
                                    self._jpeg_seq += 1
                            last_preview = now
                        except Exception:
                            log.exception("Preview encoding failed")
            except Exception as exc:
                message = str(exc) if isinstance(exc, CameraUnavailable) else "Camera backend error"
                self._invalidate("reconnecting", message)
                log.warning("%s; retrying in %.1fs", message, self.cfg.CAMERA_RETRY_DELAY)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        log.exception("Camera release failed")
            self._stop.wait(self.cfg.CAMERA_RETRY_DELAY)
        self._invalidate("disconnected", "Camera stopped")

    def _fresh_locked(self):
        return (self._state == "connected" and self._last_frame_at is not None and
                time.monotonic() - self._last_frame_at <= self.cfg.CAMERA_STALE_SECONDS)

    def status(self):
        with self.lock:
            fresh = self._fresh_locked()
            state = self._state if self._state != "connected" or fresh else "reconnecting"
            age = None if self._last_frame_at is None else time.monotonic() - self._last_frame_at
            return {"camera_connected": fresh, "camera_state": state,
                    "camera_resolution": self._resolution, "frame_age_seconds": age,
                    "camera_error": self._error if state != "reconnecting" or not age
                    else "Waiting for a fresh camera frame"}

    def get_frame(self):
        with self.lock:
            if not self._fresh_locked() or self.latest_frame is None:
                raise CameraUnavailable("Kamera belum terhubung atau frame sudah kedaluwarsa")
            return self.latest_frame.copy()

    def get_preview(self):
        with self.lock:
            return self._jpeg_seq, self._jpeg if self._fresh_locked() else None

    def take_snapshot(self):
        if not self.status()["camera_connected"]:
            raise CameraUnavailable("Kamera belum terhubung")
        warnings = []
        if self.cfg.SNAPSHOT_URL:
            try:
                request = Request(self.cfg.SNAPSHOT_URL, headers={"Cache-Control": "no-cache"})
                with urlopen(request, timeout=self.cfg.SNAPSHOT_TIMEOUT) as response:
                    data = response.read(self.cfg.SNAPSHOT_MAX_BYTES + 1)
                if len(data) > self.cfg.SNAPSHOT_MAX_BYTES:
                    raise ValueError("Snapshot too large")
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Snapshot endpoint did not return an image")
                video = self.status()["camera_resolution"]
                ratio = image.shape[1] / image.shape[0]
                if (self.cfg.SNAPSHOT_ROI is None and video and
                    abs(ratio / (video["width"] / video["height"]) - 1) > self.cfg.SNAPSHOT_ASPECT_TOLERANCE):
                    raise ValueError("Aspect mismatch: configure SNAPSHOT_ROI")
                return Snapshot(image, "still_endpoint", warnings)
            except Exception as exc:
                if not self.cfg.SNAPSHOT_FALLBACK_TO_STREAM:
                    raise CameraUnavailable("Still snapshot gagal; periksa URL dan SNAPSHOT_ROI") from exc
                warnings.append("Still snapshot gagal; memakai frame stream. Periksa URL/resolusi/SNAPSHOT_ROI.")
        return Snapshot(self.get_frame(), "video_stream", warnings)
