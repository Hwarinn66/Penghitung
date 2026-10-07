"""Run one process only: python app.py --cam URL. No reload/multiple workers."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from camera.camera_manager import CameraManager, CameraUnavailable
from config import ROOT, settings
from counting.counter import CountingBusy, ObjectCounter
from counting.roi import draw_preview_roi
from detection.factory import create_detector

log = logging.getLogger(__name__)


def create_app(cfg=settings, camera=None, detector=None):
    cfg.validate()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = camera if camera is not None else CameraManager(cfg.CAMERA_SOURCE, cfg,
                            preview_transform=lambda frame: draw_preview_roi(frame, cfg))
    placeholder = np.full((450, 800, 3), (28, 23, 18), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (195, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 165, 170), 2, cv2.LINE_AA)
    _, encoded = cv2.imencode(".jpg", placeholder)
    waiting_jpeg = encoded.tobytes()

    @asynccontextmanager
    async def lifespan(application):
        cam.start()
        application.state.detector_error = None
        application.state.counter = None
        try:
            active = detector if detector is not None else await asyncio.to_thread(create_detector, cfg)
            application.state.counter = ObjectCounter(cam, active, cfg)
        except Exception as exc:
            application.state.detector_error = str(exc)
            log.exception("Detector unavailable; camera/dashboard remain running")
        try:
            yield
        finally:
            await asyncio.to_thread(cam.stop)

    application = FastAPI(title="Dense Object Counter", lifespan=lifespan)
    application.state.camera = cam
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    application.mount("/output", StaticFiles(directory=cfg.OUTPUT_DIR), name="output")
    templates = Jinja2Templates(directory=str(ROOT / "templates"))

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html", context={
            "unit": cfg.OBJECT_UNIT, "poll_ms": cfg.STATUS_POLL_MS,
            "physical_tray": cfg.PHYSICAL_TRAY_CM, "effective_area": cfg.EFFECTIVE_AREA_CM})

    @application.get("/video_feed")
    async def video_feed(request: Request):
        async def frames():
            last_key = None
            last_sent = 0.0
            while not await request.is_disconnected():
                sequence, jpeg = cam.get_preview()
                key = (sequence, jpeg is not None)
                now = asyncio.get_running_loop().time()
                if key != last_key or now - last_sent >= 1.0:
                    data = jpeg if jpeg is not None else waiting_jpeg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                           str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    last_key, last_sent = key, now
                await asyncio.sleep(1 / cfg.PREVIEW_FPS)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @application.get("/api/status")
    def status():
        camera_status = cam.status()
        counter = application.state.counter
        busy = counter.busy if counter else False
        state = "processing" if busy else ("detector_unavailable" if counter is None else
                  "ready" if camera_status["camera_connected"] else "camera_unavailable")
        last = counter.last_result() if counter else None
        return {"status": state, **camera_status, "processing": busy,
                "detector_loaded": counter is not None,
                "detector": counter.detector.name if counter else cfg.DETECTOR.upper(),
                "detector_error": application.state.detector_error,
                "roi_calibrated": cfg.ROI_CALIBRATED, "roi_enabled": cfg.ROI_ENABLED,
                "roi_normalized": cfg.roi, "snapshot_roi_normalized": cfg.SNAPSHOT_ROI,
                "tiled_inference": bool(counter and cfg.TILED_INFERENCE and counter.detector.supports_tiling),
                "unit": cfg.OBJECT_UNIT,
                "last_result": {k:v for k,v in last.items() if k != "detections"} if last else None}

    @application.post("/api/count")
    def count():
        counter = application.state.counter
        if counter is None:
            return JSONResponse(status_code=503, content={"success": False,
                "detail": application.state.detector_error or "Detector belum siap"})
        try:
            return counter.count()
        except CountingBusy as exc:
            return JSONResponse(status_code=409, content={"success": False, "detail": str(exc)})
        except CameraUnavailable as exc:
            return JSONResponse(status_code=503, content={"success": False, "detail": str(exc)})
        except Exception:
            log.exception("Counting failed")
            return JSONResponse(status_code=500, content={"success": False,
                "detail": "Counting gagal. Periksa log terminal, ruang disk, dan konfigurasi detector."})

    @application.post("/api/reset")
    def reset():
        counter = application.state.counter
        try:
            if counter:
                counter.reset()
        except CountingBusy as exc:
            return JSONResponse(status_code=409, content={"success": False, "detail": str(exc)})
        return {"success": True, "count": None}

    @application.get("/api/result_image/{result_id}")
    def result_image(result_id: str):
        counter = application.state.counter
        jpeg = counter.latest_jpeg(result_id) if counter else None
        if jpeg is None:
            return JSONResponse(status_code=404, content={"detail": "Hasil tidak lagi tersedia"})
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return application


def main():
    parser = argparse.ArgumentParser(description="Snapshot-based dense object counter")
    parser.add_argument("--cam", default="0", help="IP camera URL atau webcam index")
    parser.add_argument("--detector", choices=["yolo", "opencv"], default=settings.DETECTOR)
    parser.add_argument("--model", type=Path, default=settings.MODEL_PATH)
    parser.add_argument("--snapshot-url", default=settings.SNAPSHOT_URL)
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG)
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    args = parser.parse_args()
    CAM_SOURCE = int(args.cam) if args.cam.isdigit() else args.cam
    cfg = replace(settings, CAMERA_SOURCE=CAM_SOURCE, DETECTOR=args.detector,
                  MODEL_PATH=args.model.resolve(), SNAPSHOT_URL=args.snapshot_url, DEBUG=args.debug)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
