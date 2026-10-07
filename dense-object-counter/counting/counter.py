from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from uuid import uuid4
import cv2
from detection.tiled_inference import run_inference
from .roi import inside_roi_policy, roi_bounds

class CountingBusy(RuntimeError):
    pass

def unique_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ") + "_" + uuid4().hex[:8]

def save_image(path, image, quality=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, quality]
                               if path.suffix.lower() in {".jpg", ".jpeg"} else [])
    if not ok:
        raise OSError(f"Cannot encode image: {path.name}")
    encoded.tofile(str(path))

def annotate(image, detections, bounds, cfg):
    canvas = image.copy()
    x1, y1, x2, y2 = bounds
    if cfg.ROI_ENABLED:
        cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), (80, 225, 170), 2)
    h, w = canvas.shape[:2]
    for detection in detections:
        left, top, right, bottom = map(round, detection["bbox"])
        cv2.rectangle(canvas, (left, top), (min(w-1, right), min(h-1, bottom)),
                      (60, 225, 130), cfg.ANNOTATION_THICKNESS)
        label = str(detection["id"])
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           cfg.ANNOTATION_FONT_SCALE, 1)
        lx = min(max(0, left), max(0, w-tw-5))
        ly = min(h-baseline-3, max(th+4, top))
        cv2.rectangle(canvas, (lx, ly-th-3), (lx+tw+4, ly+baseline+2), (15, 35, 28), -1)
        cv2.putText(canvas, label, (lx+2, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.ANNOTATION_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

class ObjectCounter:
    def __init__(self, camera, detector, cfg):
        self.camera, self.detector, self.cfg = camera, detector, cfg
        self._job_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_result = None
        self._latest_jpeg = None

    @property
    def busy(self):
        return self._job_lock.locked()

    def last_result(self):
        with self._state_lock:
            return self._last_result

    def latest_jpeg(self, result_id):
        with self._state_lock:
            if self._last_result and self._last_result["result_id"] == result_id:
                return self._latest_jpeg
        return None

    def reset(self):
        if not self._job_lock.acquire(blocking=False):
            raise CountingBusy("Counting sedang berjalan; tunggu sebelum RESET")
        try:
            with self._state_lock:
                self._last_result = self._latest_jpeg = None
        finally:
            self._job_lock.release()

    def count(self):
        if not self._job_lock.acquire(blocking=False):
            raise CountingBusy("Counting sedang berjalan")
        try:
            return self._count()
        finally:
            self._job_lock.release()

    def _count(self):
        started = time.perf_counter()
        snapshot = self.camera.take_snapshot()
        snapshot_time = time.perf_counter() - started
        image = snapshot.image
        bounds = roi_bounds(image.shape, self.cfg, snapshot.source)
        x1, y1, x2, y2 = bounds
        roi = image[y1:y2, x1:x2].copy()
        run_id = unique_id()
        debug_dir = Path(self.cfg.DEBUG_DIR) / run_id if self.cfg.DEBUG else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=False)
            save_image(debug_dir / "original.jpg", image)
            save_image(debug_dir / "roi.png", roi)
        def save_tile(tile, crop):
            save_image(debug_dir / f"tile_{tile.id:04d}_{tile.x1}_{tile.y1}.jpg", crop)
        inference = run_inference(roi, self.detector, self.cfg, save_tile if debug_dir else None)
        accepted = [d for d in inference["merged"] if inside_roi_policy(d.bbox, roi.shape[1], roi.shape[0], self.cfg)]
        accepted.sort(key=lambda d: ((d.bbox[1]+d.bbox[3])/2, (d.bbox[0]+d.bbox[2])/2))
        final = []
        for number, det in enumerate(accepted, 1):
            a, b, c, d = det.bbox
            final.append({"id": number, "class_id": 0, "bbox": [a+x1, b+y1, c+x1, d+y1],
                          "confidence": det.confidence})
        annotated = annotate(image, final, bounds, self.cfg)
        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.OUTPUT_JPEG_QUALITY])
        if not ok:
            raise OSError("Result image encoding failed")
        output = Path(self.cfg.OUTPUT_DIR)
        output.mkdir(parents=True, exist_ok=True)
        image_url = f"/api/result_image/{run_id}"
        original_url = None
        if self.cfg.SAVE_ORIGINAL:
            save_image(output / f"original_{run_id}.jpg", image, self.cfg.OUTPUT_JPEG_QUALITY)
            original_url = f"/output/original_{run_id}.jpg"
        if self.cfg.SAVE_RESULT:
            jpeg.tofile(str(output / f"result_{run_id}.jpg"))
            image_url = f"/output/result_{run_id}.jpg"
        warnings = snapshot.warnings + inference["warnings"]
        if not self.cfg.ROI_CALIBRATED:
            warnings.append("ROI belum dikalibrasi terhadap tanda fisik 90 x 90 cm.")
        if not self.detector.supports_tiling:
            warnings.append("OpenCV baseline menghitung blob; benda bersentuhan dapat dihitung sebagai satu.")
        excluded = len(inference["merged"]) - len(final)
        if excluded:
            warnings.append(f"{excluded} kandidat menyentuh batas ROI dan dikecualikan; pindahkan benda ke dalam.")
        if any(d.seam_clipped for d in accepted):
            warnings.append("Ada box menyentuh seam tile; verifikasi gambar, ukuran tile, dan overlap.")
        result = {"success": True, "result_id": run_id, "count": len(final),
                  "unit": self.cfg.OBJECT_UNIT, "detector": self.detector.name,
                  "image_url": image_url, "original_url": original_url,
                  "snapshot_source": snapshot.source,
                  "image_resolution": {"width": image.shape[1], "height": image.shape[0]},
                  "roi": list(bounds), "roi_resolution": {"width": roi.shape[1], "height": roi.shape[0]},
                  "tile_count": len(inference["tiles"]), "raw_detection_count": len(inference["raw"]),
                  "excluded_roi_edge_count": excluded, "detections": final,
                  "warnings": warnings,
                  "timings": {"snapshot": snapshot_time, "inference": inference["inference_time"],
                              "merge": inference["merge_time"]}}
        if debug_dir:
            save_image(debug_dir / "annotated.jpg", annotated, self.cfg.OUTPUT_JPEG_QUALITY)
            for name in ("raw", "merged"):
                (debug_dir / f"{name}_detections.json").write_text(json.dumps(
                    [d.to_dict() for d in inference[name]], indent=2), encoding="utf-8")
            (debug_dir / "tiles.json").write_text(json.dumps([asdict(t) for t in inference["tiles"]], indent=2), encoding="utf-8")
        result["processing_time"] = time.perf_counter() - started
        result["timings"]["total"] = result["processing_time"]
        result["timings"]["other"] = max(0.0, result["processing_time"] - snapshot_time -
                                        inference["inference_time"] - inference["merge_time"])
        if self.cfg.SAVE_RESULT:
            (output / f"result_{run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if debug_dir:
            (debug_dir / "final_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with self._state_lock:
            self._last_result, self._latest_jpeg = result, jpeg.tobytes()
        return result
