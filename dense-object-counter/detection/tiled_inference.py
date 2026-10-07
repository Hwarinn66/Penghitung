"""Native-resolution tiles, tile->ROI transform, global NMS and seam cleanup."""
from dataclasses import dataclass, replace
import math
import time
import numpy as np
from .base_detector import Detection

@dataclass(frozen=True)
class Tile:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int

def axis_starts(length, size, overlap):
    if length <= size:
        return [0]
    stride = max(1, int(size * (1 - overlap)))
    steps = math.ceil((length - size) / stride)
    return [round(i * (length - size) / steps) for i in range(steps + 1)]

def make_tiles(width, height, size, overlap):
    if size < 1 or not 0 <= overlap < 1 or min(width, height) < 1:
        raise ValueError("Invalid tile geometry")
    tiles = []
    for y in axis_starts(height, size, overlap):
        for x in axis_starts(width, size, overlap):
            tiles.append(Tile(len(tiles), x, y, min(x+size, width), min(y+size, height)))
    return tiles

def global_nms(detections, iou_threshold, seam_ios_threshold=None):
    if not detections:
        return []
    boxes = np.asarray([d.bbox for d in detections], dtype=np.float64)
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    order = np.array(sorted(range(len(detections)), key=lambda i:
                           (detections[i].seam_clipped, -detections[i].confidence)))
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(detections[i])
        rest = order[1:]
        if not len(rest):
            break
        low = np.maximum(boxes[i, :2], boxes[rest, :2])
        high = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        intersection = np.prod(np.maximum(0, high-low), axis=1)
        iou = intersection / np.maximum(areas[i]+areas[rest]-intersection, 1e-9)
        same_class = np.array([detections[j].class_id == detections[i].class_id for j in rest])
        suppress = (iou >= iou_threshold) & same_class
        if seam_ios_threshold is not None:
            ios = intersection / np.maximum(np.minimum(areas[i], areas[rest]), 1e-9)
            seam_pair = np.array([detections[i].tile_id != detections[j].tile_id and
                                  (detections[i].seam_clipped or detections[j].seam_clipped) for j in rest])
            suppress |= (ios >= seam_ios_threshold) & seam_pair & same_class
        order = rest[~suppress]
    return keep

def run_inference(image, detector, cfg, tile_callback=None):
    h, w = image.shape[:2]
    tiled = cfg.TILED_INFERENCE and detector.supports_tiling
    tiles = make_tiles(w, h, cfg.TILE_SIZE, cfg.TILE_OVERLAP) if tiled else [Tile(0, 0, 0, w, h)]
    raw, warnings = [], []
    inference_time = 0.0
    for tile in tiles:
        crop = image[tile.y1:tile.y2, tile.x1:tile.x2].copy()
        if tile_callback:
            tile_callback(tile, crop)
        started = time.perf_counter()
        local = detector.detect(crop)
        inference_time += time.perf_counter() - started
        if len(local) >= cfg.MAX_DETECTIONS_PER_TILE and detector.supports_tiling:
            warnings.append(f"Tile {tile.id} mencapai max_det; naikkan batas atau kecilkan tile.")
        tw, th = tile.x2 - tile.x1, tile.y2 - tile.y1
        e = cfg.TILE_EDGE_TOLERANCE_PX
        for det in local:
            if det.class_id != 0 or not math.isfinite(det.confidence) or det.confidence < cfg.CONF_THRESHOLD:
                continue
            coords = np.asarray(det.bbox, dtype=float)
            if coords.shape != (4,) or not np.isfinite(coords).all():
                continue
            x1, y1, x2, y2 = np.clip(coords, [0, 0, 0, 0], [tw, th, tw, th])
            if x2 <= x1 or y2 <= y1:
                continue
            seam = ((tile.x1 > 0 and x1 <= e) or (tile.y1 > 0 and y1 <= e) or
                    (tile.x2 < w and x2 >= tw-e) or (tile.y2 < h and y2 >= th-e))
            raw.append(replace(det, bbox=(float(x1+tile.x1), float(y1+tile.y1), float(x2+tile.x1), float(y2+tile.y1)),
                               tile_id=tile.id, seam_clipped=bool(seam)))
    started = time.perf_counter()
    merged = global_nms(raw, cfg.NMS_IOU_THRESHOLD, cfg.SEAM_IOS_THRESHOLD)
    merge_time = time.perf_counter() - started
    return {"raw": raw, "merged": merged, "tiles": tiles, "warnings": warnings,
            "inference_time": inference_time, "merge_time": merge_time}
