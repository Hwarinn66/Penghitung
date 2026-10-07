import math
import cv2

def roi_bounds(shape, cfg, snapshot_source="video_stream", override=None):
    h, w = shape[:2]
    if not cfg.ROI_ENABLED:
        return (0, 0, w, h)
    fractions = override or (cfg.SNAPSHOT_ROI if snapshot_source == "still_endpoint" else None) or cfg.roi
    x1, y1, x2, y2 = fractions
    bounds = (math.ceil(x1*w), math.ceil(y1*h), math.floor(x2*w), math.floor(y2*h))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("ROI terlalu kecil untuk resolusi gambar")
    return bounds

def draw_preview_roi(frame, cfg):
    if cfg.ROI_ENABLED:
        x1, y1, x2, y2 = roi_bounds(frame.shape, cfg)
        overlay = frame.copy()
        overlay[:] = (overlay * 0.35).astype(frame.dtype)
        overlay[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        cv2.rectangle(overlay, (x1, y1), (x2-1, y2-1), (80, 225, 170), 2)
        cv2.putText(overlay, "COUNTING ROI", (x1+8, y1+22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 225, 170), 1, cv2.LINE_AA)
        return overlay
    return frame

def inside_roi_policy(bbox, width, height, cfg):
    if not cfg.EXCLUDE_ROI_EDGE_OBJECTS:
        return True
    x1, y1, x2, y2 = bbox
    m = cfg.ROI_EDGE_MARGIN_PX
    return x1 > m and y1 > m and x2 < width-m and y2 < height-m
