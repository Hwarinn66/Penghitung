"""Edit this file, then restart the server. Paths are relative to this project."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    OBJECT_NAME: str = "Object"
    OBJECT_UNIT: str = "PCS"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DETECTOR: str = "yolo"
    MODEL_PATH: Path = ROOT / "models/best.pt"
    DEVICE: str = ""
    CONF_THRESHOLD: float = 0.40
    YOLO_IOU_THRESHOLD: float = 0.60
    NMS_IOU_THRESHOLD: float = 0.50
    MAX_DETECTIONS_PER_TILE: int = 2000
    YOLO_IMAGE_SIZE: int = 1024
    TILED_INFERENCE: bool = True
    TILE_SIZE: int = 1024
    TILE_OVERLAP: float = 0.15
    SEAM_IOS_THRESHOLD: float = 0.85
    TILE_EDGE_TOLERANCE_PX: int = 3

    PHYSICAL_TRAY_CM: tuple = (100.0, 100.0)
    EFFECTIVE_AREA_CM: tuple = (90.0, 90.0)
    ROI_ENABLED: bool = True
    ROI_X1: float = 0.246875
    ROI_Y1: float = 0.05
    ROI_X2: float = 0.753125
    ROI_Y2: float = 0.95
    ROI_CALIBRATED: bool = False
    EXCLUDE_ROI_EDGE_OBJECTS: bool = True
    ROI_EDGE_MARGIN_PX: int = 2

    CAMERA_SOURCE: object = 0
    CAMERA_RETRY_DELAY: float = 2.0
    CAMERA_OPEN_TIMEOUT_MS: int = 5000
    CAMERA_READ_TIMEOUT_MS: int = 3000
    CAMERA_STALE_SECONDS: float = 3.0
    CAMERA_WIDTH: int = 1920
    CAMERA_HEIGHT: int = 1080
    CAMERA_FPS: int = 15
    CAMERA_USB_BACKEND: int = 0
    PREVIEW_MAX_WIDTH: int = 1280
    PREVIEW_FPS: float = 8.0
    PREVIEW_JPEG_QUALITY: int = 75
    SNAPSHOT_URL: str | None = None
    SNAPSHOT_TIMEOUT: float = 8.0
    SNAPSHOT_MAX_BYTES: int = 32 * 1024 * 1024
    SNAPSHOT_FALLBACK_TO_STREAM: bool = True
    SNAPSHOT_ROI: tuple | None = None
    SNAPSHOT_ASPECT_TOLERANCE: float = 0.02

    SAVE_ORIGINAL: bool = True
    SAVE_RESULT: bool = True
    DEBUG: bool = False
    OUTPUT_DIR: Path = ROOT / "output"
    DEBUG_DIR: Path = ROOT / "debug"
    OUTPUT_JPEG_QUALITY: int = 95
    ANNOTATION_THICKNESS: int = 2
    ANNOTATION_FONT_SCALE: float = 0.48
    STATUS_POLL_MS: int = 2000

    HSV_LOWER: tuple = (90, 60, 40)
    HSV_UPPER: tuple = (135, 255, 255)
    MORPH_KERNEL: int = 3
    MORPH_ITERATIONS: int = 1
    CONTOUR_MIN_AREA: float = 40.0
    CONTOUR_MAX_AREA: float = 1_000_000.0

    MODEL_NAME: str = "yolo11n.pt"
    DATASET_YAML: Path = ROOT / "training/dataset.yaml"
    IMAGE_SIZE: int = 1024
    BATCH_SIZE: int = 4
    EPOCHS: int = 100
    TRAIN_WORKERS: int = 0
    TRAIN_SEED: int = 42
    RUNS_DIR: Path = ROOT / "runs"
    TRAIN_TILE_MIN_VISIBLE: float = 0.0

    def validate(self):
        if self.DETECTOR not in {"yolo", "opencv"}:
            raise ValueError("DETECTOR must be yolo or opencv")
        for name in ("CONF_THRESHOLD", "NMS_IOU_THRESHOLD", "YOLO_IOU_THRESHOLD"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.TILE_OVERLAP < 0.75 or self.TILE_SIZE < 32:
            raise ValueError("TILE_SIZE >=32 and TILE_OVERLAP in [0, .75) required")
        if self.YOLO_IMAGE_SIZE < 32 or self.YOLO_IMAGE_SIZE % 32:
            raise ValueError("YOLO_IMAGE_SIZE must be a positive multiple of 32")
        if self.MAX_DETECTIONS_PER_TILE < 1:
            raise ValueError("MAX_DETECTIONS_PER_TILE must be positive")
        if self.SEAM_IOS_THRESHOLD is not None and not 0 < self.SEAM_IOS_THRESHOLD <= 1:
            raise ValueError("SEAM_IOS_THRESHOLD must be None or in (0, 1]")
        for roi in (self.roi, self.SNAPSHOT_ROI):
            if roi is not None and (len(roi) != 4 or not
                (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
                raise ValueError("ROI must satisfy 0 <= x1 < x2 <= 1, 0 <= y1 < y2 <= 1")
        for name in ("PREVIEW_FPS", "PREVIEW_MAX_WIDTH", "CAMERA_RETRY_DELAY",
                     "CAMERA_OPEN_TIMEOUT_MS", "CAMERA_READ_TIMEOUT_MS",
                     "CAMERA_STALE_SECONDS", "SNAPSHOT_TIMEOUT", "SNAPSHOT_MAX_BYTES"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.MORPH_KERNEL < 1 or self.MORPH_KERNEL % 2 == 0:
            raise ValueError("MORPH_KERNEL must be positive and odd")
        if self.ROI_EDGE_MARGIN_PX < 0 or self.TILE_EDGE_TOLERANCE_PX < 0:
            raise ValueError("Edge margins cannot be negative")
        if self.CONTOUR_MIN_AREA < 0 or self.CONTOUR_MAX_AREA <= self.CONTOUR_MIN_AREA:
            raise ValueError("Invalid contour area range")
        return self

    @property
    def roi(self):
        return (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2)


settings = Settings()
