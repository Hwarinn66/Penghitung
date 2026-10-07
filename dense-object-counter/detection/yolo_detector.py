from pathlib import Path
from .base_detector import BaseDetector, Detection

class YoloDetector(BaseDetector):
    name = "YOLO"
    def __init__(self, cfg):
        self.cfg = cfg
        path = Path(cfg.MODEL_PATH)
        if not path.is_file():
            raise FileNotFoundError("Model belum tersedia. Train satu class 'object', lalu pasang models/best.pt")
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        names = self.model.names
        valid = len(names) == 1 and names.get(0) == "object" if isinstance(names, dict) else list(names) == ["object"]
        if not valid or self.model.task != "detect":
            raise ValueError("Butuh model detection satu class: 0=object; model COCO/multi-class ditolak")
    def detect(self, image):
        params = dict(source=image, imgsz=self.cfg.YOLO_IMAGE_SIZE, conf=self.cfg.CONF_THRESHOLD,
                      iou=self.cfg.YOLO_IOU_THRESHOLD, max_det=self.cfg.MAX_DETECTIONS_PER_TILE,
                      classes=[0], agnostic_nms=True, verbose=False, save=False)
        if self.cfg.DEVICE:
            params["device"] = self.cfg.DEVICE
        result = self.model.predict(**params)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [Detection(tuple(map(float, xyxy)), float(conf))
                for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()) if int(cls) == 0]
