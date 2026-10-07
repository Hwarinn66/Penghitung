from .opencv_detector import OpenCVDetector

def create_detector(cfg):
    if cfg.DETECTOR == "opencv":
        return OpenCVDetector(cfg)
    from .yolo_detector import YoloDetector
    return YoloDetector(cfg)
