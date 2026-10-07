import cv2
import numpy as np
from .base_detector import BaseDetector, Detection

class OpenCVDetector(BaseDetector):
    name = "OpenCV baseline"
    supports_tiling = False
    def __init__(self, cfg):
        self.cfg = cfg
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.cfg.HSV_LOWER, np.uint8), np.array(self.cfg.HSV_UPPER, np.uint8))
        if self.cfg.MORPH_ITERATIONS:
            kernel = np.ones((self.cfg.MORPH_KERNEL, self.cfg.MORPH_KERNEL), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.cfg.MORPH_ITERATIONS)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if self.cfg.CONTOUR_MIN_AREA <= cv2.contourArea(contour) <= self.cfg.CONTOUR_MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(Detection((float(x), float(y), float(x+w), float(y+h)), 1.0))
        return detections
