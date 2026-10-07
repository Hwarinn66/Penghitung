from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import numpy as np

@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    tile_id: int = -1
    seam_clipped: bool = False
    def to_dict(self):
        return asdict(self)

class BaseDetector(ABC):
    name = "base"
    supports_tiling = True
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """One image -> instances. No camera, ROI, disk I/O, or counting state."""
