import numpy as np 
from typing import Callable, Optional
from dataclasses import dataclass, field

@dataclass
class BottleResult:
    track_id: int
    bbox: tuple[int, int, int, int]
    inspected: bool
    defect: Optional[bool] = None
    score: Optional[float] = None
    mask: Optional[np.ndarray] = None          # defect mask (only when defect)
    bottle_mask: Optional[np.ndarray] = None   # bottle silhouette, bool, bbox-sized (seg models only)

# ---------- common data type (map this to your own Detection schema if you have one) ----------
@dataclass
class Detections:
    xyxy: np.ndarray                      # (N, 4) float32, pixels
    conf: np.ndarray                      # (N,)
    cls: np.ndarray                       # (N,) int
    track_id: Optional[np.ndarray] = None  # (N,) int, set by the tracker
    names: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.xyxy)

    @staticmethod
    def empty():
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))