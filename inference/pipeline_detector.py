from module.detection import Detector, Tracker, Detections, crop_box
from typing import Callable, OrderedDict, Optional
import time
import numpy as np
from dataclasses import dataclass
 

@dataclass
class BottleResult:
    """One tracked bottle in the current frame."""
    track_id: int
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2 in frame pixels (live position)
    inspected: bool                      # False = not inspected yet ("waiting")
    defect: Optional[bool] = None        # None until inspected
    score: Optional[float] = None
    mask: Optional[np.ndarray] = None    # only when defect is True

class Pipeline:
    def __init__(self, detector: Detector, tracker: Tracker, trigger, analyzer: Callable, pad=0.1,
                 ttl=30.0, max_results=25, on_result=None):
        """
        analyzer(crop, tid, i, dets) -> small dict (verdict), NOT the full anomalib prediction.
        on_result(tid, verdict, crop): optional hook for saving defect images, reject logic, logging.
        """
        self.detector, self.tracker, self.trigger, self.analyzer = detector, tracker, trigger, analyzer
        self.pad, self.ttl, self.max_results, self.on_result = pad, ttl, max_results, on_result
        self.results = OrderedDict()             # tid -> (verdict, time)

    def process(self, frame) -> list[BottleResult]:
        dets = self.tracker.update(self.detector(frame), frame)
        out: list[BottleResult] = []
        if dets.track_id is not None:
            for i, tid in enumerate(dets.track_id):
                tid = int(tid)
                box = dets.xyxy[i]

                if tid not in self.results and self.trigger(tid, box, frame.shape):
                    crop = crop_box(frame, box, self.pad)
                    if crop.size:
                        verdict = self.analyzer(crop, tid, i, dets)
                        self.results[tid] = (verdict, time.monotonic())
                        if self.on_result:
                            self.on_result(tid, verdict, crop)

                bbox = tuple(int(v) for v in box)
                cached = self.results.get(tid)
                if cached is None:
                    out.append(BottleResult(tid, bbox, inspected=False))
                else:
                    v = cached[0]
                    out.append(BottleResult(tid, bbox, True, v.defect, v.score, v.mask))
        self._prune()
        return out

    def _prune(self):
        now = time.monotonic()
        while self.results and (
            now - next(iter(self.results.values()))[1] > self.ttl
            or len(self.results) > self.max_results
        ):
            self.results.popitem(last=False)
        self.trigger.prune()