from module.detection import Detector, Tracker, Detections
from ..module.utils import crop_box
from typing import Callable, OrderedDict, Optional
import time
import numpy as np
from dataclasses import dataclass
from ..module.schema import BottleResult
from ..module.utils import masked_crop, crop_box

class Pipeline:
    def __init__(self, detector, tracker, trigger, analyzer, pad=0.1,
             ttl=30.0, max_results=500, on_result=None, mask_background=False, bg_value=0):
        """
        analyzer(crop, tid, i, dets) -> small dict (verdict), NOT the full anomalib prediction.
        on_result(tid, verdict, crop): optional hook for saving defect images, reject logic, logging.
        """
        self.detector, self.tracker, self.trigger, self.analyzer = detector, tracker, trigger, analyzer
        self.pad, self.ttl, self.max_results, self.on_result = pad, ttl, max_results, on_result
        self.results = OrderedDict()             # tid -> (verdict, time)
        self.mask_background, self.bg_value = mask_background, bg_value

    def process(self, frame) -> list[BottleResult]:
        dets = self.tracker.update(self.detector(frame), frame)
        H, W = frame.shape[:2]
        out: list[BottleResult] = []
        if dets.track_id is not None:
            for i, tid in enumerate(dets.track_id):
                tid = int(tid)
                box = dets.xyxy[i]

                if tid not in self.results and self.trigger(tid, box, frame.shape):
                    if self.mask_background and dets.masks is not None:
                        crop = masked_crop(frame, dets.masks[i], box, self.pad, self.bg_value)
                    else:
                        crop = crop_box(frame, box, self.pad)
                    if crop.size:
                        verdict = self.analyzer(crop, tid, i, dets)
                        self.results[tid] = (verdict, time.monotonic())
                        if self.on_result:
                            self.on_result(tid, verdict, crop)

                x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
                x2, y2 = min(int(box[2]), W), min(int(box[3]), H)
                bbox = (x1, y1, x2, y2)
                bm = dets.masks[i][y1:y2, x1:x2] if dets.masks is not None else None

                cached = self.results.get(tid)
                if cached is None:
                    out.append(BottleResult(tid, bbox, False, bottle_mask=bm))
                else:
                    v = cached[0]
                    out.append(BottleResult(tid, bbox, True, v.defect, v.score, v.mask, bm))
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