from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol
import numpy as np
import supervision as sv


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


# ---------- ports ----------
class Detector(Protocol):
    def __call__(self, frame_bgr: np.ndarray) -> Detections: ...

class Tracker(Protocol):
    def update(self, dets: Detections, frame_bgr: np.ndarray) -> Detections: ...


# ---------- detector adapters ----------
class YoloDetector:
    def __init__(self, weights, conf=0.4, imgsz=640, device=None, classes=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.kw = dict(conf=conf, imgsz=imgsz, device=device, classes=classes, verbose=False)

    def __call__(self, frame_bgr):
        r = self.model.predict(frame_bgr, **self.kw)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections.empty()
        return Detections(
            b.xyxy.cpu().numpy().astype(np.float32),
            b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(int),
            names=r.names,
        )


class RFDetrDetector:
    """size: nano | small | medium | base | large (depends on your rfdetr version)."""
    def __init__(self, size="base", checkpoint=None, conf=0.4, optimize=True):
        import rfdetr
        cls = getattr(rfdetr, f"RFDETR{size.capitalize()}")
        self.model = cls(pretrain_weights=checkpoint) if checkpoint else cls()
        if optimize and hasattr(self.model, "optimize_for_inference"):
            self.model.optimize_for_inference()
        self.conf = conf

    def __call__(self, frame_bgr):
        sd = self.model.predict(frame_bgr[..., ::-1].copy(), threshold=self.conf)  # RGB in, sv.Detections out
        if len(sd) == 0:
            return Detections.empty()
        return Detections(
            sd.xyxy.astype(np.float32),
            sd.confidence.astype(np.float32),
            sd.class_id.astype(int),
        )

class RTDetrOnnxDetector:
    """
    RT-DETR via ONNX Runtime. Handles two export formats:
      - Ultralytics export: 1 input, 1 output (1, 300, 4+nc), boxes = normalized cx,cy,w,h, class scores after sigmoid
      - Official RT-DETR export: inputs (images, orig_target_sizes), outputs (labels, boxes, scores)
    DETR models are NMS-free, so no NMS is applied here.
    """
    def __init__(self, onnx_path, conf=0.4, names=None, classes=None, providers=None, imgsz=640):
        import onnxruntime as ort
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inputs = self.sess.get_inputs()
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.official = len(self.inputs) == 2          # official export takes orig_target_sizes

        shape = self.inputs[0].shape                   # [1, 3, H, W], may be dynamic
        h = shape[2] if isinstance(shape[2], int) else imgsz
        w = shape[3] if isinstance(shape[3], int) else imgsz
        self.size = (w, h)

        self.conf = conf
        self.classes = None if classes is None else np.asarray(classes)
        self.names = names if isinstance(names, dict) else dict(enumerate(names or []))
        print("RT-DETR ONNX providers:", self.sess.get_providers())

    def _preprocess(self, frame_bgr):
        # RT-DETR uses a plain resize (no letterbox), RGB, 0-1 scaling, no mean/std
        import cv2
        img = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_LINEAR)
        img = img[..., ::-1].transpose(2, 0, 1)        # BGR->RGB, HWC->CHW
        return np.ascontiguousarray(img, dtype=np.float32)[None] / 255.0

    def __call__(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        x = self._preprocess(frame_bgr)

        if self.official:
            feeds = {
                self.inputs[0].name: x,
                self.inputs[1].name: np.array([[self.size[0], self.size[1]]], dtype=np.int64),
            }
            labels, boxes, scores = self.sess.run(None, feeds)
            labels, boxes, scores = labels[0], boxes[0], scores[0]
            # boxes are xyxy in model-input pixels -> scale to the original frame
            boxes = boxes * np.array([W / self.size[0], H / self.size[1]] * 2, dtype=np.float32)
        else:
            out = self.sess.run(None, {self.inputs[0].name: x})[0][0]   # (300, 4+nc)
            if out.shape[0] < out.shape[1]:            # (4+nc, N) -> (N, 4+nc)
                out = out.T
            cxcywh, cls_scores = out[:, :4], out[:, 4:]
            labels = cls_scores.argmax(1)
            scores = cls_scores.max(1)
            cx, cy, bw, bh = cxcywh.T
            if cxcywh.max() <= 1.5:                    # normalized -> pixels of the original frame
                cx, bw, cy, bh = cx * W, bw * W, cy * H, bh * H
            else:                                      # already in model-input pixels
                cx, bw = cx * W / self.size[0], bw * W / self.size[0]
                cy, bh = cy * H / self.size[1], bh * H / self.size[1]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

        keep = scores >= self.conf
        if self.classes is not None:
            keep &= np.isin(labels, self.classes)
        if not keep.any():
            return Detections.empty()

        boxes = boxes[keep].astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return Detections(boxes, scores[keep].astype(np.float32), labels[keep].astype(int), names=self.names)


# ---------- tracker adapter (detector-agnostic) ----------
class ByteTrackTracker:
    def __init__(self, fps=30, **kw):
        self.t = sv.ByteTrack(frame_rate=fps, **kw)

    def update(self, dets, frame_bgr=None):
        if len(dets) == 0:
            self.t.update_with_detections(sv.Detections.empty())
            return Detections.empty()
        sd = sv.Detections(xyxy=dets.xyxy, confidence=dets.conf, class_id=dets.cls)
        out = self.t.update_with_detections(sd)
        return Detections(out.xyxy, out.confidence, out.class_id, out.tracker_id, dets.names)


# ---------- trigger: fires once per track when its center crosses a line ----------
class LineCrossTrigger:
    def __init__(self, pos=0.5, axis=0, direction=0):
        """pos: line position as fraction of frame width (axis=0) or height (axis=1).
        direction: +1 / -1 to fire only one way, 0 for either."""
        self.pos, self.axis, self.direction = pos, axis, direction
        self.last = {}
        self.fired = set()

    def __call__(self, tid, box, frame_shape):
        h, w = frame_shape[:2]
        line = self.pos * (w if self.axis == 0 else h)
        c = (box[0] + box[2]) / 2 if self.axis == 0 else (box[1] + box[3]) / 2
        prev = self.last.get(tid)
        self.last[tid] = c
        if prev is None or tid in self.fired:
            return False
        crossed = (prev - line) * (c - line) < 0
        if crossed and (self.direction == 0 or np.sign(c - prev) == self.direction):
            self.fired.add(tid)
            return True
        return False

    def prune(self, alive_ids):
        alive = set(alive_ids)
        self.last = {k: v for k, v in self.last.items() if k in alive}
        if len(self.fired) > 5000:           # bound memory on long runs
            self.fired = {k for k in self.fired if k in alive}


# ---------- pipeline ----------
def crop_box(frame, box, pad=0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(int(x1 - px), 0), max(int(y1 - py), 0)
    x2, y2 = min(int(x2 + px), w), min(int(y2 + py), h)
    return frame[y1:y2, x1:x2]


