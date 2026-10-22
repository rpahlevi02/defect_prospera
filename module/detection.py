from __future__ import annotations
from typing import Callable, Optional, Protocol
import numpy as np
import supervision as sv
import cv2
from schema import Detections

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
    """seg=True loads a segmentation model. Class names vary between rfdetr versions."""
    def __init__(self, size="small", seg=False, checkpoint=None, conf=0.4, optimize=True):
        import rfdetr
        if seg:
            cls = getattr(rfdetr, f"RFDETRSeg{size.capitalize()}", None) or getattr(rfdetr, "RFDETRSegPreview")
        else:
            cls = getattr(rfdetr, f"RFDETR{size.capitalize()}")
        self.model = cls(pretrain_weights=checkpoint) if checkpoint else cls()
        if optimize and hasattr(self.model, "optimize_for_inference"):
            self.model.optimize_for_inference()
        self.conf = conf

    def __call__(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        sd = self.model.predict(frame_bgr[..., ::-1].copy(), threshold=self.conf)
        if len(sd) == 0:
            return Detections.empty()

        masks = getattr(sd, "mask", None)
        if masks is not None:
            masks = masks.astype(bool)
            if masks.shape[1:] != (H, W):          # normalize to frame size if needed
                masks = np.stack([
                    cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in masks
                ])
        return Detections(
            sd.xyxy.astype(np.float32),
            sd.confidence.astype(np.float32),
            sd.class_id.astype(int),
            masks=masks,
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
        if len(out) == 0:
            return Detections.empty()

        masks = None
        if dets.masks is not None:
            idx = sv.box_iou_batch(out.xyxy, dets.xyxy).argmax(1)   # output i came from input idx[i]
            masks = dets.masks[idx]
        return Detections(out.xyxy, out.confidence, out.class_id, out.tracker_id, dets.names, masks)