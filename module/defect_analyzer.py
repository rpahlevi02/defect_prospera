import torch
from anomalib.deploy import TorchInferencer
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Verdict:
    """Result of one inspection. Cached per track, so keep it small."""
    defect: bool
    score: float
    mask: Optional[np.ndarray] = None   # uint8 0/1, model resolution, only set when defect


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

class DefectAnalyzer:
    def __init__(self, model_path, device="auto"):
        # model_path = exported .pt (engine.export(model, ExportType.TORCH, ...))
        self.inferencer = TorchInferencer(path=model_path, device=device)

    def predict(self, frame_bgr):  # numpy HxWx3, uint8, from OpenCV
        rgb = frame_bgr[..., ::-1].copy()
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # CHW, [0,1]
        pred = self.inferencer.predict(img)
        return pred  # pred.pred_score, pred.pred_label, pred.anomaly_map

    def __call__(self, crop, tid, i, dets):
        r = self.predict(crop)
        defect = bool(to_numpy(r.pred_label).squeeze())
        mask = None
        if defect:
            mask = to_numpy(r.pred_mask).squeeze().astype(np.uint8)   # 0/1, model input size
        return Verdict(defect=defect, score=float(to_numpy(r.pred_score).squeeze()), mask=mask)