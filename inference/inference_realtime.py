import torch
from anomalib.deploy import TorchInferencer

class RealtimeInference:
    def __init__(self, model_path, device="auto"):
        # model_path = exported .pt (engine.export(model, ExportType.TORCH, ...))
        self.inferencer = TorchInferencer(path=model_path, device=device)

    def predict(self, frame_bgr):  # numpy HxWx3, uint8, from OpenCV
        rgb = frame_bgr[..., ::-1].copy()
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # CHW, [0,1]
        pred = self.inferencer.predict(img)
        return pred  # pred.pred_score, pred.pred_label, pred.anomaly_map


if __name__ == "__main__":
    import os
    os.environ["TRUST_REMOTE_CODE"] = "1"

    import time
    import cv2
    import numpy as np
    import torch

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    source = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try 1, 2 if still black
    source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    source.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not source.isOpened():
        raise RuntimeError("Camera could not be opened")
    for _ in range(30):  # let auto-exposure settle
        source.read()

    model_path = r"C:\Users\AQI\JOB\galaria_prospera\training\models\patchcore_2e9827ba\weights\torch\model.pt"
    inference = RealtimeInference(model_path=model_path, device="auto")

    kernel = np.ones((3, 3), np.uint8)
    prev = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = source.read()
        if not ret:
            break
        if frame.mean() < 1:  # skip black frames
            continue

        result = inference.predict(frame_bgr=frame)
        h, w = frame.shape[:2]
        is_anomalous = bool(to_numpy(result.pred_label).squeeze())
        score = float(to_numpy(result.pred_score).squeeze())

        mask = to_numpy(result.pred_mask).squeeze().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if not is_anomalous:
            mask[:] = 0

        # Frame with translucent red segmentation + outline
        out = frame.copy()
        red = np.zeros_like(frame)
        red[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(out, 1.0, red, 0.5, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        color = (0, 0, 255) if is_anomalous else (0, 200, 0)
        text = f"{'DEFECT' if is_anomalous else 'OK'}  score={score:.3f}  {fps:.1f} FPS"
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Segmentation", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.release()
    cv2.destroyAllWindows()