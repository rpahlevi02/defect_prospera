from anomalib.deploy import TorchInferencer  # or OpenVINOInferencer
from anomalib.deploy import OpenVINOInferencer 

class InferenceAnomaly:
    def __init__(self, model_path):
        from anomalib.models import Patchcore
        from anomalib.engine import Engine

        self.model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
        )
        self.engine = Engine()

        self.model_path = model_path

    def predict(self, image_path):
        predictions = self.engine.predict(
            model=self.model,
            ckpt_path=self.model_path,   # same .ckpt path you were passing to TorchInferencer
            data_path=image_path,
        )
        return predictions[0]

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from anomalib.visualization.image.functional import visualize_anomaly_map

    model_path = r"C:\Users\AQI\JOB\galaria_prospera\results\Patchcore\bottle_defect\latest\weights\lightning\model.ckpt"  # Replace with your actual model path
    image_path = r"C:\Users\AQI\JOB\galaria_prospera\training\dataset\bottle\test\broken_small\004.png"  # Replace with your actual image path

    inference = InferenceAnomaly(model_path=model_path)
    result = inference.predict(image_path=image_path)

    print(result.pred_label)  # Predicted label (0 for normal, 1 for anomalous
    print(result.pred_score)  # Predicted anomaly score

    heat_map = visualize_anomaly_map(
        result.anomaly_map.squeeze(),   # raw pixel-level anomaly scores
        colormap=True,
        normalize=True,
    )
    img_array = np.array(heat_map)


    plt.imshow(img_array, cmap="jet")
    # plt.imshow(result.image.squeeze())
    plt.colorbar()
    plt.show()