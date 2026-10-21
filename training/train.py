from anomalib.engine import Engine
from anomalib.data import Folder
from pathlib import Path
from anomalib.deploy import ExportType

import uuid

BASE_DIR = Path(__file__).resolve().parents[1]

class PatchCoreLoader:
    def __init__(
            self,
            # data loader
            root_dir_name,
            normal_dir="good", 
            abnormal_dir=None, 
            name=None,

            # model params
            backbone = "wide_resnet50_2",
            layers = ["layer2", "layer3"],
            coreset_sampling_ratio = 0.1, 
        ):
        from anomalib.models import Patchcore

        self.model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
        )
        self.engine = Engine(
            accelerator="auto",
            devices=1,
            max_epochs=1,          # PatchCore is memory-bank based, not gradient-trained — 1 epoch is enough to fill the bank
        )
        
        self._load_data(normal_dir=normal_dir, abnormal_dir=abnormal_dir,name=name,root_dir_name=root_dir_name)

    def _load_data(self, root_dir_name, normal_dir="good", abnormal_dir=None, name=None):
        if not name:
            name = uuid.uuid4().hex[:8]

        self.datamodule = Folder(
            name=name,
            root=BASE_DIR / "training/dataset" / root_dir_name,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            train_batch_size=8,
            eval_batch_size=8,
        )

    def _training(self):
        self.engine.fit(model=self.model, datamodule=self.datamodule)

        self._export_torch(self.engine)

    def _export_onnx(self, engine):
        export_root = BASE_DIR / "training" / Path("models") / f"patchcore_{uuid.uuid4().hex[:8]}"

        engine.export(
            model=self.model,
            export_type=ExportType.ONNX,
            export_root=export_root,
        )

    def _export_torch(self, engine):
        export_root = BASE_DIR / "training" / Path("models") / f"patchcore_{uuid.uuid4().hex[:8]}"

        engine.export(
            model=self.model,
            export_type=ExportType.TORCH,
            export_root=export_root,
        )

if __name__ == "__main__":
    loader = PatchCoreLoader(
        root_dir_name="bottle",
        normal_dir="train/good",
        abnormal_dir=None,
        name="bottle_defect",
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )
    loader._training()