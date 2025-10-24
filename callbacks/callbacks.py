from typing import Any, List, Optional, Literal

import torch
import wandb
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from utils import predictions_generator


class LogPredictionCallback(Callback):
    """
        This callback logs prediction results to a wandb Table.
        The prediction results contain:
            - scan id
            - slice idx
            - lesion size category
            - ground_truth
            - prediction
    """

    def __init__(self, num_images: int, log_every_n_val_epochs: int,
                 task: Literal["segmentation", "reconstruction"] = "segmentation",
                 slices_per_scan: Optional[int] = None):
        self.num_images = num_images
        self.task = task
        self.scans = []
        self.masks = []
        self.log_every_n_val_epochs = log_every_n_val_epochs
        self.slices_per_scan = slices_per_scan
        self.columns = []

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0):
        needed = self.num_images - len(self.scans)

        if needed > 0:
            take = min(needed, batch[0].shape[0])
            self.scans.append(batch[0][:take])
            self.masks.append(batch[1][:take])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.sanity_checking or trainer.current_epoch % self.log_every_n_val_epochs != 0:
            return

        pl_module.eval()
        if len(self.columns) == 0:
            self.columns = ["id", "slice idx", "lesion size", "ground-truth", "prediction"]

            metrics = list(pl_module.metrics.keys())

            if "overlap_metrics" in metrics:
                metrics.extend(["dice", "accuracy", "accuracy_background", "accuracy_foreground"])
                metrics.remove("overlap_metrics")

            if "boundary_metrics" in metrics:
                metrics.extend(["hausdorff95", "hausdorff95_1_to_2", "hausdorff95_2_to_1", "assd"])
                metrics.remove("boundary_metrics")

            metrics = sorted(metrics)

            self.columns.extend(metrics)

        table = wandb.Table(columns=self.columns)

        if isinstance(self.scans, List):
            self.scans = torch.cat(self.scans)
            self.masks = torch.cat(self.masks)

        i = 0
        for j, result in enumerate(predictions_generator(model=pl_module, scans=self.scans, masks=self.masks, metrics=pl_module.metrics, slices_per_scan=self.slices_per_scan), task=task):
            table.add_data(f"scan_{i}", result["slice_idx"], result["lesion_size"], wandb.Image(result["gt"]), wandb.Image(result["pd"]), *result["scores"].values())
            if j % self.slices_per_scan == 0:
                i += 1

        pl_module.train()
        # assuming wandb logger
        trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/predictions": table})
