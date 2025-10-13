from typing import Any, List, Optional

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

    def __init__(self, num_images: int, log_every_n_val_epochs: int, slices_per_scan: Optional[int] = None):
        self.num_images = num_images
        self.scans = []
        self.masks = []
        self.log_every_n_val_epochs = log_every_n_val_epochs
        self.slices_per_scan = slices_per_scan

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
        columns = ["id", "slice idx", "lesion size", "ground-truth", "prediction"]

        columns.extend(list(pl_module.metrics.keys()))

        table = wandb.Table(columns=columns)

        if isinstance(self.scans, List):
            self.scans = torch.cat(self.scans)
            self.masks = torch.cat(self.masks)

        i = 0
        for j, result in enumerate(predictions_generator(model=pl_module, scans=self.scans, masks=self.masks, metrics=pl_module.metrics, slices_per_scan=self.slices_per_scan, lesions_only=False)):
            table.add_data(f"scan_{i}", result["slice_idx"], result["lesion_size"], wandb.Image(result["gt"]), wandb.Image(result["pd"]), *result["scores"].values())
            if j % self.slices_per_scan == 0:
                i += 1

        pl_module.train()
        # assuming wandb logger
        trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/predictions": table})