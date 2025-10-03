import random
from typing import Any, List

import numpy as np
import torch
import wandb
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2.functional import to_pil_image

from utils import overlay_img


class LogPredictionCallback(Callback):
    """
        This callback logs prediction results to a wandb Table.
        The prediction results contain:
            - scan id
            - slice idx
            - ground_truth
            - prediction
    """

    def __init__(self, num_images: int, log_every_n_val_epochs: int, slices_per_scan: int):
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
        pl_module.eval()
        columns = ["id", "slice_idx", "ground-truth", "prediction"]

        columns.extend(list(pl_module.metrics.keys()))

        table = wandb.Table(columns=columns)

        if isinstance(self.scans, List):
            self.scans = torch.cat(self.scans)
            self.masks = torch.cat(self.masks)

        with torch.no_grad():
            preds = pl_module(self.scans, return_preds=True)

        preds = (preds >= 0.5).float()

        # randomly sample slices_per_scan
        if self.slices_per_scan < self.scans.shape[-3]:
            random_slices = random.choices(np.arange(self.masks.shape[-3]), k=self.slices_per_scan)
            masks = self.masks[:, :, random_slices]
            scans = self.scans[:, :, random_slices]
            preds = preds[:, :, random_slices]
        else:
            masks = self.masks
            scans = self.scans

        i = 0
        for scan, predicted_mask, mask in zip(scans, preds, masks):
            for slice_idx in range(scan.shape[-3]):
                scan_slice = scan[0, slice_idx, ...]
                mask_slice = mask[0, slice_idx, ...]
                predicted_slice = predicted_mask[0, slice_idx, ...]

                scores = {}

                for metric in pl_module.metrics:
                    scores[f"{metric}"] = pl_module.metrics[metric](predicted_slice, mask_slice).cpu().item()  # noqa

                scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min())

                scan_slice = np.asarray(to_pil_image(scan_slice).convert("RGB"))
                mask_slice = np.asarray(to_pil_image(mask_slice).convert("RGB"))
                predicted_slice = np.asarray(to_pil_image(predicted_slice).convert("RGB"))

                gt = overlay_img(scan_slice, mask_slice, color=(0, 255, 0))
                pd = overlay_img(scan_slice, predicted_slice, color=(255, 0, 0))

                table.add_data(f"scan_{i}", slice_idx, wandb.Image(gt), wandb.Image(pd), *scores.values())
            i += 1

        pl_module.train()
        # assuming wandb logger
        trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/predictions": table})
