from typing import Any, Optional, Tuple, List, Literal

import numpy as np
import torch
import wandb
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2.functional import to_pil_image

from constants import SCAN_DIM
from utils import reconstruct_volume, wb_mask


class LogPredictionCallback(Callback):
    """
        This callback logs prediction results to a wandb Table.
        The prediction results contain:
            - scan id
            - slice idx
            - image (contains both prediction and ground truth overlays)
        Only slices with lesions are logged.
    """
    def __init__(self, num_images: int, log_every_n_val_epochs: int, slices_per_scan: int,
                 scan_dim: Tuple[Optional[int], int, int, int] = SCAN_DIM):
        self.num_images = num_images
        self.patches = []
        self.scans = None
        self.masks = []
        self.origins = []
        self.log_every_n_val_epochs = log_every_n_val_epochs
        self.slices_per_scan = slices_per_scan
        self.scan_dim = scan_dim

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0):
        needed = self.num_images - len(self.patches)

        if needed > 0:
            take = min(needed, batch[0].shape[0])
            self.patches.append(batch[0][:take])
            self.origins.append(batch[1][:take])
            self.masks.append(batch[2][:take])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pl_module.eval()
        table = wandb.Table(columns=["slice_idx", "image"])

        if isinstance(self.patches, List):
            self.patches = torch.cat(self.patches)
            self.origins = torch.cat(self.origins)
            self.masks = torch.cat(self.masks)

        if self.scans is None:
            self.scans = reconstruct_volume(self.patches, self.scan_dim, self.origins)

        with torch.no_grad():
            preds = pl_module(self.patches, return_preds=True)  # shape (B*P, N, D, H, W)

        preds = (preds >= 0.5).float()
        preds = preds.view(self.patches.shape[0], self.patches.shape[1], *preds.shape[1:])

        predicted_masks = reconstruct_volume(preds, self.scan_dim, self.origins)

        for scan, predicted_mask, mask in zip(self.scans, predicted_masks, self.masks):
            for slice_idx in range(scan.shape[-3]):
                scan_slice = scan[0, slice_idx, ...]
                scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min())
                scan_slice = np.asarray(to_pil_image(scan_slice).convert("RGB"))

                predicted_slice = predicted_mask[0, slice_idx, ...]
                mask_slice = mask[0, slice_idx, ...]

                table.add_data(slice_idx, wb_mask(scan_slice, predicted_slice.cpu().numpy(), mask_slice.cpu().numpy()))

        pl_module.train()
        # TODO: tables is not uploaded to wandb
        # trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}": table})
        # wandb.log({f"val_epoch_{trainer.current_epoch}": table})
