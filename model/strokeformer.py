from typing import Optional, Literal, Union, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from monai.losses import DiceCELoss

from model.segformer3d import SegFormer3D
from utils import reconstruct_volume, build_metrics, extract_patches


class StrokeFormer(LightningModule):
    def __init__(self,
                 num_classes: int = 1,
                 in_channels: int = 1,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 warmup_steps: int = 10_000,
                 start_lr: float = 4e-6,
                 end_lr: float = 4e-4,
                 weight_decay: float = 1e-3):
        super().__init__()

        self.model = SegFormer3D(num_classes=num_classes, in_channels=in_channels)

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.betas = betas
        self.eps = eps
        # TODO: review learning rate schedulers
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay

        self.loss = DiceCELoss(sigmoid=num_classes <= 2, softmax=num_classes > 2, squared_pred=True)

        self.metrics = build_metrics(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, return_preds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # (B, P, C, D, H, W) -> (B*P, C, D, H, W)
        x = x.view(-1, *x.shape[2:])
        logits = self.model(x) # shape (B*P, N, D, H, W)

        if return_preds:
            if self.num_classes > 2:
                preds = logits.softmax(dim=1)
            else:
                preds = logits.sigmoid()
            return preds
        return logits


    def _common_step(self, batch, prefix: Literal['train', 'val', 'test']):
        scan_patches, origins, masks = batch

        logits = self.forward(scan_patches)
        # (B*P, N, D, H, W) -> (B, P, N, D, H, W)
        logits = logits.view(scan_patches.shape[0], scan_patches.shape[1], *logits.shape[1:])

        # reconstruct volumes from patches
        # currently, they only hold the reconstructed logits
        predicted_masks = reconstruct_volume(logits, masks[0].shape, origins)

        # TODO: check if gradients propagate through reconstructed volumes,
        #  otherwise compute loss per patch and aggregate after

        loss = self.loss(predicted_masks, masks)

        log_dict = {f'{prefix}_loss': loss}

        if self.num_classes > 2:
            predicted_masks = predicted_masks.softmax(dim=1)
        else:
            predicted_masks = predicted_masks.sigmoid()

        for metric in self.metrics:
            log_dict[f"{prefix}_{metric}"] = self.metrics[metric](predicted_masks, masks)  # noqa

        self.log_dict(dictionary=log_dict, on_step=True, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch):
        return self._common_step(batch, prefix='train')

    def validation_step(self, batch):
        return self._common_step(batch, prefix='val')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.end_lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        self.warmup_steps = min(self.warmup_steps, self.trainer.estimated_stepping_batches - 1)

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.start_lr / self.end_lr, end_factor=self.end_lr)

        poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, poly_scheduler], milestones=[self.warmup_steps])

        return [optimizer], [scheduler]
