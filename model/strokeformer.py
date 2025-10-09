from typing import Literal, Union, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from losses import SegmentationLoss
from model.segformer3d import SegFormer3D
from utils import build_metrics, compute_metrics, reset_metrics


class StrokeFormer(LightningModule):
    def __init__(self,
                 segmentation_loss: str = "DiceLoss",
                 segmentation_loss_config: dict = {},
                 prediction_loss: str = "BCEWithLogitsLoss",
                 prediction_loss_config: dict = {},
                 weights: Tuple[float, float] = (0.5, 0.5),
                 reduction: Literal["mean", "sum", "none"] = "mean",
                 num_classes: int = 1,
                 in_channels: int = 1,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 lr: float = 4e-4,
                 init_lr: float = 4e-6,
                 warmup_steps: int = 10_000,
                 lr_scheduler_interval: Literal["step", "epoch"] = "step",
                 weight_decay: float = 1e-3):
        super().__init__()

        self.loss = SegmentationLoss(segmentation_loss=segmentation_loss,
                                     segmentation_loss_config=segmentation_loss_config,
                                     prediction_loss=prediction_loss, prediction_loss_config=prediction_loss_config,
                                     weights=weights, reduction=reduction)

        self.model = SegFormer3D(num_classes=num_classes, in_channels=in_channels)

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.betas = betas
        self.eps = eps

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.start_factor = init_lr / lr if lr > init_lr else lr / init_lr
        self.end_factor = init_lr / lr if lr > init_lr else 1.0
        self.lr_scheduler_interval = lr_scheduler_interval
        self.weight_decay = weight_decay

        self.metrics = build_metrics(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, return_preds: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        logits = self.model(x)

        if return_preds:
            if self.num_classes > 2:
                preds = logits.softmax(dim=1)
            else:
                preds = logits.sigmoid()
            return preds

        return logits

    def _common_step(self, batch, prefix: Literal['train', 'val', 'test']):
        scans, masks = batch

        logits = self.forward(scans)

        loss_dict = self.loss(logits, masks, prefix=prefix, return_dict=True)

        if self.num_classes > 2:
            predicted_masks = logits.softmax(dim=1)
        else:
            predicted_masks = logits.sigmoid()

        log_dict = {
            **loss_dict,
            **compute_metrics(predicted_masks, masks, metrics=self.metrics, prefix=prefix),
        }

        self.log_dict(dictionary=log_dict, on_step=True, prog_bar=True, on_epoch=True)

        return loss_dict[f'{prefix}_loss']

    def training_step(self, batch):
        return self._common_step(batch, prefix='train')

    def validation_step(self, batch):
        return self._common_step(batch, prefix='val')

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        reset_metrics(self.metrics)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
                                      weight_decay=self.weight_decay)

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor,
                                                   end_factor=self.end_factor, total_iters=self.warmup_steps)

        total_training_steps = self.trainer.estimated_stepping_batches
        poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0,
                                                               total_iters=total_training_steps - self.warmup_steps)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, poly_scheduler],
                                                          milestones=[self.warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
            }
        }
