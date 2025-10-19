from typing import Literal, Union, Tuple, Optional

import monai.losses
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import losses
from model.segformer3d import SegFormer3D
from utils import build_metrics, compute_metrics

_MODULES = [monai.losses, nn, losses]


def check_loss_validity(loss: str):
    loss_exists = True

    for module in _MODULES:
        loss_exists = getattr(module, loss, None) is not None
        if loss_exists:
            break

    if not loss_exists:
        raise AttributeError(
            f"{loss} is not a valid loss function. It is not available in none of  the following modules: {_MODULES}")


def instantiate_loss(loss: str, config: Optional[dict] = None):
    check_loss_validity(loss)

    loss_cls = None
    for module in _MODULES:
        loss_cls = getattr(module, loss, None)
        if loss_cls is not None:
            break

    if config is None:
        config = {}

    for key in config.keys():
        if "weight" in key and not isinstance(config[key], torch.Tensor):
            config[key] = torch.tensor(config[key])

    return loss_cls(**config)


class StrokeFormer(LightningModule):
    def __init__(self,
                 segmentation_loss: str = "DiceLoss",
                 segmentation_loss_config: Optional[dict] = None,
                 prediction_loss: str = "BCEWithLogitsLoss",
                 prediction_loss_config: Optional[dict] = None,
                 loss_weights: Tuple[float, float] = (0.5, 0.5),
                 num_classes: int = 2,
                 in_channels: int = 1,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 lr: float = 4e-4,
                 init_lr: float = 4e-6,
                 warmup_steps: int = 10_000,
                 lr_scheduler_interval: Literal["step", "epoch"] = "step",
                 weight_decay: float = 1e-3):
        super().__init__()

        self.segmentation_loss = instantiate_loss(segmentation_loss, segmentation_loss_config)
        self.prediction_loss = instantiate_loss(prediction_loss, prediction_loss_config)
        self.loss_weights = loss_weights

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

        logits = self.model(x)  # [B, N, D, H, W]

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

        seg_loss = self.segmentation_loss(logits, masks)
        ce_loss = self.prediction_loss(logits, masks.to(dtype=logits.dtype))

        loss = self.loss_weights[0] * seg_loss + self.loss_weights[1] * ce_loss

        log_dict = {
            f"{prefix}_loss": loss,
            f"{self.segmentation_loss.__class__.__name__}": seg_loss,
            f"{self.prediction_loss.__class__.__name__}": ce_loss,
            **compute_metrics(logits, masks, metrics=self.metrics, prefix=prefix),
        }

        self.log_dict(dictionary=log_dict, on_step=False, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch):
        return self._common_step(batch, prefix='train')

    def validation_step(self, batch):
        return self._common_step(batch, prefix='val')

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
