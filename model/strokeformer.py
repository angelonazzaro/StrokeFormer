from typing import Optional, Literal, Union, Tuple, List

import monai.losses
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from model.segformer3d import SegFormer3D
from utils import reconstruct_volume, build_metrics


class StrokeFormer(LightningModule):
    def __init__(self,
                 losses: Union[dict, List[str], str] = ["DiceLoss", "BCEWithLogitsLoss"],
                 losses_weights: Union[dict, List[float], float] = [0.5, 0.5],
                 losses_configs: Optional[Union[dict, List[dict]]] = None,
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

        # TODO: handle dict for different heads
        if isinstance(losses, str):
            losses = [losses]
            losses_weights = [losses_weights]
            losses_configs = [losses_configs if losses_configs is not None else {}]

        if losses_configs is None:
            losses_configs = [{} for _ in range(len(losses))]

        for loss_config in losses_configs:
            for key in loss_config.keys():
                if "weight" in key and not isinstance(loss_config[key], torch.Tensor):
                    loss_config[key] = torch.tensor(loss_config[key]).to(self.device)

        if len(losses) != len(losses_configs) or len(losses) != len(losses_weights):
            raise ValueError("StrokeFormer:  losses, losses_configs and losses_weights must have same length")

        if sum(losses_weights) != 1:
            raise ValueError("StrokeFormer: losses_weights must sum up to 1")

        self.losses = []
        self.losses_configs = losses_configs
        self.losses_weights = losses_weights

        for loss, loss_config in zip(losses, losses_configs):
            loss_cls = getattr(monai.losses, loss, None)

            if loss_cls is None:
                loss_cls = getattr(torch.nn, loss, None)

            if loss_cls is None:
                raise ValueError(
                    f"StrokeFormer: Loss function '{loss}' is not supported in local module `monai.losses` or `torch.nn`.")

            self.losses.append(loss_cls(**loss_config))

        # this function is necessary to plot individual losses alongside the combination
        def compute_loss(values, weights=None):
            weights = weights if weights is not None else self.losses_weights
            assert len(values) == len(weights), f"StrokeFormer: values and weights must have the same length when loss"

            return sum(w * loss_val for w, loss_val in zip(weights, values))

        self.loss = compute_loss

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
        # (B, P, C, D, H, W) -> (B*P, C, D, H, W)
        x = x.view(-1, *x.shape[2:])
        logits = self.model(x)  # shape (B*P, N, D, H, W)

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
        loss_values = []
        for loss in self.losses:
            loss_val = loss(predicted_masks, masks)
            loss_values.append(loss_val)

            if len(self.losses) > 1:
                self.log(f"{prefix}_{loss.__class__.__name__}", loss_val, prog_bar=True, on_step=True, on_epoch=True)

        if len(self.losses) > 1:
            loss = self.loss(loss_values)
        else:
            loss = loss_values[0]

        log_dict = {f"{prefix}_loss": loss}

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
