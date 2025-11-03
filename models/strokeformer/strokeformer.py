from typing import Literal, Optional, Union

import einops
import torch.nn.functional as f
from lightning import LightningModule
from torch import Tensor, optim

from losses import SegmentationLoss
from utils import compute_metrics, build_metrics, load_anoddpm_checkpoint
from .segformer3d import SegFormer3D
from ..anoddpm import AnoDDPM


class StrokeFormer(LightningModule):
    def __init__(self,
                 seg_loss: str = "ICILoss",
                 seg_loss_cfg: Optional[dict] = None,
                 cls_loss: str = "BCEWithLogitsLoss",
                 cls_loss_cfg: Optional[dict] = None,
                 loss_weights: tuple[float, float] = (0.5, 0.5),
                 in_channels: int = 1,
                 num_classes: int = 2,
                 eps: float = 1e-8,
                 opt_lr: float = 3e-5,
                 warmup_lr: float = 4e-6,
                 max_lr: float = 4e-4,
                 warmup_steps: int = 10,
                 betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-3,
                 lr_scheduler_interval: Literal["epoch", "step"] = "epoch",
                 detection_model: Optional[Union[LightningModule, str]] = None):
        super().__init__()

        self.model = SegFormer3D(num_classes=num_classes, in_channels=in_channels)
        self.loss = SegmentationLoss(seg_loss=seg_loss, seg_loss_cfg=seg_loss_cfg,
                                     cls_loss=cls_loss, cls_loss_cfg=cls_loss_cfg,
                                     loss_weights=loss_weights)

        self.num_classes = num_classes

        if warmup_lr > max_lr:
            max_lr = warmup_lr * 100

        self.opt_lr = opt_lr
        self.warmup_lr = warmup_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_scheduler_interval = lr_scheduler_interval
         
        self.detection_model = detection_model

        if isinstance(detection_model, str):
            # load detection model that will be later used for region proposal
            # here, we assume it's AnoDDPM which is composed of a base UNet model and its EMA
            # since it will be used in inference mode, we only need the EMA
            self.detection_model = load_anoddpm_checkpoint(AnoDDPM, detection_model, inference=True)
        
        # freeze detection model
        if self.detection_model is not None:
            for param in self.detection_model.parameters():
                param.requires_grad = False
            self.detection_model.eval()

        self.metrics = build_metrics(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x: Tensor, return_preds: bool = False, return_dict: bool = False) -> Union[Tensor, dict]:
        # TODO: handle region proposal logic from the detection model
        if self.detection_model is not None:
            pass
        logits = self.model(x)  # (B, N, D, H, W)
        logits = logits.view(logits.shape[0], logits.shape[1], x.shape[1], *logits.shape[2:])  # (B, N, C, D, H, W)

        if logits.shape[-3] != x.shape[-3]:
            # segformer padded the depth dimension to output a cubic tensor
            logits = logits[:, :, :, :x.shape[-3]]

        preds = logits.softmax(dim=1)
        # cast to x.dtype is necessary for MONAI inferer
        preds = preds.argmax(dim=1).to(dtype=x.dtype)

        if return_preds:
            return preds
        elif return_dict:
            return {"logits": logits, "preds": preds}

        return logits

    def _common_step(self, batch, prefix: Literal["train", "val"]):
        scans, masks = batch["scans"], batch["masks"]  # (B, C, D, H, W)

        logits = self.forward(scans, return_dict=True)  # (B, N, D, H, W)
        logits, preds = logits["logits"], logits["preds"]

        if masks.shape != logits.shape:
            # convert masks to one-hot format to match logits
            masks = f.one_hot(masks, num_classes=self.num_classes).to(dtype=scans.dtype)
            masks = einops.rearrange(masks, "b c d h w n -> b n c d h w")

        loss_dict = self.loss(logits, masks, prefix=prefix, return_dict=True)

        # convert masks back to index tensors
        masks = masks.argmax(dim=1)

        log_dict = {
            **loss_dict,
            **compute_metrics(preds, masks, metrics=self.metrics, prefix=prefix) # noqa
        }

        self.log_dict(dictionary=log_dict, on_step=False, prog_bar=True, on_epoch=True)

        return loss_dict[f"{prefix}_loss"]

    def training_step(self, batch):
        return self._common_step(batch, "train")

    def validation_step(self, batch):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        # follow SegFormer3D's original optimization strategy:
        # AdamW + linear LR schedule followed by a polynomial one
        optimizer = optim.AdamW(self.parameters(), lr=self.opt_lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        start_factor = self.warmup_lr / self.max_lr
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=self.warmup_steps)

        total_training_steps = self.trainer.estimated_stepping_batches
        poly_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=total_training_steps - self.warmup_steps)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, poly_scheduler], milestones=[self.warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
            }
        }