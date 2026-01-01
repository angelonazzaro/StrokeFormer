from typing import Literal, Optional, Union

import torch
from lightning import LightningModule
from torch import Tensor, optim

from losses import SegmentationLoss
from utils import compute_metrics, build_metrics, sliding_window_inference_3d, propose_regions
from .segformer3d import SegFormer3D
from ..rpn import RPN


class StrokeFormer(LightningModule):
    def __init__(self,
                 seg_loss: str = "ICILoss",
                 seg_loss_cfg: Optional[dict] = None,
                 cls_loss: str = "BCEWithLogitsLoss",
                 cls_loss_cfg: Optional[dict] = None,
                 seg_loss_weight: float = 0.5,
                 cls_loss_weight: float = 0.5,
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
                 rpn_model: Optional[Union[LightningModule, str]] = None,
                 roi_size: tuple[int, int, int] = (64, 64, 64)):
        super().__init__()

        self.model = SegFormer3D(num_classes=num_classes, in_channels=in_channels)
        self.loss = SegmentationLoss(seg_loss=seg_loss, seg_loss_cfg=seg_loss_cfg,
                                     cls_loss=cls_loss, cls_loss_cfg=cls_loss_cfg,
                                     seg_loss_weight=seg_loss_weight, cls_loss_weight=cls_loss_weight)

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

        self.roi_size = roi_size

        if isinstance(rpn_model, str):
            # load detection model that will be later used for region proposal
            self.rpn_model = RPN.load_from_checkpoint(rpn_model)
            for param in self.rpn_model.parameters():
                param.requires_grad = False
            self.rpn_model.eval()
        else:
            self.rpn_model = None

        self.metrics = build_metrics(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, scans: Tensor, masks: Optional[Tensor] = None, head_masks: Optional[Tensor] = None, regions: Optional[Tensor] = None,
                return_preds: bool = False, return_dict: bool = False) -> Union[Tensor, dict]:
        # if regions are passed and rpn is None, use regions otherwise use RPN
        # if both are None, default to full volume

        if self.rpn_model is not None or regions is not None:
            if head_masks is None:
                raise ValueError("`head_masks` is None. `head_masks` are required for Region Proposal processing.")

            B, C, D, H, W = scans.shape
            # WARNING: right now we backpropagate through the full volume
            logits_sum = torch.zeros(B, self.num_classes, D, H, W, device=scans.device, dtype=scans.dtype)
            logits_count = torch.zeros(B, 1, D, H, W, device=scans.device)

            if self.rpn_model is not None and regions is None:
                for i in range(B):
                    for (final_box, group_dict) in propose_regions(self.rpn_model, scans[i], head_masks[i], self.roi_size):
                        # assign same box to all slices in the group
                        start, end = group_dict["region_start"], group_dict["region_end"]
                        xmin, ymin, xmax, ymax = final_box[:, 0], final_box[:, 1], final_box[:, 2], final_box[:, 3]
                        # group has shape [1, D_g, H, W]
                        region = group_dict["group"][:, ymin:ymax, xmin:xmax]
                        curr_logits = sliding_window_inference_3d(region, self.model, *self.roi_size)
                        logits_sum[i, :, start:end, ymin:ymax, xmin:xmax] += curr_logits
                        logits_count[i, :, start:end, ymin:ymax, xmin:xmax] += 1
            elif regions is not None:
                # regions are assumed to be already divided into groups and to have the bounding box standardized
                # according to the above procedure
                for i in range(B):
                    scan = scans[i]
                    for region in regions[i]:
                        start, end = region["region_start"], region["region_end"]
                        xmin, ymin, xmax, ymax = region["xmin"], region["ymin"], region["xmax"], region["ymax"]
                        region_input = scan[:, start:end, ymin:ymax, xmin:xmax]
                        curr_logits = sliding_window_inference_3d(region_input, self.model, *self.roi_size)
                        logits_sum[i, :, start:end, ymin:ymax, xmin:xmax] += curr_logits
                        logits_count[i, :, start:end , ymin:ymax, xmin:xmax] += 1

            logits = logits_sum / logits_count.clamp(min=1)
        else:
            logits = self.model(scans)  # (B, N, D, H, W)

        preds = logits.softmax(dim=1)
        # cast to x.dtype is necessary for MONAI inferer
        preds = preds.argmax(dim=1).to(dtype=scans.dtype)

        if return_preds:
            return preds
        elif return_dict:
            logits_dict = {"logits": logits, "preds": preds}

            if masks is not None:
                logits_dict["masks"] = masks

            return logits_dict

        return logits

    def _common_step(self, batch, prefix: Literal["train", "val"]):
        scans, head_masks, masks = batch["scans"], batch["head_masks"], batch["masks"]  # (B, D, H, W)
        regions = batch.get("regions", None)

        out = self.forward(scans=scans, masks=masks, head_masks=head_masks, regions=regions, return_dict=True)
        logits, preds = out["logits"], out["preds"] # (B, N, D, H, W) and (B, D, H, W)

        if logits.shape[-3] > masks.shape[-3]:
            # segformer pads the dimensions to output a cube
            logits = logits[:, :, :masks.shape[-3]]
            preds = preds[:, masks.shape[-3]]

        masks = out.get("masks", masks)

        loss_dict = self.loss(logits, masks, prefix=prefix, return_dict=True)

        log_dict = {
            **loss_dict,
            **compute_metrics(preds, masks, metrics=self.metrics, prefix=prefix)  # noqa
        }

        self.log_dict(dictionary=log_dict, batch_size=scans.shape[0], on_step=False, prog_bar=True, on_epoch=True)

        return loss_dict[f"{prefix}_loss"]

    def training_step(self, batch):
        return self._common_step(batch, "train")

    def validation_step(self, batch):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        # follow SegFormer3D's original optimization strategy:
        # AdamW + linear LR schedule followed by a polynomial one
        optimizer = optim.AdamW(self.parameters(), lr=self.opt_lr, betas=self.betas, eps=self.eps,
                                weight_decay=self.weight_decay)
        start_factor = self.warmup_lr / self.max_lr
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=self.warmup_steps)

        total_training_steps = self.trainer.estimated_stepping_batches
        poly_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, power=1.0,
                                                         total_iters=total_training_steps - self.warmup_steps)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, poly_scheduler], milestones=[self.warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
            }
        }
