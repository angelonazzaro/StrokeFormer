from typing import Literal, Optional, Union

import einops
import torch
import torch.nn.functional as f
from lightning import LightningModule
from torch import Tensor, optim

from losses import SegmentationLoss
from utils import compute_metrics, build_metrics, expand_boxes, compare_head_sizes, compare_head_shapes, round_half_up, \
    sliding_window_inference_3d
from .segformer3d import SegFormer3D
from ..rpn import RPN


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
                 rpn_model: Optional[Union[LightningModule, str]] = None,
                 roi_size: tuple[int, int, int] = (64, 64, 64)):
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

    def forward(self, x: Tensor, head_masks: Optional[Tensor] = None, return_preds: bool = False, return_dict: bool = False) -> Union[Tensor, dict]:
        if self.rpn_model is not None:
            # TODO: optimize
            # the RPN model works in 2D mode so we need to:
            # for each scan tensor in the batch:
            #   1. decompose the 3D tensor into 2D slices and perform detection on each
            #   2. multiple regions may be proposed per slice, for simplicity we create a bigger bounding box
            #   encompassing all proposed boxes
            #   3. we need to propose a 3D region that is anatomically consistent. To achieve this, we can:
            #       3.1 track the first slice with a valid bounding box
            #       3.2 keep considering slices (going upward) that have a similar head mask (size and shape)
            #   4. segment on this 3D region

            # x has shape (B, 1, D, H, W)
            B, C, D, H, W = x.shape
            logits = torch.zeros(B, 2, D, H, W, device=x.device)
            # regions = torch.empty(B, C, *self.roi_size, device=self.model.device)
            for i in range(B):
                # step 1: decompose 3D tensor into 2D slices.
                # since each scan is of shape [D, H, W], the RPN model will see D as the batch dimension
                # WARNING: decrease the batch dimension when extracting subvolumes to reduce memory requirements
                with torch.no_grad():
                    # (1, D, H, W) -> (D, 1, H, W)
                    # this is a list of dicts: boxes, labels, scores
                    proposals = self.rpn_model(x[i].permute(1, 0, 2, 3), None)

                boxes = [torch.empty((0, 4), device=x.device)] * len(proposals)
                min_slice_idx, max_slice_idx = None, None
                max_area, max_box = 0, None
                for j in range(len(proposals)):
                    # step 2: expand smaller boxes to as big as the roi size
                    # curr_boxes = expand_boxes(proposals[j]["boxes"], x.shape[3:], self.roi_size[1:]) # noqa
                    # boxes.append(curr_boxes)
                    curr_boxes = proposals[j]["boxes"]
                    # a shape of (0, 4) means no bounding box
                    if curr_boxes.shape[0] > 0:
                        # if multiple boxes, create a bigger box to encompass all
                        xmin, xmax, ymin, ymax = float("inf"), float("-inf"), float("inf"), float("-inf")
                        for box in curr_boxes:
                            xmin = min(xmin, box[0])
                            xmax = max(xmax, box[2])
                            ymin = min(ymin, box[1])
                            ymax = max(ymax, box[3])

                        curr_boxes = torch.tensor([xmin, ymin, xmax, ymax]).unsqueeze(0) # (1, 4)
                        curr_boxes = expand_boxes(curr_boxes, x.shape[3:], self.roi_size[1:]) # noqa
                        curr_area = (curr_boxes[0, 3] - curr_boxes[0, 1]) * (curr_boxes[0, 2] - curr_boxes[0, 0])
                    else:
                        curr_area = torch.tensor(0, device=x.device)

                    if min_slice_idx is None and curr_area > 0:
                        min_slice_idx = j
                        max_area = curr_area
                        max_box = curr_boxes
                    elif min_slice_idx is not None:
                        max_slice_idx = j
                        # step 3: compare shape and size to keep anatomically plausible regions
                        # NOTE: in this way, multiple regions per volume may be proposed
                        min_head_mask = head_masks[i, :, min_slice_idx]
                        curr_head_mask = head_masks[i, :, j]
                        if compare_head_sizes(min_head_mask, curr_head_mask, 0.1) and compare_head_shapes(min_head_mask, curr_head_mask, 0.9):
                            if curr_area > max_area:
                                max_area = curr_area
                                max_box = curr_boxes
                        else:
                            # curr slices offers a differ anatomical view, so we cut the region here
                            boxes[min_slice_idx:j] = ([max_box] * (j - min_slice_idx))
                            min_slice_idx = j
                            max_area = curr_area
                            max_box = curr_boxes
                # boxes now holds a bounding box for each slice
                # we filter the empty boxes and concentrate only on valid boxes
                # if multiple values are present, it means that there are different anatomical regions
                region_start = 0
                for r in range(1, len(boxes)):
                    prev_box = boxes[r - 1]
                    curr_box = boxes[r]

                    same_box = (
                            prev_box.shape == curr_box.shape and
                            (prev_box.numel() == 0 and curr_box.numel() == 0 or torch.equal(prev_box, curr_box))
                    )

                    if not same_box:
                        if prev_box.numel() > 0: # noqa
                            xmin, ymin, xmax, ymax = [int(v.item()) for v in prev_box[0]]
                            region = x[i, :, region_start:r, ymin:ymax, xmin:xmax]
                            region_logits = sliding_window_inference_3d(region, self.model, *self.roi_size)
                            logits[i, :, region_start:r, ymin:ymax, xmin:xmax] = region_logits
                        region_start = r

                last_box = boxes[-1]
                if last_box.numel() > 0: # noqa
                    xmin, ymin, xmax, ymax = [int(v.item()) for v in last_box[0]]
                    region = x[i, :, region_start:r, ymin:ymax, xmin:xmax]
                    region_logits = sliding_window_inference_3d(region, self.model, *self.roi_size)
                    logits[i, :, region_start:r, ymin:ymax, xmin:xmax] = region_logits
        else:
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
        scans, head_masks, masks = batch["scans"], batch["head_masks"], batch["masks"]  # (B, C, D, H, W)

        logits = self.forward(scans, head_masks=head_masks, return_dict=True)  # (B, N, D, H, W)
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