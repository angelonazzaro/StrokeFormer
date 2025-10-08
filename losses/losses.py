from typing import Tuple, Union, Optional, Literal

import monai.losses
import torch
import torch.nn as nn


def check_loss_validity(loss: str):
    if getattr(monai.losses, loss, None) is None and getattr(nn, loss, None) is None:
        raise AttributeError(
            f"{loss} is not a valid loss function. It is not available neither in `monai` nor `torch.nn`")


def instantiate_loss(loss: str, config: dict):
    check_loss_validity(loss)

    loss_cls = getattr(monai.losses, loss, None)

    if loss_cls is None:
        loss_cls = getattr(nn, loss)

    return loss_cls(**config)


def solve_reduction(loss: torch.Tensor, reduction: str):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise Exception("Unexpected reduction {}".format(reduction))


class SegmentationLoss(nn.Module):
    def __init__(self,
                 segmentation_loss: str = "DiceLoss",
                 segmentation_loss_config: dict = {},
                 prediction_loss: str = "BCEWithLogitsLoss",
                 prediction_loss_config: dict = {},
                 weights: Tuple[float, float] = (0.5, 0.5),
                 reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()

        for w in weights:
            if w < 0 or w > 1:
                raise ValueError(f"Weights should be between 0 and 1, but got {w}")

        self.segmentation_loss = instantiate_loss(segmentation_loss, segmentation_loss_config)
        self.prediction_loss = instantiate_loss(prediction_loss, prediction_loss_config)
        self.weights = weights
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            prefix: Optional[Literal['train', 'val', 'test']] = None,
            return_dict: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Args:
            predictions: segmentation predictions (B, 1, D, H, W)
            targets: Ground-truth segmentation mask (B, 1, D, H, W)
            prefix: Prefix of training/inference step. Can be either 'train', 'val', 'test'.
            return_dict: If True, returns dict with separate losses.
        """
        assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"

        seg_loss_total, cls_loss_total = torch.tensor(0.0, device=predictions.device), torch.tensor(0.0, device=predictions.device)
        seg_loss_count = 0

        for p, tgt in zip(predictions, targets):
            has_lesion = tgt.sum() > 0

            # prediction loss (always)
            cls_loss = self.prediction_loss(p, tgt)
            cls_loss_total += cls_loss

            # segmentation loss (only if lesion exists). this should help the model to focus on segmenting
            # slices that actually contain lesions
            if has_lesion:
                seg_loss = self.segmentation_loss(p, tgt)
                seg_loss_total += seg_loss
                seg_loss_count += 1

        cls_loss = cls_loss_total / len(targets)
        seg_loss = seg_loss_total / max(seg_loss_count, 1)

        total_loss = self.weights[0] * seg_loss + self.weights[1] * cls_loss

        prefix = f"{prefix}_" if prefix is not None else ""

        if return_dict:
            return {
                f"{prefix}loss": total_loss,
                f"{prefix}{self.prediction_loss.__class__.__name__}": cls_loss,
                f"{prefix}{self.segmentation_loss.__class__.__name__}": seg_loss,
            }

        return total_loss
