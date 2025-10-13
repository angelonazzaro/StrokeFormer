import sys
from typing import Tuple, Union, Optional, Literal

import monai.losses
import torch
import torch.nn as nn


_MODULES = [monai.losses, nn, sys.modules[__name__]]


def check_loss_validity(loss: str):
    loss_exists = True

    for module in _MODULES:
        loss_exists = getattr(module, loss, None) is not None
        if loss_exists:
            break

    if not loss_exists:
        raise AttributeError(
            f"{loss} is not a valid loss function. It is not available in none of  the following modules: {_MODULES}")


def instantiate_loss(loss: str, config: dict):
    check_loss_validity(loss)

    loss_cls = None
    for module in _MODULES:
        loss_cls = getattr(module, loss, None)
        if loss_cls is not None:
            break

    for key in config.keys():
        if "weight" in key and not isinstance(config[key], torch.Tensor):
            config[key] = torch.tensor(config[key])

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
                 loss_weights: Tuple[float, float] = (0.5, 0.5),
                 reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()

        for w in loss_weights:
            if w < 0 or w > 1:
                raise ValueError(f"Weights should be between 0 and 1, but got {w}")

        self.segmentation_loss = instantiate_loss(segmentation_loss, segmentation_loss_config)
        self.prediction_loss = instantiate_loss(prediction_loss, prediction_loss_config)
        self.loss_weights = loss_weights
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

        total_loss = self.loss_weights[0] * seg_loss + self.loss_weights[1] * cls_loss

        prefix = f"{prefix}_" if prefix is not None else ""

        if return_dict:
            return {
                f"{prefix}loss": total_loss,
                f"{prefix}{self.prediction_loss.__class__.__name__}": cls_loss,
                f"{prefix}{self.segmentation_loss.__class__.__name__}": seg_loss,
            }

        return total_loss


def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1, p: float = 2.0,
               apply_sigmoid: bool = False) -> torch.Tensor:
    if apply_sigmoid:
        preds = preds.sigmoid()

    b = preds.shape[0]

    # flatten label and prediction tensors
    preds = preds.reshape(b, -1).contiguous()
    targets = targets.reshape(b, -1).contiguous()

    intersection = (preds * targets).sum()
    union = (preds.pow(p) + targets.pow(p)).sum() # elevating to p should expedite convergence
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean(dim=0)


class BinaryDiceLoss(torch.nn.Module):
    """
    Binary Dice loss
    Args:
        p: Denominator value: `math: \sum{x^p} + \sum{y^p}`, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean', return sum if 'sum', return a tensor
            of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth: float = 1.0, p: float = 2.0, reduction: str = "mean", apply_sigmoid: bool = False):
        super(BinaryDiceLoss, self).__init__()

        if smooth <= 0:
            raise ValueError("Smooth must be greater than 0")

        self.smooth = smooth
        self.p: float = p
        self.reduction: str = reduction
        self.apply_sigmoid = apply_sigmoid

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice loss between the predicted and target binary tensors.

        Args:
            predict (torch.Tensor): A tensor of shape [N, C, *], where N is the batch size, C is the number of classes,
                and * is the spatial dimensions.
            target (torch.Tensor): A tensor of the same shape as the predicted tensor.

        Returns:
            Loss tensor according to arg reduction.

        Raises:
            AssertionError: If the batch sizes of the predicted and target tensors do not match.
            Exception: If unexpected reduction.
        """
        assert predict.shape[0] == target.shape[0], "BinaryDiceLoss: predict & target batch size don't match"

        dice = dice_score(predict, target, smooth=self.smooth, p=self.p, apply_sigmoid=self.apply_sigmoid)

        loss = 1 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))