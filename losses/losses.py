import sys

from functools import partial
from typing import Optional, Literal, Union

import torch
import torch.nn as nn
import monai.losses
from torchmetrics.functional.segmentation import dice_score

from utils import get_slices_with_lesions

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


class SegmentationLoss(nn.Module):
    def __init__(self,
                 seg_loss: str = "BinaryDiceLoss",
                 seg_loss_config: Optional[dict] = None,
                 cls_loss: str = "BCEWithLogitsLoss",
                 cls_loss_config: Optional[dict] = None,
                 loss_weights: tuple[float, float] = (0.5, 0.5)):
        super().__init__()

        for w in loss_weights:
            if w < 0 or w > 1:
                raise ValueError(f"Weights should be between 0 and 1, but got {w}")

        self.seg_loss = instantiate_loss(seg_loss, seg_loss_config)
        self.cls_loss = instantiate_loss(cls_loss, cls_loss_config)
        self.loss_weights = loss_weights

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            prefix: Optional[Literal['train', 'val', 'test']] = None,
            return_dict: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Args:
            predictions: segmentation predictions (B, N, D, H, W) or (B, C, N, D, H, W)
            targets: Ground-truth segmentation mask (B, N, D, H, W) or (B, C, N, D, H, W)
            prefix: Prefix of training/inference step. Can be either 'train', 'val', 'test'.
            return_dict: If True, returns dict with separate losses.
        """
        assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"

        # always compute classification loss
        cls_loss = self.cls_loss(predictions, targets)

        # segmentation loss (only if lesion exists). this should help the model to focus on segmenting
        # slices that actually contain lesions
        seg_loss = torch.tensor(0.0, requires_grad=True, device=predictions.device)
        slices_with_lesions = get_slices_with_lesions(targets)

        if predictions.ndim == 5:
            predictions = predictions[:, :, slices_with_lesions]
            targets = targets[:, :, slices_with_lesions]
        else:
            predictions = predictions[:, :, :, slices_with_lesions]
            targets = targets[:, :, :, slices_with_lesions]

        if slices_with_lesions.any():
            seg_loss = self.seg_loss(predictions, targets)

        total_loss = self.loss_weights[0] * seg_loss + self.loss_weights[1] * cls_loss

        prefix = f"{prefix}_" if prefix is not None else ""

        if return_dict:
            return {
                f"{prefix}loss": total_loss,
                f"{prefix}{self.cls_loss.__class__.__name__}": cls_loss,
                f"{prefix}{self.seg_loss.__class__.__name__}": seg_loss,
            }

        return total_loss


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self,
                 num_classes: int,
                 include_background: bool = False,
                 input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
                 average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
                 aggregation_level: Optional[Literal["samplewise", "global"]] = "global"):
        super(BinaryDiceLoss, self).__init__()

        self.num_classes = num_classes
        self.include_background = include_background
        self.average = average
        self.aggregation_level = aggregation_level

        self.dice = partial(dice_score, include_background=include_background, num_classes=num_classes,
                            input_format=input_format, average=average, aggregation_level=aggregation_level)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice loss between the predicted and target binary tensors.

        Args:
            predictions (torch.Tensor): A tensor of shape [N, C, *], where N is the batch size, C is the number of classes,
                and * is the spatial dimensions.
            targets (torch.Tensor): A tensor of the same shape as the predicted tensor.

        Returns:
            Loss tensor according to arg reduction.

        Raises:
            AssertionError: If the batch sizes of the predicted and target tensors do not match.
            Exception: If unexpected target not in [0, 1] range.
        """
        assert predictions.shape[0] == targets.shape[0], "BinaryDiceLoss: predictions & targets' batch sizes don't match"

        if targets.unique().tolist() != [0, 1]:
            raise ValueError(f"BinaryDiceLoss: target expected in [0, 1] range but got {targets.unique()}")

        return 1 - self.dice(predictions, targets)