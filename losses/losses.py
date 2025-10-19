from functools import partial
from typing import Optional, Literal

import torch
from torchmetrics.functional.segmentation import dice_score


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
            Exception: If unexpected target not in [0, 1] range.
        """
        assert predict.shape[0] == target.shape[0], "BinaryDiceLoss: predict & target batch size don't match"

        if target.unique().tolist() != [0, 1]:
            raise ValueError(f"BinaryDiceLoss: target expected in [0, 1] range but got {target.unique()}")

        return 1 - self.dice(predict, target)
