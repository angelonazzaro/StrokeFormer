from collections import defaultdict
from functools import partial
from typing import Literal, Optional

import torch
from torch import Tensor
from torchmetrics.functional.segmentation import dice_score

from constants import HEAD_MASK_THRESHOLD
from utils import get_lesion_size, generate_overlayed_slice, compute_head_mask


def compute_metrics(preds: Tensor,
                    targets: Tensor,
                    metrics: dict,
                    prefix: Optional[Literal["train", "val"]] = None) -> dict:
    prefix = f"{prefix}_" if prefix else ""

    scores = defaultdict(float)
    preds = preds.to(dtype=torch.long)
    targets = targets.to(dtype=torch.long)

    for metric_name, metric_fn in metrics.items():
        scores[prefix + metric_name] = torch.nan_to_num(metric_fn(preds, targets), nan=0.0).item()

    return scores


def build_metrics(num_classes: int,
                  average: Literal["micro", "macro", "weighted", "none"] = "macro"):
    return {
        "dice": partial(dice_score,
                        include_background=False,
                        num_classes=num_classes,
                        average=average,
                        aggregation_level="global",
                        input_format="index")
    }


def get_per_slice_segmentation_preds(model,
                                     scans: Tensor,
                                     masks: Tensor,
                                     metrics: dict,
                                     means: Optional[Tensor] = None,
                                     stds: Optional[Tensor] = None):
    """
        It is an iterator function that computes predictions and scores per single slice.
        Returns a dictionary containing:
            - current slice idx
            - current scan idx
            - current ground truth
            - current predicted lesion
            - current lesion size
            - metrics scores for the current slice

        Args:
            model: Segmentation model
            scans: Tensor of shape (B, C, D, H, W)
            masks: Tensor of shape (B, C, D, H, W)
            metrics: Metrics to compute
            means: Tensor containing the original means needed to adjust the threshold for the head mask
            stds: Tensor containing the original stds needed to adjust the threshold for the head mask
    """

    with torch.no_grad():
        preds = model(scans, return_preds=True)

    for i in range(scans.shape[0]):
        if means is not None and stds is not None:
            adjusted_threshold = (HEAD_MASK_THRESHOLD - means[i]) / stds[i]
            head_mask = compute_head_mask(scans[i], threshold=adjusted_threshold)
        else:
            head_mask = compute_head_mask(scans[i])

        for slice_idx in range(scans.shape[-3]):
            head_mask_slice = head_mask[:, slice_idx] # C, D, H, W
            scan_slice = scans[i][:, slice_idx]  # C, D, H, W
            mask_slice = masks[i][:, slice_idx]  # C, D, H, W
            pred_slice = preds[i][:, slice_idx]  # C, D, H, W

            lesion_size_str = get_lesion_size(head_mask_slice, mask_slice)
            scores = compute_metrics(pred_slice, mask_slice, metrics)

            ground_truth = generate_overlayed_slice(scan_slice, mask_slice, color=(0, 255, 0))
            prediction = generate_overlayed_slice(scan_slice, pred_slice, color=(255, 0, 0))

            yield {
                "slice_idx": slice_idx,
                "scan_idx": i,
                "lesion_size": lesion_size_str,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "scores": scores
            }