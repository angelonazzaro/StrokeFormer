from functools import partial
from typing import Literal, Optional
from collections import OrderedDict

import torch
from torch import Tensor
from torchmetrics.functional import accuracy, precision, recall, f1_score, peak_signal_noise_ratio as psnr, jaccard_index, matthews_corrcoef as mcc
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.segmentation import dice_score

from distorch import pixel_center_metrics

from constants import HEAD_MASK_THRESHOLD
from utils import get_lesion_size, generate_overlayed_slice, compute_head_mask, filter_sick_slices_per_volume


def compute_metrics(preds: Tensor,
                    targets: Tensor,
                    metrics: dict,
                    prefix: Optional[Literal["train", "val"]] = None,
                    task: Literal["segmentation", "region_proposal"] = "segmentation") -> dict:
    prefix = f"{prefix}_" if prefix else ""

    scores = {f"{prefix}{metric_name}": metric["default_value"] for metric_name, metric in metrics.items()}

    if task == "segmentation":
        preds = preds.to(dtype=torch.long)
        targets = targets.to(dtype=torch.long)

        preds, targets = filter_sick_slices_per_volume(preds, targets, "index")

    if preds.shape[0] > 0:
        for metric_name, metric in metrics.items():
            score = torch.nan_to_num(metric["fn"](preds, targets), nan=metric["default_value"]).item()
            if metric["default_value"] == float("inf") and score != float("inf"):
                scores[prefix + metric_name] = score
            else:
                scores[prefix + metric_name] = score

    return scores


def build_metrics(num_classes: Optional[int] = None,
                  average: Literal["micro", "macro", "weighted", "none"] = "macro",
                  task: Literal["segmentation", "region_proposal"] = "segmentation",
                  inference: bool = False,
                  lesions_only: bool = True):
    metrics = {}

    if task == "segmentation":
        if num_classes is None:
            raise ValueError("`num_classes` must be specified if `task` is `segmentation`.")
        task_type = "binary" if num_classes <= 2 else "multiclass"
        task_specific_metrics = {
            "dice": {
                "fn": partial(dice_score,
                              include_background=False,
                              num_classes=num_classes,
                              average=average,
                              aggregation_level="global",
                              input_format="index"),
                "default_value": 0.0
            },
            "iou": {
                "fn": partial(jaccard_index,
                              num_classes=num_classes,
                              task=task_type,
                              ignore_index=0 if lesions_only else None,  # background is not included in metric computation
                              average=average),
                "default_value": 0.0,
            },
            "mcc": {
                "fn": partial(mcc,
                              num_classes=num_classes,
                              task=task_type,
                              ignore_index=0 if lesions_only else None),
                "default_value": 0.0,
            },
            "accuracy": {
                "fn": partial(accuracy,
                              num_classes=num_classes,
                              task=task_type,
                              ignore_index=0 if lesions_only else None,
                              average=average),
                "default_value": 0.0,
            },
            "f1": {
                "fn": partial(f1_score,
                              num_classes=num_classes,
                              task=task_type,
                              ignore_index=0 if lesions_only else None,
                              average=average),
                "default_value": 0.0,
            },
            "precision": {
                "fn": partial(precision,
                              num_classes=num_classes,
                              task=task_type,
                              ignore_index=0 if lesions_only else None,
                              average=average),
                "default_value": 0.0,
            },
            "recall": {
                "fn": partial(recall,
                              num_classes=num_classes,
                              task=task_type,
                              ignore_index=0 if lesions_only else None,
                              average=average),
                "default_value": 0.0,
            },
        }

        if inference:
            dummy_result = pixel_center_metrics(torch.ones(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1))
            metric_names = list(dummy_result.__dataclass_fields__.keys())

            def center_metrics_fn(pred, target):
                """Compute all pixel-center metrics once and cache them."""
                results = pixel_center_metrics(pred, target)
                return results

            _cache = {"results": None, "pred_id": None, "target_id": None}

            def get_center_metric(name):
                """Return a callable that extracts one metric but uses cached computation."""
                def metric_fn(pred, target):
                    key = (id(pred), id(target))
                    if _cache["results"] is None or _cache["pred_id"] != key[0] or _cache["target_id"] != key[1]:
                        _cache["results"] = center_metrics_fn(pred, target)
                        _cache["pred_id"], _cache["target_id"] = key
                    return getattr(_cache["results"], name)

                return metric_fn

            for metric_name in metric_names:
                task_specific_metrics[metric_name.lower()] = {
                    "fn": get_center_metric(metric_name),
                    "default_value": float("inf"),
                }
    else:
        task_specific_metrics = {
            "ssim": {
                "fn": partial(ssim),
                "default_value": 0.0
            },
            "psnr": {
                "fn": partial(psnr),
                "default_value": 0.0
            }
        }

    metrics.update(task_specific_metrics)
    metrics = OrderedDict(sorted(metrics.items(), key=lambda kv: kv[0]))

    return metrics


def get_per_slice_segmentation_preds(model,
                                     scans: Tensor,
                                     masks: Tensor,
                                     metrics: dict,
                                     means: Optional[Tensor] = None,
                                     stds: Optional[Tensor] = None,
                                     mins_maxs: Optional[Tensor] = None,):
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
            mins_maxs: Tensor containing the standardize min and max values needed to adjust the threshold for the head mask
    """

    with torch.no_grad():
        preds = model(scans, return_preds=True)

    for i in range(scans.shape[0]):
        if means is not None and stds is not None:
            adjusted_threshold = (HEAD_MASK_THRESHOLD - means[i]) / stds[i]

            if mins_maxs is not None:
                z_min, z_max = mins_maxs[i]
                adjusted_threshold = (adjusted_threshold - z_min) / (z_max - z_min)

            head_mask = compute_head_mask(scans[i], threshold=adjusted_threshold)
        else:
            head_mask = compute_head_mask(scans[i])

        for slice_idx in range(scans.shape[-3]):
            head_mask_slice = head_mask[:, slice_idx]  # C, H, W
            scan_slice = scans[i][:, slice_idx]  # C, H, W
            mask_slice = masks[i][:, slice_idx]  # C, H, W
            pred_slice = preds[i][:, slice_idx]  # C, H, W

            lesion_size_str = get_lesion_size(head_mask_slice, mask_slice)
            scores = compute_metrics(pred_slice.unsqueeze(0), mask_slice.unsqueeze(0), metrics)

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