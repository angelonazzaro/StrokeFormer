from functools import partial
from typing import Optional, Literal, Union

import numpy as np
import torch
from torchmetrics.functional import accuracy, recall, f1_score, precision, jaccard_index, matthews_corrcoef
from torchmetrics.functional.segmentation import dice_score
from monai.metrics import compute_hausdorff_distance


def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: dict,
        prefix: Optional[Literal["train", "val", "test"]] = None):
    assert 4 <= predictions.ndim <= 6, \
        "predictions and targets must have shape (B, N, H, W), (B, N, D, H, W) or (B, C, N, D, H, W)"
    assert predictions.shape == targets.shape, "predictions and targets must have same shape"

    prefix = f"{prefix}_" if prefix is not None else ""
    scores = {f"{prefix}{m}": 0.0 if m != "hausdorff_distance" else float("inf") for m in metrics.keys()}

    argmax_dim = 1 if predictions.ndim < 6 else 2

    # (B, H, W), (B, D, H, W), (B, C, D, H, W)
    indexed_predictions = predictions.argmax(argmax_dim)
    indexed_targets = targets.argmax(argmax_dim)

    if indexed_targets.ndim == 3:
        dims = (1, 2)
    elif indexed_targets.ndim == 4:
        dims = (0, 2, 3)
    else:
        dims = (0, 1, 3, 4)

    slices_with_lesions = indexed_targets.any(dims)

    if not slices_with_lesions.any():
        return scores

    if indexed_predictions.ndim == 3:
        indexed_predictions = indexed_predictions[slices_with_lesions]
        indexed_targets = indexed_targets[slices_with_lesions]
        predictions = predictions[:, slices_with_lesions]
        targets = targets[:, slices_with_lesions]
    elif indexed_predictions.ndim == 4:
        indexed_predictions = indexed_predictions[:, slices_with_lesions]
        indexed_targets = indexed_targets[:, slices_with_lesions]
        predictions = predictions[:, :, slices_with_lesions]
        targets = targets[:, :, slices_with_lesions]
    else:
        indexed_predictions = indexed_predictions[:, :, slices_with_lesions]
        indexed_targets = indexed_targets[:, :, slices_with_lesions]
        predictions = predictions[:, :, :, slices_with_lesions]
        targets = targets[:, :, :, slices_with_lesions]

    for metric_name, metric_fn in metrics.items():
        if metric_name == "dice" or metric_name == "hd95":
            score = metric_fn(predictions, targets)
            if torch.isnan(score).any() or torch.isinf(score).any():
                # replace invalid scores with zero for dice and inf for hd95
                score = torch.nan_to_num(score, nan=0.0, posinf=float('inf'), neginf=0.0)
            if metric_name == "hd95":
                score = score.mean()
        else:
            score = metric_fn(indexed_predictions, indexed_targets)

        scores[f"{prefix}{metric_name}"] = score.item()

    return scores



def build_metrics(num_classes: int, average: Literal["micro", "macro", "weighted", "none"] = "macro"):
    task = "binary" if num_classes <= 2 else "multiclass"

    return {
        "accuracy": partial(
            accuracy,
            task=task,
            num_classes=num_classes,
            average=average,
            ignore_index=None
        ),
        "precision": partial(
            precision,
            task=task,
            num_classes=num_classes,
            average=average,
            ignore_index=None
        ),
        "recall": partial(
            recall,
            task=task,
            num_classes=num_classes,
            ignore_index=None,
            average=average
        ),
        "f1": partial(
            f1_score,
            task=task,
            num_classes=num_classes,
            average=average,
            ignore_index=None
        ),
        "iou": partial(
            jaccard_index,
            task=task,
            num_classes=num_classes,
            average=average,
            ignore_index=None,
        ),
        "dice": partial(
            dice_score,
            num_classes=num_classes,
            average=average,
            include_background=False,
            aggregation_level="global",
        ),
        "mcc": partial(
            matthews_corrcoef,
            task=task,
            num_classes=num_classes,
            ignore_index=None
        ),
        "hd95": partial(
            compute_hausdorff_distance,
            percentile=95,
            include_background=False,
            spacing=1
        )
    }


def slice_wise_fp_fn(prediction: Union[torch.Tensor, np.ndarray],
                     target: Union[torch.Tensor, np.ndarray]):
    """
    Compute normalized true positives, false positives and false negatives for a single brain slice.

    Args:
        prediction: 2D binary mask (H, W)
        target: 2D binary ground truth mask (H, W)
    Returns:
        dict with normalized 'tp', 'fp', 'fn'
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    assert prediction.shape == target.shape and prediction.ndim == 2, \
        "Prediction and target must have 2D shape of equal dimensions (single slice)."

    # ensure binary masks
    prediction = (prediction > 0.5).int()
    target = (target > 0.5).int()

    prediction_flat = prediction.flatten()
    target_flat = target.flatten()

    total_voxels = target_flat.numel()
    lesion_size = target_flat.sum()

    tp = (prediction_flat * target_flat).sum()
    fp = (prediction_flat * (1 - target_flat)).sum()
    fn = ((1 - prediction_flat) * target_flat).sum()

    fp_norm = fp.float() / total_voxels
    fn_norm = fn.float() / total_voxels
    tp_norm = tp.float() / lesion_size if lesion_size > 0 else torch.tensor(0.0)

    return {"tp": tp_norm.item(), "fp": fp_norm.item(), "fn": fn_norm.item()}
