import random
from collections import defaultdict
from functools import partial
from typing import Optional, Literal, Union

import numpy as np
import torch
from distorch import boundary_metrics, overlap_metrics
from torchmetrics.functional import recall, f1_score, precision, jaccard_index, matthews_corrcoef, auroc, roc, \
    structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr
from torchvision.transforms.v2.functional import to_pil_image

from utils import overlay_img, get_lesion_size_category, get_slices_with_lesions


def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: Optional[dict] = None,
        prefix: Optional[Literal["train", "val", "test"]] = None,
        lesions_only: bool = False,
        task: Literal["segmentation", "reconstruction"] = "segmentation",
):
    assert 4 <= predictions.ndim <= 6, "predictions and targets must have shape (B, N, H, W), (B, N, D, H, W), or (B, C, N, D, H, W)"
    assert predictions.shape == targets.shape, "predictions and targets must have the same shape"

    prefix = f"{prefix}_" if prefix is not None else ""

    # (B, H, W), (B, D, H, W), (B, C, D, H, W)
    argmax_dim = 1 if predictions.ndim < 6 else 2

    if metrics is None:
        metrics = build_metrics(num_classes=predictions.shape[argmax_dim])

    index_predictions = predictions.argmax(dim=argmax_dim)
    index_targets = targets.argmax(dim=argmax_dim)

    slices_with_lesions = get_slices_with_lesions(targets)

    if index_predictions.ndim == 3:
        # boundary metrics must be always computed on valid sets
        bo_index_predictions = index_predictions[slices_with_lesions]
        bo_index_targets = index_targets[slices_with_lesions]
    elif index_predictions.ndim == 4:
        bo_index_predictions = index_predictions[:, slices_with_lesions]
        bo_index_targets = index_targets[:, slices_with_lesions]
    else:
        bo_index_predictions = index_predictions[:, :, slices_with_lesions]
        bo_index_targets = index_targets[:, :, slices_with_lesions]

    if lesions_only:
        index_predictions = bo_index_predictions
        index_targets = bo_index_targets

    if task == "reconstruction":
        mse = predictions[:len(predictions) // 2]
        recons = predictions[mse.shape[0]:]

        masks = targets[:len(targets) // 2]
        scans = targets[masks.shape[0]:]

    scores = defaultdict(lambda: 0.0)

    for metric_name, metric_fn in metrics.items():
        if metric_name == 'boundary_metrics':
            if bo_index_predictions.numel() == 0 or bo_index_targets.numel() == 0:
                scores[f"{prefix}hausdorff95_1_to_2"] = float("inf")
                scores[f"{prefix}hausdorff95_2_to_1"] = float("inf")
                scores[f"{prefix}hausdorff95"] = float("inf")
                scores[f"{prefix}assd"] = float("inf")
                continue

            bo_metrics = metric_fn(bo_index_predictions, bo_index_targets)
            for bo_metric, value in bo_metrics.__dict__.items():
                value = torch.nan_to_num(value, 0).mean(dim=0)
                if bo_metric in ["Hausdorff95_1_to_2", "Hausdorff95_2_to_1", "AverageSymmetricSurfaceDistance"]:
                    name = bo_metric.lower() if bo_metric != "AverageSymmetricSurfaceDistance" else "assd"
                    scores[f"{prefix}{name}"] = round(value.item(), 3)
                    if "Hausdorff95" in bo_metric:
                        scores[f"{prefix}hausdorff95"] += value.item()
            scores[f"{prefix}hausdorff95"] = round(scores[f"{prefix}hausdorff95"] / 2, 3)

        elif metric_name == 'overlap_metrics':
            ov_metrics = metric_fn(index_predictions, index_targets)
            for ov_metric, value in ov_metrics.__dict__.items():
                value = torch.nan_to_num(value, 0).mean(dim=0) if ov_metric != "ConfusionMatrix" else value
                if ov_metric == "PixelAccuracy":
                    scores[f"{prefix}accuracy_background"] = round(value[0].item(), 3)
                    scores[f"{prefix}accuracy_foreground"] = round(value[1].item(), 3)
                elif ov_metric == "OverallPixelAccuracy":
                    scores[f"{prefix}accuracy"] = round(value.item(), 3)
                elif ov_metric == "Dice":
                    scores[f"{prefix}dice"] = round(value[1].item(), 3)
        else:
            if metric_name == "roc":
                fpr, _, _ = metric_fn(index_predictions, index_targets)
                scores[f"{prefix}fpr"] = round(fpr.mean().item(), 3)
            elif metric_name == "ssim" and task == "reconstruction":
                scores[f"{prefix}fpr"] = round(metric_fn(recons, scans).item(), 3)
            else:
                scores[f"{prefix}{metric_name}"] = round(metric_fn(index_predictions, index_targets.long() if metric_name == "auroc" else index_targets).item(), 3)

    return scores

def build_metrics(num_classes: int = 2, task: Literal["segmentation", "reconstruction"] = "segmentation",
                  average: Literal["micro", "macro", "weighted", "none"] = "macro"):
    task_type = "binary" if num_classes <= 2 else "multiclass"

    metrics = {
        "overlap_metrics": partial(overlap_metrics, num_classes=num_classes),
        "iou": partial(
            jaccard_index,
            task=task_type,
            num_classes=num_classes,
            average=average,
            ignore_index=None,
        ),
        "mcc": partial(
            matthews_corrcoef,
            task=task_type,
            num_classes=num_classes,
            ignore_index=None
        ),
        "precision": partial(
            precision,
            task=task_type,
            num_classes=num_classes,
            average=average,
            ignore_index=None
        ),
        "recall": partial(
            recall,
            task=task_type,
            num_classes=num_classes,
            ignore_index=None,
            average=average
        ),
        "f1": partial(
            f1_score,
            task=task_type,
            num_classes=num_classes,
            average=average,
            ignore_index=None
        ),
    }

    if task == "segmentation":
        metrics["boundary_metrics"] = partial(boundary_metrics)
    elif task == "reconstruction":
        metrics["auroc"] = partial(auroc, num_classes=num_classes, task=task_type, average=average)
        metrics["ssim"] = partial(ssim)
        metrics["psnr"] = partial(psnr)
        metrics["roc"] = partial(roc, num_classes=num_classes, task_type=task_type, average=average)

    metrics = {m: metrics[m] for m in sorted(metrics.keys())}

    return metrics


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


def predictions_generator(model, scans, masks, metrics: dict[partial], slices_per_scan: Optional[int] = None, task: Literal["segmentation", "reconstruction"] = "segmentation"):
    with torch.no_grad():
        if task == "segmentation":
            preds = model(scans, return_preds=True)
        else:
            preds = model(scans, run_backwards=True)

    preds = (preds >= 0.5).float()

    # randomly sample slices_per_scan
    if slices_per_scan is not None and slices_per_scan < scans.shape[-3]:
        random_slices = random.choices(np.arange(masks.shape[-3]), k=slices_per_scan)
        masks = masks[:, :, random_slices]
        scans = scans[:, :, random_slices]
        preds = preds[:, :, random_slices]

    for i in range(scans.shape[0]):
        for slice_idx in range(scans[i].shape[-3]):
            scan_slice = scans[i][0, slice_idx, ...]
            mask_slice = masks[i][:, slice_idx, ...]
            predicted_slice = preds[i][:, slice_idx, ...]

            lesion_size = get_lesion_size_category(mask_slice)

            scores = compute_metrics(predicted_slice.unsqueeze(0), mask_slice.unsqueeze(0), metrics, task=task)

            mask_slice = torch.argmax(mask_slice, dim=0).to(dtype=torch.uint8)  # shape: (H, W)
            predicted_slice = torch.argmax(predicted_slice, dim=0).to(dtype=torch.uint8)  # shape: (H, W)

            # normalize scan as RGB conversion requires [0,1] range
            scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min())

            scan_slice = np.asarray(to_pil_image(scan_slice).convert("RGB"))
            mask_slice = np.asarray(to_pil_image(mask_slice).convert("RGB"))

            predicted_slice = np.asarray(to_pil_image(predicted_slice).convert("RGB"))

            gt = overlay_img(scan_slice, mask_slice, color=(0, 255, 0))
            pd = overlay_img(scan_slice, predicted_slice, color=(255, 0, 0))

            results = {
                "scan_slice": scan_slice,
                "mask_slice": mask_slice,
                "slice_idx": slice_idx,
                "lesion_size": lesion_size,
                "predicted_slice": predicted_slice,
                "gt": gt,
                "pd": pd,
                "scores": scores,
            }

            yield results