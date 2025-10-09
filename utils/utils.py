import math
import random
from functools import partial
from typing import Optional, Tuple, List, Union, Literal

import matplotlib.pyplot as plt
import monai.metrics.metric
import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import label
from torchmetrics.functional import accuracy, recall, f1_score, auroc, precision
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import tqdm

from constants import LESION_SIZES


def get_lesion_distribution_metadata(masks_filepaths: List[str], labels: List[str] = LESION_SIZES,
                                     return_masks: bool = False):
    metadata = {
        "patients_count": 0,
        "lesions_per_patient": [],
        "tot_voxels": 0,
        "lesion_voxels": 0,
        "tot_slices": 0,
        "slices_without_lesions": 0,
        "slices_with_lesions": 0,
    }

    for size in labels:
        metadata[size] = {
            "lesion_area": [],
            "count": 0,
            "filepaths": []
        }

    masks = [nib.load(filepath).get_fdata() if filepath.endswith(".nii.gz") else np.load(filepath) for filepath in
             tqdm(masks_filepaths, desc="Loading masks")]

    for i, mask in tqdm(enumerate(masks), desc="Getting lesion distribution metadata"):
        tot_voxels = mask.size

        slices_with_lesions = mask.any(axis=(0, 1))

        metadata["patients_count"] += 1
        metadata["lesions_per_patient"].append(label(mask)[-1])
        metadata["tot_voxels"] += tot_voxels
        metadata["slices_without_lesions"] += np.sum(~slices_with_lesions)
        metadata["slices_with_lesions"] += np.sum(slices_with_lesions)

        for slice_idx in range(mask.shape[-1]):
            category, lesion_voxels, lesion_area = get_lesion_size_category(mask[..., slice_idx], return_all=True)
            metadata["lesion_voxels"] += lesion_voxels
            metadata[category]["lesion_area"].append(lesion_area)
            metadata[category]["count"] += 1
            metadata[category]["filepaths"].append((masks_filepaths[i], slice_idx))

    if return_masks:
        return metadata, masks

    return metadata


def plot_lesion_size_distribution(counts, labels, figsize=(8, 6), title=None, return_distribution=False):
    assert len(counts) == len(labels), "counts and labels must have same length"

    items = zip(counts, labels)
    sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
    counts, labels = zip(*sorted_items)

    colors = plt.cm.tab20.colors[:len(labels)]
    total = np.sum(counts)

    plt.figure(figsize=figsize)
    plt.bar(labels, counts, color=colors)
    distributions = {}

    for i, v in enumerate(counts):
        p = v * 100 / total
        distributions[labels[i]] = p
        plt.text(i, v + 7.5, f"{p:.2f}%", ha="center")

    plt.ylabel("Slices Without Lesions")
    plt.xlabel("Lesion Size Categories")
    plt.xticks(rotation=45)
    if title is None:
        title = "Lesion Size Distribution Across Slices"
    plt.title(title)
    plt.show()

    if return_distribution:
        return distributions
    return None


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SemanticSegmentationTarget:
    def __init__(self, mask):
        if isinstance(mask, torch.Tensor):
            self.mask = mask
        else:
            self.mask = torch.from_numpy(mask)

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output * self.mask).sum()


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, metrics: dict[partial], prefix: Optional[Literal['train', 'val', 'test']] = None):
    scores = {}
    prefix = f"{prefix}_" if prefix is not None else ""

    for name, metric_fn in metrics.items():
        is_cumulative = isinstance(metric_fn, monai.metrics.metric.Cumulative)

        if is_cumulative:
            # Hausdorff distance needs at least 4D: (B, C, H, W)
            # other MONAI metrics need at least 3D: (C, H, W)
            min_dim = 4 if name == "hausdorff_distance" else 3
            if predictions.ndim < min_dim:
                # prepend singleton dimensions to reach required rank
                pad_dims = (min_dim - predictions.ndim)
                new_shape = (1,) * pad_dims + tuple(predictions.shape)
                preds = predictions.view(new_shape)
                targs = targets.view(new_shape)
            else:
                preds, targs = predictions, targets

            metric_fn(preds, targs)
            raw_score = metric_fn.aggregate().mean()
        else:
            if name == "auroc":
                targs = targets.long()
            else:
                targs = targets

            raw_score = metric_fn(predictions, targs)

        scores[f"{prefix}{name}"] = float(raw_score)

    return scores


def reset_metrics(metrics):
    for name, metric_fn in metrics.items():
        if isinstance(metric_fn, monai.metrics.metric.Cumulative):
            metric_fn.reset()

def predictions_generator(model, scans, masks, metrics: dict[partial], slices_per_scan: Optional[int] = None, cam_model=None):
    with torch.no_grad():
        preds = model(scans, return_preds=True)

    preds = (preds >= 0.5).float()
    grayscale_cam = torch.zeros_like(preds[0])

    if cam_model is not None:
        grayscale_cam = cam_model(scans, [SemanticSegmentationTarget(mask) for mask in masks],
                                  eigen_smooth=False)  # noqa

    # randomly sample slices_per_scan
    if slices_per_scan is not None and slices_per_scan < scans.shape[-3]:
        random_slices = random.choices(np.arange(masks.shape[-3]), k=slices_per_scan)
        masks = masks[:, :, random_slices]
        scans = scans[:, :, random_slices]
        preds = preds[:, :, random_slices]
        grayscale_cam = grayscale_cam[:, random_slices]

    for i in range(scans.shape[0]):
        for slice_idx in range(scans[i].shape[-3]):
            scan_slice = scans[i][0, slice_idx, ...]
            mask_slice = masks[i][0, slice_idx, ...]
            predicted_slice = preds[i][0, slice_idx, ...]

            lesion_size = get_lesion_size_category(mask_slice)

            scores = compute_metrics(predicted_slice, mask_slice, metrics)

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

            if cam_model is not None:
                cam_image = show_cam_on_image(scan_slice / 255, grayscale_cam[i, slice_idx], use_rgb=True)
                results["cam_image"] = cam_image

            yield results


def get_lesion_size_category(mask: Union[torch.Tensor, np.ndarray], return_all: bool = False):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    slice_area = mask.size
    lesion_voxels = np.sum(mask)
    lesion_area = lesion_voxels / slice_area

    if lesion_voxels == 0:
        size_category = "No Lesion"
    elif lesion_area <= 0.01:
        size_category = "Small"
    elif lesion_area <= 0.05:
        size_category = "Medium"
    else:
        size_category = "Large"

    if return_all:
        return size_category, lesion_voxels, lesion_area

    return size_category


def lesion_wise_fp_fn(prediction: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]):
    """
    Computes lesion-wise false positives and false negatives.

    Args:
        prediction: Ground truth binary mask (C x D x H x W)
        target: Predicted binary mask (C x D H x W)

    Returns:
        dict with lesion-wise TP, FP, FN counts and derived rates
    """
    # Label connected components
    gt_labeled, gt_count = label(target, structure=np.ones((3,) * target.ndim))
    pred_labeled, pred_count = label(prediction, structure=np.ones((3,) * target.ndim))

    tp = 0
    matched_gt = set()

    for pred_id in range(1, pred_count + 1):
        pred_component = (pred_labeled == pred_id)
        overlap_ids = np.unique(gt_labeled[pred_component])
        overlap_ids = overlap_ids[overlap_ids != 0]

        if len(overlap_ids) > 0:
            tp += 1
            matched_gt.update(overlap_ids)

    fn = gt_count - len(matched_gt)
    fp = pred_count - tp

    return {
        "lesion_TP": tp,
        "lesion_FP": fp,
        "lesion_FN": fn,
        "lesion_sensitivity": tp / (tp + fn + 1e-8),
        "lesion_precision": tp / (tp + fp + 1e-8),
    }


def build_metrics(num_classes: int, average: Literal["micro", "macro", "weighted", "none"] = "micro"):
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
        "auroc": partial(
            auroc,
            task=task,
            num_classes=num_classes,
            average=average,
            ignore_index=None
        ),
        "iou": MeanIoU(get_not_nans=False),
        "dice": DiceMetric(num_classes=num_classes, get_not_nans=False),
        "hausdorff_distance": HausdorffDistanceMetric(
            get_not_nans=False,
            percentile=95,
            distance_metric='euclidean',
        )
    }


def overlay_img(scan: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5):
    assert (scan.ndim == 3 and mask.ndim == 3
            and mask.shape[-1] == 3 and scan.shape[-1] == 3), "image and overlay must be RGB images"

    scan = scan.copy()
    mask = mask.astype(bool).copy()

    overlay = np.zeros_like(scan)
    overlay[:, :] = color

    scan[mask] = (1 - alpha) * scan[mask] + alpha * overlay[mask]
    return scan.astype(np.uint8)


def round_half_up(x):
    return math.floor(x + 0.5)


def check_patch_dim(patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]],
                    scan_dim: Tuple[Optional[int], int, int, int]):
    D, H, W = scan_dim[-3], scan_dim[-2], scan_dim[-1]

    patch_D, patch_H, patch_W = patch_dim or (None, None, None)
    patch_D = patch_D or D
    patch_H = patch_H or H
    patch_W = patch_W or W

    return patch_D, patch_H, patch_W


def extract_patches(scan: Union[torch.Tensor, np.ndarray],
                    overlap: Optional[Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]] = None,
                    patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
                    return_origins: bool = False):
    patches = []
    origins = []
    patch_D, patch_H, patch_W = check_patch_dim(patch_dim, scan.shape)

    if isinstance(overlap, float):
        overlap = (overlap, overlap, overlap)
    else:
        tmp = [ov if ov is not None else 1.0 for ov in overlap]
        overlap = tuple(tmp)

    stride_D = round_half_up(patch_D * overlap[-3]) if overlap is not None else patch_D
    stride_H = round_half_up(patch_H * overlap[-2]) if overlap is not None else patch_H
    stride_W = round_half_up(patch_W * overlap[-1]) if overlap is not None else patch_W

    H_dim = scan.shape[-2]
    W_dim = scan.shape[-1]
    D_dim = scan.shape[-3]

    patch_shape = (patch_D, patch_H, patch_W)

    if scan.ndim == 4:
        patch_shape = (1, *patch_shape)

    for y in range(0, H_dim, stride_H):
        for x in range(0, W_dim, stride_W):
            for z in range(0, D_dim, stride_D):
                if scan.ndim == 4:
                    patch = scan[..., z:z + patch_D, y:y + patch_H, x:x + patch_W]
                else:
                    patch = scan[z:z + patch_D, y:y + patch_H, x:x + patch_W]

                if patch.shape != patch_shape:
                    pad_depth = patch_D - patch.shape[-3]
                    pad_height = patch_H - patch.shape[-2]
                    pad_width = patch_W - patch.shape[-1]

                    if isinstance(patch, torch.Tensor):
                        # torch expects padding in the order: (W_left, W_right, H_left, H_right, D_left, D_right)
                        padding = (0, pad_width, 0, pad_height, 0, pad_depth)

                        if scan.ndim == 4:
                            padding = (*padding, 0, 0)

                        patch = torch.nn.functional.pad(patch, padding, mode="constant")
                    else:
                        # numPy expects ((D_before, D_after), (H_before, H_after), (W_before, W_after))
                        pad_width_np = ((0, pad_depth), (0, pad_height), (0, pad_width))
                        if scan.ndim == 4:
                            pad_width_np = ((0, 0), *pad_width_np)

                        patch = np.pad(patch, pad_width_np, mode="constant")

                patches.append(patch)
                origins.append((z, y, x))

    if return_origins:
        return patches, origins
    return patches


def reconstruct_volume(
        patches: Union[torch.Tensor, List[np.ndarray]],
        scan_dim: Tuple[Optional[int], int, int, int],
        origins: List[Tuple[int, int, int]],
        to_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(patches, torch.Tensor) and patches.ndim == 6:
        # multi-scan case (list of lists)
        volumes = [
            reconstruct_volume(p, scan_dim, o, to_tensor=to_tensor)
            for p, o in zip(patches, origins)
        ]

        return torch.stack(volumes, dim=0) if isinstance(volumes[0], torch.Tensor) else np.stack(volumes, axis=0)

    D, H, W = scan_dim[-3], scan_dim[-2], scan_dim[-1]

    patch_D, patch_H, patch_W = patches[0].shape[-3], patches[0].shape[-2], patches[0].shape[-1]

    is_numpy = isinstance(patches[0], np.ndarray)

    if is_numpy:
        recon_volume = np.zeros(scan_dim, dtype=np.float32)
        count_volume = np.zeros(scan_dim, dtype=np.float32)  # keep track of overlaps
    else:
        recon_volume = torch.zeros(scan_dim, dtype=torch.float32, device=patches[0].device)
        count_volume = torch.zeros(scan_dim, dtype=torch.float32, device=patches[0].device)

    for patch, (z, y, x) in zip(patches, origins):
        z_end = min(z + patch_D, D)
        y_end = min(y + patch_H, H)
        x_end = min(x + patch_W, W)

        if patch.ndim == 4:
            patch_cropped = patch[..., :z_end - z, :y_end - y, :x_end - x]
        else:
            patch_cropped = patch[:z_end - z, :y_end - y]

        if patch.ndim == 4:
            recon_volume[..., z:z_end, y:y_end, x:x_end] += patch_cropped
            count_volume[..., z:z_end, y:y_end, x:x_end] += 1
        else:
            recon_volume[z:z_end, y:y_end, x:x_end] += patch_cropped
            count_volume[z:z_end, y:y_end, x:x_end] += 1

    # normalize overlaps
    if is_numpy:
        recon_volume /= np.maximum(count_volume, 1)
    else:
        recon_volume /= torch.clamp(count_volume, min=1)

    return torch.from_numpy(recon_volume) if to_tensor and is_numpy else recon_volume
