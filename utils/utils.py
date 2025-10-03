import math
import random
from functools import partial
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
import wandb
from torchmetrics.functional import accuracy, recall, f1_score, fbeta_score, matthews_corrcoef
from torchvision.transforms.v2.functional import to_pil_image

from constants import CLASS_LABELS


# from monai.metrics import DiceMetric, MeanIoU, AveragePrecisionMetric


@torch.no_grad()
def predictions_generator(model, scans, masks, slices_per_scan: int, metrics: dict[partial]):
    preds = model(scans, return_preds=True)
    preds = (preds >= 0.5).float()

    # randomly sample slices_per_scan
    if slices_per_scan < scans.shape[-3]:
        random_slices = random.choices(np.arange(masks.shape[-3]), k=slices_per_scan)
        masks = masks[:, :, random_slices]
        scans = scans[:, :, random_slices]
        preds = preds[:, :, random_slices]
    else:
        masks = masks
        scans = scans

    for scan, predicted_mask, mask in zip(scans, preds, masks):
        for slice_idx in range(scan.shape[-3]):
            scan_slice = scan[0, slice_idx, ...]
            mask_slice = mask[0, slice_idx, ...]
            predicted_slice = predicted_mask[0, slice_idx, ...]

            scores = {metric: metrics[metric](predicted_slice, mask_slice).cpu().item() for metric in metrics}

            # normalize scan as RGB conversion requires [0,1] range
            scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min())

            scan_slice = np.asarray(to_pil_image(scan_slice).convert("RGB"))
            mask_slice = np.asarray(to_pil_image(mask_slice).convert("RGB"))
            predicted_slice = np.asarray(to_pil_image(predicted_slice).convert("RGB"))

            lesion_size = get_lesion_size_category(mask_slice)

            gt = overlay_img(scan_slice, mask_slice, color=(0, 255, 0))
            pd = overlay_img(scan_slice, predicted_slice, color=(255, 0, 0))

            # yield results for this slice
            yield {
                "scan_slice": scan_slice,
                "mask_slice": mask_slice,
                "slice_idx": slice_idx,
                "lesion_size": lesion_size,
                "predicted_slice": predicted_slice,
                "gt": gt,
                "pd": pd,
                "scores": scores,
            }


def get_lesion_size_category(mask, return_all: bool = False):
    slice_area = mask.size
    voxels = np.sum(mask)
    lesion_size = voxels / slice_area

    if voxels == 0:
        size_category = "No Lesion"
    elif lesion_size <= 0.01:
        size_category = "Small"
    elif lesion_size <= 0.05:
        size_category = "Medium"
    else:
        size_category = "Large"

    if return_all:
        return size_category, voxels, lesion_size

    return size_category


def wb_mask(bg_img, mask, prediction_mask, class_labels=CLASS_LABELS):
    return wandb.Image(bg_img, masks={
        "predictions": {"mask_data": prediction_mask, "class_labels": class_labels},
        "ground truth": {"mask_data": mask, "class_labels": class_labels}
    })


def build_metrics(num_classes: int):
    task = "binary" if num_classes <= 2 else "multiclass"
    return {
        "accuracy": partial(
            accuracy,
            task=task,
            num_classes=num_classes,
            average='macro',
            ignore_index=None
        ),
        # "precision": AveragePrecisionMetric(),
        "recall": partial(
            recall,
            task=task,
            num_classes=num_classes,
            ignore_index=None,
            average='macro'
        ),
        "f1": partial(
            f1_score,
            task=task,
            num_classes=num_classes,
            average='macro',
            ignore_index=None
        ),
        "f2": partial(
            fbeta_score,
            beta=2.0,
            task=task,
            num_classes=num_classes,
            average='macro',
            ignore_index=None
        ),
        # "iou": MeanIoU(),
        "mcc": partial(
            matthews_corrcoef,
            task=task,
            num_classes=num_classes,
            ignore_index=None  # semantic_loss_ignore_index
        ),
        # "dice": DiceMetric(num_classes=num_classes),
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
                    stride: Optional[float] = None,
                    patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
                    return_origins: bool = False):
    patches = []
    origins = []
    patch_D, patch_H, patch_W = check_patch_dim(patch_dim, scan.shape)

    stride_D = round_half_up(patch_D * stride) if stride is not None else patch_D
    stride_H = round_half_up(patch_H * stride) if stride is not None else patch_H
    stride_W = round_half_up(patch_W * stride) if stride is not None else patch_W

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