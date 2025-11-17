import math
from typing import Union, List, Optional, Literal

import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as f
from torch import Tensor
from torchvision.transforms.v2.functional import to_pil_image

from constants import HEAD_MASK_THRESHOLD


def to_3channel(img):
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def resize(scans, masks, new_h: int, new_w: int):
    # with align_corners = False and antialias = True this is equivalent to PIL downsample method
    # shape must be (B, D*C, H, W) as anti-alias is restricted to 4-D tensors
    B, C, D, H, W = scans.shape
    scans = scans.view(B, C * D, H, W)
    masks = masks.view(B, C * D, H, W)

    scans = f.interpolate(scans,
                          size=(new_h, new_w),
                          mode="bilinear",
                          align_corners=False,
                          antialias=True)

    masks = f.interpolate(masks.to(dtype=scans.dtype),
                          size=(new_h, new_w),
                          align_corners=False,
                          mode="bilinear",
                          antialias=True).long()

    # restore original shape of (B, C, D, resize_h, resize_w)
    scans = scans.view(B, C, D, new_h, new_w)
    masks = masks.view(B, C, D, new_h, new_w)

    return scans, masks


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def compute_head_mask(scan: Union[np.ndarray, Tensor],
                      threshold: float = HEAD_MASK_THRESHOLD) -> Union[np.ndarray, Tensor]:
    return scan > threshold


def round_half_up(num: float):
    # round to the nearest integer, by excess
    return math.floor(num + 0.5)


def load_volume(volume_path: str):
    if volume_path.endswith(".nii.gz"):
        volume = nib.load(volume_path).get_fdata()
    elif volume_path.endswith(".npy"):
        volume = np.load(volume_path)
    else:
        raise ValueError(f"Extension not supported: {volume_path}")

    return volume


def convert_to_rgb(scan_slice: Union[np.ndarray, Tensor]):
    if scan_slice.min() < 0 or scan_slice.max() > 1:
        # normalize between [0,1] for RGB conversion
        scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min())

    if scan_slice.dtype == np.int64 or scan_slice.dtype == torch.int64:
        scan_slice = scan_slice.float()

    rgb_slice = np.asarray(to_pil_image(scan_slice).convert("RGB"))
    return rgb_slice


def generate_overlayed_slice(scan_slice: Union[np.ndarray, Tensor],
                             mask_slice: Union[np.ndarray, Tensor],
                             color: tuple[int, int, int] = (255, 0, 0),
                             alpha: float = 0.5,
                             return_tensor: bool = False,
                             return_rgbs: bool = False):
    scan_slice = convert_to_rgb(scan_slice)
    mask_slice = convert_to_rgb(mask_slice)

    overlay = overlay_img(scan_slice, mask_slice, color=color, alpha=alpha)

    if return_tensor:
        overlay = torch.from_numpy(overlay)
        overlay = einops.rearrange(overlay, "h w c -> c h w")

    if return_rgbs:
        return overlay, scan_slice, mask_slice
    return overlay


def overlay_img(scan: np.ndarray,
                mask: np.ndarray,
                color: tuple[int, int, int] = (255, 0, 0),
                alpha: float = 0.5):
    assert scan.ndim == 3 and mask.ndim == 3, "scan and mask must have the same shape"
    assert scan.shape[-1] == 3 and mask.shape[-1] == 3, "scan and mask must be RGB images"

    scan = scan.copy()
    mask = mask.astype(bool).copy()

    overlay = np.zeros_like(scan)
    overlay[:, :] = color

    scan[mask] = (1 - alpha) * scan[mask] + alpha * overlay[mask]

    return scan.astype(np.uint8)


def filter_sick_slices_per_volume(
        scans: Tensor,
        masks: Tensor,
        input_format: Literal["one-hot", "index"] = "one-hot"
) -> tuple[Tensor, Tensor]:
    B = scans.shape[0]

    if input_format == "one-hot":
        # one-hot format: mask shape (B, N, C, D, H, W)
        # foreground assumed to be at channel index 1
        if masks.ndim == 6:
            fg_sick = masks[:, 1].any(dim=(1, -2, -1))  # (B, D)
        else:
            raise ValueError("One-hot input should have 6 dims (B, N, C, D, H, W)")
    elif input_format == "index":
        # index format: mask shape (B, D, H, W) or (B, 1, D, H, W)
        # foreground = non-zero voxels
        if masks.ndim == 5:
            fg_sick = masks[:, 0].bool().any(dim=(-2, -1))  # (B, D)
        elif masks.ndim == 4:
            fg_sick = masks.bool().any(dim=(-2, -1))  # (B, D)
        else:
            raise ValueError("Index input should have 4 or 5 dims")
    else:
        raise ValueError(f"Unknown input_format '{input_format}'. Use 'one-hot' or 'index'.")

    max_sick_slices = fg_sick.sum(dim=-1).max().item()
    if max_sick_slices == 0:
        return torch.empty((0, *scans.shape[1:]), device=scans.device), torch.empty((0, *masks.shape[1:]),
                                                                                    device=masks.device)

    preds = []
    tgts = []

    for b in range(B):
        sick_idx = fg_sick[b]

        if len(sick_idx) == 0:
            continue

        if scans.ndim == 6:  # (B, N, C, D, H, W)
            pred_slices = scans[b][:, :, sick_idx]
            tgt_slices = masks[b][:, :, sick_idx]
        elif scans.ndim == 5:  # (B, C, D, H, W)
            pred_slices = scans[b][:, sick_idx]
            tgt_slices = masks[b][:, sick_idx]
        else: # (B, D, H, W)
            pred_slices = scans[b][sick_idx]
            tgt_slices = masks[b][sick_idx ]

        current_slices = pred_slices.shape[-3]

        if current_slices == 0:
            continue

        if current_slices < max_sick_slices:
            pad_amount = max_sick_slices - current_slices
            pad_dims = pred_slices.ndim
            if pad_dims == 5:
                pad = [0, 0] * ((pad_dims - 3)) + [0, pad_amount] + [0, 0, 0, 0]
            elif pad_dims == 4:
                pad = [0, 0] * ((pad_dims - 2)) + [0, pad_amount] + [0, 0]
            else:
                pad = [0, 0] * ((pad_dims - 1)) + [0, pad_amount] + [0, 0]
            pred_slices = f.pad(pred_slices, pad, mode='constant', value=0)
            tgt_slices = f.pad(tgt_slices, pad, mode='constant', value=0)

        preds.append(pred_slices)
        tgts.append(tgt_slices)

    if len(preds) == 0:
        return torch.empty((0, *scans.shape[1:]), device=scans.device), torch.empty((0, *masks.shape[1:]), device=masks.device)

    return torch.stack(preds, dim=0), torch.stack(tgts, dim=0)


def get_lesion_size(scan: Union[np.ndarray, Tensor],
                    mask: Union[np.ndarray, Tensor],
                    threshold: float = HEAD_MASK_THRESHOLD,
                    get_head_mask: bool = False,
                    return_all: bool = False):
    if get_head_mask:
        scan = compute_head_mask(scan, threshold)  # consider only brain area in the scan

    if isinstance(scan, Tensor):
        scan = scan.cpu().numpy()

    brain_area = np.sum(scan)

    if isinstance(mask, Tensor):
        # assuming mask is an index tensor of shape (C, H, W, D) or similar
        mask = mask.cpu().numpy()

    lesion_size = np.sum(mask)
    lesion_area = np.nan_to_num(np.divide(lesion_size, brain_area), nan=0)

    if lesion_area == 0:
        lesion_size_str = "No Lesion"
    elif lesion_area <= 0.01:
        lesion_size_str = "Small"
    elif lesion_area <= 0.05:
        lesion_size_str = "Medium"
    else:
        lesion_size_str = "Large"

    if return_all:
        return lesion_size_str, lesion_size, lesion_area

    return lesion_size_str


def check_patch_dim(patch_dim: Optional[tuple[Optional[int], Optional[int], Optional[int]]],
                    scan_dim: tuple[Optional[int], int, int, int]):
    D, H, W = scan_dim[-3], scan_dim[-2], scan_dim[-1]

    patch_D, patch_H, patch_W = patch_dim or (None, None, None)
    patch_D = patch_D or D
    patch_H = patch_H or H
    patch_W = patch_W or W

    return patch_D, patch_H, patch_W


def extract_patches(scan: Union[Tensor, np.ndarray],
                    overlap: Optional[Union[float, tuple[Optional[float], Optional[float], Optional[float]]]] = None,
                    patch_dim: Optional[tuple[Optional[int], Optional[int], Optional[int]]] = None,
                    return_origins: bool = False):
    patches = []
    origins = []
    patch_D, patch_H, patch_W = check_patch_dim(patch_dim, scan.shape)

    if overlap is not None:
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

                    if isinstance(patch, Tensor):
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
        scan_dim: tuple[Optional[int], int, int, int],
        origins: List[tuple[int, int, int]],
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