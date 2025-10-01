import math
from typing import Optional, Tuple, List, Union

import numpy as np
import torch


def round_half_up(x):
    return math.floor(x + 0.5)

def check_patch_dim(patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]], scan_dim: Tuple[int, int, int]):
    D, H, W = scan_dim

    patch_D, patch_H, patch_W = patch_dim or (None, None, None)
    patch_D = patch_D or D
    patch_H = patch_H or H
    patch_W = patch_W or W

    return patch_D, patch_H, patch_W


def extract_patches(scan: Union[torch.Tensor, np.ndarray], stride: Optional[float] = None,
                    patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None) -> List[
    Union[torch.Tensor, np.ndarray]]:
    patches = []
    patch_D, patch_H, patch_W = check_patch_dim(patch_dim, scan.shape)

    stride_D = round_half_up(patch_D * stride) if stride is not None else patch_D
    stride_H = round_half_up(patch_H * stride) if stride is not None else patch_H
    stride_W = round_half_up(patch_W * stride) if stride is not None else patch_W

    for y in range(0, scan.shape[1], stride_H):
        for x in range(0, scan.shape[2], stride_W):
            for z in range(0, scan.shape[0], stride_D):
                patch = scan[z:z + patch_D, y:y + patch_H, x:x + patch_W]

                if patch.shape != (patch_D, patch_H, patch_W):
                    pad_depth = patch_D - patch.shape[0]
                    pad_height = patch_H - patch.shape[1]
                    pad_width = patch_W - patch.shape[2]

                    if isinstance(patch, torch.Tensor):
                        # torch expects padding in the order: (W_left, W_right, H_left, H_right, D_left, D_right)
                        padding = (0, pad_width, 0, pad_height, 0, pad_depth)
                        patch = torch.nn.functional.pad(patch, padding, mode="constant")
                    else:
                        # numPy expects ((D_before, D_after), (H_before, H_after), (W_before, W_after))
                        pad_width_np = ((0, pad_depth), (0, pad_height), (0, pad_width))
                        patch = np.pad(patch, pad_width_np, mode="constant")

                patches.append(patch)

    return patches


def reconstruct_volume(
        patches: List[Union[torch.Tensor, np.ndarray]], scan_dim: Tuple[int, int, int],
        origins: List[Tuple[int, int, int]],
        patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
        to_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    D, H, W = scan_dim

    patch_D, patch_H, patch_W = check_patch_dim(patch_dim, scan_dim)

    is_numpy = isinstance(patches[0], np.ndarray)

    if is_numpy:
        recon_volume = np.zeros(scan_dim, dtype=np.float32)
        count_volume = np.zeros(scan_dim, dtype=np.float32) # keep track of overlaps
    else:
        recon_volume = torch.zeros(scan_dim, dtype=torch.float32, device=patches[0].device)
        count_volume = torch.zeros(scan_dim, dtype=torch.float32, device=patches[0].device)

    for patch, (z, y, x) in zip(patches, origins):
        z_end = min(z + patch_D, D)
        y_end = min(y + patch_H, H)
        x_end = min(x + patch_W, W)

        patch_cropped = patch[:z_end - z, :y_end - y, :x_end - x]

        recon_volume[z:z_end, y:y_end, x:x_end] += patch_cropped
        count_volume[z:z_end, y:y_end, x:x_end] += 1

    # normalize overlaps
    if is_numpy:
        recon_volume /= np.maximum(count_volume, 1)
    else:
        recon_volume /= torch.clamp(count_volume, min=1)

    return torch.from_numpy(recon_volume) if to_tensor and is_numpy else recon_volume