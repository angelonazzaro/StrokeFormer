import os
from glob import glob
from typing import Callable, List, Union, Optional, Tuple

import einops
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from constants import SCAN_DIM
from utils import extract_patches, round_half_up, check_patch_dim


class MRIDataset(Dataset):
    def __init__(self,
                 scans: Union[List[str], str],
                 masks: Optional[Union[List[str], str]] = None,
                 ext: str = ".npy",
                 scan_dim: Tuple[int, int, int, int,] = SCAN_DIM,
                 patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
                 stride: Optional[float] = 0.5,
                 transforms: Optional[List[Callable]] = None,
                 augment: bool = False,
                 ):
        super().__init__()

        """
        Args:
            scans (Union[List[str], str]):
                MRI scan sources. Can be:
                  - A directory containing scans with the given extension (`ext`).
                  - A list of file paths.

            masks (Optional[Union[List[str], str]], default=None):
                Corresponding masks for the scans. Can be:
                  - A directory containing masks with the given extension (`ext`).
                  - A list of file paths.
                Must be aligned with `scans` in ordering. If None, no masks are used.

            ext (str, default=".npy"):
                File extension for scans and masks (only used if `scans` or `masks` are directories).

            scan_dim (Tuple[int, int, int, int], default=(1, 189, 192, 192)):
                Expected shape of each MRI scan in (Channels, Depth, Height, Width).

            patch_dim (Optional[Tuple[Optional[int], Optional[int], Optional[int]]], default=None):
                Shape of extracted patches in (Depth, Height, Width).
                If any dimension is None, the full dimension of that axis is used.

            stride (float, default=0.5):
                Stride size (fraction of patch size) when sliding the window over scans
                to extract patches. If None, no stride is applied.

            transforms (Optional[List[Callable]], default=None):
                A list of transformations applied to each scan (and mask).
                Each transform should take a scan (and optionally a mask) as input
                and return the transformed version.

            augment (bool, default=False):
                If True, applies data augmentation strategies during training.
        """

        if stride < 0.0 or stride > 1.0:
            raise ValueError(f"`stride` must be within the range [0, 1]: {stride}")

        self.scans = scans

        if isinstance(scans, str):
            # load file paths from directory
            if not os.path.exists(scans) or not os.path.isdir(scans):
                raise ValueError(f"`scans` must be a valid directory path: {scans}")

            self.scans = sorted(glob(os.path.join(scans, "*" + ext)))

        if masks is not None:
            if isinstance(masks, str):
                if not os.path.exists(masks) or not os.path.isdir(masks):
                    raise ValueError(f"`masks` must be a valid directory path: {masks}")

                self.masks = sorted(glob(os.path.join(masks, "*" + ext)))

            assert len(self.masks) == len(self.scans), "MRIDataset: scans and masks must have the same length"
        else:
            self.masks = None

        self.scan_dim = scan_dim
        self.patch_dim = patch_dim
        self.stride = stride

        self.patch_num = None

        if self.patch_dim is not None:
            _, D, H, W = scan_dim
            patch_D, patch_H, patch_W = check_patch_dim(patch_dim, scan_dim)

            # check for 'padded' patches that may be added during patch extraction
            padded_patches = 0

            if D % patch_D != 0:
                padded_patches += patch_D - (D % patch_D)
            if H % patch_H != 0:
                padded_patches += patch_H - (H % patch_H)
            if W % patch_W != 0:
                padded_patches += patch_W - (W % patch_W)

            patch_D = round_half_up(patch_D * stride) if stride is not None else patch_D
            patch_H = round_half_up(patch_H * stride) if stride is not None else patch_H
            patch_W = round_half_up(patch_W * stride) if stride is not None else patch_W

            self.patch_num = round_half_up((D * H * W) / (patch_D * patch_H * patch_W)) + padded_patches

        if transforms is not None:
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = None

        self.augment = augment

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        scan = np.load(self.scans[index])  # expected shape: (C, H, W, D)
        mask = np.zeros_like(scan)

        if self.masks is not None:
            mask = np.load(self.masks[index])

        scan, mask = torch.from_numpy(scan), torch.from_numpy(mask)

        scan = einops.rearrange(scan, "c h w d -> c d h w")
        mask = einops.rearrange(mask, "c h w d -> c d h w")

        if scan.shape != self.scan_dim or mask.shape != self.scan_dim:
            raise ValueError(
                f"`scan`,`mask` have an unusual shape: {scan.shape}, {mask.shape}. Expected {self.scan_dim}.")

        if self.augment and self.transforms is not None:
            scan, mask = self.transforms(scan, mask)

        scan = (scan - scan.mean()) / scan.std()
        mask = mask.long().to(dtype=scan.dtype)

        if self.patch_dim is not None:
            patches, origins = extract_patches(scan=scan, stride=self.stride, patch_dim=self.patch_dim,
                                               return_origins=True)
            return torch.stack(patches, dim=0), torch.tensor(origins), mask

        return scan, mask
