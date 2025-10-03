import os
from glob import glob
from typing import Callable, List, Union, Optional, Tuple

import einops
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

from constants import SCAN_DIM
from utils import round_half_up


class MRIDataset(IterableDataset):
    def __init__(self,
                 scans: Union[List[str], str],
                 masks: Optional[Union[List[str], str]] = None,
                 ext: str = ".npy",
                 scan_dim: Tuple[int, int, int, int,] = SCAN_DIM,
                 slices_per_scan: int = SCAN_DIM[-3],
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
            
            slices_per_scan (Tuple[int, int, int, int], default=(1, 189, 192, 192)):
                Depth/N. of slices to consider for each volume (Channels, Depth, Height, Width).
            
            resize_to: (dict, default= None):
                A dictionary containing 'height' and 'width' keys that correspond 
                to the height and width images will be resized to

            transforms (Optional[List[Callable]], default=None):
                A list of transformations applied to each scan (and mask).
                Each transform should take a scan (and optionally a mask) as input
                and return the transformed version.

            augment (bool, default=False):
                If True, applies data augmentation strategies during training.
        """

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
        self.slices_per_scan = slices_per_scan

        if scan_dim[-3] < self.slices_per_scan:
            raise ValueError(f"MRIDataset: `slices_per_scan` must be smaller than `depth` axis in scan_dim: {self.slices_per_scan} - {scan_dim[-3]}")

        if transforms is not None:
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = None

        self.augment = augment

    def __len__(self):
        return len(self.scans) * round_half_up(self.scan_dim[-3] / self.slices_per_scan)

    def _load_data(self, index):
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

        mask = mask.long().to(dtype=scan.dtype)

        return scan, mask

    def __iter__(self):
        # TODO: implement multiple workers logic to avoid data duplication
        indexes = list(range(len(self.scans)))
        depth = self.scan_dim[-3]

        for index in indexes:
            scan, mask = self._load_data(index)

            # TODO: right now, 3D continuity is broken. Implement slicing window or overlap patching approach
            for i in range(0, depth, self.slices_per_scan):
                scan_chunk = scan[:, i: i + self.slices_per_scan]
                mask_chunk = mask[:, i: i + self.slices_per_scan]

                if scan_chunk.shape[-3] != self.slices_per_scan:
                    padding = (0, 0, 0, 0, 0, self.slices_per_scan - scan_chunk.shape[-3], 0, 0)

                    scan_chunk = torch.nn.functional.pad(scan_chunk, padding)
                    mask_chunk = torch.nn.functional.pad(mask_chunk, padding)

                # per-volume z-score normalization
                scan_chunk = (scan_chunk - scan_chunk.mean()) / scan_chunk.std()

                yield scan_chunk, mask_chunk
