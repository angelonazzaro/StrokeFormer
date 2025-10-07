import os
from glob import glob
from typing import Callable, List, Union, Optional, Tuple

import einops
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

from constants import SCAN_DIM
from utils import round_half_up, extract_patches


class MRIDataset(IterableDataset):
    def __init__(self,
                 scans: Union[List[str], str],
                 masks: Optional[Union[List[str], str]] = None,
                 ext: str = ".npy",
                 scan_dim: Tuple[int, int, int, int,] = SCAN_DIM,
                 subvolume_dim: Optional[int] = None,
                 overlap: Optional[Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]] = 0.5,
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
            
            subvolume_dim (int, default=None):
                Size to consider along Z dimension when extracting subvolume.
                If  None, the full Z dimension is used.

            overlap (Union[float, Tuple[float, float, float]], default=0.5):
                Amount of overlap between scans along each spatial dimension. If None, no overlap is applied. 
                If float, the overlap wll be applied for all spatial dimensions. Currently, only the Z dimension is considered.
            
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

        # TODO: remove 1.0 in future when X, Y dimensions will be supported for patch/subvolume extraction
        if isinstance(overlap, float):
            overlap = (overlap, 1.0, 1.0)

        if isinstance(overlap, tuple):
            overlap = tuple([o if o is not None else 1.0 for o in overlap])

            for o in overlap:
                if o is not None and (o < 0.0 or o > 1.0):
                    raise ValueError(f"`overlap` must be within the range [0, 1]: {o}")

        # TODO: remove when X, Y dimensions will be supported for patch/subvolume extraction
        overlap = (overlap[-3], 1.0, 1.0)

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
        self.subvolume_dim = subvolume_dim
        self.overlap = overlap

        self.subvolumes_num = 1

        if self.overlap is not None:
            patch_D = round_half_up(self.subvolume_dim * overlap[-3]) if overlap is not None else self.subvolume_dim
            self.subvolumes_num = round_half_up(scan_dim[-3] / patch_D)

        if transforms is not None:
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = None

        self.augment = augment

    def __len__(self):
        return len(self.scans) * self.subvolumes_num

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

        for index in indexes:
            scan, mask = self._load_data(index)

            scan_chunks = extract_patches(scan=scan, overlap=self.overlap, patch_dim=(self.subvolume_dim, None, None))
            mask_chunks = extract_patches(scan=mask, overlap=self.overlap, patch_dim=(self.subvolume_dim, None, None))

            # per-subvolume z-score standardization
            for scan_chunk, mask_chunk in zip(scan_chunks, mask_chunks):
                scan_chunk = (scan_chunk - scan_chunk.mean()) / scan_chunk.std()
                yield scan_chunk, mask_chunk
