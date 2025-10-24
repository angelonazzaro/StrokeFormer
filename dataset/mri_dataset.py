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


def init_scans_masks_filepaths(scans: Union[List[str], str], masks: Optional[Union[List[str], str]] = None,
                               ext: str = ".npy"):
    if isinstance(scans, str):
        # load file paths from directory
        if not os.path.exists(scans) or not os.path.isdir(scans):
            raise ValueError(f"`scans` must be a valid directory path: {scans}")

        scans = sorted(glob(os.path.join(scans, "*" + ext)))

    if masks is not None:
        if isinstance(masks, str):
            if not os.path.exists(masks) or not os.path.isdir(masks):
                raise ValueError(f"`masks` must be a valid directory path: {masks}")

            masks = sorted(glob(os.path.join(masks, "*" + ext)))

        assert len(masks) == len(scans), "MRIDataset: scans and masks must have the same length"
    else:
        masks = None

    return scans, masks


def load_data(scans: List[str],
              masks: Optional[List[str]],
              index: int,
              augment: bool,
              transforms: Optional[Callable],
              scan_dim: Tuple[int, int, int, int,] = SCAN_DIM):
    scan = np.load(scans[index])  # expected shape: (C, H, W, D)
    mask = np.zeros_like(scan)

    if masks is not None:
        mask = np.load(masks[index])

    scan, mask = torch.from_numpy(scan), torch.from_numpy(mask)

    scan = einops.rearrange(scan, "c h w d -> c d h w")
    mask = einops.rearrange(mask, "c h w d -> c d h w")

    if scan.shape != scan_dim or mask.shape != scan_dim:
        raise ValueError(
            f"`scan`,`mask` have an unusual shape: {scan.shape}, {mask.shape}. Expected {scan_dim}.")

    if augment and transforms is not None:
        scan, mask = transforms(scan, mask)

    mask = mask.long().to(dtype=scan.dtype)

    return scan, mask


class ReconstructionDataset(IterableDataset):
    def __init__(
            self,
            scans: Union[List[str], str],
            masks: Optional[Union[List[str], str]] = None,
            ext: str = ".npy",
            slice_stride: int = 1,
            slices_per_scan: int = 1,
            scan_dim: Tuple[int, int, int, int,] = SCAN_DIM,
            transforms: Optional[List[Callable]] = None,
            augment: bool = False,
    ):
        super().__init__()

        self.scans, self.masks = init_scans_masks_filepaths(scans, masks, ext)

        self.slice_stride = slice_stride
        self.slices_per_scan = slices_per_scan
        self.scan_dim = scan_dim

        if transforms is not None:
            self.transforms = v2.Compose(transforms)
        else:
            # apply AnoDDPM default transforms
            self.transforms = v2.Compose([
                v2.ToPILImage(),
                # transforms.CenterCrop((175, 240)),
                # transforms.RandomAffine(0, translate=(0.02, 0.1)),
                # transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(256),
                v2.ToTensor(),
                v2.Normalize((0.5, ) * len(scan_dim), (0.5, ) * len(scan_dim)) # noqa
            ])

        self.augment = augment

    def __len__(self):
        return len(self.scans) * round_half_up(self.scan_dim[1] / (self.slices_per_scan * self.slice_stride))

    def __iter__(self):
        indexes = list(range(len(self.scans)))

        for idx in indexes:
            scan, mask = load_data(self.scans, self.masks, idx, self.augment, self.transforms, self.scan_dim)

            for i in range(0, scan.shape[0], self.slice_stride):
                scan_slice = scan[:, i:i + self.slices_per_scan]
                mask_slice = mask[:, i:i + self.slices_per_scan]

                slice_mean, slice_std = scan_slice.mean(), scan_slice.std()
                slice_range = (slice_mean - 1 * slice_std, slice_mean + 2 * slice_std)
                scan_slice = torch.clip(scan_slice, slice_range[0], slice_range[1])

                scan_slice = scan_slice / (slice_range[1] - slice_range[0])

                yield scan_slice, mask_slice


class SegmentationDataset(IterableDataset):
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
        if overlap is not None:
            overlap = (overlap[-3], 1.0, 1.0)

        self.scans, self.masks = init_scans_masks_filepaths(scans, masks, ext)

        self.scan_dim = scan_dim
        self.subvolume_dim = subvolume_dim
        self.overlap = overlap

        self.subvolumes_num = 1

        if self.overlap is not None and self.subvolume_dim is not None:
            patch_D = round_half_up(self.subvolume_dim * overlap[-3]) if overlap is not None else self.subvolume_dim
            self.subvolumes_num = round_half_up(scan_dim[-3] / patch_D)

        if transforms is not None:
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = None

        self.augment = augment

    def __len__(self):
        return len(self.scans) * self.subvolumes_num

    def __iter__(self):
        # TODO: implement multiple workers logic to avoid data duplication
        indexes = list(range(len(self.scans)))

        for index in indexes:
            scan, mask = load_data(self.scans, self.masks, index, self.augment, self.transforms, self.scan_dim)

            scan_chunks = extract_patches(scan=scan, overlap=self.overlap, patch_dim=(self.subvolume_dim, None, None))
            mask_chunks = extract_patches(scan=mask, overlap=self.overlap, patch_dim=(self.subvolume_dim, None, None))

            # per-subvolume z-score standardization
            for scan_chunk, mask_chunk in zip(scan_chunks, mask_chunks):
                scan_chunk = (scan_chunk - scan_chunk.mean()) / scan_chunk.std()
                yield scan_chunk, mask_chunk
