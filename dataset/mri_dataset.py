import os
from glob import glob
from typing import Optional, List, Callable, Union

import einops
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import v2

from constants import SLICE_DIM
from utils import extract_patches, round_half_up, compute_head_mask


def init_scans_masks_filepaths(scans: Union[List[str], str],
                               masks: Optional[Union[List[str], str]] = None,
                               ext: str = ".npy") -> tuple[List[str], Optional[List[str]]]:
    # if scans is a directory, load files from directory
    if isinstance(scans, str):
        if not os.path.exists(scans) or not os.path.isdir(scans):
            raise ValueError(f"`scans` must a be a valid directory path: {scans}")

        scans = sorted(glob(os.path.join(scans, f"*{ext}"), recursive=True))

    if masks is not None:
        if isinstance(masks, str):
            if not os.path.exists(masks) or not os.path.isdir(masks):
                raise ValueError(f"`mask` must a be a valid directory path: {masks}")

            masks = sorted(glob(os.path.join(masks, f"*{ext}"), recursive=True))

        assert len(masks) == len(scans), "scans and masks must have the same length"

    return scans, masks


def load_data(scans: List[str],
              masks: Optional[List[str]],
              index: int,
              transforms: Optional[Callable] = None) -> tuple[Tensor, Tensor]:
    scan = np.load(scans[index])  # (C, H, W, D) or (C, H, W)
    mask = np.zeros_like(scan)

    if masks is not None:
        mask = np.load(masks[index])

    scan, mask = torch.from_numpy(scan), torch.from_numpy(mask).to(dtype=torch.uint8)

    if scan.ndim == 4:
        scan = einops.rearrange(scan, "c h w d -> c d h w")
        mask = einops.rearrange(mask, "c h w d -> c d h w")

    if transforms is not None:
        scan, mask = transforms(scan, mask)

    return scan, mask


class ReconstructionDataset(Dataset):
    def __init__(
            self,
            scans: Union[List[str], str],
            ext: str = ".npy",
            transforms: Optional[List[Callable]] = None,
            slice_dim: tuple[int, int, int] = SLICE_DIM,
    ):
        super().__init__()

        self.scans, _ = init_scans_masks_filepaths(scans, None, ext)

        # permute slices to destroy 'brain ordering', i.e., slices are loaded in order within the same brain
        # this could lead the models to learn a sort of ordering bias
        perm = np.random.permutation(len(self.scans))
        self.scans = [self.scans[i] for i in perm]

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
                v2.ToImage(),  # same as ToTensor
                v2.ToDtype(torch.float32, scale=True),  # same as ToTensor
                v2.Normalize((0.5, ) * slice_dim[-3], (0.5, ) * slice_dim[-3]) # noqa
            ])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_slice, mask_slice = load_data(self.scans, None, idx, self.transforms)

        head_mask = compute_head_mask(scan_slice)

        slice_mean, slice_std = scan_slice.mean(), scan_slice.std()
        slice_range = (slice_mean - 1 * slice_std, slice_mean + 2 * slice_std)
        scan_slice = torch.clip(scan_slice, slice_range[0], slice_range[1])

        scan_slice = scan_slice / (slice_range[1] - slice_range[0])

        return {"scan_slice": scan_slice, "head_mask": head_mask.to(dtype=mask_slice.dtype)}


class SegmentationDataset(IterableDataset):
    """
        Create an IterableDataset that yields 3D subvolumes from 3D MRI scans.
    """

    def __init__(self,
                 scans: Union[List[str], str],
                 masks: Optional[Union[List[str], str]] = None,
                 ext: str = ".npy",
                 subvolume_depth: Optional[int] = None,
                 overlap: Optional[float] = 0.5,
                 transforms: Optional[List[Callable]] = None):
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
                
                subvolume_depth (int, default=None):
                    Size to consider along Z dimension when extracting subvolumes.
                    If None, the full Z dimension is used.

                overlap (Union[float, Tuple[float, float, float]], default=0.5):
                    Amount of overlap between scans along each spatial dimension. 
                    If None, no overlap is applied.
                    If float, the overlap wll be applied for all spatial dimensions. 
                    Currently, only the Z dimension is considered.
                 
                transforms (Optional[List[Callable]], default=None):
                    A list of transformations applied to each scan (and mask).
                    Each transform should take a scan (and optionally a mask) as input
                    and return the transformed version.
        """

        # currently, only the Z dimension is supported for subvolume extraction.
        if isinstance(overlap, float):
            overlap = (overlap, 1.0, 1.0)  # TODO: substitute with (overlap, overlap, overlap)
        
        if isinstance(overlap, tuple):
            overlap = tuple([ov if ov is not None else 1.0 for ov in overlap])

            for ov in overlap:
                if ov > 1.0 or ov < 0.0:
                    raise ValueError(f"overlap values must be within the range [0,1], got: {overlap}")

        self.overlap = overlap

        self.scans, self.masks = init_scans_masks_filepaths(scans, masks, ext)

        self.subvolume_depth = subvolume_depth
        self.subvolumes_num = 1  # number of subvolumes that will be generated from each volume

        if self.subvolume_depth is not None:
            scan, _ = load_data(self.scans, None, 0)
            patch_D = round_half_up(self.subvolume_depth * overlap[-3] if overlap is not None else scan.shape[-3])
            self.subvolumes_num = round_half_up(scan.shape[-3] / patch_D)

        if transforms is not None:
            transforms = v2.Compose(transforms)

        self.transforms = transforms

    def __len__(self):
        return len(self.scans) * self.subvolumes_num

    def __iter__(self):
        # TODO: handle multiple workers to avoid data duplication when working with more than one worker
        indexes = list(range(len(self.scans)))

        for index in indexes:
            scan, mask = load_data(self.scans, self.masks, index, self.transforms)

            # extract [overlapped] subvolumes along Z dimension
            scan_chunks = extract_patches(scan=scan, overlap=self.overlap,
                                          patch_dim=(self.subvolume_depth, None, None))
            mask_chunks = extract_patches(scan=mask, overlap=self.overlap,
                                          patch_dim=(self.subvolume_depth, None, None))

            # per-volume z-score standardization
            # when computing the head mask for getting the lesion size, the optimal threshold is 8
            # however, this was found on raw intensity values and not on normalized intensity values
            # since we apply per-volume z-score normalization, we need the original mean and std to
            # scale the threshold accordingly when computing the head mask
            for scan_chunk, mask_chunk in zip(scan_chunks, mask_chunks):
                scan_chunk_mean, scan_chunk_std = scan_chunk.mean(), scan_chunk.std()
                scan_chunk = (scan_chunk - scan_chunk_mean) / scan_chunk_std
                yield {"scan": scan_chunk, "mask": mask_chunk, "mean": scan_chunk_mean, "std": scan_chunk_std}