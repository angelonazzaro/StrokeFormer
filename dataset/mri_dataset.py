import os
from glob import glob
from typing import Optional, List, Callable, Union

import einops
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2

from constants import SLICE_DIM
from utils import extract_patches, round_half_up, compute_head_mask, resize


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

    scan, mask = torch.from_numpy(scan).to(dtype=torch.float32), torch.from_numpy(mask).to(dtype=torch.uint8)

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
            masks: Optional[Union[List[str], str]] = None,
            ext: str = ".npy",
            transforms: Optional[List[Callable]] = None,
            resize_to: Optional[tuple[int, int]] = None,
            bb_min_size: tuple[int, int] = (5, 5),
            augment: bool = False,
    ):
        super().__init__()

        self.scans, self.masks = init_scans_masks_filepaths(scans, masks, ext)

        self.augment = augment

        # permute slices to destroy 'brain ordering', i.e., slices are loaded in order within the same brain
        # this could lead the models to learn a sort of ordering bias
        perm = np.random.permutation(len(self.scans))
        self.scans = [self.scans[i] for i in perm]
        self.masks = [self.masks[i] for i in perm]

        self.bb_min_size = bb_min_size
        self.resize_to = resize_to

        if transforms is not None:
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = v2.RandomApply([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-30, 30))
            ], p=0.5)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_slice, mask = load_data(self.scans, self.masks, idx, None)

        if self.resize_to is not None:
            scan_slice, mask = resize(scan_slice.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0), *self.resize_to)
            scan_slice, mask = scan_slice.squeeze(0).squeeze(0), mask.squeeze(0).squeeze(0).to(torch.int64)

        head_mask = compute_head_mask(scan_slice)

        # if slice is healthy, create empty bounding boxes
        if mask.sum() == 0:
            boxes = torch.empty((0, 4), dtype=torch.int64)
            labels = torch.zeros((1,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = masks_to_boxes(mask)
            # if lesion is too small to have a valid bounding box, expand it to bb_min_size
            boxes = boxes.clone()  # avoid modifying original

            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]

            small_width = widths < self.bb_min_size[1]
            # adjust xmin and xmax to create a bb_min_size pixel width box centered at original center
            x_centers = (boxes[small_width, 0] + boxes[small_width, 2]) / 2
            boxes[small_width, 0] = x_centers - self.bb_min_size[1] / 2
            boxes[small_width, 2] = x_centers + self.bb_min_size[1] / 2

            small_height = heights < self.bb_min_size[0]
            # adjust ymin and ymax to create a bb_min_size pixel height box centered at original center
            y_centers = (boxes[small_height, 1] + boxes[small_height, 3]) / 2
            boxes[small_height, 1] = y_centers - self.bb_min_size[0] / 2
            boxes[small_height, 3] = y_centers + self.bb_min_size[0] / 2

            labels = torch.ones((1,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        scan_slice = (scan_slice - scan_slice.mean()) / scan_slice.std()
        scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min())

        scan_slice = tv_tensors.Image(scan_slice)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=v2.functional.get_size(scan_slice)),
            "mask": tv_tensors.Mask(mask),
            "labels": labels,
            "area": area
        }

        if self.transforms and self.augment:
            scan_slice = self.transforms(scan_slice)

        return {"scan_slice": scan_slice, "target": target, "head_mask": head_mask.to(dtype=torch.uint8)}


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
                 transforms: Optional[List[Callable]] = None,
                 augment: bool = False):
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
                    
                augment: (bool, default=False):
                    Whether to perform data augmentation.
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
        else:
            transforms = v2.RandomApply([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-30, 30))
            ], p=0.5)

        self.transforms = transforms
        self.augment = augment

    def __len__(self):
        return len(self.scans) * self.subvolumes_num

    def __iter__(self):
        # TODO: handle multiple workers to avoid data duplication when working with more than one worker
        indexes = list(range(len(self.scans)))

        for index in indexes:
            scan, mask = load_data(self.scans, self.masks, index, self.transforms if self.augment else None)

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
                z_min, z_max = scan_chunk.min(), scan_chunk.max()
                scan_chunk = (scan_chunk - z_min) / (z_max - z_min)
                yield {"scan": scan_chunk, "mask": mask_chunk, "mean": scan_chunk_mean, "std": scan_chunk_std, "min_max": (z_min, z_max)}