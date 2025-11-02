import os
from typing import Optional, List, Callable

import torch
import torch.nn.functional as f
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .mri_dataset import SegmentationDataset, ReconstructionDataset


def validate_paths_structure(paths: dict):
    """
    `paths` must have the following structure:
    {
        "train": {
            "scans": scans/scans_dir,
            "masks": masks/masks_dir
        },
        "val": {
            "scans": scans/scans_dir,
            "masks": masks/masks_dir
        }
        "test": {
            "scans": scans/scans_dir,
            "masks": masks/masks_dir
        }
    }
    """
    keys = set(paths.keys())

    expected_train_keys = {"train", "val"}
    expected_test_keys = {"test"}

    if not (expected_train_keys.issubset(keys) or expected_test_keys.issubset(keys)):
        raise ValueError(
            f"`paths` must contain either the keys {expected_train_keys} and/or {expected_test_keys}, but got {keys}")

    for split in keys:
        if not isinstance(paths[split], dict):
            raise TypeError(f"`paths['{split}']` must be a dictionary")

        for subkey in ["scans", "masks"]:
            if subkey not in paths[split] and subkey not in expected_test_keys:
                raise ValueError(f"`paths['{split}']` must contain the key '{subkey}'")
            if not isinstance(paths[split][subkey], str) and not isinstance(paths[split][subkey], List):
                raise TypeError(
                    f"`paths['{split}']['{subkey}']` must be a string or a List of strings, "
                    f"got {type(paths[split][subkey]).__name__}")


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


class ReconstructionDataModule(LightningDataModule):
    def __init__(self,
                 scans: str,
                 ext: str = ".npy",
                 resize_to: Optional[tuple[int, int]] = None,
                 transforms: Optional[List[Callable]] = None,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()

        self.scans = scans
        self.ext = ext
        self.resize_to = resize_to
        self.transforms = transforms

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set, self.val_set, self.test_set = (None, None, None)

    def setup(self, stage):
        if stage == "fit":
            transforms = self.transforms
            splits = ["train", "val"]
        else:
            transforms = None
            splits = ["test"]

        for split in splits:
            setattr(self, f"{split}_set", ReconstructionDataset(scans=self.scans,
                                                                ext=self.ext,
                                                                transforms=transforms))

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.custom_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def custom_collate(self, batch):
        slices, head_masks = [], []

        for el in batch:
            slices.append(el["scan_slice"])
            head_masks.append(el["head_mask"])

        slices, head_masks = torch.stack(slices), torch.stack(head_masks)

        if self.resize_to is not None:
            slices, head_masks = resize(slices.unsqueeze(2), head_masks.unsqueeze(2), *self.resize_to)

        return {"slices": slices.squeeze(2), "head_masks": head_masks.squeeze(2)}  # (B, C, H, W)


class SegmentationDataModule(LightningDataModule):
    def __init__(self,
                 paths: dict,
                 ext: str = ".npy",
                 subvolume_depth: Optional[int] = None,
                 overlap: Optional[float] = 0.5,
                 transforms: Optional[List[Callable]] = None,
                 resize_to: Optional[tuple[int, int]] = None,
                 batch_size: int = 32,
                 num_workers: Optional[int] = 0):
        super().__init__()

        validate_paths_structure(paths)

        self.paths = paths

        self.ext = ext
        self.subvolume_depth = subvolume_depth
        self.overlap = overlap
        self.transforms = transforms

        self.resize_to = resize_to
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers

        self.train_set, self.val_set, self.test_set = None, None, None
        self.stage = None

    def setup(self, stage: str):
        if stage == "fit":
            transforms = self.transforms
            splits = ["train", "val"]
        else:
            transforms = None
            splits = ["test"]

        for split in splits:
            setattr(self, f"{split}_set", SegmentationDataset(scans=self.paths[split]["scans"],
                                                              masks=self.paths[split]["masks"],
                                                              ext=self.ext,
                                                              overlap=self.overlap,
                                                              subvolume_depth=self.subvolume_depth,
                                                              transforms=transforms))
        self.stage = stage

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,  # does not work with iterable dataset
            collate_fn=self.custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
        )

    def custom_collate(self, batch):
        scans, masks, means, stds = [], [], [], []

        for el in batch:
            scans.append(el["scan"])
            masks.append(el["mask"])
            means.append(el["mean"])
            stds.append(el["std"])

        scans, masks = torch.stack(scans), torch.stack(masks)  # batched tensors (B, C, D, H, W)
        means, stds = torch.stack(means), torch.stack(stds)  # (B)

        if self.resize_to is not None:
            scans, masks = resize(scans, masks, *self.resize_to)

        return {"scans": scans, "masks": masks, "means": means, "stds": stds}