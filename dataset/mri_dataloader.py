from typing import Callable, List, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from constants import SCAN_DIM
from dataset.mri_dataset import MRIDataset


def validate_paths_structure(paths):
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


class MRIDataModule(LightningDataModule):
    def __init__(self,
                 paths: dict,
                 ext: str = ".npy",
                 scan_dim: Tuple[int, int, int, int] = SCAN_DIM,
                 patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
                 stride: Optional[float] = 0.5,
                 transforms: Optional[List[Callable]] = None,
                 augment: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()

        validate_paths_structure(paths)

        self.paths = paths

        self.ext = ext

        self.scan_dim = scan_dim
        self.patch_dim = patch_dim
        self.stride = stride

        self.transforms = transforms
        self.augment = augment

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set, self.val_set, self.test_set = (None, None, None)

    def setup(self, stage):
        augment = self.augment
        splits = ['train', 'val']

        if stage == 'test':
            augment = False
            splits = ['test']

        for split in splits:
            setattr(self, f"{split}_set", MRIDataset(scans=self.paths[split]["scans"],
                                                     masks=self.paths[split]["masks"],
                                                     ext=self.ext,
                                                     scan_dim=self.scan_dim,
                                                     patch_dim=self.patch_dim,
                                                     stride=self.stride,
                                                     transforms=self.transforms,
                                                     augment=augment))

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            # collate_fn=self.custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            # collate_fn=self.custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            # collate_fn=self.custom_collate,
        )

    def custom_collate(self, batch):
        pass
