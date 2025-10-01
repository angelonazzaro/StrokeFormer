from typing import Callable, List, Union, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from constants import SCAN_DIM
from dataset.mri_dataset import MRIDataset


class MRIDataModule(LightningDataModule):
    def __init__(self,
                 scans: Union[List[str], str],
                 masks: Optional[Union[List[str], str]] = None,
                 ext: str = ".npy",
                 scan_dim: Tuple[int, int, int] = SCAN_DIM,
                 patch_dim: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
                 stride: Optional[float] = 0.5,
                 transforms: Optional[List[Callable]] = None,
                 augment: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()

        self.scans = scans
        self.masks = masks

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
            setattr(self, f"{split}_set", MRIDataset(scans=self.scans,
                                                     masks=self.masks,
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
        pass
