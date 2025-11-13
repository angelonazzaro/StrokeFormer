from typing import Literal

import lightning as l
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from utils import compute_metrics, build_metrics


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(DownBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, encoded_x: Tensor = None) -> Tensor:
        x = self.up(x)

        if encoded_x is not None:
            x = torch.cat([x, encoded_x], dim=1)
        x = self.conv(x)

        return x


class UNet(l.LightningModule):

    def __init__(self,
                 input_channels: int,
                 base_channels: int,
                 n_blocks: int = 4,
                 loss_type: Literal["l1", "l2"] = "l2",
                 lr: float = 1e-4,
                 eps: float = 1e-8,
                 betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-4):
        super().__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks

        if loss_type == "l1":
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        for i in range(n_blocks):
            if i > 0:
                input_channels = base_channels
                base_channels = base_channels * 2
            # double the output channels at each down sampling block in the contracting path
            setattr(self, f"downblock_{i}", DownBlock(input_channels, base_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
        )

        base_channels *= 2

        for i in range(n_blocks):
            # half the output channels for the expanding path
            # if i > 0:
            setattr(self, f"upblock_{i}", UpBlock(base_channels, base_channels // 2))
            base_channels = base_channels // 2
            # else:
            #     setattr(self, f"upblock_{i}", UpBlock(base_channels * 2, base_channels))

        self.out_conv = nn.Conv2d(base_channels, self.input_channels, kernel_size=1)

        self.metrics = build_metrics(task="reconstruction")
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        # feature_maps = []

        for i in range(self.n_blocks):
            feature_map, x = getattr(self, f"downblock_{i}")(x)
            # feature_maps.append(feature_map)

        x = self.bottleneck(x)

        for i in range(self.n_blocks):
            # x = getattr(self, f"upblock_{i}")(x, feature_maps[self.n_blocks - (i + 1)])
            x = getattr(self, f"upblock_{i}")(x)

        logits = self.out_conv(x)  # (B, C, H, W)
        return logits

    def _common_step(self, batch, prefix: Literal["train", "val"]):
        scans, head_masks = batch["slices"], batch["head_masks"]  # (B, C, H, W)

        logits = self.forward(scans)  # (B, C, H, W)

        # by multiplying scans/logits with the head masks, the training signal is generated only from brain tissue
        scans = scans * head_masks
        logits = logits * head_masks

        loss = self.loss(scans, logits)

        log_dict = {
            f"{prefix}_loss": loss,
            **compute_metrics(logits, scans, metrics=self.metrics, prefix=prefix, task="reconstruction") # noqa
        }

        self.log_dict(dictionary=log_dict, on_step=False, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
                                weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }