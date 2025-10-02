from typing import Optional, Literal, Union, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from monai.losses import DiceCELoss

from model.segformer3d import SegFormer3D
from utils import reconstruct_volume, build_metrics


class StrokeFormer(LightningModule):
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 num_classes: int = 1,
                 in_channels: int = 1,
                 lr: float = 1e-6,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 warmup_steps: int = 10_000,
                 warmup_init_lr: float = 4e-6,
                 weight_decay: float = 1e-3):
        super().__init__()

        self.model = SegFormer3D(num_classes=num_classes, in_channels=in_channels)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

        self.loss = DiceCELoss()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.weight_decay = weight_decay

        self.metrics = build_metrics(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # (B, P, C, D, H, W) -> (B*P, C, D, H, W)
        x = x.view(-1, *x.shape[2:])
        logits = self.model(x) # shape (B*P, N, D, H, W)

        if self.num_classes > 2:
            preds = logits.softmax(dim=1)
        else:
            preds = logits.sigmoid()

        if return_logits:
            return preds, logits

        return preds

    def _common_step(self, batch, prefix: Literal['train', 'val', 'test']):
        patches, origins, masks = batch

        preds = self.forward(patches)
        # (B*P, N, D, H, W) -> (B, P, N, D, H, W)
        preds = preds.view(patches.shape[0], patches.shape[1], *preds.shape[1:])

        # reconstruct volumes from patches
        pred_scans = []
        for scan_patches, scan_origins in zip(preds, origins):
            scan = reconstruct_volume(patches=scan_patches, scan_dim=masks[0].shape, origins=scan_origins)
            pred_scans.append(scan)

        pred_scans = torch.stack(pred_scans, dim=0)

        loss = self.loss(pred_scans, masks)

        log_dict = {f'{prefix}_loss': loss,}
        for metric in self.metrics:
            log_dict[f"{prefix}_{metric}"] = self.metrics[metric](pred_scans, masks)  # noqa

        self.log_dict(dictionary=log_dict, on_step=True, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch):
        return self._common_step(batch, prefix='train')

    def validation_step(self, batch):
        return self._common_step(batch, prefix='val')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        self.warmup_steps = min(self.warmup_steps, self.trainer.estimated_stepping_batches - 1)
        warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.warmup_steps,
                                                            eta_min=self.warmup_init_lr)

        poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, poly_scheduler],
                                                          milestones=[self.warmup_steps])

        return [optimizer], [scheduler]
