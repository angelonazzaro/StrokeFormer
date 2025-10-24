import copy
from typing import Literal, Optional

import torch.optim
from lightning import LightningModule

from utils import compute_metrics, build_metrics
from .gaussian_diffusion import GaussianDiffusionModel, get_beta_schedule
from .unet import UNetModel, update_ema_params


class AnoDDPM(LightningModule):
    def __init__(self,
                 img_size: tuple[int, int],
                 base_channels: int,
                 conv_resample: bool = True,
                 num_classes: int = 2,
                 n_heads: int = 1,
                 n_head_channels: int = -1,
                 channel_mults: str = "",
                 num_res_blocks: int = 2,
                 dropout: float = 0,
                 attention_resolutions: str = "32,16,8",
                 biggan_updown: bool = True,
                 in_channels: int = 1,
                 diff_betas: Optional[tuple[float, float]] = None,
                 beta_schedule: Literal["linear", "cosine"] = "linear",
                 train_start: bool = False,
                 T: int = 1000,
                 diff_loss_type: str = "l2",
                 diff_loss_weight: Literal["prop t", "uniform", None] = None,
                 diff_noise: Literal["gauss", "perlin", "simplex"] = "gauss",
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 betas: tuple[float, float] = (0.9, 0.999)):
        super().__init__()

        self.unet = UNetModel(img_size=img_size[0], base_channels=base_channels,
                              conv_resample=conv_resample, n_heads=n_heads,
                              n_head_channels=n_head_channels, channel_mults=channel_mults,
                              num_res_blocks=num_res_blocks, dropout=dropout,
                              attention_resolutions=attention_resolutions, biggan_updown=biggan_updown,
                              in_channels=in_channels)

        self.ema = copy.deepcopy(self.unet)

        if diff_betas is None:
            diff_betas = get_beta_schedule(T, beta_schedule)

        self.diffusion_model = GaussianDiffusionModel(img_size=img_size, betas=diff_betas,
                                                      loss_type=diff_loss_type, loss_weight=diff_loss_weight,
                                                      noise=diff_noise, img_channels=in_channels, T=T)

        self.train_start = train_start
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

        self.metrics = build_metrics(num_classes=num_classes, task="reconstruction")

        self.save_hyperparameters()


    def forward(self, x, run_backward: bool = False):
        loss, estimates = self.diffusion_model.p_loss(self.unet, x, train_start=self.train_start)

        return_dict = {"loss": loss, "estimates": estimates}

        if run_backward:
            outputs = self.diffusion_model.forward_backward(self.unet, x, see_whole_sequence=None, t_distance=200)
            return_dict["outputs"] = outputs

        return return_dict


    def _common_step(self, batch, batch_idx, prefix: Literal['train', 'val', 'test']):
        if batch_idx > 1 and prefix == 'train':
            # making sure this is called after the first optimizer step
            update_ema_params(self.ema, self.unet)

        scans, masks = batch

        output_dict = self.forward(scans, True)
        recons = output_dict["outputs"]

        mse = (masks - recons).square()
        mse = (mse > 0.5).float()

        log_dict = {
            f"{prefix}_loss": output_dict["loss"],
            **compute_metrics(torch.cat([mse, recons], dim=0), torch.cat([masks, scans], dim=0), metrics=self.metrics, prefix=prefix, task="reconstruction"),
        }

        self.log_dict(dictionary=log_dict, on_step=False, prog_bar=True, on_epoch=True)

        return output_dict["loss"]

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, prefix='val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer
        }
