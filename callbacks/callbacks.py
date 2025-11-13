from typing import Literal, Any

import torch
import wandb
from lightning import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid

from utils import get_per_slice_segmentation_preds


class LogPrediction(Callback):
    """
        Callback to log predictions after validation.
    """

    def __init__(self,
                 log_every_n_epochs: int,
                 num_samples: int,
                 task: Literal["segmentation", "reconstruction"] = "segmentation"):
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.task = task
        self.samples = None
        self.targets = None
        self.means = None
        self.stds = None
        self.mins_maxs = None
        self.current_idx = 0
        self.columns = []

    @rank_zero_only
    def on_validation_batch_end(self,
                                trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule",
                                outputs: STEP_OUTPUT,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        needed = self.num_samples - self.current_idx
        samples_key = "scans" if self.task == "segmentation" else "slices"
        targets_key = "masks" if self.task == "segmentation" else "head_masks"
        batch_size = batch[samples_key].shape[0]
        if needed > 0:
            take = min(needed, batch_size)

            if self.samples is None:
                shape = (self.num_samples, *batch[samples_key].shape[1:])
                self.samples = torch.empty(shape, dtype=batch[samples_key].dtype, device=batch[samples_key].device)
                mask_shape = (self.num_samples, *batch[targets_key].shape[1:])
                self.targets = torch.empty(mask_shape, dtype=batch[targets_key].dtype, device=batch[targets_key].device)

                if self.task == "segmentation":
                    self.means = torch.empty(self.num_samples, dtype=batch["means"].dtype, device=batch["means"].device)
                    self.stds = torch.empty(self.num_samples, dtype=batch["stds"].dtype, device=batch["stds"].device)
                    self.mins_maxs = torch.empty((self.num_samples, 2), dtype=batch["mins_maxs"].dtype, device=batch["mins_maxs"].device)

            self.samples[self.current_idx:self.current_idx + take] = batch[samples_key][:take]
            self.targets[self.current_idx:self.current_idx + take] = batch[targets_key][:take]

            if self.task == "segmentation":
                self.means[self.current_idx:self.current_idx + take] = batch["means"][:take]
                self.stds[self.current_idx:self.current_idx + take] = batch["stds"][:take]
                self.mins_maxs[self.current_idx:self.current_idx + take] = batch["mins_maxs"][:take]

            self.current_idx += take

    def _log_segmentation_prediction(self, trainer: "pl.Trainer", model: "pl.LightningModule"):
        if len(self.columns) == 0:
            self.columns = ["slice idx", "ground truth", "prediction"]

            metrics = list(model.metrics.keys())

            self.columns.extend(metrics)

        table = wandb.Table(columns=self.columns)

        for result in get_per_slice_segmentation_preds(model, self.samples, self.targets, model.metrics, self.means, self.stds, self.mins_maxs):  # noqa
            if result["lesion_size"] != "No Lesion":
                table.add_data(result["slice_idx"], wandb.Image(result["ground_truth"]), wandb.Image(result["prediction"]), *result["scores"].values())

        # assuming wandb logger
        trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/predictions": table})

    def _log_reconstruction_prediction(self, trainer: "pl.Trainer", model: "pl.LightningModule"):
        with torch.no_grad():
            recons = model(self.samples)

        all_imgs = torch.cat([self.samples, recons], dim=0)
        min_vals = all_imgs.view(all_imgs.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1) # noqa
        max_vals = all_imgs.view(all_imgs.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        all_imgs = (all_imgs - min_vals) / (max_vals - min_vals + 1e-8)

        n_pairs = recons.shape[0]
        # interleave reference and reconstruction: ref0, rec0, ref1, rec1, ...
        paired_images = torch.empty((n_pairs * 2, 1, *recons.shape[2:]), dtype=self.samples.dtype)
        paired_images[0::2] = all_imgs[:n_pairs]
        paired_images[1::2] = all_imgs[n_pairs:]
        grid = make_grid(paired_images, padding=2, pad_value=1.0)
        grid = to_pil_image(grid)
        # assuming wandb logger
        trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/recon": wandb.Image(grid)})

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.sanity_checking or trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        pl_module.eval()

        if self.task == "segmentation":
            self._log_segmentation_prediction(trainer, pl_module)
        else:
            self._log_reconstruction_prediction(trainer, pl_module)

        pl_module.train()