import tempfile
from typing import Literal, Any, List

import torch
import wandb
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from matplotlib import animation, pyplot as plt

from utils import get_per_slice_segmentation_preds, gridify_output


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
        self.samples = []
        self.targets = []
        self.columns = []

    @rank_zero_only
    def on_validation_batch_end(self,
                                trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule",
                                outputs: STEP_OUTPUT,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        needed = self.num_samples - len(self.samples)

        if needed > 0:
            take = min(needed, batch[0].shape[0])
            self.samples.append(batch["scans" if self.task == "segmentation" else "slices"][:take])
            self.targets.append(batch["masks" if self.task == "segmentation" else "head_masks"][:take])

    def _log_segmentation_prediction(self, trainer: "pl.Trainer", model: "pl.LightningModule"):
        if len(self.columns) == 0:
            self.columns = ["slice idx", "ground truth", "prediction"]

            metrics = list(model.metrics.keys())

            self.columns.extend(metrics)

        table = wandb.Table(columns=self.columns)

        for result in get_per_slice_segmentation_preds(model, self.samples, self.targets, model.metrics):  # noqa
            table.add_data(result["slice_idx"], wandb.Image(result["ground_truth"]), wandb.Image(result["prediction"]), *result["scores"].values())

        # assuming wandb logger
        trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/predictions": table})

    def _log_reconstruction_prediction(self, trainer: "pl.Trainer", model: "pl.LightningModule"):
        with torch.no_grad():
            recons = model.diffusion_model.forward_backward(model.ema, self.samples, see_whole_sequence="whole")

        fig, ax = plt.subplots()

        imgs = [[ax.imshow(gridify_output(x, 5), animated=True)] for x in recons]
        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            ani.save(video.name, fps=30)
            # assuming wandb logger
            trainer.logger.experiment.log({f"val_epoch_{trainer.current_epoch}/recon": wandb.Video(video.name)})

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.sanity_checking or trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        pl_module.eval()

        if isinstance(self.samples, List):
            self.samples = torch.stack(self.samples)
            self.targets = torch.stack(self.targets)

            if self.samples.ndim == 3:
                self.samples = self.samples.unsqueeze(0)  # add batch dimension in case it's only one sample
                self.targets = self.targets.unsqueeze(0)

        if self.task == "segmentation":
            self._log_segmentation_prediction(trainer, pl_module)
        else:
            self._log_reconstruction_prediction(trainer, pl_module)

        pl_module.train()