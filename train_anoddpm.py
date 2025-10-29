import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from dataset.mri_dataloader import ReconstructionDataModule
from models import AnoDDPM


class MyCLI(LightningCLI):
    def after_instantiate_classes(self, **kwargs):
        model = self.model
        logger = self.trainer.logger

        if isinstance(logger, WandbLogger):
            logger.watch(
                model=model,
                log="all",  # gradients + parameter histograms + models topology
                log_freq=100,
                log_graph=False,  # in case models graph is large or breaks wandb
            )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    MyCLI(
        model_class=AnoDDPM,
        datamodule_class=ReconstructionDataModule,
        save_config_kwargs={"overwrite": True},
    )