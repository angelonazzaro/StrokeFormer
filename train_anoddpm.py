import os

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
            logger.experiment.config.update(dict(self.config))
            self.save_config_kwargs["config_filename"] = os.path.join(self.trainer.default_root_dir, logger.experiment.id,
                                                                  "anoddpm_config.yaml")
            os.makedirs(os.path.join(self.trainer.default_root_dir, logger.experiment.id), exist_ok=True)
            # reinstantiate trainer
            extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
            trainer_config = {**self._get(self.config_init, "trainer", default={}), **kwargs}
            self.trainer = self._instantiate_trainer(trainer_config, extra_callbacks) # noqa


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    MyCLI(
        model_class=AnoDDPM,
        datamodule_class=ReconstructionDataModule,
        save_config_kwargs={"overwrite": True},
    )