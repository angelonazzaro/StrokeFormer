import json
import logging
import random
import subprocess
import os

from argparse import ArgumentParser
from glob import glob

import pandas as pd
import wandb

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from callbacks import LogPrediction
from models import StrokeFormer
from dataset import SegmentationDataModule

from utils import get_lesion_size_distribution_metadata, round_half_up, get_lesion_size_distribution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args):
    if sum(args.splits) != 100:
        raise ValueError("The sum of splits must be exactly 100.")

    logger.info("=" * 80)
    logger.info(f"Starting Cross-Validation with {args.k} folds - Model: {args.model_prefix}")

    seed_everything(args.seed)
    random.seed(args.seed)

    logger.info(f"Random seed set to {args.seed}")
    logger.info("=" * 80)

    scans_filepaths = sorted(glob(os.path.join(args.scans_dir, f"*T1w*{args.ext}"), recursive=True)) # noqa
    masks_filepaths = sorted(glob(os.path.join(args.masks_dir, f"*T1lesion_mask*{args.ext}"), recursive=True)) # noqa

    sizes_distribution = get_lesion_size_distribution(scans_filepaths, masks_filepaths)
    sizes_distribution = {size: sizes_distribution[size]["percentage"] for size in sizes_distribution.keys()}
    sizes_distribution_metadata = get_lesion_size_distribution_metadata(scans_filepaths, masks_filepaths)

    dataframe = [{"size": size, "filepath": fp[1]} for size in sizes_distribution.keys() for fp in sizes_distribution_metadata[size]["filepaths"]]
    df = pd.DataFrame(dataframe)

    # compute volume-level distribution
    volume_dist = (
        df.groupby("filepath")["size"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # ensure 'filepath' is preserved
    volume_dist.to_csv("volume_dist.csv")
    volume_dist = pd.read_csv("volume_dist.csv")
    os.remove("volume_dist.csv")

    volume_dist = volume_dist.sample(frac=1, random_state=args.seed)

    total_slices = len(volume_dist) * args.scan_dim[-3]
    slices_per_fold = total_slices // args.k

    logger.info(f"Total scans: {len(volume_dist)}, Total slices: {total_slices}")

    logger.info("Assigning scans to folds...")

    assigned = set()
    folds = []

    # manually split the data into folds to approximately reach each lesion size threshold
    for fold_idx in range(args.k):
        fold_target = {k: slices_per_fold * v // 100 for k, v in sizes_distribution.items()}

        for filepath, row in volume_dist.iterrows():
            if filepath in assigned:
                continue

            contrib = {k: row.get(k, 0) * args.scan_dim[-3] for k in sizes_distribution.keys()}
            if any(v > 0 for v in fold_target.values()):
                folds.append(filepath)
                assigned.add(filepath)
                for k in fold_target.keys():
                    fold_target[k] = max(0, fold_target[k] - contrib[k])

    random.shuffle(folds)

    n_val_folds = round_half_up(len(folds) * args.splits[1] / 100)
    n_test_folds = round_half_up(len(folds) * args.splits[2] / 100)
    logger.info(f"Fold distribution — Train: {args.splits[0]}%, Val: {args.splits[1]}%, Test: {args.splits[2]}%")

    for k in range(args.k):
        logger.info("=" * 80)
        logger.info(f"[Fold {k + 1}/{args.k}] Starting training and evaluation")
        logger.info("=" * 80)

        # sample test/val folds
        test_folds = random.sample(folds, k=n_test_folds)
        remaining_folds = [f for f in folds if f not in test_folds]

        val_folds = random.sample(remaining_folds, k=n_val_folds)
        train_folds = [f for f in remaining_folds if f not in val_folds]

        train_masks_paths = list(volume_dist.loc[train_folds].filepath)
        val_masks_paths = list(volume_dist.loc[val_folds].filepath)
        test_masks_paths = list(volume_dist.loc[test_folds].filepath)

        def _make_paths(filepaths):
            masks = [fp for fp in filepaths]
            scans = [fp.replace(args.masks_dir, args.scans_dir).replace("label-L_desc-T1lesion_mask", "T1w") for fp in
                     filepaths]

            return {"scans": scans, "masks": masks}

        datamodule = SegmentationDataModule(
            paths={
                "train": _make_paths(train_masks_paths),
                "val": _make_paths(val_masks_paths),
            },
            resize_to=args.resize_to,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            overlap=args.overlap,
            subvolume_depth=args.subvolume_depth,
            transforms=args.transforms,
            augment=True
        )

        seg_loss_cfg = json.loads(args.seg_loss_cfg)
        cls_loss_cfg = json.loads(args.cls_loss_cfg)

        model = StrokeFormer(num_classes=args.num_classes, seg_loss=args.seg_loss,
                             seg_loss_cfg=seg_loss_cfg, cls_loss=args.cls_loss,
                             cls_loss_cfg=cls_loss_cfg, loss_weights=args.loss_weights,
                             opt_lr=args.opt_lr, warmup_lr=args.warmup_lr, max_lr=args.max_lr,
                             weight_decay=args.weight_decay, betas=args.betas, eps=args.eps,
                             lr_scheduler_interval=args.lr_logging_interval,
                             in_channels=args.in_channels)

        callbacks = [
            LogPrediction(num_samples=args.num_samples,
                          log_every_n_epochs=args.log_every_n_val_epochs, task="segmentation"),
            LearningRateMonitor(logging_interval=args.lr_logging_interval),
            EarlyStopping(
                patience=args.patience,
                monitor="val_loss",
                min_delta=args.min_delta,
                verbose=False,
                mode="min",
            ),
            ModelCheckpoint(
                filename=f"{args.model_prefix}-fold-{k + 1}-{{epoch:02d}}-{{val_loss:.2f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]

        wandb_logger = WandbLogger(
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            group=args.group,
            name=f"{args.model_prefix}-fold-{k + 1}",
        )

        trainer = Trainer(
            callbacks=callbacks,
            logger=wandb_logger,
            max_epochs=args.max_epochs,
            default_root_dir=args.default_root_dir,
        )

        logger.info(f"[Fold {k + 1}/{args.k}] Starting Training...")
        trainer.fit(model, datamodule=datamodule)
        wandb.finish()
        logger.info(f"[Fold {k + 1}/{args.k}] Training Ended...")

        logger.info(f"[Fold {k + 1}/{args.k}] Starting Testing...")

        # get last model checkpoint from the current run
        ckpt_dir = os.path.join(args.default_root_dir, trainer.logger.experiment.id, "checkpoints") # noqa
        ckpt_path = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][-1] # noqa

        test_paths = _make_paths(test_masks_paths)

        cmd = [
            "python", "test_strokeformer.py",
            "--seed", str(args.seed),
            "--batch_size", str(1),
            "--scans", *test_paths["scans"],
            "--masks", *test_paths["masks"],
            "--ckpt_path", ckpt_path,
            "--model_name", f"{args.model_prefix}-{k + 1}-fold",
            "--subvolume_depth", str(args.subvolume_depth),
            "--scores_dir", "./cross_scores",
            "--scores_file", "cross_scores.csv",
            "--per_size_scores_file", "per_size_cross_scores.csv",
        ]

        if args.resize_to is not None:
            cmd.extend(["--resize_to", str(args.resize_to[0]), str(args.resize_to[1])])

        if args.target_layers is not None:
            cmd.append("--target_layers")
            for target_layer in args.target_layers:
                cmd.append(target_layer)

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Cross-validation pipeline for StrokeFormer training")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--splits", nargs=3, type=int, default=(70, 10, 20), help="Percentage split (train, val, test)")

    # dataloader
    parser.add_argument("--scans_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, default=".npy")
    parser.add_argument("--transforms", nargs="+", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subvolume_depth", type=int, default=189)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)
    parser.add_argument("--scan_dim", nargs=4, type=int, default=(1, 189, 192, 192))

    # models
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--opt_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_lr", type=float, default=4e-6)
    parser.add_argument("--max_lr", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--seg_loss", type=str, default="ICILoss")
    parser.add_argument("--seg_loss_cfg", type=str, default="{}")
    parser.add_argument("--cls_loss", type=str, default="BCEWithLogitsLoss")
    parser.add_argument("--cls_loss_cfg", type=str, default="{}")
    parser.add_argument("--loss_weights", nargs=2, type=float, default=(0.5, 0.5))

    # training
    parser.add_argument("--default_root_dir", type=str, default="StrokeFormer")
    parser.add_argument("--project", type=str, default="StrokeFormer")
    parser.add_argument("--entity", type=str, default="neurone-lab")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--lr_logging_interval", type=str, default="epoch", choices=["epoch", "step"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--log_every_n_val_epochs", type=int, default=5)
    parser.add_argument("--model_prefix", type=str, required=True)

    # testing
    parser.add_argument("--target_layers", help="Target layers for Grad-CAM. If None, Grad-CAM will not be executed.",
                        nargs="+", default=None)
    parser.add_argument("--scores_dir", help="Directory to save predictions and scores.", type=str, default="./scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")

    main(parser.parse_args())
