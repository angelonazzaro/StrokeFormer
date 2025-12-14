import json
import logging
import os
import random
import subprocess
from argparse import ArgumentParser
from glob import glob

import shutil
import pandas as pd
import numpy as np
import csv
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
from utils import resize

import wandb
from callbacks import LogPrediction
from dataset import SegmentationDataModule, RegionProposalDataModule
from models import StrokeFormer, RPN
from utils import get_lesion_size_distribution_metadata, round_half_up, get_lesion_size_distribution, propose_regions, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(model, datamodule, model_prefix, project, entity, offline, group, patience, min_delta, k, num_samples,
          log_every_n_val_epochs, lr_logging_interval, devices, max_epochs, default_root_dir, task):
    callbacks = [
        LogPrediction(num_samples=num_samples,
                      log_every_n_epochs=log_every_n_val_epochs, task=task),
        LearningRateMonitor(logging_interval=lr_logging_interval),
        EarlyStopping(
            patience=patience,
            monitor="val_loss",
            min_delta=min_delta,
            verbose=False,
            mode="min",
        ),
        ModelCheckpoint(
            filename=f"{model_prefix}-fold-{k + 1}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        offline=offline,
        group=group,
        name=f"{model_prefix}-fold-{k + 1}",
    )

    trainer = Trainer(
        devices=devices,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()

    return trainer.logger.experiment.id # noqa

def decompose_3d_volumes(volumes_paths, base_dir, k, K, split):
    OUTPUT_SCAN2D_DIR = os.path.join(base_dir, f"FOLD-2D-{k + 1}", split, "Scans")
    OUTPUT_MASK2D_DIR = os.path.join(base_dir, f"FOLD-2D-{k + 1}", split, "Masks")

    os.makedirs(OUTPUT_SCAN2D_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK2D_DIR, exist_ok=True)

    for i in tqdm(range(len(volumes_paths)),
                  desc=f"Decomposing scans for region proposal task {k + 1}/{K}"):
        scan = np.load(volumes_paths["scans"][i])  # shape: (C, H, W, D)
        mask = np.load(volumes_paths["scans"][i])  # shape: (C, H, W, D)

        scan_filepath_basename = os.path.basename(volumes_paths["scans"][i])
        masks_filepath_basename = os.path.basename(volumes_paths["masks"][i])

        for j in range(scan.shape[-1]):
            output_scan2d_filepath = scan_filepath_basename.replace(".npy", f"_slice_{j}.npy")
            output_mask2d_filepath = masks_filepath_basename.replace(".npy", f"_slice_{j}.npy")
            np.save(os.path.join(OUTPUT_SCAN2D_DIR, output_scan2d_filepath), scan[..., j]) # noqa
            np.save(os.path.join(OUTPUT_MASK2D_DIR, output_mask2d_filepath), mask[..., j]) # noqa

    return OUTPUT_SCAN2D_DIR, OUTPUT_MASK2D_DIR


def main(args):
    if sum(args.splits) != 100:
        raise ValueError("The sum of splits must be exactly 100.")

    logger.info("=" * 80)
    logger.info(f"Starting Cross-Validation with {args.k} folds - Model: {args.model_prefix}")

    seed_everything(args.seed)
    random.seed(args.seed)

    logger.info(f"Random seed set to {args.seed}")
    logger.info("=" * 80)

    scans_filepaths = sorted(glob(os.path.join(args.scans_dir, f"*T1w*{args.ext}"), recursive=True))  # noqa
    masks_filepaths = sorted(glob(os.path.join(args.masks_dir, f"*T1lesion_mask*{args.ext}"), recursive=True))  # noqa

    sizes_distribution = get_lesion_size_distribution(scans_filepaths, masks_filepaths)
    sizes_distribution = {size: sizes_distribution[size]["percentage"] for size in sizes_distribution.keys()}
    sizes_distribution_metadata = get_lesion_size_distribution_metadata(scans_filepaths, masks_filepaths)

    dataframe = [{"size": size, "filepath": fp[1]} for size in sizes_distribution.keys() for fp in
                 sizes_distribution_metadata[size]["filepaths"]]
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

    volume_dist.sample(frac=1, random_state=args.seed)

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

        if args.use_augmented:
            # replace training normal scans and masks with augmented ones
            for i in range(len(train_masks_paths)):
                mask_filepath = train_masks_paths[i]

                aug_mask_filepath = mask_filepath.replace(args.masks_dir, args.augmented_dir + "/Masks").replace(
                    "T1lesion_mask",
                    "T1lesion_mask_augmented")
                if os.path.exists(aug_mask_filepath):
                    train_masks_paths[i] = aug_mask_filepath

        def _make_paths(filepaths):
            masks = [fp for fp in filepaths]

            scans = [fp.replace("Masks", "Scans").replace("label-L_desc-T1lesion_mask", "T1w") for fp in
                     filepaths]

            return {"scans": scans, "masks": masks}

        # right now, these contain 3D volumes, in order to use them for training the RPN we need to split them into their
        # 2D slices
        train_paths = _make_paths(train_masks_paths)
        val_paths = _make_paths(val_masks_paths)
        test_paths = _make_paths(test_masks_paths)

        base_dir = os.path.dirname(train_paths["scans"][0])
        train_scans_2d_dir, train_masks_2d_dir = decompose_3d_volumes(train_paths, base_dir, k, args.k, "train")
        val_scans_2d_dir, val_masks_2d_dir = decompose_3d_volumes(val_paths, base_dir, k, args.k, "val")
        test_scans_2d_dir, test_masks_2d_dir = decompose_3d_volumes(test_paths, base_dir, k, args.k, "test")

        dataset_type = args.dataset_anchors.split("-")[0]

        rpn = RPN(lr=args.rpn_lr, weight_decay=args.rpn_weight_decay, betas=args.rpn_betas, eps=args.rpn_eps,
                  backbone_weights=args.rpn_backbone_weights, dataset_anchors=args.dataset_anchors)
        rp_datamodule = RegionProposalDataModule(
            paths={
                "train": {
                    "scans": train_scans_2d_dir,
                    "masks": train_masks_2d_dir,
                },
                "val": {
                    "scans": val_scans_2d_dir,
                    "masks": val_masks_2d_dir,
                }
            },
            resize_to=args.resize_to,
            num_workers=args.num_workers,
            batch_size=args.rpn_batch_size,
            dataset_type=dataset_type,
            augment=True)

        logger.info(f"[Fold {k + 1}/{args.k}] Starting RPN Training...")
        experiment_id = train(rpn, rp_datamodule, args.rpn_model_prefix, args.rpn_project, args.entity, args.offline,
              args.rpn_group, args.rpn_patience, args.min_delta, k, args.num_samples, args.log_every_n_val_epochs,
              args.lr_logging_interval, args.devices, args.max_epochs, args.rpn_default_root_dir, "region_proposal")

        logger.info(f"[Fold {k + 1}/{args.k}] RPN Training Ended...")

        logger.info(f"[Fold {k + 1}/{args.k}] RPN Starting Testing...")

        # get last model checkpoint from the current run
        ckpt_dir = os.path.join(args.rpn_default_root_dir, experiment_id, "checkpoints")  # noqa
        ckpt_path = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][
            -1]  # noqa

        cmd = [
            "python", "test_rpn.py",
            "--seed", str(args.seed),
            "--batch_size", str(args.rpn_batch_size),
            "--scans", test_scans_2d_dir,
            "--masks", test_masks_2d_dir,
            "--ckpt_path", ckpt_path,
            "--model_name", f"{args.rpn_model_prefix}-{k + 1}-fold",
            "--scores_dir", "./rpn_cross_scores",
            "--scores_file", "cross_scores.csv",
        ]

        if args.resize_to is not None:
            cmd.extend(["--resize_to", str(args.resize_to[0]), str(args.resize_to[1])])

        subprocess.run(cmd, check=True)

        # save region proposals to speedup segformer training and testing
        csv_file = f"regions_{dataset_type}_{args.roi_size[1:]}.csv"
        headers = ["scan_name", "region_start", "region_end", "xmin", "ymin", "xmax", "ymax"]
        stages = ["train", "val", "test"]

        rpn_model = RPN.load_from_checkpoint(ckpt_path)
        rpn_model = rpn_model.to(get_device())
        rpn_model.eval()

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"')
            writer.writerow(headers)

            segmentation_datamodule = SegmentationDataModule(
                paths={
                    "train": train_paths,
                    "val": val_paths,
                    "test": test_paths
                },
                resize_to=args.resize_to,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                overlap=args.overlap,
                subvolume_depth=args.subvolume_depth,
                transforms=args.transforms,
                augment=True
            )

            for stage in stages:
                segmentation_datamodule.setup("fit" if stage != "test" else stage)
                data_set = getattr(segmentation_datamodule, f"{stage}_set")

                for idx, data in tqdm(enumerate(data_set), total=len(data_set), desc=f"Proposing regions for {stage} stage - {k + 1}/{args.k}"):
                    scan, head_mask = data["scan"].to(rpn_model.device), data["head_mask"].to(rpn_model.device)
                    scan, head_mask = resize(scan.unsqueeze(0), head_mask.unsqueeze(0), *args.resize_to)
                    scan, head_mask = scan.squeeze(0), head_mask.squeeze(0)
                    for final_box, group_dict in propose_regions(rpn_model, scan, head_mask, args.roi_size):
                        start, end = group_dict["region_start"], group_dict["region_end"]
                        xmin, ymin, xmax, ymax = round_half_up(final_box[:, 0].item()), round_half_up(
                            final_box[:, 1].item()), round_half_up(final_box[:, 2].item()), round_half_up(
                            final_box[:, 3].item())

                        writer.writerow([os.path.basename(data_set.scans[idx]), start, end, xmin, ymin, xmax, ymax])

        df = pd.read_csv(csv_file)

        # Group rows by scan_name and convert each group to a list of dicts
        grouped = (
            df.groupby("scan_name")
            .apply(lambda g: g.drop(columns=["scan_name"]).to_dict(orient="records"))
            .to_dict()
        )

        os.remove(csv_file)

        regions_json_file = f"regions_grouped_{dataset_type}_{args.roi_size[1:]}.json"
        with open(regions_json_file, "w") as f:
            json.dump(grouped, f, indent=2)

        # remove 2d paths to save space
        shutil.rmtree(train_scans_2d_dir)
        shutil.rmtree(train_masks_2d_dir)
        shutil.rmtree(val_scans_2d_dir)
        shutil.rmtree(val_masks_2d_dir)
        shutil.rmtree(test_scans_2d_dir)
        shutil.rmtree(test_masks_2d_dir)

        # SegFormer
        seg_loss_cfg = json.loads(args.seg_loss_cfg)
        cls_loss_cfg = json.loads(args.cls_loss_cfg)

        segformer = StrokeFormer(num_classes=args.num_classes, seg_loss=args.seg_loss,
                                 seg_loss_cfg=seg_loss_cfg, cls_loss=args.cls_loss,
                                 cls_loss_cfg=cls_loss_cfg, seg_loss_weight=args.seg_loss_weight,
                                 cls_loss_weight=args.cls_loss_weight,
                                 opt_lr=args.opt_lr, warmup_lr=args.warmup_lr, max_lr=args.max_lr,
                                 weight_decay=args.weight_decay, betas=args.betas, eps=args.eps,
                                 lr_scheduler_interval=args.lr_logging_interval,
                                 in_channels=args.in_channels)

        segmentation_datamodule = SegmentationDataModule(
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
            regions=regions_json_file,
            augment=True
        )

        logger.info(f"[Fold {k + 1}/{args.k}] Starting SegFormer Training...")
        experiment_id = train(segformer, segmentation_datamodule, args.model_prefix, args.project, args.entity, args.offline,
              args.group, args.patience, args.min_delta, k, args.num_samples, args.log_every_n_val_epochs,
              args.lr_logging_interval, args.devices, args.max_epochs, args.default_root_dir, "segmentation")
        logger.info(f"[Fold {k + 1}/{args.k}] SegFormer Training Ended...")

        logger.info(f"[Fold {k + 1}/{args.k}] SegFormer Starting Testing...")

        # get last model checkpoint from the current run
        ckpt_dir = os.path.join(args.default_root_dir, experiment_id, "checkpoints")  # noqa
        ckpt_path = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][
            -1]  # noqa

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
            "--regions", regions_json_file,
            "--per_size_scores_file", "per_size_cross_scores.csv",
        ]

        if args.resize_to is not None:
            cmd.extend(["--resize_to", str(args.resize_to[0]), str(args.resize_to[1])])

        if args.target_layers is not None:
            cmd.append("--target_layers")
            for target_layer in args.target_layers:
                cmd.append(target_layer)

        subprocess.run(cmd, check=True)
        os.remove(regions_json_file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Cross-validation pipeline for StrokeFormer training")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--splits", nargs=3, type=int, default=(70, 10, 20), help="Percentage split (train, val, test)")
    parser.add_argument("--use_augmented", default=False, action="store_true",
                        help="Whether to use augmented scans for training")
    parser.add_argument("--augmented_dir", default="data/ATLAS_2/Processed/Augmented/")

    # dataloader
    parser.add_argument("--scans_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, default=".npy")
    parser.add_argument("--transforms", nargs="+", type=str, default=None)
    parser.add_argument("--rpn_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subvolume_depth", type=int, default=189)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)
    parser.add_argument("--scan_dim", nargs=4, type=int, default=(1, 189, 192, 192))

    # SegFormer
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--opt_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_lr", type=float, default=4e-6)
    parser.add_argument("--max_lr", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--roi_size", nargs=3, type=int, default=(64, 64, 64))
    parser.add_argument("--seg_loss", type=str, default="DiceCELoss")
    parser.add_argument("--seg_loss_cfg", type=str, default='{"softmax": true, "squared_pred": true, "include_background": false}')
    parser.add_argument("--cls_loss", type=str, default=None)
    parser.add_argument("--cls_loss_cfg", type=str, default="{}")
    parser.add_argument("--seg_loss_weight", type=float, default=0.5)
    parser.add_argument("--cls_loss_weight", type=float, default=0.5)
    parser.add_argument("--default_root_dir", type=str, default="StrokeFormer")
    parser.add_argument("--project", type=str, default="StrokeFormer")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--model_prefix", type=str, required=True)

    # RPN
    parser.add_argument("--rpn_lr", type=float, default=1e-4)
    parser.add_argument("--rpn_eps", type=float, default=1e-8)
    parser.add_argument("--rpn_betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--rpn_weight_decay", type=float, default=1e-4)
    parser.add_argument("--dataset_anchors", type=str, choices=["ATLAS", "ISLES-DWI", "ISLES-FLAIR"], default="ATLAS")
    parser.add_argument("--rpn_backbone_weights", default=None, type=str, choices=["DEFAULT"])
    parser.add_argument("--rpn_default_root_dir", type=str, default="RPN")
    parser.add_argument("--rpn_project", type=str, default="RPN")
    parser.add_argument("--rpn_max_epochs", type=int, default=250)
    parser.add_argument("--rpn_patience", type=int, default=50)
    parser.add_argument("--rpn_group", type=str, default=None)
    parser.add_argument("--rpn_model_prefix", type=str, required=True)

    # training
    parser.add_argument("--entity", type=str, default="neurone-lab")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--lr_logging_interval", type=str, default="epoch", choices=["epoch", "step"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--log_every_n_val_epochs", type=int, default=5)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])

    # testing
    parser.add_argument("--target_layers", help="Target layers for Grad-CAM. If None, Grad-CAM will not be executed.",
                        nargs="+", default=None)
    parser.add_argument("--scores_dir", help="Directory to save segmentation predictions and scores.", type=str,
                        default="./scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--rpn_scores_dir", help="Directory to save RPN predictions and scores.", type=str,
                        default="./rpn_scores")
    parser.add_argument("--rpn_scores_file", type=str, default="scores.csv")

    main(parser.parse_args())
