

import csv
import logging
import os
from argparse import ArgumentParser
from collections import defaultdict

import einops
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from constants import LESION_SIZES
from dataset import MRIDataModule
from model import StrokeFormer
from utils import generate_overlayed_slice, get_lesion_size_category, build_metrics, compute_metrics, slice_wise_fp_fn

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SemanticSegmentationTarget:
    def __init__(self, mask):
        if isinstance(mask, torch.Tensor):
            self.mask = mask
        else:
            self.mask = torch.from_numpy(mask)

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        if model_output.shape[-3] != self.mask.shape[-3]:
            self.mask = torch.nn.functional.pad(self.mask,
                                                (0, 0, 0, 0, 0, model_output.shape[-3] - self.mask.shape[-3]))
        return (model_output * self.mask).sum()


def test(args):
    logger.info("=== Starting test evaluation ===")

    logger.info(f"Loading model checkpoint from {args.ckpt_path}")
    model = StrokeFormer.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name or "_".join(args.ckpt_path.split(os.path.sep)[-1].split("-")[:2])
    logger.info(f"Using model name: {model_name}")

    cam_model = None
    if args.target_layers:
        logger.info(f"Setting up GradCAM for layers: {args.target_layers}")
        target_layers = []
        for layer_path in args.target_layers:
            layer = model
            for attr in layer_path.split("."):
                layer = getattr(layer, attr)
            target_layers.append(layer)
        cam_model = GradCAM(model, target_layers=target_layers)

    if len(args.scans) == 1 and os.path.isdir(args.scans[0]):
        args.scans = args.scans[0]
    if len(args.masks) == 1 and os.path.isdir(args.masks[0]):
        args.masks = args.masks[0]

    datamodule = MRIDataModule(
        paths={"test": {"scans": args.scans, "masks": args.masks}},
        batch_size=args.batch_size,
        overlap=None,
        num_workers=args.num_workers
    )
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    logger.info(f"Loaded test dataloader: {len(dataloader.dataset)} volumes")

    os.makedirs(args.scores_dir, exist_ok=True)
    model_prediction_dir = os.path.join(args.scores_dir, model_name, "predictions")

    os.makedirs(model_prediction_dir, exist_ok=True)

    logger.info(f"Predictions will be saved to {model_prediction_dir}")

    per_size_scores = {size: defaultdict(lambda: {"sum": 0.0, "count": 0}) for size in LESION_SIZES}
    global_scores = defaultdict(lambda: {"sum": 0.0, "count": 0})

    roi_size = (args.subvolume_dim, *args.resize_to) if args.resize_to else (args.subvolume_dim, *args.scan_dim[3:])
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=args.batch_size,
        overlap=(args.overlap, 0, 0),
        progress=True,
        mode="gaussian"
    )

    metrics_fns = build_metrics(num_classes=args.num_classes)

    logger.info(f"SlidingWindowInferer configured with ROI {roi_size} and overlap {(args.overlap, 0, 0)}")

    logger.info("Starting inference over test set...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Sliding window inference")):
        scans, masks = batch
        scans, masks = scans.to(model.device), masks.to(model.device)

        with torch.no_grad():
            preds = inferer(scans, model)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        grayscale_cam = torch.zeros_like(preds[0])

        if cam_model is not None:
            targets = [SemanticSegmentationTarget(mask) for mask in masks]
            grayscale_cam = cam_model(scans, targets, eigen_smooth=False)  # noqa
            grayscale_cam = torch.from_numpy(grayscale_cam)
            del targets

        predictions_until_now = batch_idx * args.batch_size
        for i in range(preds.shape[0]):
            if args.n_predictions > predictions_until_now:
                pred_dir = os.path.join(model_prediction_dir, f"scan_{predictions_until_now + i}")  # noqa
                os.makedirs(pred_dir, exist_ok=True)

            for slice_idx in range(preds.shape[-3]):
                scan_slice = scans[i][0, slice_idx]
                mask_slice = masks[i][0, slice_idx]
                pred_slice = preds[i][0, slice_idx]

                gt, scan_slice, _ = generate_overlayed_slice(scan_slice, mask_slice, color=(0, 255, 0), return_tensor=True, return_rgbs=True)
                pd = generate_overlayed_slice(scan_slice, pred_slice, color=(255, 0, 0), return_tensor=True)

                # compute per size metrics
                lesion_size = get_lesion_size_category(mask_slice)
                scores = compute_metrics(pred_slice, mask_slice, metrics_fns, lesions_only=False)
                tp_fp_dict = slice_wise_fp_fn(pred_slice, mask_slice)
                scores = {**scores, **tp_fp_dict}

                for metric_name, value in scores.items():
                    per_size_scores[lesion_size][metric_name]["sum"] += value
                    per_size_scores[lesion_size][metric_name]["count"] += 1
                    # exclude from global metrics as free-lesion slices are in majority and could push
                    # the overall scores much higher than they actually are
                    if lesion_size != "No Lesion":
                        global_scores[metric_name]["sum"] += value
                        global_scores[metric_name]["count"] += 1

                if args.n_predictions > predictions_until_now:
                    images = [gt, pd]

                    if cam_model is not None:
                        cam_overlay = torch.from_numpy(show_cam_on_image(scan_slice / 255,
                                                                         grayscale_cam[i, slice_idx],
                                                                         use_rgb=True))
                        cam_overlay = einops.rearrange(cam_overlay, "h w c -> c h w")
                        images.append(cam_overlay)
                        del cam_overlay

                    grid = make_grid(images, nrow=len(images))
                    to_pil_image(grid).save(os.path.join(pred_dir, f"{lesion_size.replace(' ', '_')}_{predictions_until_now + i}_{slice_idx}.png"))  # noqa
                    del grid

                # free memory
                del gt, pd, scan_slice, mask_slice, pred_slice
                torch.cuda.empty_cache()

    logger.info("Inference complete. Computing metrics...")

    local_metrics = {}
    for size, metric_dict in per_size_scores.items():
        local_metrics[size] = {metric: (values["sum"] / values["count"] if values["count"] > 0 else 0.0)
                               for metric, values in metric_dict.items()}

    global_metrics = {metric: round(values["sum"] / values["count"], 4) if values["count"] > 0 else 0.0
                      for metric, values in global_scores.items()}
    global_metrics = {"model_name": model_name, **global_metrics}

    scores_path = os.path.join(args.scores_dir, args.scores_file)
    per_size_path = os.path.join(args.scores_dir, args.per_size_scores_file)

    logger.info(f"Writing global metrics to: {scores_path}")
    file_exists = os.path.exists(scores_path)
    with open(scores_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(global_metrics)

    logger.info(f"Writing per-size metrics to: {per_size_path}")
    file_exists = os.path.exists(per_size_path)
    with open(per_size_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_name", "size"] + list(next(iter(local_metrics.values())).keys()))
        for size, metric_dict in local_metrics.items():
            row = [model_name, size] + [round(v, 4) for v in metric_dict.values()]
            writer.writerow(row)

    logger.info(f"=== Test evaluation completed ===")
    logger.info(f"Global metrics: {global_metrics}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scans", type=str, required=True, nargs="+")
    parser.add_argument("--masks", type=str, default=None, nargs="+")
    parser.add_argument("--subvolume_dim", type=int, default=189)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scan_dim", nargs=4, type=int, default=(1, 189, 192, 192))
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--target_layers", nargs="+", default=None)

    parser.add_argument("--n_predictions", help="Number of predictions to save", type=int, default=30)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--scores_dir", type=str, default="./scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--per_size_scores_file", type=str, default="per_size_scores.csv")

    test(parser.parse_args())