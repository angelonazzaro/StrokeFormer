import csv
import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial

import einops
import torch
from monai.inferers import SlidingWindowInferer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from constants import LESION_SIZES
from dataset import SegmentationDataModule
from models import StrokeFormer
from utils import build_metrics, get_per_slice_segmentation_preds, convert_to_rgb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class SemanticSegmentationTarget:
    def __init__(self, mask):
        if isinstance(mask, torch.Tensor):
            self.mask = mask
        else:
            self.mask = torch.from_numpy(mask)

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output * self.mask).sum()


def test(args):
    logger.info("=== Starting test evaluation ===")

    model = StrokeFormer.load_from_checkpoint(args.ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.cudnn.benchmark else "cpu"
    model = model.to(device=device)
    model.eval()
    model_name = args.model_name or "_".join(args.ckpt_path.split(os.path.sep)[-1].split("-")[:2])

    logger.info(f"Loading model {model_name} from {args.ckpt_path}")

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

    datamodule = SegmentationDataModule(
        paths={"test": {"scans": args.scans, "masks": args.masks}},
        batch_size=args.batch_size,
        overlap=None,
        num_workers=args.num_workers,
    )

    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    logger.info(f"Loaded test dataloader: {len(dataloader.dataset)} volumes")

    roi_size = (args.subvolume_depth, *args.resize_to) if args.resize_to else (args.subvolume_depth, *args.scan_dim[2:])
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=args.batch_size,
        overlap=(args.overlap, 0, 0),
        progress=True,
        mode="gaussian"
    )

    inferer = partial(inferer, network=model)

    metrics = build_metrics(num_classes=args.num_classes)

    logger.info(f"SlidingWindowInferer configured with ROI {roi_size} and overlap {(args.overlap, 0, 0)}")
    logger.info("=== Starting inference over test set ===")

    # scores as saved as cumulative average (CA)
    # structure is:
    # - size:
    #   - metric:
    #       - ca: 0.0
    #         n: 0.0
    per_size_scores = {}
    for size in LESION_SIZES:
        per_size_scores[size] = {}
        for metric_name in metrics.keys():
            per_size_scores[size][metric_name] = {
                "ca": 0.0,
                "n": 0,
            }

    os.makedirs(args.scores_dir, exist_ok=True)
    model_prediction_dir = os.path.join(args.scores_dir, model_name, "predictions")

    os.makedirs(model_prediction_dir, exist_ok=True)

    logger.info(f"Predictions will be saved to {model_prediction_dir}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Segmenting lesions")):
        scans, masks, means, stds = batch["scans"], batch["masks"], batch["means"], batch["stds"]
        scans, masks = scans.to(device=model.device), masks.to(device=model.device)
        means, stds = means.to(device=model.device), stds.to(device=model.device)

        preds_until_now = batch_idx * scans.shape[0]
        grayscale_cam = torch.zeros_like(scans[0], device=model.device)

        if cam_model is not None and args.n_predictions > preds_until_now:
            targets = [SemanticSegmentationTarget(mask) for mask in masks]
            grayscale_cam = cam_model(scans, targets, eigen_smooth=False)  # noqa
            grayscale_cam = grayscale_cam  # (B, D, H, W)

        for result in get_per_slice_segmentation_preds(inferer, scans, masks, metrics, means, stds):
            ground_truth = torch.from_numpy(result["ground_truth"]).to(device=model.device)  # (H, W, C)
            prediction = torch.from_numpy(result["prediction"]).to(device=model.device)

            ground_truth = einops.rearrange(ground_truth, "h w c -> c h w")
            prediction = einops.rearrange(prediction, "h w c -> c h w")

            lesion_size = result["lesion_size"]
            scores = result["scores"]

            # CA update rule: (x_n+1 + n * CA_n) / (n + 1)
            for metric_name in metrics.keys():
                curr_ca = per_size_scores[lesion_size][metric_name]["ca"]
                curr_n = per_size_scores[lesion_size][metric_name]["n"]
                per_size_scores[lesion_size][metric_name] = {
                    "ca": (scores[metric_name] + curr_n * curr_ca) / (curr_n + 1),
                    "n": curr_n + 1
                }

            if args.n_predictions > preds_until_now:
                images = [ground_truth, prediction]
                scan_idx = result["scan_idx"]
                slice_idx = result["slice_idx"]

                if cam_model is not None:
                    scan_slice = scans[scan_idx][:, slice_idx]
                    scan_slice = convert_to_rgb(scan_slice) / 255
                    cam_slice = grayscale_cam[scan_idx, slice_idx]
                    cam_overlay = show_cam_on_image(scan_slice, cam_slice, use_rgb=True)
                    cam_overlay = torch.from_numpy(cam_overlay).to(device=model.device)
                    cam_overlay = einops.rearrange(cam_overlay, "h w c -> c h w")
                    images.append(cam_overlay)

                grid = make_grid(images, padding=4, pad_value=255, nrow=len(images))
                grid = to_pil_image(grid)
                lesion_size = lesion_size.replace(" ", "_")
                preds_dir = os.path.join(model_prediction_dir, f"scan_{preds_until_now + scan_idx}")  # noqa
                os.makedirs(preds_dir, exist_ok=True)
                grid.save(os.path.join(preds_dir, f"{lesion_size}_{slice_idx}.png"))  # noqa

            torch.cuda.empty_cache()

    logger.info("Inference complete. Aggregating metrics...")
    global_metrics = defaultdict(float)

    for size in per_size_scores.keys():
        for metric_name in metrics.keys():
            # do not include healthy slices
            if size != "No Lesion":
                global_metrics[metric_name] += per_size_scores[size][metric_name]["ca"]

    for metric_name, metric_value in global_metrics.items():
        global_metrics[metric_name] = round(metric_value / (len(per_size_scores.keys()) - 1), 4)

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
            writer.writerow(["model_name", "size"] + list(next(iter(per_size_scores.values())).keys()))
        for size, metric_dict in per_size_scores.items():
            row = [model_name, size] + [round(v["ca"], 4) for v in metric_dict.values()]
            writer.writerow(row)

    logger.info(f"=== Test evaluation completed ===")
    logger.info(f"Global metrics: {global_metrics}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scans", type=str, required=True, nargs="+")
    parser.add_argument("--masks", type=str, default=None, nargs="+")
    parser.add_argument("--subvolume_depth", type=int, default=189)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scan_dim", nargs=4, type=int, default=(1, 189, 192, 192))
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--target_layers", nargs="+", default=None)

    parser.add_argument("--n_predictions", help="Number of predictions to save", type=int, default=30)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--scores_dir", type=str, default="./scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--per_size_scores_file", type=str, default="per_size_scores.csv")

    test(parser.parse_args())