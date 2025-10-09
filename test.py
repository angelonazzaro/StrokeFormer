import csv
import os
import logging
import time
from argparse import ArgumentParser
from collections import defaultdict

import einops
import numpy as np
import torch
from pytorch_grad_cam import GradCAM

from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from constants import LESION_SIZES
from dataset import MRIDataModule
from model import StrokeFormer
from utils import predictions_generator


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def test(args):
    start_time = time.time()
    logger.info("=== Starting test evaluation ===")

    logger.info(f"Loading model checkpoint from {args.ckpt_path}")
    model = StrokeFormer.load_from_checkpoint(args.ckpt_path)
    model.eval()

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
        subvolume_dim=args.subvolume_dim,
        overlap=args.overlap,
        resize_to=args.resize_to,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    logger.info(f"Loaded test dataloader: {len(dataloader.dataset)} samples")

    os.makedirs(args.scores_dir, exist_ok=True)
    model_prediction_dir = os.path.join(args.scores_dir, model_name, "predictions")
    model_cam_dir = os.path.join(args.scores_dir, model_name, "gradcam")
    os.makedirs(model_prediction_dir, exist_ok=True)
    if cam_model is not None:
        os.makedirs(model_cam_dir, exist_ok=True)

    logger.info(f"Predictions will be saved to {model_prediction_dir}")
    if cam_model:
        logger.info(f"GradCAM visualizations will be saved to {model_cam_dir}")

    per_size_scores = {size: defaultdict(list) for size in LESION_SIZES}
    local_metrics = {size: defaultdict(float) for size in LESION_SIZES}
    global_metrics = defaultdict(list)

    total_batches = len(dataloader)
    logger.info(f"Starting testing on {total_batches} batches")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting lesions")):
        scans, masks = batch
        scans, masks = scans.to(model.device), masks.to(model.device)
        pred_dir = os.path.join(model_prediction_dir, f"scan_{batch_idx}")
        grad_dir = os.path.join(model_cam_dir, f"scan_{batch_idx}")
        os.makedirs(pred_dir, exist_ok=True)

        if cam_model:
            os.makedirs(grad_dir, exist_ok=True)

        for j, result in enumerate(predictions_generator(model=model, scans=scans, masks=masks, metrics=model.metrics, cam_model=cam_model)):
            lesion_size = result["lesion_size"]

            for metric_name, value in result["scores"].items():
                per_size_scores[lesion_size][metric_name].append(value)
                global_metrics[metric_name].append(value)

            if args.n_predictions > batch_idx:
                gt, pd = map(torch.tensor, (result["gt"], result["pd"]))
                gt, pd = map(lambda x: einops.rearrange(x, "h w c -> c h w"), (gt, pd))

                grid = make_grid([gt, pd], nrow=2)
                to_pil_image(grid).save(os.path.join(pred_dir, f"{lesion_size.replace(' ', '_')}_{batch_idx}_{j}.png"))

                if cam_model:
                    cam_image = torch.tensor(result["cam_image"])
                    cam_image = einops.rearrange(cam_image, "h w c -> c h w")
                    grid = make_grid([gt, pd, cam_image], nrow=3)
                    to_pil_image(grid).save(os.path.join(grad_dir, f"{lesion_size.replace(' ', '_')}_{batch_idx}_{j}.png"))

    logger.info(f"Testing complete.")

    for size, metric_dict in per_size_scores.items():
        for metric, values in metric_dict.items():
            local_metrics[size][metric] = float(np.mean(values))

    global_metrics = {metric: round(float(np.mean(values)), 4) for metric, values in global_metrics.items()}
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

    elapsed = time.time() - start_time
    logger.info(f"=== Test evaluation complete in {elapsed:.2f}s ===")
    logger.info(f"Global metrics: {global_metrics}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument('--scans', type=str, required=True, nargs='+')
    parser.add_argument('--masks', type=str, default=None, nargs='+')
    parser.add_argument('--subvolume_dim', help='subvolume dimension for training', type=int, default=189)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scan_dim", nargs=4, type=int, default=(1, 189, 192, 192))
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name. If None, it will be inferred from the ckpt_path.")
    parser.add_argument("--target_layers", help="Target layers for Grad-CAM. If None, Grad-CAM will not be executed.",
                        nargs="+", default=None)

    parser.add_argument("--n_predictions", help="Number of predictions to save", type=int, default=30)
    parser.add_argument("--scores_dir", help="Directory to save predictions and scores.", type=str, default="./scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--per_size_scores_file", type=str, default="per_size_scores.csv")

    test(parser.parse_args())