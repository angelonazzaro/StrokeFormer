import csv
import os
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


def test(args):
    model = StrokeFormer.load_from_checkpoint(args.ckpt_path)
    model_name = args.model_name

    if model_name is None:
        model_name = args.ckpt_path.split(os.path.sep)[-1].split("-")[:2]

    cam_model = None

    if args.target_layers is not None:
        target_layers = []
        for target_layer in args.target_layers:
            layer = model
            for attr in target_layer.split("."):
                layer = getattr(layer, attr)
            target_layers.append(layer)

        cam_model = GradCAM(model, target_layers=target_layers)

    if len(args.scans) == 1 and os.path.isdir(args.scans[0]):
        args.scans = args.scans[0]

    if len(args.masks) == 1 and os.path.isdir(args.masks[0]):
        args.masks = args.masks[0]

    datamodule = MRIDataModule(paths={"test": {"scans": args.scans, "masks": args.masks}},
                               subvolume_dim=args.subvolume_dim, overlap=args.overlap,
                               resize_to=args.resize_to,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    model.eval()

    os.makedirs(args.scores_dir, exist_ok=True)
    model_prediction_dir = os.path.join(args.scores_dir, model_name, "predictions")
    model_cam_dir = os.path.join(args.scores_dir, model_name, "gradcam")
    os.makedirs(model_prediction_dir, exist_ok=True)

    if cam_model is not None:
        os.makedirs(model_cam_dir, exist_ok=True)

    per_size_scores = {}
    local_metrics = {}
    global_metrics = defaultdict(list)

    # TODO: should I keep the No Lesion?
    for size in LESION_SIZES:
        per_size_scores[size] = defaultdict(list)
        local_metrics[size] = defaultdict(float)

    for batch in tqdm(dataloader, desc="Predicting lesions"):
        scans, masks = batch
        i = 0
        for j, result in enumerate(predictions_generator(model, scans, masks, args.slices_per_scan, model.metrics, cam_model)):
            os.makedirs(os.path.join(model_prediction_dir, f"scan_{i}"), exist_ok=True)

            if cam_model is not None:
                os.makedirs(os.path.join(model_cam_dir, f"scan_{i}"), exist_ok=True)

            for metric in result['scores']:
                per_size_scores[result["lesion_size"]][metric].append(result['scores'][metric])

            if j % args.slices_per_scan == 0 and j > 0:
                i += 1

            if args.n_predictions > 0:
                gt, pd = torch.tensor(result['gt']), torch.tensor(result['pd'])
                gt, pd = einops.rearrange(gt, 'h w c -> c h w'), einops.rearrange(pd, 'h w c -> c h w')

                grid = make_grid([gt, pd], nrow=2, normalize=False, scale_each=False)
                grid_img = to_pil_image(grid)
                grid_img.save(os.path.join(model_prediction_dir, f"scan_{i}", f"{result['lesion_size']}.png"))  # noqa

                if cam_model is not None:
                    cam_image = torch.tensor(result["cam_image"])
                    cam_image = einops.rearrange(cam_image, 'h w c -> c h w')

                    grid = make_grid(torch.stack([gt, pd, cam_image]), nrow=3, normalize=False, scale_each=False)
                    grid_img = to_pil_image(grid)
                    grid_img.save(os.path.join(model_cam_dir, f"scan_{i}", f"gradcam_{result['lesion_size']}.png"))  # noqa

                args.n_predictions -= 1

    scores_path = os.path.join(args.scores_dir, args.scores_file)
    file_exists = os.path.exists(scores_path)

    for size in per_size_scores.keys():
        for metric in per_size_scores[size].keys():
            local_metrics[size][metric] = np.mean(per_size_scores[size][metric])
            global_metrics[metric].extend(per_size_scores[size][metric])

    for metric in global_metrics.keys():
        global_metrics[metric] = round(np.mean(global_metrics[metric]), 4)

    global_metrics = {"model_name": args.model_name, **global_metrics}

    with open(scores_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(global_metrics)

    scores_path = os.path.join(args.scores_dir, args.per_size_scores_file)
    file_exists = os.path.exists(scores_path)

    with open(scores_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model_name", "size"] + list(local_metrics[size].keys()))

        for size, metric_dict in local_metrics.items():
            row = [args.model_name, size] + [round(local_metrics[size][m], 4) for m in metric_dict.keys()]
            writer.writerow(row)


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

    parser.add_argument("--n_predictions", help="Number of predictions to save", type=int, default=100)
    parser.add_argument("--slices_per_scan",
                        help="Number of slices to save from the predictions for quality inspection", type=int,
                        default=20)
    parser.add_argument("--scores_dir", help="Directory to save predictions and scores.", type=str, default="./scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--per_size_scores_file", type=str, default="per_size_scores.csv")

    test(parser.parse_args())