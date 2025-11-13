import csv
import os
from argparse import ArgumentParser

import torch
from lightning import seed_everything
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import ReconstructionDataModule
from models import UNet
from utils import build_metrics, compute_metrics, get_device


def test(args):
    seed_everything(args.seed)

    model = UNet.load_from_checkpoint(args.ckpt_path)
    model = model.to(device=get_device())
    model.eval()
    model_name = args.model_name or "_".join(args.ckpt_path.split(os.path.sep)[-1].split("-")[:2])
    
    datamodule = ReconstructionDataModule(scans=args.scans,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers, resize_to=args.resize_to)

    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    metrics = build_metrics(num_classes=args.num_classes, task="reconstruction")

    os.makedirs(args.scores_dir, exist_ok=True)
    model_prediction_dir = os.path.join(args.scores_dir, model_name, "predictions")

    os.makedirs(model_prediction_dir, exist_ok=True)
    global_scores = {}
    for metric_name in metrics.keys():
        global_scores[metric_name] = {
            "ca": 0.0,
            "n": 0,
        }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Reconstructing brains")):
        slices, head_masks = batch["slices"], batch["head_masks"]
        slices, head_masks = slices.to(device=model.device), head_masks.to(device=model.device)

        preds_until_now = batch_idx * slices.shape[0]

        with torch.no_grad():
            recons = model(slices)

        # focus the metrics computation on brain regions only
        scores = compute_metrics(recons * head_masks, slices * head_masks, metrics, task="reconstruction")

        # CA update rule: (x_n+1 + n * CA_n) / (n + 1)
        for metric_name in global_scores.keys():
            curr_ca = global_scores[metric_name]["ca"]
            curr_n = global_scores[metric_name]["n"]
            global_scores[metric_name] = {
                "ca": (scores[metric_name] + curr_n * curr_ca) / (curr_n + 1),
                "n": curr_n + 1
            }

        if args.n_predictions > preds_until_now:
            n_pairs = min(25, recons.shape[0], slices.shape[0])
            recons_to_show = recons[:n_pairs]
            preds_to_show = slices[:n_pairs]
            if recons_to_show.ndim == 3:
                recons_to_show = recons_to_show.unsqueeze(1)
            if preds_to_show.ndim == 3:
                preds_to_show = preds_to_show.unsqueeze(1)
            recons_to_show = recons_to_show.detach().cpu().float()
            preds_to_show = preds_to_show.detach().cpu().float()

            all_imgs = torch.cat([preds_to_show, recons_to_show], dim=0)
            min_vals = all_imgs.view(all_imgs.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
            max_vals = all_imgs.view(all_imgs.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            all_imgs = (all_imgs - min_vals) / (max_vals - min_vals + 1e-8)

            # interleave reference and reconstruction: ref0, rec0, ref1, rec1, ...
            paired_images = torch.empty((n_pairs * 2, 1, *recons_to_show.shape[2:]), dtype=all_imgs.dtype)
            paired_images[0::2] = all_imgs[:n_pairs]
            paired_images[1::2] = all_imgs[n_pairs:n_pairs * 2]

            # create the grid (5 rows, 10 columns = 5 pairs per row)
            grid = make_grid(paired_images, nrow=10, padding=2, pad_value=1.0)
            img_pil = to_pil_image(grid)
            img_pil.save(os.path.join(model_prediction_dir, f"{batch_idx}_paired_grid.png"))

    scores_path = os.path.join(args.scores_dir, args.scores_file)
    file_exists = os.path.exists(scores_path)

    with open(scores_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_scores.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(global_scores)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scans", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--n_predictions", help="Number of predictions to save", type=int, default=30)
    parser.add_argument("--scores_dir", type=str, default="./recon_scores")
    parser.add_argument("--scores_file", type=str, default="recon_scores.csv")

    test(parser.parse_args())