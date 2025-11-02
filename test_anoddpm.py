import os
import csv
from argparse import ArgumentParser
from datetime import time

import torch
from lightning import seed_everything
from matplotlib import animation, pyplot as plt
from tqdm import tqdm

from models import AnoDDPM
from dataset import ReconstructionDataModule
from utils import get_device, build_metrics, compute_metrics, gridify_output, load_anoddpm_checkpoint


def test(args):
    seed_everything(args.seed)

    model = load_anoddpm_checkpoint(AnoDDPM, args.ckpt_path, device='cpu', inference=True)
    # model = model.to(device=get_device())
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
    global_scores = {m: metrics[m]["default_value"] for m in metrics.keys()}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Reconstructing brains")):
        slices = batch["slices"]
        slices = slices.to(device=model.device)
        preds_until_now = batch_idx * slices.shape[0]

        with torch.no_grad():
            recons = model.forward_backward(slices)

        recons = recons[-1]

        scores = compute_metrics(recons, slices, metrics)

        if args.n_predictions > preds_until_now:
            fig, ax = plt.subplots()

            imgs = [[ax.imshow(gridify_output(x, 5), animated=True)] for x in recons]
            ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

            ani.save(os.path.join(model_prediction_dir, f"{time.time()}.mp4")) # noqa

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
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--scores_dir", type=str, default="./recon_scores")
    parser.add_argument("--scores_file", type=str, default="recon_scores.csv")

    test(parser.parse_args())