import csv
import os
from argparse import ArgumentParser

import torch
from lightning import seed_everything
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid, draw_bounding_boxes
from tqdm import tqdm

from dataset import RegionProposalDataModule
from models import RPN
from utils import build_metrics, compute_metrics, get_device, to_3channel


def test(args):
    seed_everything(args.seed)

    model = RPN.load_from_checkpoint(args.ckpt_path)
    model = model.to(device=get_device())
    model.eval()
    model_name = args.model_name or "_".join(args.ckpt_path.split(os.path.sep)[-1].split("-")[:2])

    datamodule = RegionProposalDataModule(
        paths={
            "test": {
                "scans": args.scans,
                "masks": args.masks,
            }
        },
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_to=args.resize_to
    )

    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    metrics = build_metrics(num_classes=args.num_classes, task="region_proposal")

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
        slices, targets = batch["slices"], batch["targets"]
        targets = [{k: v.to(model.device) for k, v in t.items()} for t in targets]
        slices = slices.to(device=model.device)

        preds_until_now = batch_idx * slices.shape[0]

        with torch.no_grad():
            proposals = model(slices, None)

        preds = [
            {
                "boxes": p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu()
            }
            for p in proposals
        ]
        targets = [
            {
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu()
            }
            for t in targets
        ]

        # focus the metrics computation on brain regions only
        scores = compute_metrics(preds, targets, metrics, task="region_proposal")

        # CA update rule: (x_n+1 + n * CA_n) / (n + 1)
        for metric_name in global_scores.keys():
            curr_ca = global_scores[metric_name]["ca"]
            curr_n = global_scores[metric_name]["n"]
            global_scores[metric_name] = {
                "ca": (scores[metric_name] + curr_n * curr_ca) / (curr_n + 1),
                "n": curr_n + 1
            }

        if args.n_predictions > preds_until_now:
            ground_truths = [draw_bounding_boxes(to_3channel(img), target["boxes"], colors="green").cpu() for img, target in zip(slices, targets)]
            predictions = [draw_bounding_boxes(to_3channel(img), proposal["boxes"], colors="red").cpu() for img, proposal in zip(slices, proposals)]
            ground_truths = torch.stack(ground_truths)
            predictions = torch.stack(predictions)

            all_imgs = torch.cat([ground_truths, predictions], dim=0)

            n_pairs = slices.shape[0]
            # interleave reference and reconstruction: ref0, rec0, ref1, rec1, ...
            paired_images = torch.empty((n_pairs * 2, 3, *slices.shape[2:]), dtype=slices.dtype)
            paired_images[0::2] = all_imgs[:n_pairs]
            paired_images[1::2] = all_imgs[n_pairs:]
            grid = make_grid(paired_images, padding=2, pad_value=1.0)
            grid = to_pil_image(grid)
            grid.save(os.path.join(model_prediction_dir, f"{batch_idx}_preds.png"))

    scores_path = os.path.join(args.scores_dir, args.scores_file)
    file_exists = os.path.exists(scores_path)

    global_scores = {m: v["ca"].item() for m, v in global_scores.items()}

    with open(scores_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_scores.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(global_scores)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scans", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resize_to", nargs=2, type=int, default=None)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--n_predictions", help="Number of predictions to save", type=int, default=30)
    parser.add_argument("--scores_dir", type=str, default="./rpn_scores")
    parser.add_argument("--scores_file", type=str, default="rpn_scores.csv")

    test(parser.parse_args())