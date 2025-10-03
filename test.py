import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import MRIDataModule
from model import StrokeFormer
from utils import predictions_generator


def test(args):
    # TODO: implement Grad-CAM
    model = StrokeFormer.load_from_checkpoint(args.ckpt_path)
    model_name = args.model_name

    if model_name is None:
        model_name = args.ckpt_path.split(os.path.sep)[-1].split("-")[:2]

    datamodule = MRIDataModule(paths={"test": {"scans": args.scans, "masks": args.masks}},
                               subvolume_dim=args.subvolume_dim, stride=args.stride,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    model.eval()

    os.makedirs(args.scores_dir, exist_ok=True)
    model_prediction_dir = os.path.join(args.scores_dir, model_name, "predictions")
    os.makedirs(model_prediction_dir, exist_ok=True)

    per_size_scores = {}

    for size in ["No Lesion", "Small", "Medium", "Large"]:
        per_size_scores[size] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting lesions"):
            scans, masks = batch
            i = 0
            for j, result in enumerate(predictions_generator(model, scans, masks, args.slices_per_scan)):
                scan_dir = os.path.join(model_prediction_dir, f"scan_{i}")
                os.makedirs(scan_dir, exist_ok=True)

                for metric in result['scores']:
                    per_size_scores[result["lesion_size"]][metric].append(result['scores'][metric])

                if j % args.slices_per_scan == 0:
                    i += 1

                if args.n_predictions > 0:
                    grid = make_grid(torch.stack([result["gt"], result["pd"]]), nrow=2, normalize=False, scale_each=False)
                    grid_img = to_pil_image(grid)
                    grid_img.save(osp.join(model_prediction_dir, f"{result['lesion_size']}.png"))  # noqa
                    args.n_predictions -= 1

    # TODO: compute global metrics, save results file


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument('--scans', type=str, required=True, nargs='+')
    parser.add_argument('--masks', type=str, default=None, nargs='+')
    parser.add_argument("--subvolume_dim",
                        help="Optional tuple of three integers to resize (e.g., --subvolume_dim 128 128 128",
                        nargs=3, type=int, required=False)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=0)

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