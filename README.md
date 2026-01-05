<div align="center">
    <!-- <img src="https://github.com/user-attachments/assets/52b8223d-5740-47c4-b699-9497c6c52a9b" alt="StrokeFormer logo" width="300"/> -->
    <h1>StrokeFormer</h1>
    <h3>A lightweight approach for stroke lesion segmentation</h3>
</div>

<p align="center">
 <a href="#"><img src="https://img.shields.io/github/contributors/angelonazzaro/StrokeFormer?style=for-the-badge" alt="Contributors"/></a>
 <img src="https://img.shields.io/github/last-commit/angelonazzaro/StrokeFormer?style=for-the-badge" alt="last commit">
</p>
<p align="center">
 <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"/></a>
 <a href="#"><img src="https://img.shields.io/github/languages/top/angelonazzaro/StrokeFormer?style=for-the-badge" alt="Languages"/></a>
</p>

# Introduction

Stroke is a common cerebrovascular disease and one of the leading causes of death in the
world. Accurate segmentation of stroke lesions is critical for diagnosis, treatment planning
and disease monitoring. Over the last decade, various segmentation approaches have been
proposed, with deep learning based methods becoming more and more essential for clinicians.
Despite their rapid growth, current methods are often highly specialized for a particular
imaging modality and stroke stage and result in heavy and complex pipelines that are
difficult to deploy in clinical settings.

In this work, we propose *StrokeFormer*, a _lightweight_ two-stage pipeline for stroke lesion
segmentation, designed for clinical deployment. StrokeFormer combines a 2D region proposal
mechanism with a 3D native segmentation model to reduce computational complexity,
mitigate severe class imbalance, and better exploit anatomical context.

# Objectives

Despite promising theoretical results, the practical effectiveness of the current pipeline is majorly limited
by the region proposal stage, which has to deal with _extreme_ class imbalance. In response, this work 
investigates and discusses strategies to combat overfitting and undesired biases 
in stroke lesion segmentation, with the goal of providing insights and guidance for future research efforts.


> [!WARNING]
> **Documentation Release Status:**
> The documentation is currently **under embargo**. The full source documentation will be released publicly in **August 2026**.
>
> This README serves as a general overview of the proposed method and documents its usage. 
> 

# StrokeFormer

The two phases of StrokeFormer are organized as follows:

- *Region Proposal*: Given a 3D volume, it is first
  decomposed into its 2D slices, then region proposal is performed independently on each. Since multiple
  regions can be proposed within a single slice, for example, when several lesions are present, we
  consolidate these into a single larger bounding box, that encompasses all proposed regions.   Because the RPN treats each slice independently, predicted regions may vary across consecutive
  slices, especially when anatomical perspectives vary between them. Therefore, for each 3D scan, we
  identify and group slices corresponding to distinct "anatomical regions" based on brain shape and
  size. To enforce spatial consistency, within each group, we enforce within each group the most frequent proposed region. If there is no most frequent region, a
  bigger bounding box is created to encompass all proposals. Region proposals are therefore aggregated within
  each group, and segmentation is performed independently for each anatomical region;


- *Segmentation*: The proposed regions are expanded to match a fixed RoI size (e.g. $64 \times 64
  \times 64$). When the 3D regions exceed this fixed resolution, we apply a sliding window approach
  over the larger region, processing overlapping sub-volume sequentially. The segmentation outputs
  from these overlapping windows are then averaged to produce the final prediction. Regions that
  were not proposed by the RPN model, and therefore not segmented, are masked to prevent the
  training signal from being distorted by non-relevant regions. The segmented outputs are finally
  concatenated back into a full 3D volume, reconstituting the whole scan.

  <br>
  <img style="mix-blend-mode: multiply;" src="https://github.com/angelonazzaro/StrokeFormer/blob/main/readme-misc/strokeformer-pipeline.svg" />

# Data Augmentation

To mitigate class imbalance, we propose a data augmentation strategy to generate _anatomically plausible_ lesions. 
The strategy involves four main steps:

1. We identify similar healthy-sick pairs by comparing: brain size, shape and
  overall structure. Brain size and brain shape are compared using IoU, while overall structure
  similarity is computed through SSIM;

2. To ensure anatomical plausbility, we refine the previously selected pairs by retaining only
  those whose slice depth differs by no more than $30$ slices;

3. To induce and increase lesion variability, for each healthy slice, we randomly select lesions
  from its paired sick counterparts. Each lesion undergoes random roation and/or erosion;

4. Finally, the transformed lesions are transferred onto the healthy slice using Poisson image
  editing. To preserve the original slice intensities and avoid
  excessive blurring, we perform the transfer considering only the sick region on both slices.

<br>
<img src="https://github.com/angelonazzaro/StrokeFormer/blob/main/readme-misc/data_augmentation_pipeline.svg" />
  
# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.11` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/angelonazzaro/StrokeFormer.git
```

## Installing Requirements

If you have an NVIDIA GPU with cuda installed on your system, run the following command in
your shell:

```shell
pip install -r requirements-cuda.txt
```

Otherwise, if your machine does not support cuda, run the following command in your shell to make
the code run on CPU:
```shell
pip install -r requirements.txt
```

# Usage

This repository provides three main entry-point scripts for brain lesion segmentation:

- **`cross_validation.py`**: End-to-end K-fold cross-validation pipeline (RPN + StrokeFormer) on 3D brain volumes.
- **`test_rpn.py`**: Evaluation and visualization of a trained RPN on 2D slices.
- **`test_strokeformer.py`**: Evaluation and visualization of a trained StrokeFormer model on 3D volumes, with optional Grad-CAM.
- **`train_rpn.py`**: Trains the RPN.
- **`train_strokeformer.py`**: Train the segmentation model.

All scripts are intended to be run from the project root with Python 3.

## 1. `cross_validation.py`

Cross-validation pipeline that:

1. Splits 3D volumes into K folds with balanced lesion-size distribution;
2. Trains an RPN on 2D slices derived from each fold;
3. Uses the RPN to generate region proposals and saves them;
4. Trains StrokeFormer using those proposals;
5. Tests both RPN and StrokeFormer and logs metrics and predictions.

### Basic usage

```bash
python cross_validation.py \
  --scans_dir /path/to/Scans \
  --masks_dir /path/to/Masks \
  --model_prefix StrokeFormerATLAS \
  --rpn_model_prefix RPNATLAS
```

### Required arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--scans_dir` | `str` | Root directory containing 3D scan volumes (e.g., `*T1w*.npy`) |
| `--masks_dir` | `str` | Root directory containing corresponding lesion masks (e.g., `*T1lesion_mask*.npy`) |
| `--model_prefix` | `str` | Prefix for naming StrokeFormer runs and checkpoints |
| `--rpn_model_prefix` | `str` | Prefix for naming RPN runs and checkpoints |

### Full argument reference

#### General / cross-validation
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | `int` | `42` | Random seed for all random number generators |
| `--k` | `int` | `5` | Number of folds for cross-validation |
| `--splits` | `3 ints` | `70 10 20` | Percentage split (train, val, test); must sum to 100 |
| `--use_augmented` | `flag` | `False` | Use augmented scans for training when available |
| `--augmented_dir` | `str` | `data/ATLAS_2/Processed/Augmented/` | Base directory containing augmented masks |

#### Data loading
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ext` | `str` | `.npy` | File extension for scan/mask volumes |
| `--transforms` | `list[str]` | `None` | Data augmentation/transformation functions |
| `--rpn_batch_size` | `int` | `32` | Batch size for RPN training/testing |
| `--batch_size` | `int` | `8` | Batch size for StrokeFormer training/testing |
| `--num_workers` | `int` | `0` | Number of workers for PyTorch dataloaders |
| `--subvolume_depth` | `int` | `189` | Depth of 3D subvolumes for StrokeFormer |
| `--overlap` | `float` | `None` | Overlap ratio for sliding-window inference |
| `--resize_to` | `2 ints` | `None` | In-plane resize `(H, W)` |
| `--scan_dim` | `4 ints` | `1 189 192 192` | Expected scan shape `(C, D, H, W)` |

#### StrokeFormer model & optimization
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_classes` | `int` | `2` | Number of segmentation classes |
| `--in_channels` | `int` | `1` | Input channels for StrokeFormer |
| `--opt_lr` | `float` | `3e-5` | Base learning rate |
| `--warmup_lr` | `float` | `4e-6` | Warmup learning rate |
| `--max_lr` | `float` | `4e-4` | Maximum learning rate |
| `--warmup_steps` | `int` | `10` | Warmup steps |
| `--weight_decay` | `float` | `1e-3` | Weight decay |
| `--eps` | `float` | `1e-8` | Optimizer epsilon |
| `--betas` | `2 floats` | `0.9 0.999` | Adam beta coefficients |
| `--roi_size` | `3 ints` | `64 64 64` | 3D ROI size for region proposals |
| `--seg_loss` | `str` | `DiceCELoss` | Segmentation loss function |
| `--seg_loss_cfg` | `str` | JSON config | Segmentation loss hyperparameters |
| `--cls_loss` | `str` | `None` | Classification loss (optional) |
| `--cls_loss_weight` | `float` | `0.5` | Classification loss weight |
| `--seg_loss_weight` | `float` | `0.5` | Segmentation loss weight |

#### Training & logging
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--default_root_dir` | `str` | `StrokeFormer` | StrokeFormer checkpoints/logs directory |
| `--project` | `str` | `StrokeFormer` | WandB project name |
| `--group` | `str` | `None` | WandB group name |
| `--max_epochs` | `int` | `250` | Maximum training epochs |
| `--patience` | `int` | `50` | Early stopping patience |
| `--entity` | `str` | `neurone-lab` | WandB entity |
| `--offline` | `flag` | `False` | Run WandB offline |
| `--devices` | `list[int]` | `[0]` | GPU device indices |

#### RPN model & optimization
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rpn_lr` | `float` | `1e-4` | RPN learning rate |
| `--rpn_eps` | `float` | `1e-8` | RPN optimizer epsilon |
| `--rpn_betas` | `2 floats` | `0.9 0.999` | RPN Adam betas |
| `--rpn_weight_decay` | `float` | `1e-4` | RPN weight decay |
| `--dataset_anchors` | `str` | `ATLAS` | Dataset type for anchor config |
| `--rpn_backbone_weights` | `str` | `None` | RPN backbone weights |
| `--rpn_default_root_dir` | `str` | `RPN` | RPN checkpoints/logs directory |
| `--rpn_project` | `str` | `RPN` | RPN WandB project |
| `--rpn_max_epochs` | `int` | `250` | RPN maximum epochs |
| `--rpn_patience` | `int` | `50` | RPN early stopping patience |
| `--rpn_group` | `str` | `None` | RPN WandB group |
| `--rpn_model_prefix` | `str` | **required** | RPN model prefix |

#### Training / logging
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min_delta` | `float` | `0.001` | Minimum validation loss improvement |
| `--lr_logging_interval` | `str` | `epoch` | LR logging interval (`epoch`/`step`) |
| `--num_samples` | `int` | `5` | Number of prediction samples to log |
| `--log_every_n_val_epochs` | `int` | `5` | Prediction logging frequency |

#### Testing / scores
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--target_layers` | `str+` | `None` | Grad-CAM target layers for StrokeFormer |
| `--scores_dir` | `str` | `./scores` | StrokeFormer scores directory |
| `--scores_file` | `str` | `scores.csv` | StrokeFormer scores filename |
| `--rpn_scores_dir` | `str` | `./rpn_scores` | RPN scores directory |
| `--rpn_scores_file` | `str` | `scores.csv` | RPN scores filename |

## 2. `test_rpn.py`

Evaluate a trained RPN on 2D slices and save metrics/visualizations.

### Basic usage
```bash
python test_rpn.py \
  --scans /path/to/2D/Scans \
  --masks /path/to/2D/Masks \
  --ckpt_path /path/to/rpn.ckpt \
  --model_name RPNATLAS-test
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | `int` | `42` | Random seed |
| `--scans` | `str` | **required** | Directory of 2D scan slices |
| `--masks` | `str` | **required** | Directory of 2D masks |
| `--batch_size` | `int` | `32` | Test batch size |
| `--num_workers` | `int` | `0` | Dataloader workers |
| `--resize_to` | `2 ints` | `None` | Resize `(H, W)` |
| `--ckpt_path` | `str` | **required** | Path to RPN checkpoint |
| `--model_name` | `str` | `None` | Name for scores/predictions |
| `--num_classes` | `int` | `2` | Number of classes |
| `--n_predictions` | `int` | `30` | Number of prediction visualizations |
| `--scores_dir` | `str` | `./rpn_scores` | Output directory |
| `--scores_file` | `str` | `rpn_scores.csv` | Metrics CSV filename |

## 3. `test_strokeformer.py`

Evaluate StrokeFormer on 3D volumes with per-lesion-size metrics and optional Grad-CAM.

### Basic usage
```bash
python test_strokeformer.py \
  --scans /path/to/Scans \
  --masks /path/to/Masks \
  --ckpt_path /path/to/strokeformer.ckpt \
  --model_name StrokeFormerATLAS-test
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | `int` | `42` | Random seed |
| `--scans` | `str+` | **required** | One or more scan paths/directories |
| `--masks` | `str+` | `None` | One or more mask paths/directories |
| `--subvolume_depth` | `int` | `189` | Sliding window depth |
| `--overlap` | `float` | `None` | Sliding window overlap |
| `--batch_size` | `int` | `8` | Test batch size |
| `--num_workers` | `int` | `0` | Dataloader workers |
| `--scan_dim` | `4 ints` | `1 189 192 192` | Expected scan shape |
| `--regions` | `str` | `None` | Precomputed regions JSON |
| `--resize_to` | `2 ints` | `None` | In-plane resize `(H, W)` |
| `--ckpt_path` | `str` | **required** | StrokeFormer checkpoint |
| `--model_name` | `str` | `None` | Name for scores/predictions |
| `--target_layers` | `str+` | `None` | Grad-CAM target layers |
| `--n_predictions` | `int` | `30` | Number of visualizations |
| `--num_classes` | `int` | `2` | Number of classes |
| `--scores_dir` | `str` | `./scores` | Output directory |
| `--scores_file` | `str` | `scores.csv` | Global metrics CSV |
| `--per_size_scores_file` | `str` | `per_size_scores.csv` | Per-lesion-size metrics CSV |

## Expected directory structure

```
data/
├── Scans/
│   ├── sub-001_T1w.npy
│   └── sub-002_T1w.npy
├── Masks/
│   ├── sub-001_label-L_desc-T1lesion_mask.npy
│   └── sub-002_label-L_desc-T1lesion_mask.npy
(Optional) Augmented/
├── Scans/
└── Masks/
```

## Output structure

```
scores/                    # test_strokeformer.py & cross_validation.py
├── StrokeFormerATLAS/
│   └── predictions/
├── scores.csv             # Global metrics
└── per_size_scores.csv    # Per-lesion-size metrics

rpn_scores/                # test_rpn.py & cross_validation.py
├── RPNATLAS/
│   └── predictions/
└── rpn_scores.csv
```

---

*For complete argument details and advanced usage, see the source code argument parsers.*

# Citation

If you found this work useful, please consider citing: 
```
@article{nazzaro2026:strokeformer,
  author      = {Angelo Nazzaro},
  title       = {StrokeFormer: A lightweight approach for stroke lesion segmentation},
  year        = {2026},
  institution = {University of Salerno}
  url         = {https://github.com/angelonazzaro/StrokeFormer}
}
```

