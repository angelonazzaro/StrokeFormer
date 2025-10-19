import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

from constants import LESION_SIZES
from .helpers import get_lesion_size_category


def plot_mri_slices(mri_volume, figsize=(14, 14)):
    # remove channel dimension if present
    if mri_volume.ndim == 4 and mri_volume.shape[0] == 1:
        mri_volume = mri_volume[0]
    elif mri_volume.ndim != 3:
        raise ValueError("Input MRI must have shape (1, H, W, D) or (H, W, D).")

    H, W, D = mri_volume.shape

    # compute grid size
    cols = int(math.ceil(math.sqrt(D)))
    rows = int(math.ceil(D / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(D):
        axes[i].imshow(mri_volume[:, :, i], cmap='gray')
        axes[i].axis('off')

    # hide empty subplots
    for j in range(D, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_fold_distribution(fold_masks_paths, title):
    fold_metadata = get_lesion_distribution_metadata(fold_masks_paths)
    counts = [fold_metadata[size]["count"] for size in LESION_SIZES]
    plot_lesion_size_distribution(counts, LESION_SIZES, title=title)


def get_lesion_distribution_metadata(masks_filepaths: List[str], labels: List[str] = LESION_SIZES,
                                     return_masks: bool = False):
    metadata = {
        "patients_count": 0,
        "lesions_per_patient": [],
        "tot_voxels": 0,
        "lesion_voxels": 0,
        "tot_slices": 0,
        "slices_without_lesions": 0,
        "slices_with_lesions": 0,
    }

    for size in labels:
        metadata[size] = {
            "lesion_area": [],
            "count": 0,
            "filepaths": []
        }

    masks = [nib.load(filepath).get_fdata().astype(np.uint8) if filepath.endswith(".nii.gz") \
                 else np.load(filepath).astype(np.uint8) for filepath in tqdm(masks_filepaths, desc="Loading masks")]

    for i, mask in tqdm(enumerate(masks), desc="Getting lesion distribution metadata"):
        tot_voxels = mask.size

        slices_with_lesions = mask.any(axis=(0, 1))

        metadata["patients_count"] += 1
        metadata["tot_voxels"] += tot_voxels
        metadata["slices_without_lesions"] += np.sum(~slices_with_lesions)
        metadata["slices_with_lesions"] += np.sum(slices_with_lesions)

        lesions_per_patient = 0
        for slice_idx in range(mask.shape[-1]):
            slice_mask = mask[0, ..., slice_idx]
            category, lesion_voxels, lesion_area = get_lesion_size_category(slice_mask, return_all=True)
            metadata["lesion_voxels"] += lesion_voxels
            metadata[category]["lesion_area"].append(lesion_area)
            metadata[category]["count"] += 1
            metadata[category]["filepaths"].append((masks_filepaths[i], slice_idx))

            _, thresh = cv2.threshold(slice_mask, 0, 1, cv2.THRESH_BINARY)
            num_lesions, _, _, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

            lesions_per_patient += num_lesions - 1  # exclude background region

        metadata["lesions_per_patient"].append(lesions_per_patient)

    if return_masks:
        return metadata, masks

    return metadata


def plot_lesion_size_distribution(counts, labels, figsize=(8, 6), title=None, plot=True, return_distribution=False):
    assert len(counts) == len(labels), "counts and labels must have same length"

    items = zip(counts, labels)
    sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
    counts, labels = zip(*sorted_items)

    colors = plt.cm.tab20.colors[:len(labels)]
    total = np.sum(counts)

    plt.figure(figsize=figsize)
    plt.bar(labels, counts, color=colors)
    distributions = {}

    for i, v in enumerate(counts):
        p = v * 100 / total
        distributions[labels[i]] = p
        plt.text(i, v + 7.5, f"{p:.2f}%", ha="center")

    if plot:
        plt.ylabel("Slices Without Lesions")
        plt.xlabel("Lesion Size Categories")
        plt.xticks(rotation=45)
        if title is None:
            title = "Lesion Size Distribution Across Slices"
        plt.title(title)
        plt.show()

    if return_distribution:
        return distributions
    return None
