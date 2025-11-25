import math
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import ndimage

from constants import LESION_SIZES
from .helpers import get_lesion_size, load_volume, compute_head_mask, generate_overlayed_slice


def plot_mri_slices(mri_volume, mask_volume=None, figsize=(14, 14)):
    # remove channel dimension if present
    if mri_volume.ndim == 4 and mri_volume.shape[0] == 1:
        mri_volume = mri_volume[0]
        if mask_volume is not None:
            mask_volume = mask_volume[0]
    elif mri_volume.ndim != 3:
        raise ValueError("Input MRI must have shape (1, H, W, D) or (H, W, D).")

    H, W, D = mri_volume.shape

    # compute grid size
    cols = int(math.ceil(math.sqrt(D)))
    rows = int(math.ceil(D / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(D):
        if mask_volume is not None:
            scan_slice = generate_overlayed_slice(mri_volume[:, :, i], mask_volume[:, :, i], alpha=0.25)
        else:
            scan_slice = mri_volume[:, :, i]

        axes[i].imshow(scan_slice, cmap='gray')
        axes[i].axis('off')

    # hide empty subplots
    for j in range(D, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def get_lesion_size_distribution(scans_filepaths: List[str],
                                 masks_filepaths: List[str],
                                 labels: List[str] = LESION_SIZES):
    metadata = get_lesion_size_distribution_metadata(scans_filepaths, masks_filepaths, labels)
    per_size_counts = [len(metadata[size]["lesion_areas"]) for size in labels]
    total = np.sum(per_size_counts)

    # sort in descending order
    items = zip(per_size_counts, labels)
    sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

    distribution = {}

    for (count, label) in sorted_items:
        percentage = count * 100 / total
        distribution[label] = {"percentage": percentage, "count": count}

    return distribution


def plot_lesion_size_distribution(scans_filepaths: List[str],
                                  masks_filepaths: List[str],
                                  labels: List[str] = LESION_SIZES,
                                  figsize: tuple[int, int] = (10, 6),
                                  title: Optional[str] = None):
    distribution = get_lesion_size_distribution(scans_filepaths, masks_filepaths, labels)
    colors = plt.cm.tab20.colors[:len(labels)]

    plt.figure(figsize=figsize)
    labels, percentages, counts = [], [], []

    for label in distribution.keys():
        labels.append(label)
        percentages.append(distribution[label]["percentage"])
        counts.append(distribution[label]["count"])

    plt.bar(labels, counts, color=colors)

    for i in range(len(percentages)):
        plt.text(i, counts[i] + 7.5, f"{percentages[i]:.2f}%", ha="center")

    plt.ylabel("Counts")
    plt.xlabel("Lesion Sizes")

    if title is None:
        title = "Lesion Sizes Distribution"

    plt.title(title)
    plt.show()


def get_lesion_size_distribution_metadata(scans_filepaths: List[str],
                                          masks_filepaths: List[str],
                                          labels: List[str] = LESION_SIZES):
    """
        Get lesion size distribution data.

        Extract from masks and scans:
            - Distribution of lesion sizes across slices
            - Number of sick pixels
            - Number of healthy pixels
            - Number of lesions per patient
            - Number of sick slices
            - Number of healthy slices
    """

    metadata = {
        "sick_voxels": 0,
        "healthy_voxels": 0,
        "sick_slices_num": 0,
        "healthy_slices_num": 0,
        "num_lesions_per_patient": [],
    }

    for label in labels:
        metadata[label] = {  # noqa
            "lesion_areas": [],
            "filepaths": [],  # (scan path, mask path, slice idx)
        }

    for i in tqdm(range(len(masks_filepaths)), desc="Retrieving lesion size distribution data"):
        # shape is (1, H, W, D)
        scan = load_volume(scans_filepaths[i])
        mask = load_volume(masks_filepaths[i]).astype(np.uint8)
        head_mask = compute_head_mask(scan)

        for slice_idx in range(mask.shape[-1]):
            mask_slice = mask[..., slice_idx]
            scan_slice = head_mask[..., slice_idx]

            lesion_size_str, lesion_size, lesion_area = get_lesion_size(scan_slice, mask_slice, return_all=True)
            metadata[lesion_size_str]["lesion_areas"].append(lesion_area)  # noqa
            metadata[lesion_size_str]["filepaths"].append((scans_filepaths[i], masks_filepaths[i], slice_idx))  # noqa

            if lesion_size_str != "No Lesion":
                metadata["sick_voxels"] += lesion_size
                metadata["sick_slices_num"] += 1
            else:
                metadata["healthy_voxels"] += np.sum(scan_slice)
                metadata["healthy_slices_num"] += 1

        if mask.ndim == 4:
            mask = mask[0]
        # get number of lesions/connected components
        labeled_mask, num_lesions = ndimage.label(mask)
        # ndimage.label counts also the background as a connected component so
        # the number of lesion must be decremented by one
        num_lesions -= 1
        metadata["num_lesions_per_patient"].append(num_lesions)

    return metadata
