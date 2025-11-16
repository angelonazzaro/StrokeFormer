import torch
from torch import Tensor


def expand_boxes(boxes: Tensor, roi_size: tuple[int, int]):
    min_width, min_height = roi_size
    # if lesion is too small to have a valid bounding box, expand it to bb_min_size
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    small_width = widths < min_width
    # adjust xmin and xmax to create a bb_min_size pixel width box centered at original center
    x_centers = (boxes[small_width, 0] + boxes[small_width, 2]) / 2
    boxes[small_width, 0] = x_centers - min_width / 2
    boxes[small_width, 2] = x_centers + min_width / 2

    small_height = heights < min_height
    # adjust ymin and ymax to create a bb_min_size pixel height box centered at original center
    y_centers = (boxes[small_height, 1] + boxes[small_height, 3]) / 2
    boxes[small_height, 1] = y_centers - min_height / 2
    boxes[small_height, 3] = y_centers + min_height / 2

    return boxes


def compare_head_sizes(head_mask_a: Tensor, head_mask_b: Tensor, threshold: float = 0.20) -> Tensor:
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    total_area = torch.tensor(head_mask_a.shape[-3] * head_mask_a.shape[-2], device=head_mask_a.device).float()

    # shape [N_sick]
    sick_head_masks_sum = head_mask_a.float().sum(dim=(0, 1, 2))
    # shape [N_healthy]
    healthy_head_masks_sum = head_mask_b.float().sum(dim=(0, 1, 2))

    sick_head_masks_area = sick_head_masks_sum / total_area.repeat(head_mask_a.shape[-1])
    healthy_head_masks_area = healthy_head_masks_sum / total_area.repeat(head_mask_b.shape[-1])

    # compute pairwise absolute differences (broadcasting)
    # sick_areas: [N_sick] → [N_sick, 1]
    # healthy_areas: [N_healthy] → [1, N_healthy]
    sick_head_masks_area = sick_head_masks_area.unsqueeze(-1)
    healthy_head_masks_area = healthy_head_masks_area.unsqueeze(0)
    abs_diff = (sick_head_masks_area - healthy_head_masks_area).abs()

    return abs_diff <= threshold


def compute_pairwise_iou(A, B):
    A = A.unsqueeze(-1)  # [..., N_sick] -> [..., N_sick, 1]
    B = B.unsqueeze(-2)  # [..., N_healthy] -> [..., 1, N_healthy]

    intersection = (A & B).sum(dim=(0, 1, 2))
    union = (A | B).sum(dim=(0, 1, 2))

    return intersection / (union + 1e-8)  # avoid division by zero


def compare_head_shapes(head_mask_a: Tensor, head_mask_b: Tensor, threshold: float = 0.80) -> bool:
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    iou_score = compute_pairwise_iou(head_mask_a, head_mask_b)
    return iou_score >= threshold