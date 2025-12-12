import torch
from monai.inferers import SlidingWindowInferer
from torch import Tensor
from torchvision.ops import boxes as box_ops
from torchmetrics.functional.classification import jaccard_index


def sliding_window_inference_3d(region, model, roi_d, roi_h, roi_w, overlap=0.5):
    # apply overlap only on dimensions exceeds their roi dimension
    C, D, H, W = region.shape

    z_overlap = overlap if D > roi_d else 0
    y_overlap = overlap if H > roi_h else 0
    x_overlap = overlap if W > roi_w else 0

    inferer = SlidingWindowInferer(
        roi_size=(roi_d, roi_h, roi_w),
        sw_batch_size=1,
        overlap=(z_overlap, y_overlap, x_overlap),
        progress=False,
        mode="gaussian"
    )

    return inferer(region.unsqueeze(0), network=model).squeeze(0)


def expand_boxes(boxes: Tensor, bounds: tuple[int, int], roi_size: tuple[int, int]):
    min_width, min_height = roi_size

    boxes = boxes.clone()  # avoid modifying caller's tensor

    expanded = False
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    small_width = widths < min_width
    if small_width.any():
        x_centers = (boxes[small_width, 0] + boxes[small_width, 2]) / 2
        boxes[small_width, 0] = x_centers - min_width / 2
        boxes[small_width, 2] = x_centers + min_width / 2
        expanded = True

    small_height = heights < min_height
    if small_height.any():
        y_centers = (boxes[small_height, 1] + boxes[small_height, 3]) / 2
        boxes[small_height, 1] = y_centers - min_height / 2
        boxes[small_height, 3] = y_centers + min_height / 2
        expanded = True

    if expanded:
        boxes = box_ops.clip_boxes_to_image(boxes, bounds)

    return boxes



def compare_head_sizes(head_mask_a: Tensor, head_mask_b: Tensor, threshold: float = 0.20) -> bool:
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    total_area = torch.tensor(head_mask_a.shape[-2] * head_mask_a.shape[-1], device=head_mask_a.device).float()

    sum_a = head_mask_a.float().sum()
    sum_b = head_mask_b.float().sum()

    area_a = sum_a / total_area
    area_b = sum_b / total_area

    abs_diff = (area_a - area_b).abs().item()

    return abs_diff <= threshold


def compare_head_shapes(head_mask_a: Tensor, head_mask_b: Tensor, threshold: float = 0.80) -> bool:
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    iou_score = jaccard_index(head_mask_a, head_mask_b, task="binary", num_classes=2, ignore_index=0)
    return iou_score.item() >= threshold