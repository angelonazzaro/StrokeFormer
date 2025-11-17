import torch
from torch import Tensor

from torchmetrics.functional.classification import jaccard_index


import torch.nn.functional as F

import torch
import torch.nn.functional as F

def sliding_window_inference_3d(region, model, roi_d, roi_h, roi_w, overlap=0.5):
    C, D, H, W = region.shape

    # --- Compute strides in all dimensions ---
    stride_d = max(1, int(roi_d * (1 - overlap)))
    stride_h = max(1, int(roi_h * (1 - overlap)))
    stride_w = max(1, int(roi_w * (1 - overlap)))

    # --- Output accumulators ---
    output = torch.zeros(1, 2, D, H, W, device=region.device)
    weight = torch.zeros(1, 1, D, H, W, device=region.device)

    # --- 3D blending window (roi_d × roi_h × roi_w) ---
    bd = torch.linspace(0, 1, roi_d, device=region.device)
    bh = torch.linspace(0, 1, roi_h, device=region.device)
    bw = torch.linspace(0, 1, roi_w, device=region.device)

    bd = torch.minimum(bd, 1 - bd) * 2.0 + 1e-6
    bh = torch.minimum(bh, 1 - bh) * 2.0 + 1e-6
    bw = torch.minimum(bw, 1 - bw) * 2.0 + 1e-6

    blend = (
        bd.view(roi_d, 1,      1) *
        bh.view(1,     roi_h,  1) *
        bw.view(1,     1,      roi_w)
    )  # shape (roi_d, roi_h, roi_w)

    blend = blend.view(1, 1, roi_d, roi_h, roi_w)

    # --- Sliding window loops ---
    for z in range(0, D, stride_d):
        for y in range(0, H, stride_h):
            for x in range(0, W, stride_w):

                # Compute window ends
                z2 = z + roi_d
                y2 = y + roi_h
                x2 = x + roi_w

                window = region[:,
                                z:min(z2, D),
                                y:min(y2, H),
                                x:min(x2, W)]

                # Pad if window touches boundary
                pd = roi_d - window.shape[1]
                ph = roi_h - window.shape[2]
                pw = roi_w - window.shape[3]

                if pd > 0 or ph > 0 or pw > 0:
                    window = F.pad(
                        window,
                        (0, pw,   # width
                         0, ph,   # height
                         0, pd),  # depth
                        mode="constant",
                        value=0
                    )

                # Add batch dim
                logits = model(window.unsqueeze(0))  # (1,2,roi_d,roi_h,roi_w)

                # Determine how much is valid (avoid padded area)
                valid_d = min(roi_d, D - z)
                valid_h = min(roi_h, H - y)
                valid_w = min(roi_w, W - x)

                logits = logits[:, :, :valid_d, :valid_h, :valid_w]
                local_blend = blend[:, :, :valid_d, :valid_h, :valid_w]

                # Accumulate
                output[:, :,
                       z:z+valid_d,
                       y:y+valid_h,
                       x:x+valid_w] += logits * local_blend

                weight[:, :,
                       z:z+valid_d,
                       y:y+valid_h,
                       x:x+valid_w] += local_blend

    # Normalize
    output = output / torch.clamp(weight, min=1e-6)
    return output

def expand_boxes(boxes: Tensor, bounds: tuple[int, int], roi_size: tuple[int, int]):
    H, W = bounds
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
        # keep expanded size into bounds
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # widths/heights after resize
        new_widths = x2 - x1
        new_heights = y2 - y1

        shift = -torch.minimum(x1, torch.zeros_like(x1))
        x1 = x1 + shift
        x2 = x1 + new_widths

        shift = torch.maximum(x2 - W, torch.zeros_like(x2))
        x1 = x1 - shift
        x2 = x1 + new_widths

        shift = -torch.minimum(y1, torch.zeros_like(y1))
        y1 = y1 + shift
        y2 = y1 + new_heights

        shift = torch.maximum(y2 - H, torch.zeros_like(y2))
        y1 = y1 - shift
        y2 = y1 + new_heights

        boxes[:, 0] = x1
        boxes[:, 1] = y1
        boxes[:, 2] = x2
        boxes[:, 3] = y2

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