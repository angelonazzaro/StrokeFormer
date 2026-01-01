import torch
from monai.inferers import SlidingWindowInferer
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops, box_iou
from torchmetrics.functional.classification import jaccard_index


def sliding_window_inference_3d(region, model, roi_d, roi_h, roi_w, overlap=0.25):
    # apply overlap only on dimensions exceeds their roi dimension
    C, D, H, W = region.shape

    # Compute required padding per dimension
    pad_d = max(roi_d - D, 0)
    pad_h = max(roi_h - H, 0)
    pad_w = max(roi_w - W, 0)

    # symmetric padding (left, right)
    pad_d0, pad_d1 = pad_d // 2, pad_d - pad_d // 2
    pad_h0, pad_h1 = pad_h // 2, pad_h - pad_h // 2
    pad_w0, pad_w1 = pad_w // 2, pad_w - pad_w // 2

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        region = F.pad(
            region,
            (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1),
            mode="constant",
            value=0,
        )

    _, Dp, Hp, Wp = region.shape

    # apply overlap only if dimension exceeds roi
    z_overlap = overlap if Dp > roi_d else 0
    y_overlap = overlap if Hp > roi_h else 0
    x_overlap = overlap if Wp > roi_w else 0

    inferer = SlidingWindowInferer(
        roi_size=(roi_d, roi_h, roi_w),
        sw_batch_size=1,
        overlap=(z_overlap, y_overlap, x_overlap),
        progress=False,
        mode="gaussian"
    )

    output = inferer(region.unsqueeze(0), network=model).squeeze(0)

    # crop back to original size if padded
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        output = output[
                 :,
                 pad_d0: pad_d0 + D,
                 pad_h0: pad_h0 + H,
                 pad_w0: pad_w0 + W,
                 ]

    return output


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

def collect_boxes(group_proposals):
    boxes = []
    for p in group_proposals:
        if p["boxes"].numel() > 0:
            boxes.append(p["boxes"][0])  # assume 1 box per slice
    if len(boxes) == 0:
        return None
    return torch.stack(boxes)  # (K, 4)


def cluster_boxes_iou(boxes, iou_thresh=0.7):
    K = boxes.shape[0]
    ious = box_iou(boxes, boxes)
    used = torch.zeros(K, dtype=torch.bool, device=boxes.device)

    clusters = []
    for i in range(K):
        if used[i]:
            continue
        cluster_idx = torch.where((ious[i] >= iou_thresh) & ~used)[0]
        used[cluster_idx] = True
        clusters.append(cluster_idx)

    return clusters


def enclosing_box(boxes):
    return torch.tensor(
        [
            boxes[:, 0].min(),
            boxes[:, 1].min(),
            boxes[:, 2].max(),
            boxes[:, 3].max(),
        ],
        device=boxes.device,
    )

def propose_regions(rpn, scan, head_mask, roi_size):
    # the RPN works on 2D slices, treating each independently of another.
    # for each 3D tensor in the batch, we need to:
    #   1. decompose it into its 2D slices components and make SEQUENTIAL predictions on them.
    #   This is replicates the RPN training where slices coming from the same patient were fed sequentially to induce a sort of spatial/temporal bias (very simple concept)
    #   2. since each slice is treated independently, we may have inconsistent proposals between consecutive slices. This is not necessarily bad since in different views of the brain,
    #   the lesion(s) may appear differently. The problem is when these inconsistencies are presented in similar/equal views of the brain. We need to propose 3D plausible and consistent regions.
    #   In order to do so, we can:
    #       2.1 divide the volume into anatomical groups, based on the shape and dimension of the brain. here, we assume that in similar views of the brain, the lesion(s) may not vary greatly
    #       2.2 for each anatomical group:
    #           2.2.1 we scroll both from above and below and we refine the proposals as follows:
    #               2.2.1.1 if there are multiple proposed regions, we take the most frequent
    #               2.2.1.2 if there is not a 'most frequent region' or there are multiple (e.g., slice contains multiple lesions in multiple locations), we create a bigger bounding box to encompass all proposed region.
    #               This renders the RP approach less effective but speeds up training. For now, it is a good enough compromise between
    #               RP effectiveness and training speed

    # step 1: decompose 3D tensor into 2D slices.
    # since each scan is of shape [D, H, W], the RPN model will see D as the batch dimension
    with torch.no_grad():
        # (1, D, H, W) -> (D, 1, H, W)
        # this is a list of dicts: boxes, labels, scores
        curr_scan = scan.permute(1, 0, 2, 3)
        num_slices = curr_scan.shape[0]
        stride = num_slices // 4
        proposals = []
        for j in range(0, num_slices, stride):
            proposals.extend(rpn(curr_scan[j:j + stride], None))

    anatomical_groups = []
    anatomical_group_start = None
    prev_head_mask = None
    # step 2.1: divide the volume into anatomical groups. these will be represented as a list of dicts containing tensors and proposals for each
    for j in range(len(proposals)):
        # we consider an anatomical group a group of similar slices that start when we find a valid box and ends
        # when the next slice is different/distant from the previous
        if anatomical_group_start is None:
            curr_boxes = proposals[j]["boxes"]
            # a shape of (0, 4) means no bounding box
            if curr_boxes.shape[0] > 0:
                anatomical_group_start = j
                prev_head_mask = head_mask[:, anatomical_group_start]
                # if multiple boxes, create a bigger box to encompass all
                # this is done at slice level
                xmin, xmax, ymin, ymax = float("inf"), float("-inf"), float("inf"), float("-inf")
                for box in curr_boxes:
                    xmin = min(xmin, box[0])
                    xmax = max(xmax, box[2])
                    ymin = min(ymin, box[1])
                    ymax = max(ymax, box[3])

                curr_boxes = torch.tensor([xmin, ymin, xmax, ymax], device=rpn.device).unsqueeze(0)  # (1, 4)
                curr_boxes = expand_boxes(curr_boxes, scan.shape[2:], roi_size[1:])  # noqa
                proposals[j]["boxes"] = curr_boxes
        else:
            curr_head_mask = head_mask[:, j]
            if not (compare_head_sizes(prev_head_mask, curr_head_mask, 0.2) and compare_head_shapes(
                    prev_head_mask, curr_head_mask, 0.7)):
                # curr slices offers a differ anatomical view, so we cut the group here
                anatomical_groups.append({"group": scan[:, anatomical_group_start:j],
                                          "proposals": proposals[anatomical_group_start:j],
                                          "region_start": anatomical_group_start, "region_end": j})
                curr_boxes = proposals[j]["boxes"]
                if curr_boxes.shape[0] > 0:
                    anatomical_group_start = j
                    prev_head_mask = head_mask[:, anatomical_group_start]
                else:
                    anatomical_group_start = None
                    prev_head_mask = None

    # step 2.2: scroll each group from above and below
    for group_dict in anatomical_groups:
        group_proposals = group_dict["proposals"]

        boxes = collect_boxes(group_proposals)

        if boxes is None:
            continue  # no proposals at all in this group

        # find most frequent box via IoU clustering
        clusters = cluster_boxes_iou(boxes, iou_thresh=0.7)
        clusters = sorted(clusters, key=len, reverse=True)

        best_cluster = clusters[0]

        # if dominant box exists → use it
        if len(best_cluster) >= 2:
            final_box = boxes[best_cluster].mean(dim=0, keepdim=True)
        else:
            # fallback: create bigger box
            final_box = enclosing_box(boxes).unsqueeze(0)

        # expand once at group level
        yield expand_boxes(final_box, scan.shape[2:], roi_size[1:]), group_dict