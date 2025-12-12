from collections import OrderedDict
from typing import Literal, Optional, List

import lightning as l
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from constants import DATASET_ANCHORS
from utils import build_metrics


def detection_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    original_image_sizes: List[tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals,
                                                                   image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    # detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # noqa
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


class RPN(l.LightningModule):

    def __init__(self,
                 lr: float = 1e-4,
                 eps: float = 1e-8,
                 betas: tuple[float, float] = (0.9, 0.999),
                 dataset_anchors: Literal["ATLAS", "ISLES-DWI", "ISLES-FLAIR"] = "ATLAS",
                 backbone_weights: Optional[Literal["DEFAULT"]] = "DEFAULT",
                 weight_decay: float = 1e-4):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.dataset_anchors = dataset_anchors

        backbone = torchvision.models.mobilenet_v3_small(weights=backbone_weights).features

        if backbone_weights is not None:
            # https://www.mdpi.com/1424-8220/23/21/8763
            # https://link.springer.com/chapter/10.1007/978-3-031-37745-7_2
            averaged_weight = backbone[0][0].weight.mean(dim=1, keepdim=True)
            averaged_weight = torch.nn.Parameter(averaged_weight)

        backbone[0][0] = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if backbone_weights is not None:
            backbone[0][0].weight = averaged_weight

        x = torch.randn(1, 1, 224, 224)
        out = backbone(x)

        if isinstance(out, dict):
            first_key = list(out.keys())[0]
            channels = out[first_key].shape[1]
        else:
            channels = out.shape[1]

        # get number of input features for the classifier
        backbone.out_channels = channels
        num_classes = 2  # 1 class (lesion) + background

        anchor_generator = AnchorGenerator(
            sizes=DATASET_ANCHORS[dataset_anchors]["sizes"],
            aspect_ratios=DATASET_ANCHORS[dataset_anchors]["aspect_ratios"]
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        # replace the pre-trained head with a new one
        self.model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, roi_pooler=roi_pooler)

        self.model.transform = GeneralizedRCNNTransform(
            min_size=(800,),
            max_size=1333,
            image_mean=[0.0],  # 1-channel mean
            image_std=[1.0],  # 1-channel std
        )

        if backbone_weights is None:
            # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
            def init_weights(m):
                if isinstance(m, torch.nn.Module) and hasattr(m, "weight") and (
                        hasattr(m, "in_channels") or hasattr(m, "in_features")):
                    if isinstance(m, torch.nn.Conv2d):
                        y = m.in_channels
                    else:
                        y = m.in_features

                    m.weight.data.normal_(0.0, 1.0 / np.sqrt(y))

            self.model.apply(init_weights)

        self.metrics = build_metrics(2, task="region_proposal")

        self.save_hyperparameters()

    def forward(self, images: Tensor, targets: Optional[list] = None) -> Tensor:
        if targets is not None:
            # this function is necessary because torchvision models are supposed to return (losses, predictions)
            # during training and (predictions) during evaluation
            # for some reason, the model is returning predictions only during training
            return detection_forward(self.model, images, targets)
        else:
            return self.model(images)

    def _common_step(self, batch, prefix: Literal["train", "val"]):
        slices, targets = batch["slices"], batch["targets"]  # (B, C, H, W)

        losses_dict, proposals = self.forward(slices, targets)

        prefixed_loss_dict = {f"{prefix}_{name}": value for name, value in losses_dict.items()}
        prefixed_loss_dict[f"{prefix}_loss"] = sum(loss for loss in losses_dict.values())

        # preds = [
        #     {
        #         "boxes": p["boxes"],
        #         "scores": p["scores"],
        #         "labels": p["labels"]
        #     }
        #     for p in proposals
        # ]
        # targets = [
        #     {
        #         "boxes": t["boxes"],
        #         "labels": t["labels"]
        #     }
        #     for t in targets
        # ]
        log_dict = {
            **prefixed_loss_dict,
            # **compute_metrics(preds, targets, metrics=self.metrics, prefix=prefix, task="region_proposal")
        }

        self.log_dict(dictionary=log_dict, batch_size=slices.shape[0], on_step=False, prog_bar=True, on_epoch=True)

        return prefixed_loss_dict[f"{prefix}_loss"]

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
                                weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
