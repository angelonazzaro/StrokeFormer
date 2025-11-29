from collections import OrderedDict
from typing import Literal, Optional, List

import lightning as l
import torch
import torch.optim as optim
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

from utils import build_metrics, compute_metrics


def eval_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

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
    model.rpn.training=True

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
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
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
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # noqa
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


class RPN(l.LightningModule):

    def __init__(self,
                 lr: float = 1e-4,
                 eps: float = 1e-8,
                 betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-4):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2  # 1 class (lesion) + background
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.metrics = build_metrics(2, task="region_proposal")

        self.save_hyperparameters()

    def forward(self, images: Tensor, targets: Optional[list]) -> Tensor:
        if targets is not None:
            if self.training:
                return self.model(images, targets)
            else:
                return eval_forward(self.model, images, targets)
        else:
            return self.model(images)

    def _common_step(self, batch, prefix: Literal["train", "val"]):
        slices, targets = batch["slices"], batch["targets"]  # (B, C, H, W)

        # torchvision model return a dictionary containing the losses during training mode
        # getting the predictions would be setting the model to eval and perform another forward step
        output = self.forward(slices, targets)
        if type(output) == dict:
            losses_dict = output
        else:
            losses_dict, proposals = output

        prefixed_loss_dict = {f"{prefix}_{name}": value for name, value in losses_dict.items()}
        prefixed_loss_dict[f"{prefix}_loss"] = sum(loss for loss in losses_dict.values())

        log_dict = {
            **prefixed_loss_dict,
        }

        # if type(output) == tuple:
        #     preds = [
        #         {
        #             "boxes": p["boxes"],
        #             "scores": p["scores"],
        #             "labels": p["labels"]
        #         }
        #         for p in proposals
        #     ]
        #     targets = [
        #         {
        #             "boxes": t["boxes"],
        #             "labels": t["labels"]
        #         }
        #         for t in targets
        #     ]
        #     log_dict.update(**compute_metrics(preds, targets, metrics=self.metrics, prefix=prefix, task="region_proposal"))

        self.log_dict(dictionary=log_dict, batch_size=slices.shape[0], on_step=False, prog_bar=True, on_epoch=True)

        return prefixed_loss_dict[f"{prefix}_loss"]

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
                                weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }