from collections import OrderedDict
from typing import Literal, Optional, List

import lightning as l
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from constants import DATASET_ANCHORS
from utils import build_metrics, compute_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# =========================================================================================
# Truenet model utility functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
# =========================================================================================


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernelsize, name, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            OrderedDict([(
                name + "conv", nn.Conv2d(in_channels, mid_channels, kernel_size=kernelsize, padding=1)),
                (name + "bn", nn.BatchNorm2d(mid_channels)),
                (name + "relu", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernelsize, name, mid_channels=None):
        super().__init__()
        pad = (kernelsize - 1) // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            OrderedDict([(
                name + "conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=kernelsize, padding=pad)),
                (name + "bn1", nn.BatchNorm2d(mid_channels)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "conv2", nn.Conv2d(mid_channels, out_channels, kernel_size=kernelsize, padding=pad)),
                (name + "bn2", nn.BatchNorm2d(out_channels)),
                (name + "relu2", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, name):
        super().__init__()
        pad = (kernel_size - 1) // 2
        mid_channels = out_channels
        self.maxpool_conv = nn.Sequential(
            OrderedDict([
                (name + "maxpool", nn.MaxPool2d(2)),
                (name + "conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad)),
                (name + "bn1", nn.BatchNorm2d(mid_channels)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "conv2", nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad)),
                (name + "bn2", nn.BatchNorm2d(out_channels)),
                (name + "relu2", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, name, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, name)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, 3, name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """convolution"""

    def __init__(self, in_channels, out_channels, name):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            OrderedDict([(
                name + "conv", nn.Conv2d(in_channels, out_channels, kernel_size=1)), ])
        )

    def forward(self, x):
        return self.conv(x)


# =========================================================================================
# Triplanar U-Net ensemble network (TrUE-Net) model
# Vaanathi Sundaresan
# 09-03-2021, Oxford
# =========================================================================================

class TrUENet(nn.Module):
    '''
    TrUE-Net model definition
    '''

    def __init__(self, n_channels, n_classes, init_channels, plane='axial', bilinear=False):
        super(TrUENet, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = OutConv(n_channels, 3, name="inpconv_")
        if plane == 'axial':
            self.convfirst = DoubleConv(3, init_channels, 3, name="convfirst_")
        else:
            self.convfirst = DoubleConv(3, init_channels, 5, name="convfirst_")
        self.down1 = Down(init_channels, init_channels * 2, 3, name="down1_")
        self.down2 = Down(init_channels * 2, init_channels * 4, 3, name="down2_")
        self.down3 = Down(init_channels * 4, init_channels * 8, 3, name="down3_")
        self.up3 = Up(init_channels * 8, init_channels * 4, 3, "up1_", bilinear)
        self.up2 = Up(init_channels * 4, init_channels * 2, 3, "up2_", bilinear)
        self.up1 = Up(init_channels * 2, init_channels, 3, "up3_", bilinear)
        self.outconv = OutConv(init_channels, n_classes, name="outconv_")

    def forward(self, x):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        return logits


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
                 backbone_lr: float = 1e-4,
                 roi_head_lr: float = 1e-3,
                 rpn_lr: float = 1e-3,
                 eps: float = 1e-8,
                 betas: tuple[float, float] = (0.9, 0.999),
                 dataset_anchors: Literal["ATLAS", "ISLES-DWI", "ISLES-FLAIR"] = "ATLAS",
                 backbone_weights: Optional[Literal["DEFAULT"]] = "DEFAULT",
                 weight_decay: float = 1e-4):
        super().__init__()

        self.backbone_lr = backbone_lr
        self.roi_head_lr = roi_head_lr
        self.rpn_lr = rpn_lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.dataset_anchors = dataset_anchors

        # backbone = torchvision.models.mobilenet_v3_small(weights=backbone_weights).features
        #
        # if backbone_weights is not None:
        #     # https://www.mdpi.com/1424-8220/23/21/8763
        #     # https://link.springer.com/chapter/10.1007/978-3-031-37745-7_2
        #     averaged_weight = backbone[0][0].weight.mean(dim=1, keepdim=True)
        #
        # backbone[0][0] = torch.nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #
        # if backbone_weights is not None:
        #     averaged_weight = averaged_weight.repeat(1, 2, 1, 1)
        #     backbone[0][0].weight = torch.nn.Parameter(averaged_weight)

        axial_truenet_ckpt = torch.load(
            "/media/neurone-pc13/7FD0B7F50A4D7BEE/Angelo/strokeformer/Truenet_MWSC_T1_axial.pth", map_location="cpu")
        axial_truenet = TrUENet(n_channels=1, n_classes=2, init_channels=64)

        new_axial_state_dict = OrderedDict()
        for key, value in axial_truenet_ckpt.items():
            name = key
            name = name.replace("module.", "")
            new_axial_state_dict[name] = value

        axial_truenet.load_state_dict(new_axial_state_dict)
        old_inpcov = axial_truenet.inpconv.conv[0].weight.clone()
        new_inpconv = old_inpcov.repeat(1, 2, 1, 1)
        axial_truenet.inpconv = OutConv(2, 3, name="inpconv_")
        axial_truenet.inpconv.conv[0].weight = torch.nn.Parameter(new_inpconv)

        # encoder only
        backbone = nn.Sequential(
            axial_truenet.inpconv,
            axial_truenet.convfirst,
            axial_truenet.down1,
            axial_truenet.down2,
            axial_truenet.down3,
        )

        for pname, param in backbone.named_parameters():
            if "inpconv" in pname or "down1" in pname:
                param.requires_grad = False

        x = torch.randn(1, 2, 192, 192)
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
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes=2  # background + stroke
        )
        self.model.transform = GeneralizedRCNNTransform(
            min_size=192,
            max_size=192,
            image_mean=[0.0], # disable internal normalization
            image_std=[1.0],
            size_divisible=32  # important for FPN / RPN stability
        )

        # if backbone_weights is None:
        #     # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        #     def init_weights(m):
        #         if isinstance(m, torch.nn.Module) and hasattr(m, "weight") and (
        #                 hasattr(m, "in_channels") or hasattr(m, "in_features")):
        #             if isinstance(m, torch.nn.Conv2d):
        #                 y = m.in_channels
        #             else:
        #                 y = m.in_features
        #
        #             m.weight.data.normal_(0.0, 1.0 / np.sqrt(y))
        #
        #     self.model.apply(init_weights)

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

        # metrics = compute_metrics(preds, targets, metrics=self.metrics, prefix=prefix, task="region_proposal")

        # if "classes" in metrics:
        #     del metrics["classes"]

        log_dict = {
            **prefixed_loss_dict,
            # **metrics
        }

        self.log_dict(dictionary=log_dict, batch_size=slices.shape[0], on_step=False, prog_bar=True, on_epoch=True)

        return prefixed_loss_dict[f"{prefix}_loss"]

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {"params": [p for p in self.model.backbone.parameters() if p.requires_grad], "lr": self.backbone_lr},
            {"params": self.model.roi_heads.parameters(), "lr": self.roi_head_lr},
            {"params": self.model.rpn.parameters(), "lr": self.rpn_lr},
        ],  lr=self.rpn_lr, betas=self.betas, eps=self.eps,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
