"""
fasterrcnn_model.py — Faster R-CNN from scratch (Naveen's architecture).

Extracted from FasterRCNN_Project.ipynb so the GUI can import and load
the trained weights without running the full notebook.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms, roi_align


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

ANCHOR_SIZES = [32, 64, 128, 256]
ANCHOR_ASPECT_RATIOS = [0.5, 1.0, 2.0]
ANCHOR_STRIDES = [4, 8, 16, 32]

RPN_PRE_NMS_TOPK_TRAIN = 1000
RPN_PRE_NMS_TOPK_TEST = 1000
RPN_POST_NMS_TOPK_TRAIN = 200
RPN_POST_NMS_TOPK_TEST = 150
RPN_NMS_THRESHOLD = 0.7
RPN_FG_IOU_THRESHOLD = 0.7
RPN_BG_IOU_THRESHOLD = 0.3
RPN_BATCH_SIZE_PER_IMAGE = 256
RPN_POSITIVE_FRACTION = 0.5

ROI_OUTPUT_SIZE = 7
ROI_BATCH_SIZE_PER_IMAGE = 128
ROI_POSITIVE_FRACTION = 0.25
ROI_FG_IOU_THRESHOLD = 0.5
ROI_BG_IOU_THRESHOLD = 0.5

DETECTIONS_PER_IMAGE = 100
MODEL_SCORE_THRESHOLD = 0.05
MODEL_NMS_THRESHOLD = 0.5
CROSS_CLASS_NMS_THRESHOLD = 0.0
MIN_BOX_SIZE = 4.0


# ---------------------------------------------------------------------------
# Box utility functions
# ---------------------------------------------------------------------------

def clip_boxes_to_image(boxes, image_shape):
    if boxes.numel() == 0:
        return boxes
    height, width = image_shape
    x1 = boxes[:, 0].clamp(0, width - 1)
    y1 = boxes[:, 1].clamp(0, height - 1)
    x2 = boxes[:, 2].clamp(0, width - 1)
    y2 = boxes[:, 3].clamp(0, height - 1)
    return torch.stack((x1, y1, x2, y2), dim=1)


def box_area(boxes):
    if boxes.numel() == 0:
        return boxes.new_zeros((boxes.shape[0],))
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def remove_small_boxes(boxes, min_size):
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)
    return torch.where(keep)[0]


def encode_boxes(reference_boxes, proposals):
    if reference_boxes.numel() == 0 or proposals.numel() == 0:
        return proposals.new_zeros((proposals.shape[0], 4))
    proposal_widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    proposal_heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    proposal_ctr_x = proposals[:, 0] + 0.5 * proposal_widths
    proposal_ctr_y = proposals[:, 1] + 0.5 * proposal_heights
    gt_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    gt_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
    dx = (gt_ctr_x - proposal_ctr_x) / proposal_widths
    dy = (gt_ctr_y - proposal_ctr_y) / proposal_heights
    dw = torch.log(gt_widths / proposal_widths)
    dh = torch.log(gt_heights / proposal_heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(rel_codes, boxes):
    if rel_codes.numel() == 0 or boxes.numel() == 0:
        return boxes.new_zeros((boxes.shape[0], 4))
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = rel_codes[:, 0]
    dy = rel_codes[:, 1]
    dw = rel_codes[:, 2].clamp(max=math.log(1000.0 / 16.0))
    dh = rel_codes[:, 3].clamp(max=math.log(1000.0 / 16.0))
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)
    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=1)


def sample_binary_labels(labels, batch_size, positive_fraction):
    positive_indices = torch.where(labels == 1)[0]
    negative_indices = torch.where(labels == 0)[0]
    num_positive = min(int(batch_size * positive_fraction), positive_indices.numel())
    num_negative = min(batch_size - num_positive, negative_indices.numel())
    if positive_indices.numel() > 0:
        positive_indices = positive_indices[torch.randperm(positive_indices.numel(), device=labels.device)[:num_positive]]
    if negative_indices.numel() > 0:
        negative_indices = negative_indices[torch.randperm(negative_indices.numel(), device=labels.device)[:num_negative]]
    return torch.cat((positive_indices, negative_indices), dim=0)


def sample_roi_labels(labels, batch_size, positive_fraction):
    positive_indices = torch.where(labels > 0)[0]
    negative_indices = torch.where(labels == 0)[0]
    num_positive = min(int(batch_size * positive_fraction), positive_indices.numel())
    num_negative = min(batch_size - num_positive, negative_indices.numel())
    if positive_indices.numel() > 0:
        positive_indices = positive_indices[torch.randperm(positive_indices.numel(), device=labels.device)[:num_positive]]
    if negative_indices.numel() > 0:
        negative_indices = negative_indices[torch.randperm(negative_indices.numel(), device=labels.device)[:num_negative]]
    return torch.cat((positive_indices, negative_indices), dim=0)


# ---------------------------------------------------------------------------
# RPN helper functions
# ---------------------------------------------------------------------------

def reshape_rpn_predictions(objectness, bbox_deltas, num_anchors):
    batch_size, _, height, width = objectness.shape
    objectness = objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
    bbox_deltas = bbox_deltas.view(batch_size, num_anchors, 4, height, width)
    bbox_deltas = bbox_deltas.permute(0, 3, 4, 1, 2).reshape(batch_size, -1, 4)
    return objectness, bbox_deltas


def match_anchors_to_gt(anchors, gt_boxes, fg_iou_threshold, bg_iou_threshold):
    device = anchors.device
    labels = torch.full((anchors.shape[0],), -1.0, device=device)
    matched_indices = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=device)
    if gt_boxes.numel() == 0:
        labels.fill_(0.0)
        regression_targets = anchors.new_zeros((anchors.shape[0], 4))
        return labels, matched_indices, regression_targets
    ious = box_iou(anchors, gt_boxes)
    max_iou_per_anchor, matched_indices = ious.max(dim=1)
    labels[max_iou_per_anchor < bg_iou_threshold] = 0.0
    labels[max_iou_per_anchor >= fg_iou_threshold] = 1.0
    gt_best_anchor = ious.argmax(dim=0)
    labels[gt_best_anchor] = 1.0
    matched_indices[gt_best_anchor] = torch.arange(gt_boxes.shape[0], device=device)
    matched_gt_boxes = gt_boxes[matched_indices.clamp(min=0)]
    regression_targets = encode_boxes(matched_gt_boxes, anchors)
    return labels, matched_indices, regression_targets


def assign_targets_to_proposals(proposals, gt_boxes, gt_labels, fg_iou_threshold, bg_iou_threshold):
    device = proposals.device
    labels = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=device)
    matched_indices = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=device)
    if gt_boxes.numel() == 0:
        labels.fill_(0)
        regression_targets = proposals.new_zeros((proposals.shape[0], 4))
        return matched_indices, labels, regression_targets
    ious = box_iou(proposals, gt_boxes)
    max_iou_per_proposal, matched_indices = ious.max(dim=1)
    labels = gt_labels[matched_indices]
    labels[max_iou_per_proposal < bg_iou_threshold] = 0
    ignore_mask = (max_iou_per_proposal >= bg_iou_threshold) & (max_iou_per_proposal < fg_iou_threshold)
    labels[ignore_mask] = -1
    regression_targets = encode_boxes(gt_boxes[matched_indices.clamp(min=0)], proposals)
    return matched_indices, labels, regression_targets


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = bottleneck_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.out_channels = [256, 512, 1024, 2048]
        self._init_weights()

    def _make_layer(self, bottleneck_channels, blocks, stride):
        downsample = None
        out_channels = bottleneck_channels * Bottleneck.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self.in_channels, bottleneck_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, bottleneck_channels, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.out_channels = out_channels
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features):
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class AnchorGenerator(nn.Module):
    def __init__(self, sizes, aspect_ratios, strides):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.num_anchors_per_location = len(aspect_ratios)

    def generate_base_anchors(self, size, device):
        anchors = []
        area = float(size * size)
        for ar in self.aspect_ratios:
            width = math.sqrt(area / ar)
            height = ar * width
            anchors.append([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def grid_anchors(self, feature_shape, stride, size, device):
        height, width = feature_shape
        shifts_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)
        base_anchors = self.generate_base_anchors(size, device)
        anchors = shifts[:, None, :] + base_anchors[None, :, :]
        return anchors.reshape(-1, 4)

    def forward(self, features):
        anchors = []
        for feature, stride, size in zip(features, self.strides, self.sizes):
            anchors.append(self.grid_anchors(feature.shape[-2:], stride, size, feature.device))
        return anchors


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in [self.conv, self.objectness_logits, self.bbox_deltas]:
            nn.init.normal_(module.weight, std=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, features):
        objectness = []
        pred_bbox_deltas = []
        for feature in features:
            hidden = self.relu(self.conv(feature))
            objectness.append(self.objectness_logits(hidden))
            pred_bbox_deltas.append(self.bbox_deltas(hidden))
        return objectness, pred_bbox_deltas


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2]:
            nn.init.normal_(module.weight, std=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes + 1)
        self.bbox_pred = nn.Linear(in_channels, (num_classes + 1) * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0.0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0.0)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


class FasterRCNNFromScratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet50Backbone()
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=256)
        self.anchor_generator = AnchorGenerator(ANCHOR_SIZES, ANCHOR_ASPECT_RATIOS, ANCHOR_STRIDES)
        self.rpn_head = RPNHead(256, len(ANCHOR_ASPECT_RATIOS))
        self.box_head = TwoMLPHead(256 * ROI_OUTPUT_SIZE * ROI_OUTPUT_SIZE, 1024)
        self.box_predictor = FastRCNNPredictor(1024, num_classes)

        self.rpn_pre_nms_topk_train = RPN_PRE_NMS_TOPK_TRAIN
        self.rpn_pre_nms_topk_test = RPN_PRE_NMS_TOPK_TEST
        self.rpn_post_nms_topk_train = RPN_POST_NMS_TOPK_TRAIN
        self.rpn_post_nms_topk_test = RPN_POST_NMS_TOPK_TEST
        self.rpn_nms_threshold = RPN_NMS_THRESHOLD
        self.rpn_fg_iou_threshold = RPN_FG_IOU_THRESHOLD
        self.rpn_bg_iou_threshold = RPN_BG_IOU_THRESHOLD
        self.rpn_batch_size_per_image = RPN_BATCH_SIZE_PER_IMAGE
        self.rpn_positive_fraction = RPN_POSITIVE_FRACTION
        self.roi_batch_size_per_image = ROI_BATCH_SIZE_PER_IMAGE
        self.roi_positive_fraction = ROI_POSITIVE_FRACTION
        self.roi_fg_iou_threshold = ROI_FG_IOU_THRESHOLD
        self.roi_bg_iou_threshold = ROI_BG_IOU_THRESHOLD
        self.model_score_threshold = MODEL_SCORE_THRESHOLD
        self.model_nms_threshold = MODEL_NMS_THRESHOLD
        self.cross_class_nms_threshold = CROSS_CLASS_NMS_THRESHOLD
        self.detections_per_image = DETECTIONS_PER_IMAGE
        self.min_box_size = MIN_BOX_SIZE

    def extract_features(self, images):
        backbone_features = self.backbone(images)
        return self.fpn(backbone_features)

    def flatten_rpn_outputs(self, objectness, pred_bbox_deltas):
        flattened_objectness = []
        flattened_bbox_deltas = []
        anchors_per_level = []
        num_anchors = len(ANCHOR_ASPECT_RATIOS)
        for objectness_level, bbox_level in zip(objectness, pred_bbox_deltas):
            objectness_level, bbox_level = reshape_rpn_predictions(objectness_level, bbox_level, num_anchors)
            flattened_objectness.append(objectness_level)
            flattened_bbox_deltas.append(bbox_level)
            anchors_per_level.append(objectness_level.shape[1])
        return torch.cat(flattened_objectness, dim=1), torch.cat(flattened_bbox_deltas, dim=1), anchors_per_level

    def filter_proposals(self, objectness, pred_bbox_deltas, anchors, anchors_per_level, image_shape):
        proposals_per_image = []
        pre_nms_topk = self.rpn_pre_nms_topk_train if self.training else self.rpn_pre_nms_topk_test
        post_nms_topk = self.rpn_post_nms_topk_train if self.training else self.rpn_post_nms_topk_test

        for image_index in range(objectness.shape[0]):
            scores = objectness[image_index].sigmoid()
            deltas = pred_bbox_deltas[image_index]
            decoded = decode_boxes(deltas, anchors)
            decoded = clip_boxes_to_image(decoded, image_shape)

            level_boxes = []
            level_scores = []
            level_ids = []
            start = 0
            for level_index, count in enumerate(anchors_per_level):
                end = start + count
                boxes_level = decoded[start:end]
                scores_level = scores[start:end]
                num_topk = min(pre_nms_topk, scores_level.numel())
                topk_scores, topk_indices = scores_level.topk(num_topk)
                boxes_level = boxes_level[topk_indices]
                keep = remove_small_boxes(boxes_level, self.min_box_size)
                boxes_level = boxes_level[keep]
                topk_scores = topk_scores[keep]
                if boxes_level.numel() > 0:
                    level_boxes.append(boxes_level)
                    level_scores.append(topk_scores)
                    level_ids.append(torch.full((boxes_level.shape[0],), level_index, dtype=torch.int64, device=boxes_level.device))
                start = end

            if not level_boxes:
                fallback_count = min(16, decoded.shape[0])
                fallback_indices = scores.topk(fallback_count).indices
                proposals_per_image.append(decoded[fallback_indices])
                continue

            boxes_image = torch.cat(level_boxes, dim=0)
            scores_image = torch.cat(level_scores, dim=0)
            level_ids_image = torch.cat(level_ids, dim=0)
            keep = batched_nms(boxes_image, scores_image, level_ids_image, self.rpn_nms_threshold)
            keep = keep[:post_nms_topk]
            proposals = boxes_image[keep]
            if proposals.shape[0] == 0:
                proposals = decoded[scores.topk(min(16, decoded.shape[0])).indices]
            proposals_per_image.append(proposals)

        return proposals_per_image

    def roi_pool(self, features, proposals, image_shape):
        total_proposals = sum(p.shape[0] for p in proposals)
        if total_proposals == 0:
            return next(iter(features.values())).new_zeros((0, 256, ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE))

        rois = []
        levels = []
        for batch_index, proposals_per_image in enumerate(proposals):
            if proposals_per_image.numel() == 0:
                continue
            batch_indices = torch.full((proposals_per_image.shape[0], 1), batch_index, device=proposals_per_image.device)
            rois.append(torch.cat((batch_indices, proposals_per_image), dim=1))
            box_sizes = torch.sqrt(box_area(proposals_per_image).clamp(min=1e-6))
            target_levels = torch.floor(4 + torch.log2(box_sizes / 224.0 + 1e-6))
            target_levels = target_levels.clamp(min=2, max=5).to(torch.int64)
            levels.append(target_levels)

        rois = torch.cat(rois, dim=0)
        levels = torch.cat(levels, dim=0)
        feature_dtype = next(iter(features.values())).dtype
        feature_device = next(iter(features.values())).device
        pooled = torch.zeros((rois.shape[0], 256, ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), dtype=feature_dtype, device=feature_device)

        feature_map_lookup = {2: features["p2"], 3: features["p3"], 4: features["p4"], 5: features["p5"]}
        spatial_scale_lookup = {2: 1.0 / ANCHOR_STRIDES[0], 3: 1.0 / ANCHOR_STRIDES[1], 4: 1.0 / ANCHOR_STRIDES[2], 5: 1.0 / ANCHOR_STRIDES[3]}

        for level in [2, 3, 4, 5]:
            indices = torch.where(levels == level)[0]
            if indices.numel() == 0:
                continue
            pooled_level = roi_align(
                feature_map_lookup[level],
                rois[indices],
                output_size=(ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE),
                spatial_scale=spatial_scale_lookup[level],
                aligned=True,
            )
            pooled[indices] = pooled_level

        return pooled

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shape):
        scores = F.softmax(class_logits, dim=1)
        box_regression = box_regression.view(class_logits.shape[0], self.num_classes + 1, 4)
        proposal_counts = [p.shape[0] for p in proposals]

        detections = []
        start = 0
        for proposals_per_image, proposal_count in zip(proposals, proposal_counts):
            end = start + proposal_count
            image_scores = scores[start:end]
            image_box_regression = box_regression[start:end]
            start = end

            boxes_all = []
            scores_all = []
            labels_all = []
            for class_id in range(1, self.num_classes + 1):
                all_class_scores = image_scores[:, class_id]
                score_keep = torch.where(all_class_scores > self.model_score_threshold)[0]
                if score_keep.numel() == 0:
                    continue
                boxes = decode_boxes(image_box_regression[score_keep, class_id], proposals_per_image[score_keep])
                boxes = clip_boxes_to_image(boxes, image_shape)
                kept_scores = all_class_scores[score_keep]
                size_keep = remove_small_boxes(boxes, self.min_box_size)
                if size_keep.numel() == 0:
                    continue
                boxes = boxes[size_keep]
                kept_scores = kept_scores[size_keep]
                keep = nms(boxes, kept_scores, self.model_nms_threshold)
                boxes_all.append(boxes[keep])
                scores_all.append(kept_scores[keep])
                labels_all.append(torch.full((keep.numel(),), class_id, dtype=torch.long, device=boxes.device))

            if boxes_all:
                boxes = torch.cat(boxes_all, dim=0)
                scores_per_image = torch.cat(scores_all, dim=0)
                labels = torch.cat(labels_all, dim=0)
                if self.cross_class_nms_threshold > 0:
                    keep = nms(boxes, scores_per_image, self.cross_class_nms_threshold)
                    boxes = boxes[keep]
                    scores_per_image = scores_per_image[keep]
                    labels = labels[keep]
                if scores_per_image.numel() > self.detections_per_image:
                    top_indices = scores_per_image.topk(self.detections_per_image).indices
                    boxes = boxes[top_indices]
                    scores_per_image = scores_per_image[top_indices]
                    labels = labels[top_indices]
            else:
                device = proposals_per_image.device
                boxes = torch.zeros((0, 4), device=device)
                scores_per_image = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            detections.append({"boxes": boxes, "scores": scores_per_image, "labels": labels})

        return detections

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Targets are required during training.")

        image_shape = tuple(images.shape[-2:])
        pyramid_features = self.extract_features(images)
        feature_list = [pyramid_features["p2"], pyramid_features["p3"], pyramid_features["p4"], pyramid_features["p5"]]

        objectness, pred_bbox_deltas = self.rpn_head(feature_list)
        objectness, pred_bbox_deltas, anchors_per_level = self.flatten_rpn_outputs(objectness, pred_bbox_deltas)
        anchors = torch.cat(self.anchor_generator(feature_list), dim=0)

        proposals = self.filter_proposals(objectness, pred_bbox_deltas, anchors, anchors_per_level, image_shape)

        if self.training:
            rpn_objectness_loss, rpn_box_loss = self.compute_rpn_loss(objectness, pred_bbox_deltas, anchors, targets)
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        pooled_features = self.roi_pool(pyramid_features, proposals, image_shape)
        box_features = self.box_head(pooled_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if self.training:
            roi_classifier_loss, roi_box_loss = self.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            return {
                "loss_rpn_objectness": rpn_objectness_loss,
                "loss_rpn_box_reg": rpn_box_loss,
                "loss_roi_classifier": roi_classifier_loss,
                "loss_roi_box_reg": roi_box_loss,
            }

        return self.postprocess_detections(class_logits, box_regression, proposals, image_shape)
