import torch
import numpy as np

from .projections import extract_masks_rects
from .object_detection_metrics.BoundingBox import BoundingBox
from .object_detection_metrics.BoundingBoxes import BoundingBoxes
from .object_detection_metrics.utils import BBFormat, BBType
from .object_detection_metrics.Evaluator import Evaluator
from .iou import bb_intersection_over_union_numpy

SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, reduce=True):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = outputs.sigmoid() > 0.5
    labels = labels.squeeze(1) > 0.5

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    # print("!", intersection.shape, intersection)
    # print("#", union.shape, union)
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    if reduce:
        thresholded = thresholded.mean()

    return thresholded





def accuracy(pred, true):
    return ((pred == true).float()).mean()


def accuracy_wrapper(pred, true):
    return accuracy(pred.argmax(1).long(), true.long())


def special_accuracy(pred, true, true_pred_map):
    pred = pred.argmax(1).long().cpu()
    true = true.long().cpu()
    intersection = torch.zeros(len(true_pred_map), dtype=torch.uint8)
    should_count_mask = true_pred_map > -1
    # print(true_pred_map.dtype, pred.dtype, true.dtype, should_count_mask.dtype)
    # print(true_pred_map.shape, pred.shape, true.shape, should_count_mask.shape)
    intersection[should_count_mask] = (pred == true)[true_pred_map[should_count_mask]]
    return intersection.float().mean()


def maP_create_boxes(pred_rectangles, pred_classes, image_indeces, label_mask, relative_index=0):
    """

    :param pred_rectangles:
    :param pred_classes: torch.Tensor(N, C) logits
    :param image_indeces:
    :param label_mask:
    :return:
    """
    label_mask = label_mask.detach().cpu().numpy()
    pred_rectangles = pred_rectangles.detach().cpu().numpy()
    pred_classes = pred_classes.softmax(1).detach().cpu().numpy()

    boxes = []
    N = label_mask.shape[0]
    for image_index in range(N):
        _, label_rectangles, label_classes = extract_masks_rects(label_mask[image_index])
        for rect, class_id in zip(label_rectangles, label_classes):
            x, y, w, h = rect
            box = BoundingBox(
                imageName=str(relative_index + image_index),
                classId=class_id - 1,
                x=x,y=y,w=w,h=h,
                bbType=BBType.GroundTruth,
                classConfidence=1,
            )
            boxes.append(box)

    for image_index, rect, class_distribution in zip(image_indeces, pred_rectangles, pred_classes):
        pred_class = np.argmax(class_distribution)
        confidence = class_distribution[pred_class]
        x, y, w, h = rect
        box = BoundingBox(
            imageName=str(relative_index + image_index),
            classId=pred_class,
            x=x,y=y,w=w,h=h,
            bbType=BBType.Detected,
            classConfidence=confidence,
        )
        boxes.append(box)
    return boxes


def mAP_wrapper(pred_rectangles, pred_classes, image_indeces, label_mask, relative_index=0):
    """

    :param pred_rectangles:
    :param pred_classes: torch.Tensor(N, C) logits
    :param image_indeces:
    :param label_mask:
    :return:
    """
    boxes_ = maP_create_boxes(pred_rectangles, pred_classes, image_indeces, label_mask, relative_index)
    boxes = BoundingBoxes()
    for box in boxes_:
        boxes.addBoundingBox(box)
    res = Evaluator().GetPascalVOCMetrics(boxes, IOUThreshold=0.5)
    # print(len(res))
    return res


def mAP_wrapper_from_boxes(boxes):
    res = Evaluator().GetPascalVOCMetrics(boxes, IOUThreshold=0.5)
    # print(len(res))
    return res
