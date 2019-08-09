import cv2
import torch
import numpy as np
from skimage import measure

from .metrics import bb_intersection_over_union_numpy

def extract_masks_rects(mask):
    rectangles = []
    masks = []
    classes = []
    for i in np.unique(mask):
        if i == 0:
            continue
        classes.append(i)

        i_mask = mask == i
        contours, _ = cv2.findContours(i_mask.astype(np.uint8), 1, 2)
        biggest_cntr = max(contours, key=cv2.contourArea)
        rect = cv2.boundingRect(biggest_cntr)
        rectangles.append(rect)

        x, y, w, h = rect
        i_mask = np.zeros_like(i_mask)
        i_mask[y:y + h, x:x + w] = True
        masks.append(i_mask)
    return masks, rectangles, classes


def process_entry_numpy(in_img, pred_mask, true_mask, filter_masks=True):
    assert len(in_img.shape) == 2
    pred_mask = measure.label(pred_mask, background=0)
    _, pred_rectangles, _ = extract_masks_rects(pred_mask)
    _, true_rectangles, true_classes = extract_masks_rects(true_mask)

    pred_classes = [0] * len(pred_rectangles)
    for rect_i, rect in enumerate(pred_rectangles):
        best_iou = 0.5
        for class_id, true_rect in zip(true_classes, true_rectangles):
            iou = bb_intersection_over_union_numpy(rect, true_rect)
            if iou > best_iou:
                pred_classes[rect_i] = class_id
                best_iou = iou
    projections = []
    classes = []
    rectangles = []
    in_img = in_img.astype(np.float)
    for class_id, pred_rect in zip(pred_classes, pred_rectangles):
        if class_id == 0 and filter_masks:
            continue
        classes.append(class_id - 1)
        x, y, w, h = pred_rect
        patch = cv2.resize(in_img[y:y+h, x:x+w], (100, 100))
        x_projection = patch.sum(0)
        y_projection = patch.sum(1)
        projections.append([[x_projection], [y_projection]])
        rectangles.append(pred_rect)
    return projections, rectangles, classes


def process_batch_numpy(in_img, pred_mask, label_mask, filter_masks=True):
    assert len(pred_mask.shape) == 3
    assert len(label_mask.shape) == 3
    N = pred_mask.shape[0]

    projections, classes, rectangles, image_index = [], [], [], []
    for i in range(N):
        i_projections, i_rectangles, i_classes = process_entry_numpy(in_img[i],
                                                                     pred_mask[i],
                                                                     label_mask[i],
                                                                     filter_masks)
        projections += i_projections
        classes += i_classes
        rectangles += i_rectangles
        image_index += [i] * len(i_classes)
    projections = np.array(projections, dtype=np.float)
    classes = np.array(classes, dtype=np.int)
    rectangles = np.array(rectangles, dtype=np.int)
    return projections, rectangles, classes, image_index


def process_batch_torch_wrap(batch_img, batch_pred, batch_mask, filter_masks=True):
    batch_img = batch_img.detach().squeeze(1).numpy()
    batch_pred = batch_pred.detach().squeeze(1).numpy()
    batch_mask = batch_mask.detach().numpy()
    projections, rectangles, classes, image_index = process_batch_numpy(batch_img,
                                                           batch_pred,
                                                           batch_mask,
                                                           filter_masks)
    projections = torch.from_numpy(projections).float()
    classes = torch.from_numpy(classes).long()
    rectangles = torch.from_numpy(rectangles).long()
    return projections, rectangles, classes, image_index
