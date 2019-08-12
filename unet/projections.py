import collections

import cv2
import torch
import numpy as np
from skimage import measure

from .metrics import bb_intersection_over_union_numpy

def extract_masks_rects(mask):
    rectangles = []
    masks = []
    classes = []
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        labeled_mask = measure.label(mask == class_id, background=0)
        for i in np.unique(labeled_mask):
            if i == 0:
                continue
            classes.append(class_id)

            i_mask = labeled_mask == i
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
    _, pred_rectangles, _ = extract_masks_rects(pred_mask)
    _, true_rectangles, true_classes = extract_masks_rects(true_mask)

    pred_classes = [0] * len(pred_rectangles)
    pred_true_iou = [0.5] * len(pred_rectangles)
    true_indices = [-1] * len(pred_rectangles)
    for rect_i, rect in enumerate(pred_rectangles):
        for true_obj_i, (class_id, true_rect) in enumerate(zip(true_classes, true_rectangles)):
            iou = bb_intersection_over_union_numpy(rect, true_rect)
            if iou >= pred_true_iou[rect_i]:
                pred_classes[rect_i] = class_id
                pred_true_iou[rect_i] = iou
                true_indices[rect_i] = true_obj_i

    projections = []
    classes = []
    rectangles = []
    in_img = in_img.astype(np.float)
    is_filtered = [True] * len(pred_rectangles)
    for index, (class_id, pred_rect) in enumerate(zip(pred_classes, pred_rectangles)):
        if class_id == 0 and filter_masks:
            continue
        is_filtered[index] = False
        classes.append(class_id - 1)
        x, y, w, h = pred_rect
        patch = cv2.resize(in_img[y:y+h, x:x+w], (100, 100))
        x_projection = patch.sum(0)
        y_projection = patch.sum(1)
        projections.append([[x_projection], [y_projection]])
        rectangles.append(pred_rect)

    pred_true_iou = [iou for i, iou in enumerate(pred_true_iou) if is_filtered[i] is False]
    true_indices = [index for i, index in enumerate(true_indices) if is_filtered[i] is False]
    true_obj_pred_obj_map = [-1] * len(true_classes)
    for pred_obj_index in np.argsort(pred_true_iou):
        true_obj_index = true_indices[pred_obj_index]
        if true_obj_index > -1:
            true_obj_pred_obj_map[true_obj_index] = pred_obj_index

    return projections, rectangles, classes, true_obj_pred_obj_map


def process_batch_numpy(in_img, pred_mask, label_mask, filter_masks=True):
    assert len(pred_mask.shape) == 3
    assert len(label_mask.shape) == 3
    N = pred_mask.shape[0]

    projections, classes, rectangles, image_index, true_pred = [], [], [], [], []
    for i in range(N):
        i_projections, i_rectangles, i_classes, i_true_pred = process_entry_numpy(in_img[i],
                                                                     pred_mask[i],
                                                                     label_mask[i],
                                                                     filter_masks)
        projections += i_projections
        classes += i_classes
        rectangles += i_rectangles
        image_index += [i] * len(i_classes)
        true_pred += i_true_pred
    projections = np.array(projections, dtype=np.float)
    classes = np.array(classes, dtype=np.int)
    rectangles = np.array(rectangles, dtype=np.int)
    true_pred = np.array(true_pred, dtype=np.int)
    return projections, rectangles, classes, image_index, true_pred


def process_batch_torch_wrap(batch_img, batch_pred, batch_mask, filter_masks=True):
    batch_img = batch_img.detach().squeeze(1).numpy()
    batch_pred = batch_pred.detach().squeeze(1).numpy()
    batch_mask = batch_mask.detach().numpy()
    projections, rectangles, classes, image_index, true_pred = process_batch_numpy(batch_img,
                                                           batch_pred,
                                                           batch_mask,
                                                           filter_masks)
    projections = torch.from_numpy(projections).float()
    classes = torch.from_numpy(classes).long()
    rectangles = torch.from_numpy(rectangles).long()
    true_pred = torch.from_numpy(true_pred).long()

    return projections, rectangles, classes, image_index, true_pred


def process_patches(batch_img, rectangles, indices):
    batch_img = batch_img.cpu().detach().numpy()
    resized_patches = []
    for index, rect in zip(indices, rectangles):
        img = batch_img[index]
        x, y, w, h = rect
        patch = img[0, y:y+h, x:x+w]
        patch = cv2.resize(patch, (128, 32))
        resized_patches.append(patch)
    resized_patches = np.array(resized_patches)
    return torch.from_numpy(resized_patches).unsqueeze(1)
