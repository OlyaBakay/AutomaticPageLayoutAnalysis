import torch
import numpy as np


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


def bb_intersection_over_union_numpy(boxA, boxB):
    """
    Find out iou of two rectangles
    :param boxA: cv2.rectangle - tuple(x, y, w, h)
    :param boxB: cv2.rectangle - tuple(x, y, w, h)
    :return: iou
    """
    # from (x,y,w,h) to (xA,yA,xB,yB)
    # where A is top left and B is bottom right
    boxA, boxB = np.array(boxA), np.array(boxB)
    boxA[2:] += boxA[:2]
    boxB[2:] += boxB[:2]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
