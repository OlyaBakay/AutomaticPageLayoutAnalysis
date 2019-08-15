import numpy as np


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