import cv2
import os

import numpy as np

def iteration(image, value):
    """
    Provides RLSA smothing.
    :param image: initial image.
    :param value: smothing threshold.
    :return: 
    """

    rows, cols = image.shape
    mask = image.copy()
    for row in range(0, rows):
        try:
            start = mask[row].tolist().index(0)
        except ValueError:
            start = 0

        count = start
        for col in range(start, cols):
            if mask[row, col] == 0:
                if 0 < (col - count) <= value:
                    mask[row, count:col] = 0
                count = col
    return mask

def rlsa(image, c_h, c_v, c_a):
    pass


if __name__ == "__main__":
    # print(os.listdir("../data/supervisely/zbirnyk"))
    img = cv2.imread("../out/out.jpg", 0)
    cv2.imwrite("../out/new.jpg", iteration(img, 10))


