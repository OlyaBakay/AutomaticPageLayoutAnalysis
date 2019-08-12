import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

from utils import supervisely


def semisuper_contour_gt(img):
    mask = img != 255
    coords = np.argwhere(mask)
    # indices in unique/sorted by first coord
    indeces = np.unique(coords[:,0], return_index=True)[1]
    # for each Y first in line, by X
    bounds1 = coords[indeces]
    # last of current line
    # then reverse so we can create a controur
    # going from left top to left bottom
    # then from right bottom to right top !!!
    bounds2 = coords[np.roll(indeces - 1, -1)][::-1]

    bounds1[:,[1,0]] = bounds1[:,[0,1]]
    bounds2[:,[1,0]] = bounds2[:,[0,1]]
    hull = cv2.convexHull(np.concatenate((bounds1, bounds2)))
    mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, (1), -1)
    return mask


class MaskDataset(Dataset):
    ANNOTATION_FOLDER = supervisely.ANNOTATION_FOLDER
    def __init__(self, files, categories=("text", "maths", "separator"),
                 transform_img=None, transform_mask=None, augmentations=None):
        super().__init__()
        self.files = files
        self.categories = set(categories)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.augmentations = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image_object = supervisely.parse_json(file)
        image_object = image_object.scale((736, 1024))
        grey = cv2.cvtColor(image_object.image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(grey, dtype=np.int32)

        for region in image_object.regions:
            if region.category in self.categories:
                mn = np.min(region.contour, axis=0)
                mx = np.max(region.contour, axis=0)
                img_patch = grey[mn[1]:mx[1], mn[0]:mx[0]]
                semi_mask = semisuper_contour_gt(img_patch).astype(np.int32)
                semi_mask[semi_mask > 0.5] = region.category_id
                mask[mn[1]:mx[1], mn[0]:mx[0]] = semi_mask
        # mask = (mask > 0).astype(np.uint)

        if self.augmentations:
            augmented = self.augmentations(image=grey, mask=mask)
            grey = augmented['image']
            mask = augmented['mask']
        # else:
            # grey = cv2.resize(grey, (736, 1024))
            # mask = cv2.resize(mask.astype(np.float), (736, 1024))

        # mask.dtype has been bool.
        mask_with_class = mask.astype(np.int32).copy()
        mask_with_class = torch.from_numpy(mask_with_class).long()
        mask = mask > 0

        # TODO: fix
        if self.transform_img:
            grey = self.transform_img(grey)
        else:
            grey = torch.from_numpy(grey.astype(np.float) / 255.0).float().unsqueeze(0)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return grey, mask, mask_with_class
