import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

from utils import supervisely


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
        grey = cv2.cvtColor(image_object.image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(grey, dtype=np.bool)

        for region in image_object.regions:
            if region.category in self.categories:
                mn = np.min(region.contour, axis=0)
                mx = np.max(region.contour, axis=0)
                mask[mn[1]:mx[1], mn[0]:mx[0]] = True
        mask = (mask > 0).astype(np.uint)

        if self.augmentations:
            augmented = self.augmentations(image=grey, mask=mask)
            grey = augmented['image']
            mask = augmented['mask'] > 0
        else:
            grey = cv2.resize(grey, (736, 1024))
            mask = cv2.resize(mask.astype(np.float), (736, 1024)) > 0

        # TODO: fix
        if self.transform_img:
            grey = self.transform_img(grey)
        else:
            grey = torch.Tensor(grey.astype(np.float) / 255.0).float().unsqueeze(0)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.Tensor(mask).float().unsqueeze(0)

        return grey, mask
