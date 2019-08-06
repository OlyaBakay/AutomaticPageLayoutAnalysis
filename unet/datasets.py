import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

from utils import supervisely


class MaskDataset(Dataset):
    def __init__(self, in_path, categories=("text", "maths", "separator"), transform_img=None, transform_mask=None):
        super().__init__()
        self.in_path = in_path
        self.files = list(os.listdir(os.path.join(in_path, supervisely.ANNOTATION_FOLDER)))
        self.categories = set(categories)
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        file = os.path.join(self.in_path, supervisely.ANNOTATION_FOLDER, file)
        image_object = supervisely.parse_json(file)
        grey = cv2.cvtColor(image_object.image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(grey, dtype=np.bool)

        for region in image_object.regions:
            if region.category in self.categories:
                mn = np.min(region.contour, axis=0)
                mx = np.max(region.contour, axis=0)
                mask[mn[1]:mx[1], mn[0]:mx[0]] = True
        grey = cv2.resize(grey, (512, 512))
        mask = cv2.resize(mask.astype(np.float), (512, 512)) > 0

        # TODO: fix
        mask = torch.Tensor(mask).float().unsqueeze(0)
        grey = torch.Tensor(grey.astype(np.float) / 255.0).float().unsqueeze(0)

        if self.transform_img:
            grey = self.transform_img(grey)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        return grey, mask
