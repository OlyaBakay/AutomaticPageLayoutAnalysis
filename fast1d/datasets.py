import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class TrainProjectionDataset(Dataset):
    def __init__(self, path, transform=None):
        self.categories = os.listdir(path)
        self.files = {}
        for category in self.categories:
            category_path = os.path.join(path, category)
            self.files[category] = [os.path.join(path, category, file) for file in os.listdir(category_path)]
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.categories) * max([len(self.files[category]) for category in self.files])

    def __getitem__(self, idx):
        category_id = idx % len(self.categories)
        category = self.categories[category_id]
        photo_id = idx // len(self.categories) % len(self.files[category])
        img_filename = "info_{}.pickle".format(photo_id)
        with open(os.path.join(self.path, category, img_filename), "rb") as f:
            info = pickle.load(f)

        return np.expand_dims(info['x'], axis=0), np.expand_dims(info['y'], axis=0), category_id

class TestProjectionDataset(Dataset):
    def __init__(self, path, transform=None):
        self.categories = os.listdir(path)
        self.files = []
        self.labels = []
        for i, category in enumerate(self.categories):
            category_path = os.path.join(path, category)
            category_files = [os.path.join(path, category, file) for file in os.listdir(category_path)]
            self.files += category_files
            self.labels += [i] * len(category_files)
        self.transform = transform
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.files[idx]), "rb") as f:
            info = pickle.load(f)

        return np.expand_dims(info['x'], axis=0), np.expand_dims(info['y'], axis=0), self.labels[idx]