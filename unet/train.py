import random
import os
import shutil

import torch
import torch.utils.data as data_utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2

from .datasets import MaskDataset
from .collector import Collector


class Trainer(object):
    def __init__(self, exp_path, config, device):
        self.exp_path = exp_path
        self.device = device
        self.config = config
        self.train_data, self.val_data = self.load_datasets()
        self.criterion = self.load_criterion()
        self.model = self.load_model()
        self.optim = self.load_optim()
        self.writer = self.init_board()

    def init_board(self):
        return SummaryWriter(os.path.join(self.exp_path, "runs"))

    def load_optim(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config["train"]["lr"])

    def load_model(self):
        config = self.config["model"]
        # Different models
        from .model import UNet
        return UNet(**config["params"]).to(self.device)

    def load_datasets(self):
        config = self.config["data"]

        datasets = data_utils.ConcatDataset(
            [MaskDataset(data_path) for data_path in config["list"]]
        )
        indices = list(range(len(datasets)))
        random.seed(config["seed"])
        random.shuffle(indices)
        train_indices = indices[:int(len(indices) * config["train_fraction"])]
        val_indices = indices[len(train_indices):]
        return data_utils.Subset(datasets, train_indices), data_utils.Subset(datasets, val_indices)

    def load_criterion(self):
        return nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch_number):
        config = self.config["train"]
        it = data_utils.DataLoader(self.train_data, batch_size=config["batch"], num_workers=8, shuffle=True)
        it = tqdm(it, desc="train[%d]" % epoch_number)
        self.model.train()

        collection = Collector()
        for batch_index, (img, mask) in enumerate(it):
            img, mask = img.to(self.device), mask.to(self.device)

            self.optim.zero_grad()
            out = self.model(img)
            loss = self.criterion(out, mask)

            collection.add("loss", loss.item())
            self.writer.add_scalars("batch_bce_loss", dict(train=loss.item()), self.global_step)

            loss.backward()
            self.optim.step()
            it.set_postfix(loss=loss.item())
            self.global_step += 1
            if batch_index == 0:
                self._write_images("train", img, out.sigmoid(), epoch_number)

        return np.mean(collection["loss"])

    def val_epoch(self, epoch_number):
        config = self.config["val"]
        it = data_utils.DataLoader(self.val_data, batch_size=config["batch"], num_workers=8, shuffle=False)
        it = tqdm(it, desc="val[%d]" % epoch_number)
        self.model.eval()

        collection = Collector()
        for batch_index, (img, mask) in enumerate(it):
            img, mask = img.to(self.device), mask.to(self.device)

            with torch.no_grad():
                out = self.model(img)
                loss = self.criterion(out, mask)
            collection.add("loss", loss.item())
            it.set_postfix(loss=loss.item())

            if batch_index == 0:
                self._write_images("val", img, out.sigmoid(), epoch_number)

        return np.mean(collection["loss"])

    def _write_images(self, general_tag, imgs, masks, epoch):
        # B, C, W, H
        imgs = imgs.detach().cpu().squeeze(1).numpy() * 255
        # B, C, W, H
        masks = masks.detach().cpu().squeeze(1).numpy() > 0.5
        for image_index in range(len(imgs)):
            img = cv2.cvtColor(imgs[image_index], cv2.COLOR_GRAY2RGB).astype(np.float)
            mask = masks[image_index]
            img[mask] = img[mask] / 2 + [0.0, 127.5, 0.0]
            img = torch.Tensor(img.transpose(2, 0, 1) / 255.0)
            self.writer.add_image("{}/image-{}".format(general_tag, image_index + 1), img, epoch)

    def train(self):
        self.global_step = 0
        best_value = None
        for i_epoch in range(self.config["train"]["epochs"]):
            self.epoch = i_epoch
            train_loss = self.train_epoch(self.epoch)
            print("Train loss epoch[{}] = {}".format(i_epoch, train_loss))
            val_loss = self.val_epoch(self.epoch)
            print("Val loss epoch[{}] = {}".format(i_epoch, val_loss))
            self.writer.add_scalars("epoch_bce_loss", dict(train=train_loss, val=val_loss), i_epoch)

            torch.save(self.model.state_dict(), os.path.join(self.exp_path, "current_model.h5"))
            if best_value is None or val_loss < best_value:
                print("Upgrade in LOSS!")
                best_value = val_loss
                shutil.copy(os.path.join(self.exp_path, "current_model.h5"),
                            os.path.join(self.exp_path, "best_model.h5"))
