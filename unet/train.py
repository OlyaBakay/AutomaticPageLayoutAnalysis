import random
import os
import shutil

import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    RandomCrop,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    RandomScale,
    Rotate,
    Resize
)

from utils.region import Region
from .datasets import MaskDataset
from .collector import Collector
from .metrics import iou_pytorch
from .projections import process_batch_torch_wrap


class Trainer(object):
    def __init__(self, exp_path, config, device):
        self.exp_path = exp_path
        self.device = device
        self.config = config
        self.train_data, self.val_data = self.load_datasets()
        print("Train", len(self.train_data))
        print("Val", len(self.val_data))
        self.criterion = self.load_criterion()
        self.model, self.proj_model = self.load_model()
        self.optim = self.load_optim()
        self.writer = self.init_board()
        self.metrics = self.init_metrics()

    def init_metrics(self):
        metrics = dict()
        metrics["iou"] = iou_pytorch
        return metrics

    def init_augmentations(self):
        # TODO: change this
        width, height = 724, 1024
        wanted_size = 256
        aug = Compose([
            Resize(height=height, width=width),
            RandomScale(scale_limit=0.5, always_apply=True),
            RandomCrop(height=wanted_size, width=wanted_size),
            PadIfNeeded(min_height=wanted_size, min_width=wanted_size, p=0.5),
            Rotate(limit=4, p=0.5),
            VerticalFlip(p=0.5),
            GridDistortion(p=0.5),
            CLAHE(p=0.8),
            RandomBrightnessContrast(p=0.8),
            RandomGamma(p=0.8)
        ])
        return aug

    def init_board(self):
        return SummaryWriter(os.path.join(self.exp_path, "runs"))

    def load_optim(self):
        parameters = list(self.model.parameters()) + list(self.proj_model.parameters())
        return torch.optim.Adam(parameters,
                                lr=self.config["train"]["lr"])

    def load_model(self):
        config = self.config["model"]
        # Different models
        from .model import UNet, Fast1D
        # TODO: fix `5` magic constant
        return UNet(**config["params"]).to(self.device), Fast1D(5).to(self.device)

    def load_datasets(self):
        config = self.config["data"]
        aug = self.init_augmentations()

        files = []
        for in_path in config["list"]:
            in_path = os.path.join(in_path, MaskDataset.ANNOTATION_FOLDER)
            files += list(os.path.join(in_path, fn) for fn in os.listdir(in_path))

        random.seed(config["seed"])
        random.shuffle(files)
        train_files = files[:int(len(files) * config["train_fraction"])]
        val_files = files[len(train_files):]

        train_dset = MaskDataset(train_files, augmentations=aug)
        val_dset = MaskDataset(val_files)
        return train_dset, val_dset

    def load_criterion(self):
        return nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch_number):
        config = self.config["train"]
        it = data_utils.DataLoader(self.train_data, batch_size=config["batch"], num_workers=8, shuffle=True)
        it = tqdm(it, desc="train[%d]" % epoch_number)
        self.model.train()

        collection = Collector()
        for batch_index, (img, mask, class_mask) in enumerate(it):
            img, mask = img.to(self.device), mask.to(self.device)

            self.optim.zero_grad()
            out = self.model(img)
            loss = self.criterion(out, mask)

            # loss.backward()

            out_mask = out.detach().sigmoid() > 0.5
            proj, rectangles, proj_class, image_index = process_batch_torch_wrap(img.detach().cpu(), out_mask.cpu(), class_mask, filter_masks=True)

            total_loss = loss
            proj_loss = torch.zeros(1)
            if proj.shape[0] == 0:
                not_enough_rects = True
            else:
                not_enough_rects = False
                proj, proj_class = proj.to(self.device), proj_class.to(self.device)
                proj_out = self.proj_model(proj)

                if (proj_class >= 0).sum() > 0:
                    proj_loss = F.cross_entropy(proj_out[proj_class >= 0], proj_class[proj_class >= 0])
                    total_loss = loss + proj_loss
            total_loss.backward()

            self.optim.step()

            # collection.add("proj_loss", proj_loss.item())
            # self.writer.add_scalars("proj_batch", dict(loss=proj_loss.item()), self.global_step)
            collection.add("total_loss", total_loss.item())
            collection.add("proj_loss", proj_loss.item())
            collection.add("segm_loss", loss.item())
            self.writer.add_scalars("batch", dict(total_loss=total_loss.item(),
                                                  segm_loss=loss.item(),
                                                  proj_loss=proj_loss.item()), self.global_step)
            batch_metrics = dict()
            for metric_name, metric_f in self.metrics.items():
                metric_slug = "metric_{}".format(metric_name)
                with torch.no_grad():
                    metric_value = metric_f(out, mask, reduce=True).item()
                collection.add(metric_slug, metric_value)
                batch_metrics[metric_slug] = metric_value
            self.writer.add_scalars("batch", batch_metrics, self.global_step)


            it.set_postfix(loss=loss.item(), **batch_metrics)
            class_counts = pd.Series.value_counts(proj_class.cpu().detach().numpy()).to_dict()
            class_counts = {class_name: class_counts.get(class_index, 0) for class_index, class_name in enumerate(Region.CATEGORIES)}
            self.writer.add_scalars("proj_class_dist", class_counts, self.global_step)

            self.global_step += 1


            if batch_index == 0 and not_enough_rects is False:
                self._write_images_with_class("train", img, out_mask, rectangles, proj_out, image_index, epoch_number)
            elif batch_index == 0:
                self._write_images("train", img, out.sigmoid(), epoch_number)

        epoch_reduced_metrics = {metric_name: np.mean(collection[metric_name]) for metric_name in collection.keys()}
        epoch_loss = epoch_reduced_metrics.pop("total_loss")
        return epoch_loss, epoch_reduced_metrics

    def val_epoch(self, epoch_number):
        config = self.config["val"]
        it = data_utils.DataLoader(self.val_data, batch_size=config["batch"], num_workers=8, shuffle=False)
        it = tqdm(it, desc="val[%d]" % epoch_number)
        self.model.eval()

        collection = Collector()
        for batch_index, (img, mask, class_mask) in enumerate(it):
            img, mask = img.to(self.device), mask.to(self.device)

            with torch.no_grad():
                out = self.model(img)
                loss = self.criterion(out, mask)

                out_mask = out.detach().sigmoid() > 0.5
                proj, rectangles, proj_class, image_index = process_batch_torch_wrap(img.detach().cpu(), out_mask.cpu(), class_mask, filter_masks=False)

                total_loss = loss
                proj_loss = torch.zeros(1)
                if proj.shape[0] == 0:
                    not_enough_rects = True
                else:
                    not_enough_rects = False
                    proj, proj_class = proj.to(self.device), proj_class.to(self.device)
                    proj_out = self.proj_model(proj)

                    if (proj_class >= 0).sum() > 0:
                        proj_loss = F.cross_entropy(proj_out[proj_class >= 0], proj_class[proj_class >= 0])
                        total_loss = loss + proj_loss

            collection.add("total_loss", total_loss.item())
            collection.add("proj_loss", proj_loss.item())
            collection.add("segm_loss", loss.item())
            batch_metrics = dict()
            for metric_name, metric_f in self.metrics.items():
                metric_slug = "metric_{}".format(metric_name)
                with torch.no_grad():
                    metric_value = metric_f(out, mask, reduce=True).item()
                collection.add(metric_slug, metric_value)
                batch_metrics[metric_slug] = metric_value

            it.set_postfix(loss=loss.item(), **batch_metrics)

            if batch_index == 0 and not_enough_rects is False:
                self._write_images_with_class("val", img, out_mask, rectangles, proj_out, image_index, epoch_number)
            elif batch_index == 0:
                self._write_images("val", img, out.sigmoid(), epoch_number)

        epoch_reduced_metrics = {metric_name: np.mean(collection[metric_name]) for metric_name in collection.keys()}
        epoch_loss = epoch_reduced_metrics.pop("total_loss")
        return epoch_loss, epoch_reduced_metrics

    def _write_images_with_class(self, general_tag, imgs, pred_masks, rectangles, classes, image_index, epoch):
        # B, C, W, H
        imgs = imgs.detach().cpu().squeeze(1).numpy() * 255
        pred_masks = pred_masks.detach().cpu().squeeze(1).numpy() > 0.5
        # print(pred_masks.shape, pred_masks.dtype)
        N = imgs.shape[0]

        # print(classes.shape)
        rectangles = rectangles.detach().cpu().numpy()
        classes = classes.detach().cpu().argmax(1).numpy()

        grouped_rectangles = [[] for _ in range(N)]
        grouped_classes = [[] for _ in range(N)]
        for image_i, class_i, rect_i in zip(image_index, classes, rectangles):
            grouped_classes[image_i].append(class_i)
            grouped_rectangles[image_i].append(rect_i)

        for image_index in range(N):
            img = cv2.cvtColor(imgs[image_index], cv2.COLOR_GRAY2RGB).astype(np.float)
            mask = pred_masks[image_index]
            img[mask] = img[mask] / 2.0 + [127.5, 0.0, 0.0]
            regions = grouped_rectangles[image_index]
            classes = grouped_classes[image_index]
            for i, (region_i, class_i) in enumerate(zip(regions, classes)):
                x,y,w,h = region_i
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = Region.CATEGORIES[class_i]
                # cv2.drawContours(img, [region.contour], 0, (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.2
                lineType = 1
                t_size = cv2.getTextSize(label, font, fontScale, lineType)[0]
                top_left_corner = x, y
                right_bottom_corner = x + t_size[0] + 3, y + t_size[1] + 4

                cv2.rectangle(img, top_left_corner, right_bottom_corner, (0, 255, 0), -1)
                cv2.putText(img, label, (top_left_corner[0]+1, top_left_corner[1]+t_size[1]+2),
                            font, fontScale,
                            [225, 255, 255], lineType)

            img = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)
            self.writer.add_image("{}/image-{}".format(general_tag, image_index + 1), img, epoch)

    def _write_images(self, general_tag, imgs, masks, epoch):
        # B, C, W, H
        imgs = imgs.detach().cpu().squeeze(1).numpy() * 255
        # B, C, W, H
        masks = masks.detach().cpu().squeeze(1).numpy() > 0.5
        for image_index in range(len(imgs)):
            img = cv2.cvtColor(imgs[image_index], cv2.COLOR_GRAY2RGB).astype(np.float)
            mask = masks[image_index]
            img[mask] = img[mask] / 2 + [0.0, 127.5, 0.0]
            img = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)
            self.writer.add_image("{}/image-{}".format(general_tag, image_index + 1), img, epoch)

    def train(self):
        self.global_step = 0
        best_value = None
        for i_epoch in range(self.config["train"]["epochs"]):
            self.epoch = i_epoch
            train_loss, train_metrics = self.train_epoch(self.epoch)
            print("Train loss epoch[{}] = {}".format(i_epoch, train_loss))
            val_loss, val_metrics = self.val_epoch(self.epoch)
            print("Val loss epoch[{}] = {}".format(i_epoch, val_loss))
            self.writer.add_scalars("epoch_total_loss", dict(train=train_loss, val=val_loss), i_epoch)
            for metric_name in train_metrics.keys():
                self.writer.add_scalars("epoch_{}".format(metric_name),
                                        dict(train=train_metrics[metric_name],
                                             val=val_metrics[metric_name]), i_epoch)

            torch.save(self.model.state_dict(), os.path.join(self.exp_path, "current_model.h5"))
            if best_value is None or val_loss < best_value:
                print("Upgrade in LOSS!")
                best_value = val_loss
                shutil.copy(os.path.join(self.exp_path, "current_model.h5"),
                            os.path.join(self.exp_path, "best_model.h5"))
