import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import pydicom
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from monai.transforms import Resize
import monai.transforms as transforms

from loss_fn import bce_dice, multilabel_dice_score
from dataset import SEGDataset
from model import TimmSegModel, convert_3d

rcParams["figure.figsize"] = 20, 8
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_valid: DataLoader,
        criterion,
        scaler,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        # self.model = model.cuda()
        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.criterion = criterion
        self.scaler = scaler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(
            self.model, device_ids=[self.gpu_id], find_unused_parameters=True
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.003)
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 1000
        )
        self.train_loss = []
        self.valid_loss = []
        self.batch_metrics = [[]] * 7
        self.metrics = None
        self.p_mixup = 0.1
        self.ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.metric_best = 0
        self.best_metric_model_file = "best_snapshot.pth"
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"logs.txt")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_train_batch(self, images, gt_masks):
        def mixup(input, truth, clip=[0, 1]):
            indices = torch.randperm(input.size(0))
            shuffled_input = input[indices]
            shuffled_labels = truth[indices]

            lam = np.random.uniform(clip[0], clip[1])
            input = input * lam + shuffled_input * (1 - lam)
            return input, truth, shuffled_labels, lam

        self.optimizer.zero_grad()

        do_mixup = False
        if random.random() < self.p_mixup:
            do_mixup = True
            images, gt_masks, gt_masks_sfl, lam = mixup(images, gt_masks)

        with amp.autocast():
            logits = self.model(images)
            loss = self.criterion(logits, gt_masks)
            if do_mixup:
                loss2 = self.criterion(logits, gt_masks_sfl)
                loss = loss * lam + loss2 * (1 - lam)

        self.train_loss.append(loss.item())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _run_train_epoch(self, epoch):
        b_sz = len(next(iter(self.loader_train))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.loader_train)}"
        )
        self.loader_train.sampler.set_epoch(epoch)
        for images, gt_masks in self.loader_train:
            images = images.to(self.gpu_id)
            gt_masks = gt_masks.to(self.gpu_id)

            self._run_train_batch(images, gt_masks)

    def _run_valid_batch(self, images, gt_masks):
        logits = self.model(images)
        loss = self.criterion(logits, gt_masks)
        self.valid_loss.append(loss.item())
        for thi, th in enumerate(self.ths):
            pred = (logits.sigmoid() > th).float().detach()
            for i in range(logits.shape[0]):
                tmp = multilabel_dice_score(
                    y_pred=logits[i].sigmoid().cpu(),
                    y_true=gt_masks[i].cpu(),
                    threshold=0.5,
                )
                self.batch_metrics[thi].extend(tmp)
        self.metrics = [np.mean(this_metric) for this_metric in self.batch_metrics]

    def _run_valid_epoch(self, epoch):
        b_sz = len(next(iter(self.loader_valid))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.loader_train)}"
        )
        self.loader_valid.sampler.set_epoch(epoch)
        with torch.no_grad():
            for images, gt_masks in self.loader_valid:
                images = images.to(self.gpu_id)
                gt_masks = gt_masks.to(self.gpu_id)

                self._run_valid_batch(images, gt_masks)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _save_metric_best(self, metric):
        if metric > self.metric_best:
            print(
                f"metric_best ({self.metric_best:.6f} --> {metric:.6f}). Saving model ..."
            )
            torch.save(self.model.state_dict(), self.best_metric_model_file)
            self.metric_best = metric

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self.scheduler_cosine.step()
            self._run_train_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            self._run_valid_epoch(epoch)

            content = (
                time.ctime()
                + " "
                + f'Fold 0, Epoch {epoch}, lr: {self.optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(self.train_loss):.5f}, valid loss: {np.mean(self.valid_loss):.5f}, metric: {np.max(self.metrics):.6f}.'
            )
            print(content)

            with open(self.log_file, "a") as appender:
                appender.write(content + "\n")

            metric = np.max(self.metrics)
            self._save_metric_best(metric)


def load_train_objs():
    backbone = "resnet18d"
    fold = 0
    image_sizes = [128, 128, 128]

    transforms_train = transforms.Compose(
        [
            transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            transforms.RandAffined(
                keys=["image", "mask"],
                translate_range=[
                    int(x * y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])
                ],
                padding_mode="zeros",
                prob=0.7,
            ),
            transforms.RandGridDistortiond(
                keys=("image", "mask"),
                prob=0.5,
                distort_limit=(-0.01, 0.01),
                mode="nearest",
            ),
        ]
    )

    transforms_valid = transforms.Compose([])

    df_seg = pd.read_csv("./data/folds.csv")
    train_ = df_seg[df_seg["fold"] != fold].reset_index(drop=True)
    valid_ = df_seg[df_seg["fold"] == fold].reset_index(drop=True)
    dataset_train = SEGDataset(
        train_, "train", transform=transforms_train
    )  # load your dataset
    dataset_valid = SEGDataset(
        valid_, "valid", transform=transforms_valid
    )  # load your dataset

    model = TimmSegModel(backbone, pretrained=True)
    model = convert_3d(model)
    scaler = torch.cuda.amp.GradScaler()
    criterion = bce_dice

    return (
        dataset_train,
        dataset_valid,
        model,
        scaler,
        criterion,
    )


def prepare_dataloader_train(dataset: Dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(dataset),
    )


def prepare_dataloader_valid(dataset: Dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(dataset),
    )


def ddp_setup():
    init_process_group(backend="nccl")


def main(
    save_every: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = "snapshot.pt",
):
    ddp_setup()
    (
        dataset_train,
        dataset_valid,
        model,
        scaler,
        criterion,
    ) = load_train_objs()
    loader_train = prepare_dataloader_train(dataset_train, batch_size)
    loader_valid = prepare_dataloader_valid(dataset_valid, batch_size)

    trainer = Trainer(
        model,
        loader_train,
        loader_valid,
        criterion,
        scaler,
        save_every,
        snapshot_path,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size", default=4, help="Input batch size on each device (default: 32)"
    )
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)
