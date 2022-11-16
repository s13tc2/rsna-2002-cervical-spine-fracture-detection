import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
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
from torch.distributed import init_process_group

from config import cfg
from mixup import mixup
from loss import criterion
from model import TimmModel
from dataset import CLSDataset
from transforms import get_train_transforms, get_valid_transforms

class Trainer:
  def __init__(self, model, loader_train, loader_valid, optimizer, scheduler, criterion, scaler, save_every, snapshot_path):
    self.gpu_id = int(os.environ["LOCAL_RANK"])
    self.model = model.to(self.gpu_id)
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
    self.model = DDP(self.model, device_ids=[self.gpu_id])
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_loss = []
    self.valid_loss = []
    self.metric = None
    self.metric_best = None
    self.gts = []
    self.outputs = []
    self.log_dir = "./logs"
    self.model_dir = "./models"
    os.makedirs(self.log_dir, exist_ok=True)
    os.makedirs(self.model_dir, exist_ok=True)
    self.log_file = os.path.join(self.log_dir, f'{cfg.kernel_type}.txt')
    self.model_file = os.path.join(self.model_dir, f'{cfg.kernel_type}_best.pth')
    
  def _load_snapshot(self, snapshot_path):
    loc = f"cuda:{self.gpu_id}"
    snapshot = torch.load(snapshot_path, map_location=loc)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

  def _run_train_batch(self, images, targets):
    self.optimizer.zero_grad()

    do_mixup = False
    if random.random() < cfg.p_mixup:
      do_mixup = True
      # remember to import mixup
      images, targets, targets_mix, lam = mixup(images, targets)
    
    with amp.autocast():
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        if do_mixup:
          loss11 = self.criterion(logits, targets_mix)
          loss = loss * lam  + loss11 * (1 - lam)
    self.train_loss.append(loss.item())
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()

  def _run_train_epoch(self, epoch):
    b_sz = len(next(iter(self.loader_train))[0])
    print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.loader_train)}")
    self.loader_train.sampler.set_epoch(epoch)
    for images, targets in tqdm(self.loader_train):
      images = images.to(self.gpu_id)
      targets = targets.to(self.gpu_id)

      self._run_train_batch(images, targets)
    self.train_loss = np.mean(self.train_loss)

  def _run_valid_batch(self, images, targets):
    logits = self.model(images)
    loss = self.criterion(logits, targets)
    
    self.gts.append(targets.cpu())
    self.outputs.append(logits.cpu())
    self.valid_loss.append(loss.item())

  def _run_valid_epoch(self, epoch):
    b_sz = len(next(iter(self.loader_valid))[0])
    print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.loader_valid)}")
    self.loader_valid.sampler.set_epoch(epoch)
    with torch.no_grad():
      for images, targets in tqdm(self.loader_valid):
        images = images.to(self.gpu_id)
        targets = targets.to(self.gpu_id)

        self._run_valid_batch(images, targets)
    
    self.outputs = torch.cat(self.outputs)
    self.gts = torch.cat(self.gts)
    self.valid_loss = criterion(self.outputs, self.gts).item()

  def _save_snapshot(self, epoch):
    snapshot = {
        "MODEL_STATE": self.model.module.state_dict(),
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, self.snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

  def train(self, n_epochs: int):
    for epoch in range(self.epochs_run+1, n_epochs):
      self.scheduler.step(epoch-1)
      self._run_train_epoch(epoch)
      if self.gpu_id == 0 and epoch % self.save_every == 0:
        self._save_snapshot(epoch)
      self._run_valid_epoch(epoch)
      self.metric = self.valid_loss

      content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {self.optimizer.param_groups[0]["lr"]:.7f}, train loss: {self.train_loss:.5f}, valid loss: {self.valid_loss:.5f}, metric: {(self.metric):.6f}.'

      print(content)
      with open(self.log_file, 'a') as appender:
        appender.write(content + '\n')
      
      if self.metric < self.metric_best:
        print(f'metric_best ({self.metric_best:.6f} --> {self.epochs_runmetric:.6f}). Saving model ...')
        torch.save(self.model.state_dict(), self.model_file)
        self.metric_best = self.metric

    del self.model
    torch.cuda.empty_cache()
    gc.collect()
    
# trainer.py
def load_data():
  df = pd.read_csv(os.path.join(f'../data/train_seg.csv'))
  df = df.sample(16).reset_index(drop=True) if cfg.DEBUG else df

  sid = []
  cs = []
  label = []
  fold = []
  for _, row in df.iterrows():
      for i in [1,2,3,4,5,6,7]:
          sid.append(row.StudyInstanceUID)
          cs.append(i)
          label.append(row[f'C{i}'])
          fold.append(row.fold)

  df = pd.DataFrame({
      'StudyInstanceUID': sid,
      'c': cs,
      'label': label,
      'fold': fold
  })

  return df

    
def load_train_objs():
  df = load_data()
  train_ = df[df['fold'] != cfg.fold].reset_index(drop=True)
  valid_ = df[df['fold'] == cfg.fold].reset_index(drop=True)
  dataset_train = CLSDataset(train_, 'train', transform=get_train_transforms())
  dataset_valid = CLSDataset(valid_, 'valid', transform=get_valid_transforms())

  model = TimmModel(cfg.backbone, pretrained=True)

  optimizer = optim.AdamW(model.parameters(), lr=cfg.init_lr)
  scaler = torch.cuda.amp.GradScaler() if cfg.use_amp else None
  loss_fn = criterion
  scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cfg.n_epochs, eta_min=cfg.eta_min)

  return dataset_train, dataset_valid, model, optimizer, scaler, loss_fn, scheduler_cosine

def prepare_dataloader_train(dataset, batch_size):
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    sampler=DistributedSampler(dataset)
  )

def prepare_dataloader_valid(dataset: Dataset, batch_size: int):
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    sampler=DistributedSampler(dataset)
  )

def ddp_setup():
  init_process_group(backend="nccl")

def main(save_every, total_epochs, batch_size, snapshot_path="snapshot.pth"):
  ddp_setup()
  dataset_train, dataset_valid, model, optimizer, scaler, loss_fn, scheduler_cosine = load_train_objs()
  loader_train = prepare_dataloader_train(dataset_train, batch_size)
  loader_valid = prepare_dataloader_valid(dataset_valid, batch_size)
  trainer = Trainer(
    model, 
    loader_train, 
    loader_valid, 
    optimizer, 
    scheduler_cosine, 
    loss_fn, 
    scaler, 
    save_every, 
    snapshot_path
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
      "--batch_size", default=8, help="Input batch size on each device (default: 32)"
  )
  args = parser.parse_args()

  main(args.save_every, args.total_epochs, args.batch_size)
