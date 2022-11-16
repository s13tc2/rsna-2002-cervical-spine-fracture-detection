import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

revert_list = [
    "1.2.826.0.1.3680043.1363",
    "1.2.826.0.1.3680043.20120",
    "1.2.826.0.1.3680043.2243",
    "1.2.826.0.1.3680043.24606",
    "1.2.826.0.1.3680043.32071",
]


class SEGDataset(Dataset):
    def __init__(self, df, mode, transform):

        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_file = os.path.join('./data/train_images_npy/', f"{row.StudyInstanceUID}.npy")
        mask_file = os.path.join('./data/segmentations_npy/', f"{row.StudyInstanceUID}.npy")
        image = np.load(image_file).astype(np.float32)
        mask = np.load(mask_file).astype(np.float32)

        if row.StudyInstanceUID in revert_list:
            mask = mask[:, :, :, ::-1]

        image, mask = (
            torch.tensor(image.copy()).float(),
            torch.tensor(mask.copy()).float(),
        )

        return image, mask
