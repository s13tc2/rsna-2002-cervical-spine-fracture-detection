import os
import torch
import random
import numpy as np
import pandas as pd
from config import cfg
from torch.utils.data import Dataset

class CLSDataset(Dataset):
    def __init__(self, df, mode, transform):

        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        images = []
        
        tmp = list(range(7))
        ### random order v3
        if self.mode == 'train' and random.random() < cfg.p_rand_order:
            random.shuffle(tmp)
        ###
        for cid in (tmp):
            for ind in list(range(cfg.n_slice_per_c)):
                filepath = os.path.join(cfg.data_dir, f'{row.StudyInstanceUID}_{cid+1}_{ind}.npy')
                image = np.load(filepath)
                image = self.transform(image=image)['image']
                image = image.transpose(2, 0, 1).astype(np.float32) / 255.
                images.append(image)
        images = np.stack(images, 0)

        if self.mode != 'test':
            labels = []
            for i in row[[f'C{x+1}' for x in tmp]].tolist():
                labels += [i] * cfg.n_slice_per_c
            images = torch.tensor(images).float()
            labels = torch.tensor(labels).float()
            
            if self.mode == 'train' and random.random() < cfg.p_rand_order_v1:
                indices = torch.randperm(images.size(0))
                images = images[indices]
                labels = labels[indices]

            return images, labels
        else:
            return torch.tensor(images).float()