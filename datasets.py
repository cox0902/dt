import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class GuiVisDataset(Dataset):

    def __init__(self, data_path: str, data_name: str, transform = None):
        self.h = h5py.File(data_path + data_name + ".hdf5", "r")
        self.images = self.h["images"]
        self.masks = self.h["masks"]
        self.labels = self.h["labels"]
        self.transform = transform

    def __getitem__(self, i: int):
        img_idx, _, label = self.labels[i]
        img = torch.FloatTensor(self.images[img_idx] / 255.)
        img_mask = torch.FloatTensor(self.masks[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.cat([img, img_mask], dim=0)
        label = torch.FloatTensor([label])
        return {
            "image": img, 
            "target": label
        }

    def __len__(self) -> int:
        return len(self.masks)
    
    @property
    def sample_weights(self):
        labels_weights = 1.0 / np.bincount(self.labels)
        return labels_weights[self.labels]
