from typing import Union, Literal

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class GuiVisDataset(Dataset):

    def __init__(self, data_path: str, data_name: str, 
                 mask_mode: Literal["binary", "gray"] = "binary", 
                 transform = None):
        self.h = h5py.File(data_path + data_name + ".hdf5", "r")
        self.images = self.h["images"]
        self.ricoid = self.h["ricoid"]
        self.masks = self.h["masks"]
        self.mask_mode = mask_mode
        self.rects = self.h["rects"]
        self.labels = self.h["labels"]
        self.transform = transform
        self.mode = None
        self.extras = 0
        self.k = 10

    def summary(self):
        print(f"Mask Grayscaled: {len(np.unique(self.masks[0])) != 2}")

    def __getitem__(self, i: int):
        if i >= len(self.labels):
            return self.__getoovitem()
        img_idx = self.labels[i, 0]
        label = self.labels[i, 2]
        img = torch.from_numpy(self.images[img_idx])
        if self.mask_mode == "binary":
            img_mask = torch.FloatTensor(self.masks[i] == 255)
        else:
            img_mask = torch.FloatTensor(self.masks[i] / 255.)
        img_rect = torch.FloatTensor(self.rects[i])
        # img = torch.FloatTensor(self.images[img_idx] / 255.)
        # img_mask = torch.FloatTensor(self.masks[i] / 255.)
        if self.transform is not None:
            img, img_mask, _ = self.transform(img, img_mask, img_rect)
        img = torch.cat([img, img_mask], dim=0)
        label = torch.FloatTensor([label])
        return {
            "image": img, 
            "target": label
        }
    
    def __getoovitem(self):
        img_index = np.random.randint(len(self.images))
        img = torch.from_numpy(self.images[img_index])
        
        if self.mode == "rnd":
            msk_index = np.random.choice(np.where(self.labels[:, 0] != img_index)[0])
            img_mask = torch.FloatTensor(self.masks[msk_index] / 255.)
            img_rect = torch.FloatTensor(self.rects[msk_index])
        else:  # "neg"
            msk_indices_pos = np.where(self.labels[:, 0] != img_index)[0]
            img_masks_pos = self.masks[msk_indices_pos] / 255.
            img_mask_pos = np.sum(img_masks_pos, axis=0)

            msk_indices_neg = np.random.choice(
                np.where(self.labels[:, 0] != img_index)[0], 
                (self.k, ), replace=False)
            msk_indices_neg = np.sort(msk_indices_neg)
            img_masks_neg = self.masks[msk_indices_neg] / 255.

            img_masks_weights = img_mask_pos * img_masks_neg
            img_masks_weights = img_masks_weights.reshape(self.k, -1)
            img_masks_weights = np.average(img_masks_weights, axis=1)

            msk_index_neg = np.random.choice(np.where(img_masks_weights == img_masks_weights.min())[0])
            msk_index_neg = msk_indices_neg[msk_index_neg]

            img_mask = torch.FloatTensor(self.masks[msk_index_neg] / 255.)
            img_rect = torch.FloatTensor(self.rects[msk_index_neg])
        
        if self.transform is not None:
            img, img_mask, _ = self.transform(img, img_mask, img_rect)
        img = torch.cat([img, img_mask], dim=0)
        label = torch.FloatTensor([0])
        return { 
            "image": img, 
            "target": label
        }

    def __len__(self) -> int:
        return len(self.labels) + self.extras
    
    @property
    def sample_weights(self):
        labels = self.labels[:, 2]
        labels_weights = 1.0 / np.bincount(labels)
        return labels_weights[labels]
    
    def resample(self, mode: Literal["rnd", "neg"], k: int = 10):
        self.mode = mode
        self.k = k
        bincount = np.bincount(self.labels[:, 2])
        self.extras = bincount[1] - bincount[0]
        assert self.extras > 0
