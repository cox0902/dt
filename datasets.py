from typing import Union, Literal

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class GuiVisDataset(Dataset):

    def __init__(self, data_path: str, data_name: str, transform = None, 
                 label_smooth: bool = False):
        self.h = h5py.File(data_path + data_name + ".hdf5", "r")
        self.images = self.h["images"]
        self.ricoid = self.h["ricoid"]
        self.masks = self.h["masks"]
        self.rects = self.h["rects"]
        self.labels = self.h["labels"]
        self.transform = transform
        self.mode = None
        self.leaf_only = False
        self.extras = 0
        self.k = 10
        self.label_smooth = label_smooth

    def summary(self):
        print(f"Dataset Size: {len(self)}")
        print(f" Labels Size: {np.bincount(self.labels[:, 2])}")

    def __getitem__(self, i: int):
        if i >= len(self.labels) or -self.extras <= i < 0:
            return self.__getoovitem()
        img_idx = self.labels[i, 0]
        label = self.labels[i, 2]
        img = torch.from_numpy(self.images[img_idx])
        img_mask = torch.FloatTensor(self.masks[i] / 255.)
        img_rect = torch.FloatTensor(self.rects[i])
        # img = torch.FloatTensor(self.images[img_idx] / 255.)
        # img_mask = torch.FloatTensor(self.masks[i] / 255.)
        if self.transform is not None:
            img, img_mask, _ = self.transform(img, img_mask, img_rect)
        img = torch.cat([img, img_mask], dim=0)
        if self.label_smooth:
            label = torch.FloatTensor([np.clip(label, 0.1, 0.9)])
        else:
            label = torch.FloatTensor([label])
        return {
            "image": img, 
            "target": label
        }
    
    def __getoovitem(self):
        img_index = np.random.randint(len(self.images))
        img = torch.from_numpy(self.images[img_index])
        # print("sampled : ", self.labels[np.random.choice(np.where(self.labels[:, 0] == img_index)[0])])

        if self.leaf_only:
            msk_indices_pos = np.where(np.logical_and(
                self.labels[:, 0] == img_index, self.labels[:, 4] == 1
            ))[0]
        else:
            msk_indices_pos = np.where(self.labels[:, 0] == img_index)[0]
        img_masks_pos = self.masks[msk_indices_pos] / 255.

        if self.mode == "rnd":
            if self.leaf_only:
                msk_index = np.random.choice(np.where(np.logical_and(
                    self.labels[:, 0] != img_index, self.labels[:, 4] == 1
                ))[0])
            else:
                msk_index = np.random.choice(np.where(self.labels[:, 0] != img_index)[0])
            # print("randomed: ", self.labels[msk_index])
            img_mask = torch.FloatTensor(self.masks[msk_index] / 255.)
            img_rect = torch.FloatTensor(self.rects[msk_index])
        else:  # "neg"
            # img_mask_pos = np.sum(img_masks_pos, axis=0) / len(msk_indices_pos)

            if self.leaf_only:
                msk_indices_neg = np.random.choice(
                    np.where(np.logical_and(
                        self.labels[:, 0] != img_index, self.labels[:, 4] == 1
                    ))[0], (self.k, ), replace=False)
            else:
                msk_indices_neg = np.random.choice(
                    np.where(self.labels[:, 0] != img_index)[0], 
                    (self.k, ), replace=False)
            msk_indices_neg = np.sort(msk_indices_neg)
            img_masks_neg = self.masks[msk_indices_neg] / 255.

            ious = np.zeros((img_masks_pos.shape[0], img_masks_neg.shape[0]), dtype=np.float32)
            # print(ious.shape, img_masks_pos.shape, img_masks_neg.shape)
            for i in range(img_masks_pos.shape[0]):
                for j in range(img_masks_neg.shape[0]):
                    inter = np.sum(np.logical_and(img_masks_pos[i, 0], img_masks_neg[j, 0]))
                    union = np.sum(np.logical_or(img_masks_pos[i, 0], img_masks_neg[j, 0]))
                    ious[i, j] = inter / union

            m = np.max(ious, axis=0)
            p = None if np.allclose(m, 1) else (1 - m) / np.sum(1 - m)
            msk_index_neg = np.random.choice(len(img_masks_neg), p=p)
            iou = m[msk_index_neg]

            # img_masks_weights = img_mask_pos * img_masks_neg
            # img_masks_weights = img_masks_weights.reshape(self.k, -1)
            # img_masks_weights = np.sum(img_masks_weights, axis=1) / np.sum(img_mask_pos)

            # p = 1 - img_masks_weights
            # p = p / np.sum(p)
            # assert not np.any(np.isnan(p)), img_index
            # msk_index_neg = np.random.choice(len(img_masks_weights), p=p)
            msk_index_neg = msk_indices_neg[msk_index_neg]

            img_mask = torch.FloatTensor(self.masks[msk_index_neg] / 255.)
            img_rect = torch.FloatTensor(self.rects[msk_index_neg])

        if self.label_smooth:
            # mask_i = np.logical_and(img_masks_pos, img_mask.numpy())
            # mask_i = np.sum(mask_i, axis=(1, 2, 3))
            # mask_u = np.logical_or(img_masks_pos, img_mask.numpy())
            # mask_u = np.sum(mask_u, axis=(1, 2, 3))
            # mask_iou = mask_i / mask_u
            label = torch.FloatTensor([iou])
        else:
            label = torch.FloatTensor([0])
        
        if self.transform is not None:
            img, img_mask, _ = self.transform(img, img_mask, img_rect)
        img = torch.cat([img, img_mask], dim=0)
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
    
    def resample(self, mode: Literal["rnd", "neg"], k: int = 10, leaf_only: bool = False):
        self.mode = mode
        self.leaf_only = leaf_only
        self.k = k
        bincount = np.bincount(self.labels[:, 2])
        self.extras = bincount[1] - bincount[0]
        assert self.extras > 0
