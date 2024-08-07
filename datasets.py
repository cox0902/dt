from typing import *

import json
import h5py
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset

from .helpers import plot


class GuiMatDataset(Dataset):

    def __init__(self, data_path: str, data_name: Optional[str] = "images.hdf5", 
                 set_name: Optional[Literal["train", "valid", "test"]] = None, 
                 fold: Optional[int] = None, transform = None, 
                 full: Optional[bool] = False,
                 fill: Optional[Tuple[int, int, int, int]] = None, 
                 outline: Optional[Tuple[int, int, int, int]] = (255, 255, 255, 255)):
        self.set_name = set_name
        self.fold_num = fold

        if set_name is not None and fold is not None:
            self.split = np.load(Path(data_path) / f"split_fold_{fold}.npz")[set_name]
        elif set_name is None and fold is None:
            self.split = None
        else:
            assert False
        
        self.h = h5py.File(Path(data_path) / data_name, "r")
        self.images = self.h["images"]
        self.rects = self.h["rects"][self.split] if self.split is not None else self.h["rects"]

        self.labels = self.h["labels"][self.split] if self.split is not None else self.h["labels"]
        self.indexs = np.unique(self.labels[:, 0])

        self.transform = transform
        self.fill = fill
        self.outline = outline

    def __len__(self) -> int:
        return len(self.indexs)
    
    def summary(self):
        if self.split is not None:
            print(f"Dataset Name: {self.set_name}#{self.fold_num}")
        print(f'Dataset Time: {self.h.attrs["create_time"]}')
        print(f"Dataset Size: {len(self)}")
        count_neg, count_pos = 0, 0
        for idx in self.indexs:
            if all(self.labels[self.labels[:, 0] == idx, 2]):
                count_pos += 1
            else:
                count_neg += 1
        print(f" Labels Size: {count_neg} {count_pos}")

    @property
    def sample_weights(self):
        count_neg, count_pos = 0, 0
        labels = []
        for idx in self.indexs:
            if all(self.labels[self.labels[:, 0] == idx, 2]):
                labels.append(1)
                count_pos += 1
            else:
                labels.append(0)
                count_neg += 1
        labels_weights = 1.0 / np.array([count_neg, count_pos])
        return labels_weights[labels]

    def __getitem__(self, i: int) -> Dict:
        img_idx = self.indexs[i]
        label = 1 if all(self.labels[self.labels[:, 0] == img_idx, 2]) else 0

        rcs_idx = np.logical_and(
            self.labels[:, 0] == img_idx,
            self.labels[:, 4] == 1 
        )
        rects = self.rects[rcs_idx]

        img = torch.from_numpy(self.images[img_idx])

        if self.fill is None and self.outline is None:
            mask = Image.new("RGB", (img.size(1), img.size(2)), 0)
            mask_draw = ImageDraw.Draw(mask, "RGBA")
            for rect in rects:
                mask_draw.rectangle(rect.tolist(), fill=self.fill, outline=self.outline, width=1)
            mask = mask.convert("L")

            img_mask = torch.FloatTensor(np.asarray(mask) / 255.)
            img_mask = img_mask.unsqueeze(0)
        else:
            img_mask = np.ones((1, img.size(1), img.size(2)), dtype=np.float32)
            img_mask = torch.FloatTensor(img_mask)

        if self.transform is not None:
            img, img_mask, _ = self.transform(img, img_mask)
        img = torch.cat([img, img_mask], dim=0)

        label = torch.FloatTensor([label])
        return {
            "image": img, 
            "target": label
        }
    
    def vis(self, i: Optional[int] = None):
        if i is None:
            i = np.random.randint(len(self))
        data = self[i]
        plot([(data["image"][:-1], {}), (data["image"][-1], {}), 
              (data["image"][:-1], { "masks": data["image"][-1] })],
             [(f"{self.h['ricoid'][self.indexs[i]]}", "mask", f"{data['target'].item():.4f}")],
             figshow_kwargs={ "figsize": (12, 4) })


class GuiMatMaskDataset(Dataset):

    def __init__(self, mask_path: str, data_path: str, data_name: Optional[str] = "images.hdf5", 
                 set_name: Optional[Literal["train", "valid", "test"]] = None, 
                 fold: Optional[int] = None, transform = None):
        self.set_name = set_name
        self.fold_num = fold

        if set_name is not None and fold is not None:
            self.split = np.load(Path(data_path) / f"split_fold_{fold}.npz")[set_name]
        elif set_name is None and fold is None:
            self.split = None
        else:
            assert False
        
        self.h = h5py.File(Path(data_path) / data_name, "r")
        self.images = self.h["images"]
        self.labels = self.h["labels"][self.split] if self.split is not None else self.h["labels"]
        self.indexs = np.unique(self.labels[:, 0])

        self.m = h5py.File(mask_path, "r")
        self.masks = self.m["masks"]

        self.transform = transform


    def __len__(self) -> int:
        return len(self.indexs)
    
    def summary(self):
        if self.split is not None:
            print(f"Dataset Name: {self.set_name}#{self.fold_num}")
        print(f'Dataset Time: {self.h.attrs["create_time"]}')
        print(f"Dataset Size: {len(self)}")
        count_neg, count_pos = 0, 0
        for idx in self.indexs:
            if all(self.labels[self.labels[:, 0] == idx, 2]):
                count_pos += 1
            else:
                count_neg += 1
        print(f" Labels Size: {count_neg} {count_pos}")

    @property
    def sample_weights(self):
        count_neg, count_pos = 0, 0
        labels = []
        for idx in self.indexs:
            if all(self.labels[self.labels[:, 0] == idx, 2]):
                labels.append(1)
                count_pos += 1
            else:
                labels.append(0)
                count_neg += 1
        labels_weights = 1.0 / np.array([count_neg, count_pos])
        return labels_weights[labels]

    def __getitem__(self, i: int) -> Dict:
        img_idx = self.indexs[i]
        label = 1 if all(self.labels[self.labels[:, 0] == img_idx, 2]) else 0
        img = torch.from_numpy(self.images[img_idx])

        mask = self.masks[img_idx]

        img_mask = torch.FloatTensor(np.asarray(mask) / 255.)

        if self.transform is not None:
            img, _, _ = self.transform(img)
        img = torch.cat([img, img_mask], dim=0)

        label = torch.FloatTensor([label])
        return {
            "image": img, 
            "target": label
        }
    
    def vis(self, i: Optional[int] = None):
        if i is None:
            i = np.random.randint(len(self))
        data = self[i]
        plot([(data["image"][:-2], {}), (data["image"][-2], {}), (data["image"][-1], {})],
             [(f"{self.h['ricoid'][self.indexs[i]]}", "mask t", "mask nt")],
             figshow_kwargs={ "figsize": (12, 4) })


class GuiVisDataset(Dataset):

    def __init__(self, data_path: str, data_name: Optional[str] = "images.hdf5", 
                 set_name: Optional[Literal["train", "valid", "test"]] = None, 
                 fold: Optional[int] = None, transform = None, label_smooth: bool = False):
        self.set_name = set_name
        self.fold_num = fold

        if set_name is not None and fold is not None:
            self.split = np.load(Path(data_path) / f"split_fold_{fold}.npz")[set_name]
        elif set_name is None and fold is None:
            self.split = None
        else:
            assert False
        
        self.h = h5py.File(Path(data_path) / data_name, "r")
        self.images = self.h["images"]
        self.ricoid = self.h["ricoid"]
        if "masks" not in self.h:
            assert set_name is None  # TODO:
            self.masks = None
            self.rects = self.h["rects"]
        else:
            self.masks = self.h["masks"]
        # self.rects = self.h["rects"]
        self.labels = self.h["labels"][self.split] if self.split is not None else self.h["labels"]

        self.transform = transform
        self.mode = None
        self.leaf_only = False
        self.extras = 0
        self.k = 10
        self.label_smooth = label_smooth

    def __idx(self, idx):
        return self.split[idx] if self.split is not None else idx

    def summary(self):
        if self.split is not None:
            print(f"Dataset Name: {self.set_name}#{self.fold_num}")
        print(f"Dataset Size: {len(self)}")
        print(f" Labels Size: {np.bincount(self.labels[:, 2])}")

    def __getitem__(self, i: int) -> Dict:
        if i >= len(self.labels) or -self.extras <= i < 0:
            assert self.set_name == "train"
            return self.__getoovitem()

        img_idx = self.labels[i, 0]
        label = self.labels[i, 2]
        img = torch.from_numpy(self.images[img_idx])

        if self.masks is None:
            mask = Image.new("L", (1440, 2560), 0)
            ImageDraw.Draw(mask).rectangle(self.rects[self.__idx(i)].tolist(), 255)
            mask = mask.resize((256, 256), Image.NEAREST)
            img_mask = torch.FloatTensor(np.asarray(mask) / 255.)
            img_mask = img_mask.unsqueeze(0)
        else:
            img_mask = torch.FloatTensor(self.masks[self.__idx(i)] / 255.)
        # img_rect = torch.FloatTensor(self.rects[self.__idx(i)])
        if self.transform is not None:
            img, img_mask, _ = self.transform(img, img_mask)
            # img, img_mask, _ = self.transform(img, img_mask, img_rect)
        img = torch.cat([img, img_mask], dim=0)
        # if self.label_smooth:
        #     label = torch.FloatTensor([np.clip(label, 0.1, 0.9)])
        # else:
        label = torch.FloatTensor([label])
        return {
            "image": img, 
            "target": label
        }
    
    def __getoovitem(self):
        img_index = np.random.choice(np.unique(self.labels[:, 0]))
        img = torch.from_numpy(self.images[img_index])
        # print("sampled : ", self.labels[np.random.choice(np.where(self.labels[:, 0] == img_index)[0])])

        if self.leaf_only:
            msk_indices_pos = np.where(np.logical_and(
                self.labels[:, 0] == img_index, self.labels[:, 4] == 1
            ))[0]
        else:
            msk_indices_pos = np.where(self.labels[:, 0] == img_index)[0]
        img_masks_pos = self.masks[self.__idx(msk_indices_pos)] / 255.

        if self.mode == "rnd":
            if self.leaf_only:
                msk_index = np.random.choice(np.where(np.logical_and(
                    self.labels[:, 0] != img_index, self.labels[:, 4] == 1
                ))[0])
            else:
                msk_index = np.random.choice(np.where(self.labels[:, 0] != img_index)[0])
            # print("randomed: ", self.labels[msk_index])
            img_mask = torch.FloatTensor(self.masks[self.__idx(msk_index)] / 255.)
            img_rect = torch.FloatTensor(self.rects[self.__idx(msk_index)])
        elif self.mode == "neg":  # "neg"
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
            img_masks_neg = self.masks[self.__idx(msk_indices_neg)] / 255.

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

            img_mask = torch.FloatTensor(self.masks[self.__idx(msk_index_neg)] / 255.)
            # img_rect = torch.FloatTensor(self.rects[self.__idx(msk_index_neg)])
        else:
            assert False

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
            # img, img_mask, _ = self.transform(img, img_mask, img_rect)
            img, img_mask, _ = self.transform(img, img_mask)

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
        assert self.set_name == "train"
        self.mode = mode
        self.leaf_only = leaf_only
        self.k = k
        bincount = np.bincount(self.labels[:, 2])
        self.extras = bincount[1] - bincount[0]
        assert self.extras > 0

    def vis(self, i: Optional[int] = None):
        if i is None:
            i = np.random.randint(len(self))
        data = self[i]
        plot([(data["image"][:-1], {}), (data["image"][-1], {}), 
              (data["image"][:-1], { "masks": data["image"][-1] })],
             [("original", "mask", f"{data['target'].item():.4f}")],
             figshow_kwargs={ "figsize": (12, 4) })


class GuiCodeDataset(Dataset):

    def __init__(self, vis_feature_path: str, data_path: str, set_name: Literal["train", "valid", "test"], 
                 fold: int, need_features: bool = False, mask: float = .0, vocabs_trans: Optional[str] = None):
        super().__init__()
        self.set_name = set_name
        self.fold_num = fold

        self.split = np.load(Path(data_path) / f"split_fold_{fold}.npz")[set_name]

        self.need_features = need_features
        self.mask = mask

        if vocabs_trans is not None:
            with open(vocabs_trans, "r") as input:
                self.vt: List[List[int]] = json.load(input)
        else:
            self.vt = None
        
        self.vis = np.load(vis_feature_path)

        self.hc = h5py.File(Path(data_path) / "codes.hdf5", "r")
        self.max_len = self.hc.attrs["max_len"]

        self.ims = self.hc["ims"][self.split]
        self.ids = self.hc["ids"][self.split]
        self.iis = self.hc["iis"][self.split]
        self.eqs = self.hc["eqs"][self.split]
        self.lbs = self.hc["lbs"][self.split]
        self.ivs = self.hc["ivs"][self.split]
        self.les = self.hc["les"][self.split]

    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Dict:
        code_len = self.les[index]
        code_idx = np.where(self.ids[index, :code_len] == self.eqs[index, :code_len])[0]

        iis = self.iis[index][code_idx]
        vish = np.zeros((self.max_len, ), dtype=np.float32)
        vish[:len(code_idx)] = self.vis["predicts"][iis]

        if self.need_features:
            visf = np.zeros((self.max_len, 2048), dtype=np.float32)
            visf[:len(code_idx)] = self.vis["features"][iis]

        code = np.zeros((self.max_len, ), dtype=np.int32)
        code[:len(code_idx)] = self.ivs[index][code_idx]

        if self.vt is not None:
            # print(f"old: {' '.join([f'{each:3}' for each in code])}")
            code_uni, code_inv = np.unique(code[:len(code_idx)], return_inverse=True)
            for i, each_iu in enumerate(code_uni):
                vt = self.vt[each_iu]
                if len(vt) > 1:
                    if np.random.rand() < 0.5:
                        it = np.random.choice(vt)
                        code[np.where(code_inv == i)] = it
            # print(f"new: {' '.join([f'{each:3}' for each in code])}")
        
        if self.mask > .0:
            rand = np.random.rand(*code.shape)
            rand_mask = (rand < self.mask) * (code != 0)
            code[rand_mask] = 3

        target = np.zeros((self.max_len, ), dtype=np.int32)
        target[:len(code_idx)] = self.lbs[index][code_idx] 
        
        if self.need_features:
            return {
                "vish": vish,
                "visf": visf,
                "code": code,
                "code_len": len(code_idx),
                "target": torch.FloatTensor(target)
            }

        return {
            "vish": vish,
            "code": code,
            "code_len": len(code_idx),
            "target": torch.FloatTensor(target)
        }   


class GuiVisCodeDataset(Dataset):

    def __init__(self, vis_data_path: str, code_data_path: str, set_name: Literal["train", "valid", "test"], 
                 fold: int, transform = None, mask: float = .0, vocabs_trans: Optional[str] = None):
        super().__init__()
        self.set_name = set_name
        self.fold_num = fold

        self.split = np.load(Path(code_data_path) / f"split_fold_{fold}.npz")[set_name]

        self.mask = mask

        if vocabs_trans is not None:
            with open(vocabs_trans, "r") as input:
                self.vt: List[List[int]] = json.load(input)
        else:
            self.vt = None
        
        self.h = h5py.File(Path(vis_data_path) / "images.hdf5", "r")
        self.images = self.h["images"]
        self.masks = self.h["masks"]

        self.hc = h5py.File(Path(code_data_path) / "codes.hdf5", "r")
        self.max_len = int(self.hc.attrs["max_len"])

        self.ims = self.hc["ims"][self.split]
        self.ids = self.hc["ids"][self.split]
        self.iis = self.hc["iis"][self.split]
        self.eqs = self.hc["eqs"][self.split]
        self.lbs = self.hc["lbs"][self.split]
        self.ivs = self.hc["ivs"][self.split]
        self.les = self.hc["les"][self.split]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Dict:
        code_len = self.les[index]
        code_idx = np.where(self.ids[index, :code_len] == self.eqs[index, :code_len])[0]

        ims = self.ims[index]
        image = torch.from_numpy(self.images[ims])

        if self.transform is not None:
            image, _, _ = self.transform(image)

        iis = self.iis[index][code_idx]
        mask = torch.FloatTensor(self.masks[iis] / 255.)

        image_n_mask = torch.zeros(self.max_len, 4, 256, 256)
        image_n_mask[:len(code_idx), :3, :, :] = image
        image_n_mask[:len(code_idx), 3:, :, :] = mask

        code = np.zeros((self.max_len, ), dtype=np.int32)
        code[:len(code_idx)] = self.ivs[index][code_idx]

        if self.vt is not None:
            # print(f"old: {' '.join([f'{each:3}' for each in code])}")
            code_uni, code_inv = np.unique(code[:len(code_idx)], return_inverse=True)
            for i, each_iu in enumerate(code_uni):
                vt = self.vt[each_iu]
                if len(vt) > 1:
                    if np.random.rand() < 0.5:
                        it = np.random.choice(vt)
                        code[np.where(code_inv == i)] = it
            # print(f"new: {' '.join([f'{each:3}' for each in code])}")
        
        if self.mask > .0:
            rand = np.random.rand(*code.shape)
            rand_mask = (rand < self.mask) * (code != 0)
            code[rand_mask] = 3

        target = np.zeros((self.max_len, ), dtype=np.int32)
        target[:len(code_idx)] = self.lbs[index][code_idx] 

        return {
            "image": image_n_mask,
            "code": code,
            "code_len": len(code_idx),
            "target": torch.FloatTensor(target)
        }