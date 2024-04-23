
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors


class GuiVisPresetTrain:
    def __init__(
            self,
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225),
    ):
        transforms_img = []
        
        # transforms_img.append(TrivialAugmentGui())
        if len(transforms_img) != 0:
            self.transforms_img = T.Compose(transforms_img) 
        else:
            self.transforms_img = None

        transforms_all = []
        
        # transforms.append(T.RandomHorizontalFlip(p=1))

        transforms_all.extend([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ])

        transforms_all.append(T.ToPureTensor())
        self.transforms_all = T.Compose(transforms_all)

    def __call__(self, img, img_mask, img_rect):
        if self.transforms_img is not None:
            img = self.transforms_img(img)
        return self.transforms_all(
            img, tv_tensors.Mask(img_mask), 
            tv_tensors.BoundingBoxes(img_rect, format="XYXY", canvas_size=(256, 256)))


class GuiVisPresetEval:
    def __init__(
            self,
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225),
    ):
        self.transforms = T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])

    def __call__(self, img, img_mask, img_rect):
        return self.transforms(img), img_mask, img_rect
    