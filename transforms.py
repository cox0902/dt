
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors


class TrivialAugmentGui(T.TrivialAugmentWide):

    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),

        # "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        # "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        # "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
        # "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
        # "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
        
        # "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        # "Equalize": (lambda num_bins, height, width: None, False),
    }


class GuiVisPresetTrain:
    def __init__(
            self,
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225),
            use_ta: bool = False,
            hflip_prob = 0
    ):
        transforms_img = []
        if use_ta:
            transforms_img.append(TrivialAugmentGui())
        if len(transforms_img) != 0:
            self.transforms_img = T.Compose(transforms_img) 
        else:
            self.transforms_img = None

        transforms_all = []
        
        if hflip_prob > 0:
            transforms_all.append(T.RandomHorizontalFlip(p=hflip_prob))

        transforms_all.extend([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])
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
    