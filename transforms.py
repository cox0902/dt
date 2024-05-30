from typing import *

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors


# class TrivialAugmentGui(T.TrivialAugmentWide):

#     _AUGMENTATION_SPACE = {
#         "Identity": (lambda num_bins, height, width: None, False),

#         # "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
#         # "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
#         # "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
#         # "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
#         # "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
        
#         "Brightness": (lambda num_bins, height, width: torch.linspace(-0.5, 0.5, num_bins), True),
#         "Color": (lambda num_bins, height, width: torch.linspace(-0.99, 0.99, num_bins), True),
#         "Contrast": (lambda num_bins, height, width: torch.linspace(-0.5, 0.99, num_bins), True),
#         "Sharpness": (lambda num_bins, height, width: torch.linspace(-0.99, 0.99, num_bins), True),
#         "Posterize": (
#             lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(),
#             False,
#         ),
#         "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
#         "AutoContrast": (lambda num_bins, height, width: None, False),
#         # "Equalize": (lambda num_bins, height, width: None, False),
#     }

#     def __init__(
#         self,
#         num_magnitude_bins: int = 31,
#         interpolation = F.InterpolationMode.NEAREST,
#         fill = None,
#     ):
#         super().__init__(interpolation=interpolation, fill=fill)
#         self.num_magnitude_bins = num_magnitude_bins

#     def forward(self, *inputs):
#         flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
#         height, width = F.get_size(image_or_video)

#         transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

#         magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
#         if magnitudes is not None:
#             magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
#             # if signed and torch.rand(()) <= 0.5:
#             #     magnitude *= -1
#         else:
#             magnitude = 0.0

#         image_or_video = self._apply_image_or_video_transform(
#             image_or_video, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill
#         )
#         return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


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


def get_model_normalize_cfg(model_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if model_name in ["resnet", "resnext", "convnext"]:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif model_name.startswith("clip."):
        return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    assert False


class GuiVisPresetTrain:
    def __init__(
            self,
            model_name: str,
            use_ta: bool = False,
            hflip_prob = 0
    ):
        mean, std = get_model_normalize_cfg(model_name)

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

    def __call__(self, img, img_mask: Optional[Any] = None, img_rect: Optional[Any] = None):
        if self.transforms_img is not None:
            img = self.transforms_img(img)

        if img_mask is not None and img_rect is not None:
            return self.transforms_all(
                img, tv_tensors.Mask(img_mask), 
                tv_tensors.BoundingBoxes(img_rect, format="XYXY", canvas_size=(256, 256)))
        elif img_mask is not None:
            img, img_mask = self.transforms_all(img, tv_tensors.Mask(img_mask))
            return img, img_mask, None
        elif img_rect is not None:
            img, img_rect = self.transforms_all(
                img, tv_tensors.BoundingBoxes(img_rect, format="XYXY", canvas_size=(256, 256)))
            return img, None, img_rect
        return self.transforms_all(img), None, None


class GuiVisPresetEval:
    def __init__(self, model_name: str):
        mean, std = get_model_normalize_cfg(model_name)
        self.transforms = T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])

    def __call__(self, img, img_mask: Optional[Any] = None, img_rect: Optional[Any] = None):
        return self.transforms(img), img_mask, img_rect
    