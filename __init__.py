from typing import List, Tuple, Any, Dict

import os
import random
import numpy as np
import torch
import torchvision

from .trainer import Trainer


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Torch      : {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")


__all__ = [
    seed_everything,
    Trainer
]