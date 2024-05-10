from typing import *

import os
import random
import numpy as np

import torch
import torchvision


def seed_everything(seed, state: Optional[Dict[str, Any]] = None) -> torch.Generator:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Torch      : {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")

    g = torch.Generator()
    g.manual_seed(seed)
    
    if state is not None:
        g.set_state(state["generator"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"])
        torch.cuda.set_rng_state(state["gpu"])

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        random.seed(worker_seed) 
        np.random.seed(worker_seed)
        if state is not None:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])

    return g, seed_worker