from typing import *
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

# import sys
# sys.path.append("..")

from dt.trainer import Trainer
from dt.metrics import SimpleBinaryMetrics
from dt.datasets import GuiVisDataset
from dt.transforms import GuiVisPresetEval
from dt.models import VisModel


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")

    parser.add_argument("--model", type=str)

    parser.add_argument("--data-path", type=str)
    parser.add_argument("--fold", type=int)

    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)

    parser.add_argument("--save-path", type=str)

    return parser


def main(args):
    print(args)

    print("=" * 100)
    test_loader = DataLoader(GuiVisDataset(args.data_path, "test", args.fold, transform=GuiVisPresetEval()), 
                             batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers)
                         
    trainer = Trainer.load_checkpoint(args.model)
    _ = trainer.test(data_loader=test_loader, metrics=SimpleBinaryMetrics(), proof_of_concept=args.proof_of_concept)

    if args.save_path:
        print("=" * 100)
        data_loader = DataLoader(GuiVisDataset(args.data_path, None, None, transform=GuiVisPresetEval()), 
                                batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                num_workers=args.workers)
        
        predicts, _, logits, features = trainer.test(data_loader=data_loader, metrics=SimpleBinaryMetrics(), 
                                                    hook="avgpool", proof_of_concept=False)
        
        np.savez(Path(args.save_path) / "outputs.npz", predicts=predicts, logits=logits, features=features)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)