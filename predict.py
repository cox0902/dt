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

    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--no-logits", action="store_true")

    return parser


def main(args):
    print(args)

    print("=" * 100)
    trainer = Trainer.load_checkpoint(args.model)

    print("=" * 100)
    if args.fold is not None:
        test_loader = DataLoader(GuiVisDataset(args.data_path, "test", args.fold, transform=GuiVisPresetEval()), 
                                 batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=args.workers)
        _ = trainer.test(data_loader=test_loader, metrics=SimpleBinaryMetrics(), proof_of_concept=args.proof_of_concept)

    if args.save_path:
        print("=" * 100)
        save_path = Path(args.save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        data_loader = DataLoader(GuiVisDataset(args.data_path, None, None, transform=GuiVisPresetEval()), 
                                 batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=args.workers)
        
        hook = None if args.no_features else "avgpool"
        return_logits = not args.no_logits
        predicts, _, logits, features = trainer.test(data_loader=data_loader, metrics=SimpleBinaryMetrics(), 
                                                     hook=hook, proof_of_concept=args.proof_of_concept,
                                                     return_logits=return_logits)
        
        save_dict = { "predicts": predicts }
        if not args.no_logits:
            save_dict["logits"] = logits
        if not args.no_features:
            save_dict["features"] = features
        np.savez(save_path / "outputs.npz", **save_dict)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)