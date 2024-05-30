from typing import *
import argparse

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# import sys
# sys.path.append("..")

from dt.utils import seed_everything
from dt.trainer import Trainer
from dt.metrics import SimpleBinaryMetrics
from dt.datasets import GuiVisDataset
from dt.transforms import GuiVisPresetTrain, GuiVisPresetEval
from dt.models import VisModel


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--model", type=str)
    parser.add_argument("--load-weight", action="store_true")
    parser.add_argument("--copy-weight", action="store_true")

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)

    parser.add_argument("--data-path", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--use-ta", action="store_true")
    parser.add_argument("--label-smooth", action="store_true")
    parser.add_argument("--resample", type=str)

    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", default=120, type=int)

    return parser


def main(args):
    print(args)

    generator, seed_worker = seed_everything(args.seed)

    model = VisModel(model=args.model, load_weight=args.load_weight, copy_weight=args.copy_weight)
    criterion = nn.BCEWithLogitsLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_set = GuiVisDataset(args.data_path, "train", args.fold, 
                              transform=GuiVisPresetTrain(use_ta=args.use_ta),
                              label_smooth=args.label_smooth)
    if args.resample == "rng":
        train_set.resample(mode="rnd")
    elif args.resample == "neg":
        train_set.resample(mode="neg")

    if args.resample == "weight":
        train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, 
                                  num_workers=args.workers, worker_init_fn=seed_worker, 
                                  sampler=WeightedRandomSampler(train_set.sample_weights, len(train_set), generator=generator))
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                                  num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        
    valid_loader = DataLoader(GuiVisDataset(args.data_path, "valid", args.fold, transform=GuiVisPresetEval()), 
                              batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, generator=generator,
                      is_ema=args.ema, use_amp=args.amp)
    trainer.fit(epochs=args.epochs, train_loader=train_loader, valid_loader=valid_loader, metrics=SimpleBinaryMetrics(), 
                proof_of_concept=args.proof_of_concept)
    
    print("=" * 100)
    test_loader = DataLoader(GuiVisDataset(args.data_path, "test", args.fold, transform=GuiVisPresetEval()), 
                             batch_size=args.batch_size, shuffle=False)
                         
    trainer = Trainer.load_checkpoint("./BEST.pth.tar")
    _ = trainer.test(data_loader=test_loader, metrics=SimpleBinaryMetrics(), proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)