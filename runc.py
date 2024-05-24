from typing import *
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# import sys
# sys.path.append("..")

from dt.utils import seed_everything
from dt.trainer import Trainer
from dt.metrics import SimpleBinaryMetrics
from dt.datasets import GuiVisCodeDataset
from dt.transforms import GuiVisPresetTrain, GuiVisPresetEval
from dt.models import VisModel, VisCodeModel


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--mode", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--load-weight", action="store_true")
    parser.add_argument("--copy-weight", action="store_true")
    parser.add_argument("--embedding-size", type=int)
    parser.add_argument("--hidden-size", type=int)

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--pos-weight", type=float)

    parser.add_argument("--vis-data-path", type=str)
    parser.add_argument("--code-data-path", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--use-ta", action="store_true")
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

    #

    train_set = GuiVisCodeDataset(args.vis_data_path, args.code_data_path, "train", args.fold, 
                                  transform=GuiVisPresetTrain(use_ta=args.use_ta))
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
        
    valid_set = GuiVisCodeDataset(args.vis_data_path, args.code_data_path, "valid", args.fold, transform=GuiVisPresetEval())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # 

    if args.model in ["resnet", "resnext"]:
        vis_model = VisModel(model=args.model, load_weight=args.load_weight, copy_weight=args.copy_weight, 
                            use_logits=True)
    else:
        vis_trainer = Trainer.load_checkpoint(args.model)
        vis_model = vis_trainer.get_inner_model()

    model = VisCodeModel(vis_model, args.mode, 86, train_set.max_len, embedding_size=args.embedding_size, hidden_size=args.hidden_size)

    if args.pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight]))

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        assert False

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, generator=generator,
                      is_ema=args.ema, use_amp=args.amp)
    trainer.fit(epochs=args.epochs, train_loader=train_loader, valid_loader=valid_loader, metrics=SimpleBinaryMetrics(), 
                proof_of_concept=args.proof_of_concept)

    #
    
    print("=" * 100)
    test_set = GuiVisCodeDataset(args.vis_data_path, args.code_data_path, "test", args.fold, transform=GuiVisPresetEval())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
                         
    trainer = Trainer.load_checkpoint("./BEST.pth.tar")
    _ = trainer.test(data_loader=test_loader, metrics=SimpleBinaryMetrics(), proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)