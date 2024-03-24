from typing import Optional, List, Tuple, Any, Dict

import os
import time
import random
import numpy as np
from datetime import timedelta

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision


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


class AverageMeter:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:

    def __init__(self, metrics: List[Dict[str, Any]], scorer = None):
        self.start_time = time.perf_counter()
        self.batch_time = AverageMeter()
        self.losses = AverageMeter()
        self.metrics = [{
            'name': each_name,
            'meter': AverageMeter(),
            'scorer': each_scorer
        } for each_name, each_scorer in metrics.items()]
        self.scorer = scorer

    def update(self, loss, predicts, targets, reset: bool = True):
        self.losses.update(loss)
        for metric in self.metrics:
            if reset:
                metric['scorer'].reset()
            metric['scorer'].update(predicts, targets.long())
            metric['meter'].update(metric['scorer'].compute())
        self.batch_time.update(time.perf_counter() - self.start_time)
        self.start_time = time.perf_counter()

    def compute(self, hypotheses, references):
        scorer = self.scorer()
        scorer.update(hypotheses, references)
        return scorer.compute()

    def format(self, show_batch_time: bool = True):
        agg_metrics = []
        if show_batch_time:
            agg_metrics.append(f"Batch Time {timedelta(seconds=self.batch_time.val)} ({timedelta(seconds=self.batch_time.avg)})")
        agg_metrics.append(f"Loss {self.losses.val:.4f} ({self.losses.avg:.4f})\n")
        for metric in self.metrics:
            agg_metrics.append(f'{metric["name"]} {metric["meter"].val:.3f} ({metric["meter"].avg:.3f})')
        return '\t'.join(agg_metrics)


class Trainer:

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                 use_logits: bool = False):
        self.use_logits: bool = use_logits
        self.print_freq: int = 100
        
        self.epochs_early_stop: int = 10
        self.epochs_adjust_lr: int = 4
        self.grad_clip: float = 5.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

    def adjust_learning_rate(self, shrink_factor: float):
        print("\nDECAYING learning rate...")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * shrink_factor
        print(f"- The new learning rate is {self.optimizer.param_groups[0]['lr']}\n")

    def clip_gradient(self, grad_clip: float):
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def save_checkpoint(self, epoch: int, epochs_since_improvement: int, score, is_best: bool):
        state = {
            'use_logits': self.use_logits,

            'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'score': score,
            'model': self.model,
            'optimizer': self.optimizer,
        }
        if is_best:
            torch.save(state, 'BEST_3407.pth.tar')
        else:
            torch.save(state, 'checkpoint_3407.pth.tar')

    def load_checkpoint(self, is_best: bool):
        if is_best:
            saved = torch.load('BEST_3407.pth.tar')
        else:
            saved = torch.load('checkpoint_3407.pth.tar')
        self.model = saved['model']
        self.optimizer = saved['optimizer']

    def train(self, data_loader: DataLoader, metrics: Metrics, epoch: int, proof_of_concept: bool = False):
        self.model.train()

        # batch_time = AverageMeter()
        # losses = AverageMeter()
        # scores = AverageMeter()

        start_time = time.perf_counter()
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            predict = self.model(images)
            logits = torch.sigmoid(predict) if self.use_logits else predict
            # print(predict.shape)

            loss = self.criterion(predict, targets)

            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip is not None:
                self.clip_gradient(self.grad_clip)

            self.optimizer.step()

            # scorer = BinaryAUROC()  # BinaryF1Score()
            # scorer.update(logits.squeeze(), targets.squeeze())
            
            # losses.update(loss.item())
            # scores.update(scorer.compute())
            # batch_time.update(time.perf_counter() - start_time)
            # start_time = time.perf_counter()

            metrics.update(loss=loss.item(), predicts=logits.squeeze(), targets=targets.squeeze())

            if i % self.print_freq == 0:
                print(f"Epoch [{epoch}][{i}/{len(data_loader)}]\t{metrics.format()}")
                
            if proof_of_concept:
                break
        
    def valid(self, data_loader: DataLoader, metrics: Metrics, proof_of_concept: bool = False):
        self.model.eval()

        # batch_time = AverageMeter()
        # losses = AverageMeter()
        # scores = [
        #     { "name": "Acc", "meter": AverageMeter(), "scorer": BinaryAccuracy() },
        #     { "name": "AUC", "meter": AverageMeter(), "scorer": BinaryAUROC() },
        # ]

        references = []
        hypotheses = []

        start_time = time.perf_counter()
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                predict = self.model(images)
                logits = torch.sigmoid(predict) if self.use_logits else predict

                loss = self.criterion(predict, targets)

                # losses.update(loss.item())

                # for score in scores:
                #     score["scorer"].reset()
                #     score["scorer"].update(logits.squeeze(), targets.squeeze())
                #     score["meter"].update(score["scorer"].compute())

                # batch_time.update(time.perf_counter() - start_time)
                # start_time = time.perf_counter()

                metrics.update(loss=loss.item(), predicts=logits.squeeze(), targets=targets.squeeze())

                if i % self.print_freq == 0:
                    # print_str = f''
                    # print_str += f'Batch Time {timedelta(seconds=batch_time.val)} ({timedelta(seconds=batch_time.avg)})\t'
                    # print_str += f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    # for score in scores:
                    #     print_str += f'{score["name"]} {score["meter"].val:.3f} ({score["meter"].avg:.3f})\t'
                    print(f'\nValidation [{i}/{len(data_loader)}]\t{metrics.format()}')

                references.extend(targets.squeeze())
                hypotheses.extend(logits.squeeze())

                if proof_of_concept:
                    break

            # print_str = ", ".join([f'{score["name"]} {score["meter"].avg:.3f}' for score in scores])
            print(f'\n * {metrics.format(show_batch_time=False)}')

        # f1_score = BinaryAUROC()
        # f1_score.update(torch.Tensor(hypotheses), torch.Tensor(references))

        # bcm = BinaryConfusionMatrix()
        # bcm.update(torch.Tensor(hypotheses), torch.Tensor(references).long())
        # m = bcm.compute().long()
        # print(tabulate([["T", f"TP {m[0][0]}", f"FN {m[0][1]}"], ["F", f"FP {m[1][0]}", f"TN {m[1][1]}"]], 
        #             headers=["", "P", "N"], tablefmt="psql"))
        # print()

        # return f1_score.compute()
        return metrics.compute(torch.Tensor(hypotheses), torch.Tensor(references))

    def fit(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader, metrics: Metrics,
            proof_of_concept: bool = False):
        epochs_since_improvement = 0
        best_score = 0

        for epoch in range(epochs):
            if epochs_since_improvement == self.epochs_early_stop:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % self.epochs_adjust_lr == 0:
                self.adjust_learning_rate(0.8)

            self.train(data_loader=train_loader, metrics=metrics, epoch=epoch, proof_of_concept=proof_of_concept)

            recent_score = self.valid(data_loader=valid_loader, metrics=metrics, proof_of_concept=proof_of_concept)
            
            is_best = recent_score > best_score
            best_score = max(recent_score, best_score)
            if not is_best:
                epochs_since_improvement += 1
                print(f"\nEpochs since last improvement: {epochs_since_improvement} ({best_score})\n")
            else:
                epochs_since_improvement = 0

            # save checkpoint
            self.save_checkpoint(epoch=epoch, epochs_since_improvement=epochs_since_improvement, 
                                 score=recent_score, is_best=is_best)
            
            if proof_of_concept:
                break