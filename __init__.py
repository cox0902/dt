from typing import Optional, List, Tuple, Any, Dict

import os
import time
import random
import hashlib
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

    def update(self, predicts, targets, loss = None, reset: bool = True):
        if loss is not None:
            self.losses.update(loss)
        if predicts is not None and targets is not None:
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

    def format(self, show_scores: bool = True, show_average: bool = True, show_batch_time: bool = True, show_loss: bool = True):
        agg_metrics = []
        if show_batch_time:
            str_inline = f"Batch Time {timedelta(seconds=self.batch_time.val)}"
            if show_average:
                str_inline += f" ({timedelta(seconds=self.batch_time.avg)})"
            agg_metrics.append(str_inline)
        if show_loss:
            str_inline = f"Loss {self.losses.val:.4f}"
            if show_average:
                str_inline += f" ({self.losses.avg:.4f})"
            agg_metrics.append(str_inline)
        # if show_batch_time or show_loss:
        #     agg_metrics.append("\n")
        if show_scores:
            for i, metric in enumerate(self.metrics):
                if i % 5 == 0:
                    agg_metrics.append("\n")
                str_inline = f'{metric["name"]} {metric["meter"].val:.3f}'
                if show_average:
                    str_inline += f' ({metric["meter"].avg:.3f})'
                agg_metrics.append(str_inline)
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

        if model is not None:
            self.model = model.to(self.device)
        if criterion is not None:
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
            'criterion': self.criterion,
            'optimizer': self.optimizer,
        }
        if is_best:
            torch.save(state, 'BEST_3407.pth.tar')
        else:
            torch.save(state, 'checkpoint_3407.pth.tar')

    @staticmethod
    def load_checkpoint(save_file: str = None, is_best: bool = True):
        if save_file is None:
            if is_best:
                save_file = 'BEST_3407.pth.tar'
            else:
                save_file = 'checkpoint_3407.pth.tar'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(save_file, map_location=device)

        use_logits = saved['use_logits']

        model = saved['model']
        criterion = saved['criterion'] if 'criterion' in saved else None
        optimizer = saved['optimizer']

        md5 = hashlib.md5()  # ignore
        for arg in model.parameters():
            x = arg.data
            if hasattr(x, "cpu"):
                md5.update(x.cpu().numpy().data.tobytes())
            elif hasattr(x, "numpy"):
                md5.update(x.numpy().data.tobytes())
            elif hasattr(x, "data"):
                md5.update(x.data.tobytes())
            else:
                try:
                    md5.update(x.encode("utf-8"))
                except:
                    md5.update(str(x).encode("utf-8"))

        print(f"Loaded {md5.hexdigest()}")
        print(f"  from '{save_file}'")
        print(f"- use_logits: {saved['use_logits']}")
        print(f"- epoch     : {saved['epoch']}")
        print(f"- epochs_since_improvement: {saved['epochs_since_improvement']}")
        print(f"- score     : {saved['score']}")

        return Trainer(model=model, criterion=criterion, optimizer=optimizer,
                       use_logits=use_logits)
    
    def to_device(self, data):
        if type(data) in [tuple, list]:
            return [self.to_device(each) for each in data]
        return data.to(self.device)
    
    def collect_batch(self, samples):
        sources, targets = samples
        predicts = self.model(sources)
        logits = torch.sigmoid(predicts) if self.use_logits else predicts
        loss = self.criterion(predicts, targets)
        return loss, logits, targets

    def train(self, data_loader: DataLoader, metrics: Metrics, epoch: int, proof_of_concept: bool = False):
        self.model.train()

        for i, samples in enumerate(data_loader):
            samples = self.to_device(samples)
            loss, logits, targets = self.collect_batch(samples)
            # if type(sample) in [tuple, list]:
            #     sample = [each.to(self.device) for each in sample]   
            # else:
            #     sample = sample.to(self.device)  

            # targets = targets.to(self.device)

            # predict = self.model(sample)
            # logits = torch.sigmoid(predict) if self.use_logits else predict

            # loss = self.criterion(predict, targets)

            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip is not None:
                self.clip_gradient(self.grad_clip)

            self.optimizer.step()

            metrics.update(predicts=logits.squeeze(), targets=targets.squeeze(), loss=loss.item())

            if i % self.print_freq == 0:
                print(f"Epoch [{epoch}][{i}/{len(data_loader)}]\t{metrics.format()}")
                
            if proof_of_concept:
                break
        
    def valid(self, data_loader: DataLoader, metrics: Metrics, proof_of_concept: bool = False):
        self.model.eval()

        references = []
        hypotheses = []

        with torch.no_grad():
            for i, samples in enumerate(data_loader):
                samples = self.to_device(samples)
                loss, logits, targets = self.collect_batch(samples)
                # if type(sample) in [tuple, list]:
                #     sample = [each.to(self.device) for each in sample]     
                # else:
                #     sample = sample.to(self.device)  
                # targets = targets.to(self.device)

                # predict = self.model(sample)
                # logits = torch.sigmoid(predict) if self.use_logits else predict

                # loss = self.criterion(predict, targets)

                metrics.update(predicts=logits.squeeze(), targets=targets.squeeze(), loss=loss.item())

                if i % self.print_freq == 0:
                    print(f'\nValidation [{i}/{len(data_loader)}]\t{metrics.format()}')

                references.extend(targets.squeeze())
                hypotheses.extend(logits.squeeze())

                if proof_of_concept:
                    break

            hypotheses = torch.Tensor(hypotheses)
            references = torch.Tensor(references)
            metrics.update(predicts=hypotheses, targets=references)
            print(f'\n * {metrics.format(show_batch_time=False)}')

        return metrics.compute(hypotheses, references)
    
    def test(self, data_loader: DataLoader, metrics: Metrics, proof_of_concept: bool = False):
        self.model.eval()

        references = []
        hypotheses = []

        with torch.no_grad():
            for i, samples in enumerate(data_loader):
                samples = self.to_device(samples)
                _, logits, targets = self.collect_batch(samples)
                # if type(sample) in [tuple, list]:
                #     sample = [each.to(self.device) for each in sample]     
                # else:
                #     sample = sample.to(self.device)  
                # targets = targets.to(self.device)

                # predict = self.model(sample)
                # logits = torch.sigmoid(predict) if self.use_logits else predict

                metrics.update(None, None)  # 

                if i % self.print_freq == 0:
                    print(f'Test [{i}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')

                references.extend(targets.squeeze())
                hypotheses.extend(logits.squeeze())

                if proof_of_concept:
                    break

            hypotheses = torch.Tensor(hypotheses)
            references = torch.Tensor(references)
            metrics.update(predicts=hypotheses, targets=references)
            print(f'\n* {metrics.format(show_average=False, show_batch_time=False, show_loss=False)}')

            metrics.compute(hypotheses, references)

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