
from typing import Dict

import hashlib
import torch
from torch import nn
from torch import optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import DataLoader

from .metrics import Metrics


class Trainer:

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                 is_ema: bool = False):
        self.print_freq: int = 100
        
        self.epochs_early_stop: int = 10
        self.epochs_adjust_lr: int = 4
        self.grad_clip: float = 5.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if model is not None:
            self.model = model.to(self.device)

        self.ema_model = None
        if is_ema:
            self.ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), 
                                           use_buffers=True)

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

    def save_checkpoint(self, epoch: int, epochs_since_improvement: int, score, is_best: bool,
                        save_checkpoint: bool = True):
        state = {
            'seed': torch.initial_seed(),
            'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'score': score,
            'model': self.model,
            'criterion': self.criterion,
            'optimizer': self.optimizer,
        }
        if self.ema_model is not None:
            state['ema_model'] = self.ema_model
        if is_best:
            torch.save(state, 'BEST.pth.tar')
        elif save_checkpoint:
            torch.save(state, 'checkpoint.pth.tar')

    @staticmethod
    def load_checkpoint(save_file: str = None, is_best: bool = True) -> "Trainer":
        if save_file is None:
            if is_best:
                save_file = 'BEST.pth.tar'
            else:
                save_file = 'checkpoint.pth.tar'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(save_file, map_location=device)

        model: nn.Module = saved['model']
        is_ema = 'ema_model' in saved
        criterion = saved['criterion'] if 'criterion' in saved else None
        optimizer = saved['optimizer']

        md5 = hashlib.md5()
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
        if 'seed' in saved:
            print(f"- seed      : {saved['seed']}")
        if is_ema:
            print(f"- is_ema    : True")
        print(f"- epoch     : {saved['epoch']}")
        print(f"- epochs_since_improvement: {saved['epochs_since_improvement']}")
        print(f"- score     : {saved['score']}")

        trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer)
        if is_ema:
            trainer.ema_model = saved['ema_model']
        return trainer
    
    def to_device(self, data: Dict) -> Dict:
        for k, v in data.items():
            data[k] = v.to(self.device)
        return data
    
    # def collect_batch(self, samples):
    #     sources, targets = samples
    #     predicts = self.model(sources)
    #     logits = torch.sigmoid(predicts) if self.use_logits else predicts
    #     loss = self.criterion(predicts, targets)
    #     return loss, logits, targets

    def train(self, data_loader: DataLoader, metrics: Metrics, epoch: int, proof_of_concept: bool = False):
        self.model.train()

        for i, batch in enumerate(data_loader):
            batch = self.to_device(batch)
            targets = batch["target"]

            self.optimizer.zero_grad()
            # loss, logits, targets = self.collect_batch(samples)
            logits, predicts = self.model(batch)
            loss = self.criterion(logits, targets)
            loss.backward()

            if self.grad_clip is not None:
                self.clip_gradient(self.grad_clip)

            self.optimizer.step()
            
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            metrics.update(predicts=predicts.squeeze(), targets=targets.squeeze(), loss=loss.item())

            if i % self.print_freq == 0:
                print(f"Epoch [{epoch}][{i}/{len(data_loader)}]\t{metrics.format()}")
                
            if proof_of_concept:
                break
        
    def valid(self, data_loader: DataLoader, metrics: Metrics, proof_of_concept: bool = False) -> float:
        model = self.model
        if self.ema_model is not None:
            model = self.ema_model

        model.eval()

        references = []
        hypotheses = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = self.to_device(batch)
                targets = batch["target"]
                # loss, logits, targets = self.collect_batch(samples)
                logits, predicts = model(batch)
                loss = self.criterion(logits, targets)

                metrics.update(predicts=predicts.squeeze(), targets=targets.squeeze(), loss=loss.item())

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
        model = self.model
        if self.ema_model is not None:
            model = self.ema_model

        model.eval()

        references = []
        hypotheses = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = self.to_device(batch)
                targets = batch["target"]
                # _, logits, targets = self.collect_batch(samples)
                _, predicts = model(batch)
               
                metrics.update(None, None)  # 

                if i % self.print_freq == 0:
                    print(f'Test [{i}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')

                references.extend(targets.squeeze())
                hypotheses.extend(predicts.squeeze())

                if proof_of_concept:
                    break

            hypotheses = torch.Tensor(hypotheses)
            references = torch.Tensor(references)
            metrics.update(predicts=hypotheses, targets=references)
            print(f'\n* {metrics.format(show_average=False, show_batch_time=False, show_loss=False)}')

            metrics.compute(hypotheses, references)
        return hypotheses, references

    def fit(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader, metrics: Metrics,
            save_checkpoint: bool = True, proof_of_concept: bool = False):
        epochs_since_improvement: int = 0
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
                print(f"\nEpochs since last improvement: {epochs_since_improvement} ({best_score})\n")  # [OK]
            else:
                epochs_since_improvement = 0

            # save checkpoint
            self.save_checkpoint(epoch=epoch, epochs_since_improvement=epochs_since_improvement, 
                                 score=recent_score, is_best=is_best, save_checkpoint=save_checkpoint)
            
            if proof_of_concept:
                break
