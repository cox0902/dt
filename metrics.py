from typing import List, Tuple, Any, Dict, Type

import time
from datetime import timedelta
from tabulate import tabulate
from torcheval.metrics import (
    BinaryAccuracy, BinaryConfusionMatrix, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
)
from torcheval.metrics.metric import Metric


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

    def __init__(self, metrics: Dict[str, Metric], scorer: Type[Metric] = None):
        self.start_time = time.perf_counter()
        self.batch_time = AverageMeter()
        self.losses = AverageMeter()
        self.metrics = [{
            'name': each_name,
            'meter': AverageMeter(),
            'scorer': each_scorer
        } for each_name, each_scorer in metrics.items()]
        self.scorer: Type[Metric] = scorer

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

    def compute(self, hypotheses, references) -> float:
        scorer: Metric = self.scorer()
        scorer.update(hypotheses, references)
        return scorer.compute()

    def format(self, show_scores: bool = True, show_average: bool = True, 
               show_batch_time: bool = True, show_loss: bool = True) -> str:
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
    

class BinaryMetrics(Metrics):

    def __init__(self):
        super().__init__({ 
            "Acc": BinaryAccuracy(),
            "Pre": BinaryPrecision(), 
            "Rec": BinaryRecall(),
            "F-1": BinaryF1Score(),
            "AUC": BinaryAUROC() 
        }, BinaryAUROC)

    def compute(self, hypotheses, references) -> float:
        score = super().compute(hypotheses, references)

        bcm = BinaryConfusionMatrix()
        bcm.update(hypotheses, references.long())
        m = bcm.compute().long()
        TP = m[0][0]
        FN = m[0][1]
        FP = m[1][0]
        TN = m[1][1]
        print(tabulate([["T", f"TP {m[0][0]}", f"FN {m[0][1]}"], ["F", f"FP {m[1][0]}", f"TN {m[1][1]}"]], 
                       headers=["", "P", "N"], tablefmt="psql"))
        print(f'* POS * Pre {TP / (TP + FP):.5f} Rec {TP / (TP + FN):.5f} F-1 {2 * TP / (2 * TP + FP + FN):.5f}')
        print(f'* NEG * Pre {TN / (TN + FN):.5f} Rec {TN / (TN + FP):.5f} F-1 {2 * TN / (2 * TN + FP + FN):.5f}')
        return score
