from typing import List, Tuple, Any, Dict, Type

import time
from collections import deque
from datetime import timedelta
from tabulate import tabulate
import torch
from torcheval.metrics import (
    BinaryAccuracy, BinaryConfusionMatrix, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
)
from torcheval.metrics.metric import Metric


class AverageMeter:
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.sum, self.count = 0, 0
        self.deque = deque(maxlen=self.window_size)

    def update(self, val, n: int = 1):
        self.deque.append(val)
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def smoothed_avg(self):
        return sum(self.deque) / len(self.deque)

    @property
    def val(self):
        return self.deque[-1]


def _tf(seconds) -> str:
    td = timedelta(seconds=seconds)
    mm, ss = divmod(td.seconds, 60)
    hh, mm = divmod(mm, 60)
    s = f'{hh}:{mm:02}:{ss:02}'
    if td.days:
        def plural(n):
            return n, abs(n) != 1 and "s" or ""
        s = ("%d day%s, " % plural(td.days)) + s
    return s
    
class Metrics:

    def __init__(self, metrics: Dict[str, Metric], scorer: Type[Metric] = None):
        self.start_time = time.perf_counter()
        self.batch_time = AverageMeter()
        self.batch_count = 0
        self.losses = AverageMeter(window_size=10)
        self.metrics = [{
            'name': each_name,
            'meter': AverageMeter(),
            'scorer': each_scorer
        } for each_name, each_scorer in metrics.items()]
        self.scorer: Type[Metric] = scorer

    def reset(self, batch_count):
        self.batch_time.reset()
        self.start_time = time.perf_counter()
        self.batch_count = batch_count
        for metric in self.metrics:
            metric['scorer'].reset()
            metric['meter'].reset()

    def update(self, predicts, targets, loss = None):
        if loss is not None:
            self.losses.update(loss)
        if predicts is not None and targets is not None:
            for metric in self.metrics:
                metric['scorer'].update(predicts, targets.long())
                metric['meter'].update(metric['scorer'].compute())
        self.batch_time.update(time.perf_counter() - self.start_time)
        self.start_time = time.perf_counter()

    def compute(self, hypotheses, references) -> float:
        if self.scorer is None:
            return 0.0
        scorer: Metric = self.scorer()
        scorer.update(hypotheses, references)
        return scorer.compute()
    
    def format(self, show_scores: bool = True, show_average: bool = True, 
               show_batch_time: bool = True, show_loss: bool = True) -> str:
        agg_metrics = []
        if torch.cuda.is_available():
            MB = 1024.0 * 1024.0
            ma, mr = torch.cuda.mem_get_info()
            # ma = torch.cuda.max_memory_allocated()
            # mr = torch.cuda.max_memory_reserved()
            agg_metrics.append(f"{int(ma / MB)} MB / {int(mr / MB)} MB")
        if show_batch_time:
            str_inline = f"ETA {_tf(self.batch_time.sum)}"
            if self.batch_count > 0 and self.batch_count > self.batch_time.count:
                str_inline += f" / FIN {_tf(self.batch_time.avg * (self.batch_count - self.batch_time.count))}"
            agg_metrics.append(str_inline)
        if show_loss:
            str_inline = f"Loss {self.losses.val:.4f}"
            if show_average:
                str_inline += f" ({self.losses.smoothed_avg:.4f})"
            agg_metrics.append(str_inline)
        # if show_batch_time or show_loss:
        #     agg_metrics.append("\n")
        if show_scores:
            for i, metric in enumerate(self.metrics):
                if i % 5 == 0:
                    agg_metrics.append("\n")
                str_inline = f'{metric["name"]} {metric["meter"].val:.5f}'
                if show_average:
                    str_inline += f' ({metric["meter"].avg:.5f})'
                agg_metrics.append(str_inline)
        return '\t'.join(agg_metrics)
    

class SimpleBinaryMetrics(Metrics):

    def __init__(self, metrics: Dict[str, Metric] = None, scorer: Type[Metric] = BinaryAUROC):
        if metrics is None:
            super().__init__({ 
                "Acc": BinaryAccuracy(),
                "AUC": BinaryAUROC() 
            }, scorer)
        else:
            super().__init__(metrics, scorer)

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


class SimpleBinaryMetricsAcc(SimpleBinaryMetrics):

    def __init__(self):
        super().__init__(scorer=BinaryAccuracy)


class BinaryMetrics(SimpleBinaryMetrics):

    def __init__(self):
        super().__init__({ 
            "Acc": BinaryAccuracy(),
            "Pre": BinaryPrecision(), 
            "Rec": BinaryRecall(),
            "F-1": BinaryF1Score(),
            "AUC": BinaryAUROC() 
        })


class EmptyMetrics(Metrics):

    def __init__(self):
        super().__init__({})