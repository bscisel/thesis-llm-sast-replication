from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


from analysis.dataset import Finding

SEVERITY_ORDER = {"BLOCKER": 0, "CRITICAL": 1, "MAJOR": 2, "MINOR": 3, "INFO": 4}


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def accuracy(self) -> float | None:
        return _ratio(self.tp + self.tn, self.total)

    @property
    def fp_reduction(self) -> float | None:
        return _ratio(self.tn, self.fp + self.tn)

    @property
    def lost_tp(self) -> int:
        return self.fn

    @property
    def lost_tp_share(self) -> float | None:
        return _ratio(self.fn, self.tp + self.fn)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "evaluated": self.total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "fp_reduction": self.fp_reduction,
            "lost_tp": self.lost_tp,
            "lost_tp_share": self.lost_tp_share,
        }


def confusion(pairs: Iterable[tuple[bool, bool]]) -> Confusion:
    tp = fp = fn = tn = 0
    for label, predicted in pairs:
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


Predictor = Callable[[Finding], bool]


