from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Generic, Sequence, Tuple, TypeVar

import numpy as np
from instancelib.instances.base import Instance
from instancelib.labels.base import LabelProvider
from instancelib.labels.memory import MemoryLabelProvider

from typing_extensions import Self
from ..activelearning import ActiveLearner
from ..activelearning.ml_based import MLBased
from ..utils.func import union
from instancelib.analysis.base import BinaryModelMetrics

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")


class ResultUnit(Enum):
    PERCENTAGE = "Percentage"
    ABSOLUTE = "Absolute"
    FRACTION = "Fraction"


def loss_er(pos_found: int, effort: int, pos_size: int, dataset_size: int) -> float:
    recall_perc = pos_found / pos_size
    inability_loss = (1 - recall_perc) ** 2
    effort_loss = (100 / dataset_size) ** 2 * (effort / (pos_found + 100)) ** 2
    return inability_loss + effort_loss


@dataclass(frozen=True)
class BinaryPerformance(BinaryModelMetrics[KT, LT], Generic[KT, LT]):
    loss_er: float = field()  # type: ignore

    @property
    def loss_er(self) -> float:
        pos_found = len(self.true_positives)
        dataset_size = len(
            union(
                self.true_positives,
                self.false_positives,
                self.false_negatives,
                self.true_negatives,
            )
        )
        effort = len(union(self.true_positives, self.false_positives))
        n_pos_truth = len(union(self.true_positives, self.false_negatives))
        score = loss_er(pos_found, effort, n_pos_truth, dataset_size)
        return score

    @loss_er.setter
    def loss_er(self, value: float) -> None:
        pass

    @classmethod
    def from_bm_metrics(cls, metrics: BinaryModelMetrics[KT, LT]) -> Self:
        return cls(
            metrics.pos_label,
            metrics.neg_label,
            metrics.true_positives,
            metrics.true_negatives,
            metrics.false_positives,
            metrics.false_negatives,
        )


def process_performance(
    learner: ActiveLearner[Any, KT, Any, Any, Any, LT], label: LT
) -> BinaryPerformance[KT, LT]:
    labeled = frozenset(learner.env.labeled)
    labeled_positives = frozenset(
        learner.env.get_subset_by_labels(learner.env.labeled, label)
    )
    labeled_negatives = labeled.difference(labeled_positives)

    truth_positives = learner.env.truth.get_instances_by_label(label)

    unlabeled = frozenset(learner.env.unlabeled)
    unlabeled_positives = unlabeled.intersection(
        learner.env.truth.get_instances_by_label(label)
    )
    unlabeled_negatives = unlabeled.difference(unlabeled_positives)

    true_positives = labeled_positives.intersection(truth_positives)
    false_positives = labeled_positives.difference(truth_positives).union(
        labeled_negatives
    )
    false_negatives = truth_positives.difference(labeled_positives)
    true_negatives = unlabeled_negatives

    return BinaryPerformance(
        label,
        None,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    )
