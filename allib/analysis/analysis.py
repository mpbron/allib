from numpy.lib.function_base import average
from allib.utils.func import union
import itertools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Generic, Sequence, TypeVar, Any, Tuple

import numpy as np  # type: ignore
from scipy import stats  # type: ignore

from ..activelearning import ActiveLearner
from ..activelearning.ml_based import MLBased
from ..history import BaseLogger, MemoryLogger
from instancelib.instances.base import Instance
from instancelib.labels.base import LabelProvider
from instancelib.labels.memory import MemoryLabelProvider

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




@dataclass
class BinaryPerformance(Generic[KT]):
    true_positives: FrozenSet[KT]
    true_negatives: FrozenSet[KT]
    false_positives: FrozenSet[KT]
    false_negatives: FrozenSet[KT]

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        recall = tp / ( tp + fn)
        return recall

    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            return 0.0
        return precision
    
    @property
    def accuracy(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    @property
    def wss(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        n = tp + fp + fn + tn
        wss = ((tn + fn) / n) - (1 - (tp / ( tp + fn)))
        return wss

    @property
    def f1(self) -> float:
        return self.f_beta(beta=1)

    def f_beta(self, beta: int=1) -> float:
        b2 = beta*beta
        try:
            fbeta = (1 + b2) * (
                (self.precision * self.recall) /
                ((b2 * self.precision) + self.recall))
        except ZeroDivisionError:
            fbeta = 0.0
        return fbeta

class MultilabelPerformance(Generic[KT, LT]):
    def __init__(self, *label_performances: Tuple[LT, BinaryPerformance[KT]]):
        self.label_dict = {
            label: performance for (label, performance) in label_performances}
    
    @property
    def true_positives(self) -> FrozenSet[KT]:
        keys = union(*(pf.true_positives for pf in self.label_dict.values()))
        return keys

    @property
    def true_negatives(self) -> FrozenSet[KT]:
        keys = union(*(pf.true_negatives for pf in self.label_dict.values()))
        return keys

    @property
    def false_negatives(self) -> FrozenSet[KT]:
        keys = union(*(pf.false_negatives for pf in self.label_dict.values()))
        return keys
    
    @property
    def false_positives(self) -> FrozenSet[KT]:
        keys = union(*(pf.false_positives for pf in self.label_dict.values()))
        return keys

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        recall = tp / ( tp + fn)
        return recall

    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        precision = tp / (tp + fp)
        return precision
    
    @property
    def accuracy(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    @property
    def f1(self) -> float:
        return self.f_beta(beta=1)

    def f_beta(self, beta: int=1) -> float:
        b2 = beta*beta
        fbeta = (1 + b2) * (
            (self.precision * self.recall) /
            ((b2 * self.precision) + self.recall)) 
        return fbeta

    @property
    def f1_macro(self) -> float:
        return self.f_macro(beta=1)

    def f_macro(self, beta=1) -> float:
        average_recall = np.mean([pf.recall for pf in self.label_dict.values()])
        average_precision = np.mean([pf.precision for pf in self.label_dict.values()])
        b2 = beta*beta
        fbeta = (1 + b2) * (
            (average_precision * average_recall) /
            ((b2 * average_precision) + average_recall)) 
        return fbeta
    
def label_metrics(truth: LabelProvider[KT, LT], 
                  prediction: LabelProvider[KT, LT], 
                  keys: Sequence[KT], label: LT):
    included_keys = frozenset(keys)
    ground_truth_pos = truth.get_instances_by_label(label).intersection(included_keys)
    pred_pos = prediction.get_instances_by_label(label)
    true_pos = pred_pos.intersection(ground_truth_pos)
    false_pos = pred_pos.difference(true_pos)
    false_neg = ground_truth_pos.difference(true_pos)
    true_neg = included_keys.difference(true_pos, false_pos, false_neg)
    return BinaryPerformance[KT](true_pos, true_neg, false_pos, false_neg)

def classifier_performance(learner: MLBased[Any, KT, DT, VT, RT, LT, Any, Any], 
              ground_truth: LabelProvider[KT, LT],
              instances: Sequence[Instance[KT, DT, VT, RT]]) -> Dict[LT, BinaryPerformance[KT]]:
    keys = [ins.identifier for ins in instances]
    labelset = learner.env.labels.labelset
    pred_provider = MemoryLabelProvider[KT, LT](labelset, {}, {})
    predictions = learner.predict(instances)
    for ins, pred in zip(instances, predictions):
        pred_provider.set_labels(ins, *pred)
    performance = {
        label: label_metrics(
            ground_truth, 
            pred_provider, 
            keys,
            label
        ) for label in labelset
    }
    return performance

def classifier_performance_ml(learner: MLBased[Any, KT, DT, VT, RT, LT, Any, Any], 
              ground_truth: LabelProvider[KT, LT],
              instances: Sequence[Instance[KT, DT, VT, RT]]) -> MultilabelPerformance[KT, LT]:
    keys = [ins.identifier for ins in instances]
    labelset = learner.env.labels.labelset
    pred_provider = MemoryLabelProvider[KT, LT](labelset, {}, {})
    predictions = learner.predict(instances)
    for ins, pred in zip(instances, predictions):
        pred_provider.set_labels(ins, *pred)
    performances = [
        (label, label_metrics(ground_truth, 
                              pred_provider, 
                              keys, 
                              label)) for label in labelset]
    performance = MultilabelPerformance[KT, LT](*performances)    
    return performance

def process_performance(learner: ActiveLearner[Any, KT, Any, Any, Any, LT], label: LT) -> BinaryPerformance[KT]:
    labeled = frozenset(learner.env.labeled)
    labeled_positives = learner.env.labels.get_instances_by_label(label)
    labeled_negatives = labeled.difference(labeled_positives)
    
    truth_positives = learner.env.truth.get_instances_by_label(label)
   
    unlabeled  = frozenset(learner.env.unlabeled)
    unlabeled_positives = unlabeled.intersection(learner.env.truth.get_instances_by_label(label))
    unlabeled_negatives = unlabeled.difference(unlabeled_positives)
    
    true_positives = labeled_positives.intersection(truth_positives)
    false_positives = labeled_positives.difference(truth_positives).union(labeled_negatives)
    false_negatives = truth_positives.difference(labeled_positives)
    true_negatives = unlabeled_negatives

    return BinaryPerformance(true_positives, true_negatives, false_positives, false_negatives)