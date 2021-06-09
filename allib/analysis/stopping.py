from allib.estimation.rcapture import AbundanceEstimator
import collections
import itertools
from abc import ABC, abstractmethod
from typing import Deque, Generic, TypeVar, Any

import numpy as np # type: ignore

from ..activelearning import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from ..activelearning.estimator import Estimator
from .analysis import process_performance
from ..utils.func import all_equal, intersection



KT = TypeVar("KT")
VT = TypeVar("VT")
DT = TypeVar("DT")
RT = TypeVar("RT")
LT = TypeVar("LT")

class AbstractStopCriterion(ABC, Generic[LT]):
    @abstractmethod
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        pass

    @property
    @abstractmethod
    def stop_criterion(self) -> bool:
        pass

class DocCountStopCritertion(AbstractStopCriterion[LT], Generic[LT]):
    def __init__(self, max_docs: int):
        self.max_docs = max_docs
        self.doc_count = 0
    
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        self.doc_count = len(learner.env.labeled)
    
    @property
    def stop_criterion(self) -> bool:
        return self.doc_count >= self.max_docs

class RecallStopCriterion(AbstractStopCriterion[LT], Generic[LT]):
    def __init__(self, label: LT, target_recall: float):
        self.label = label
        self.target_recall = target_recall
        self.recall = 0.0

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        self.recall = process_performance(learner, self.label).recall

    @property
    def stop_criterion(self) -> bool:
        if self.recall >= 0 and self.recall <= 1:
            return self.recall >= self.target_recall
        return False


class SameStateCount(AbstractStopCriterion[LT], Generic[LT]):
    def __init__(self, label: LT, same_state_count: int):
        self.label = label
        self.same_state_count = same_state_count
        self.pos_history: Deque[int] = collections.deque()
        self.has_been_different = False

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        performance = process_performance(learner, self.label)
        self.add_count(len(performance.true_positives))

    def add_count(self, value: int) -> None:
        if len(self.pos_history) > self.same_state_count:
            self.pos_history.pop()
        if self.pos_history and not self.has_been_different:
            previous_value = self.pos_history[0]
            if previous_value != value:
                self.has_been_different = True
        self.pos_history.appendleft(value)

    @property
    def count(self) -> int:
        return self.pos_history[0]

    @property
    def same_count(self) -> bool:
        return all_equal(self.pos_history)

    @property
    def stop_criterion(self) -> bool:
        if len(self.pos_history) < self.same_state_count:
            return False 
        return self.has_been_different and self.same_count


class CaptureRecaptureCriterion(SameStateCount[LT], Generic[LT]):
    def __init__(self, calculator: AbundanceEstimator[Any, Any, Any, Any, LT], label: LT, same_state_count: int, margin: float):
        self.calculator = calculator
        super().__init__(label, same_state_count)
        self.estimate_history: Deque[float] = collections.deque()
        self.margin: float = margin

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        super().update(learner)
        if isinstance(learner, Estimator):
            self.add_count(learner.env.labels.document_count(self.label))
            estimate, lower, upper = self.calculator(learner, self.label)
            self.add_estimate(upper)
        
    def add_estimate(self, value: float) -> None:
        if len(self.estimate_history) > self.same_state_count:
            self.estimate_history.pop()
        self.estimate_history.appendleft(value)

    @property
    def estimate(self) -> float:
        return self.estimate_history[0]

    @property
    def estimate_match(self) -> bool:
        difference = abs(self.estimate - self.count)
        return difference < self.margin
        
    
    @property
    def same_count(self) -> bool:
        return all_equal(self.pos_history)

    @property
    def same_estimate(self) -> bool:
        return all_equal(self.estimate_history)

    @property
    def stop_criterion(self) -> bool:
        if len(self.estimate_history):
            if len(self.pos_history) < self.same_state_count:
                return False 
            return self.has_been_different and self.same_count and self.estimate_match
        return super().stop_criterion

class CaptureRecaptureCriterion2(CaptureRecaptureCriterion[LT], Generic[LT]):
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        self.add_count(learner.env.labels.document_count(self.label))
        if isinstance(learner, Estimator):
            estimate, lower, upper = self.calculator(learner, self.label)
            self.add_estimate(estimate)

class RaschCaptureCriterion(CaptureRecaptureCriterion[LT], Generic[LT]):
    @property
    def estimate(self) -> float:
        history = np.array([*self.estimate_history])
        return float(np.mean(history))

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        self.add_count(learner.env.labels.document_count(self.label))
        if isinstance(learner, Estimator):
            estimate, lower, upper = self.calculator(learner, self.label)
            dataset_size = len(learner.env.dataset)
            if estimate < dataset_size:
                self.add_estimate(estimate)

class EnsembleConvergenceCriterion(SameStateCount[LT], Generic[LT]):
    def __init__(self, label: LT, same_state_count: int, convergence_margin: float):
        super().__init__(label, same_state_count)
        self.margin = convergence_margin
        self.missing_history: Deque[float] = collections.deque()

    def add_missing(self, value: float) -> None:
        if len(self.missing_history) > self.same_state_count:
            self.missing_history.pop()
        self.missing_history.appendleft(value)

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        super().update(learner)
        if isinstance(learner, Estimator):
            common_positive = learner.env.labels.get_instances_by_label(self.label)
            positive_sets = intersection(*[
                    (member.env.labels
                        .get_instances_by_label(self.label)
                        .intersection(member.env.labeled)) 
                    for member in learner.learners
                ])
            missing_percentage = 1.0 - len(positive_sets) / len(common_positive)
            self.add_missing(missing_percentage)

    @property
    def missing_ratio(self) -> float:
        return self.missing_history[0]

    @property
    def within_margin(self) -> bool:
        return self.missing_ratio < self.margin

    @property
    def stop_criterion(self) -> bool:
        if self.missing_history:
            if len(self.pos_history) < self.same_state_count:
                return False 
            return self.has_been_different and self.same_count and self.within_margin
        return super().stop_criterion