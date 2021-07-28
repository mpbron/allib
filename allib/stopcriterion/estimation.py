import collections
from typing import Any, Deque, Generic

from instancelib.utils.func import all_equal, intersection
import numpy as np

from ..activelearning import ActiveLearner
from ..typehints import LT
from .heuristic import SameStateCount

from ..activelearning.estimator import Estimator
from ..estimation.base import AbstractEstimator

class CaptureRecaptureCriterion(SameStateCount[LT], Generic[LT]):
    def __init__(self, calculator: AbstractEstimator[Any, Any, Any, Any, Any, LT], label: LT, same_state_count: int, margin: float):
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

class CombinedStopCriterion(RaschCaptureCriterion[LT], Generic[LT]):
    def __init__(self, 
                 calculator: AbstractEstimator[Any ,Any, Any, Any, Any, LT],
                 label: LT, same_state_count: int, margin: float, convergence_margin: float):
        super().__init__(calculator, label, same_state_count, margin)
        self.convergence_margin = convergence_margin
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
    def estimate(self) -> float:
        return self.estimate_history[0]

    @property
    def missing_ratio(self) -> float:
        return self.missing_history[0]

    @property
    def within_margin(self) -> bool:
        return self.missing_ratio <= self.convergence_margin

    @property
    def stop_criterion(self) -> bool:
        if self.missing_history:
            if len(self.pos_history) < self.same_state_count:
                return False 
            return self.has_been_different and self.same_count and self.within_margin or super().stop_criterion
        return super().stop_criterion

class TwoCombinedStopCriterion(CombinedStopCriterion, Generic[LT]):
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        super().update(learner)
        if isinstance(learner, Estimator):
            common_positive = learner.env.labels.get_instances_by_label(self.label)
            common_positives = np.array(([len(common_positive) * len(learner.learners)]))
            positive_sets = np.array([len(member.env.labels
                                .get_instances_by_label(self.label)
                                .intersection(member.env.labeled)) 
                            for member in learner.learners
                            ])
            missing_percentages = 1.0 - positive_sets / common_positives
            missing_percentage = 1.0 - len(positive_sets) / len(common_positive)
            self.add_missing(missing_percentage)

class UpperboundCombinedCritertion(CombinedStopCriterion[LT], Generic[LT]):
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        self.add_count(learner.env.labels.document_count(self.label))
        if isinstance(learner, Estimator):
            estimate, lower, upper = self.calculator(learner, self.label)
            dataset_size = len(learner.env.dataset)
            if upper < dataset_size:
                self.add_estimate(upper)
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
    def estimate_match(self) -> bool:
        projected_recall = (self.count / self.estimate)
        return projected_recall >= 0.95