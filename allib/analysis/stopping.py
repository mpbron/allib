from allib.estimation.abundance import AbundanceEstimator
import collections
import itertools
from abc import ABC, abstractmethod
from typing import Deque, Generic, TypeVar, Any

from ..activelearning import ActiveLearner
from ..activelearning.estimator import Estimator
from .analysis import process_performance
from ..utils.func import all_equal

KT = TypeVar("KT")
VT = TypeVar("VT")
DT = TypeVar("DT")
RT = TypeVar("RT")
LT = TypeVar("LT")

class AbstractStopCriterion(ABC, Generic[LT]):
    @abstractmethod
    def update(self, learner: ActiveLearner[KT, DT, VT, RT, LT]) -> None:
        pass

    @property
    @abstractmethod
    def stop_criterion(self) -> bool:
        pass

class RecallStopCriterion(AbstractStopCriterion[LT], Generic[LT]):
    def __init__(self, label: LT, target_recall: float):
        self.label = label
        self.target_recall = target_recall
        self.recall = 0.0

    def update(self, learner: ActiveLearner[KT, DT, VT, RT, LT]) -> None:
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

    def update(self, learner: ActiveLearner[KT, DT, VT, RT, LT]):
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
    def __init__(self, label: LT, same_state_count: int, margin: float):
        self.calculator = AbundanceEstimator[KT, DT, VT, RT, LT]()
        super().__init__(label, same_state_count)
        self.estimate_history: Deque[float] = collections.deque()
        self.margin: float = margin

    def update(self, learner: ActiveLearner[KT, DT, VT, RT, LT]):
        super().update(learner)
        if isinstance(learner, Estimator):
            self.add_count(learner.env.labels.document_count(self.label))
            estimate, error = self.calculator(learner, self.label)
            self.add_estimate(estimate + error)
            print(f"Found {self.pos_history[0]} positive documents. The current estimate is"
                f"{estimate:.2f} (+- {error:.2f})")
        
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
