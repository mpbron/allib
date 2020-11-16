import collections
import itertools
from abc import ABC, abstractmethod
from typing import Deque, Generic, TypeVar

from ..activelearning import ActiveLearner
from ..activelearning.estimator import Estimator
from .analysis import process_performance
from ..utils.func import all_equal

KT = TypeVar("KT")
VT = TypeVar("VT")
DT = TypeVar("DT")
RT = TypeVar("RT")
LT = TypeVar("LT")

class AbstractStopCriterion(ABC):
    @abstractmethod
    def update(self, learner: ActiveLearner[KT, DT, VT, RT, LT]) -> None:
        pass

    @property
    @abstractmethod
    def stop_criterion(self) -> bool:
        pass

class RecallStopCriterion(AbstractStopCriterion, Generic[LT]):
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



class StopCriterion(AbstractStopCriterion, Generic[LT]):
    def __init__(self, label: LT, same_state_count: int):
        self.label = label
        self.same_state_count = same_state_count
        self.pos_history: Deque[int] = collections.deque()
        self.estimate_history: Deque[float] = collections.deque()

    def update(self, learner: Estimator):
        self.add_count(learner.env.labels.document_count(self.label))
        abd = learner.get_abundance(self.label)
        if abd is not None:
            estimate, error = abd
            if estimate > 0 and error < estimate and error < 50:
                self.add_estimate(estimate + error)
        print(f"Found positive: {self.pos_history[0]} % with estimate "
              f"{self.estimate_history[0]:.2f}")
        

    def add_count(self, value: int) -> None:
        if len(self.pos_history) > self.same_state_count:
            self.pos_history.pop()
        self.pos_history.appendleft(value)

    def add_estimate(self, value: float) -> None:
        if len(self.estimate_history) > self.same_state_count:
            self.estimate_history.pop()
        self.estimate_history.appendleft(value)
    
    @property
    def same_count(self) -> bool:
        return all_equal(self.pos_history)

    @property
    def same_estimate(self) -> bool:
        return all_equal(self.estimate_history)

    @property
    def stop_criterion(self) -> bool:
        if len(self.pos_history) < self.same_state_count:
            return False 
        return self.same_count and self.same_estimate
