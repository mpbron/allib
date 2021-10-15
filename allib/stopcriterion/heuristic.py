import collections
from typing import Any, Deque, Generic

from instancelib.utils.func import all_equal

from ..activelearning import ActiveLearner
from ..analysis.analysis import process_performance
from ..typehints import LT
from .base import AbstractStopCriterion

class AllDocsCriterion(AbstractStopCriterion[LT], Generic[LT]):
    def __init__(self) -> None:
        self.remaining = 2000

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        self.remaining = len(learner.env.unlabeled)
    
    @property
    def stop_criterion(self) -> bool:
        return self.remaining <= 0

class DocCountStopCritertion(AbstractStopCriterion[LT], Generic[LT]):
    def __init__(self, max_docs: int):
        self.max_docs = max_docs
        self.doc_count = 0
    
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        self.doc_count = len(learner.env.labeled)
    
    @property
    def stop_criterion(self) -> bool:
        return self.doc_count >= self.max_docs

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
