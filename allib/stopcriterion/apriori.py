from typing import Any, Generic

from ..activelearning import ActiveLearner
from ..analysis.analysis import process_performance
from ..typehints import LT
from .base import AbstractStopCriterion




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






