from typing import Any, Generic
from instancelib.typehints.typevars import LT
from allib.activelearning import ActiveLearner

from allib.activelearning.base import ActiveLearner

from ..activelearning.learnersequence import LearnerSequence

from .base import AbstractStopCriterion

from ..activelearning.base import ActiveLearner


class LastSequence(AbstractStopCriterion[LT], Generic[LT]):
    last_status: bool

    def __init__(self) -> None:
        super().__init__()
        self.last_status = False

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, LearnerSequence) and learner.stopcriteria:
            if learner.current_learner == len(learner.learners) - 1:
                self.last_status = learner.stopcriteria[-1].stop_criterion

    @property
    def stop_criterion(self) -> bool:
        return self.last_status


class PriorToLast(LastSequence[LT], Generic[LT]):
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, LearnerSequence) and learner.stopcriteria:
            if learner.current_learner >= len(learner.learners) - 2:
                self.last_status = learner.stopcriteria[-2].stop_criterion


class DummyFalse(AbstractStopCriterion[LT], Generic[LT]):
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        pass

    @property
    def stop_criterion(self) -> bool:
        return False
