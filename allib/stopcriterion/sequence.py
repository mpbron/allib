from typing import Any, Generic
from instancelib.typehints.typevars import LT

from ..activelearning.learnersequence import LearnerSequence

from .base import AbstractStopCriterion

from ..activelearning.base import ActiveLearner


class LastSequence(AbstractStopCriterion, Generic[LT]):
    last_status: bool

    def __init__(self) -> None:
        super().__init__()
        self.last_status = False

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, LearnerSequence) and learner.stopcriteria:
            self.last_status = learner.stopcriteria[-1].stop_criterion

    @property
    def stop_criterion(self) -> bool:
        return self.last_status
