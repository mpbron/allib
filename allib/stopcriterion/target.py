from typing import Any, FrozenSet, Generic
from ..activelearning.base import ActiveLearner
from ..activelearning.target import TargetMethod
from .base import AbstractStopCriterion
from ..typehints.typevars import LT


class TargetCriterion(AbstractStopCriterion[LT], Generic[LT]):
    found: FrozenSet[LT]
    targetset: FrozenSet[LT]
    phase: int
    pos_label: LT

    def __init__(self, pos_label: LT) -> None:
        self.phase = 0
        self.targetset = frozenset()
        self.found = frozenset()
        self.pos_label = pos_label

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, TargetMethod):
            self.phase = learner.current_learner
            if self.phase == 1:
                self.targetset = frozenset(
                    learner.learners[0].env.get_subset_by_labels(
                        learner.learners[0].env.labeled, self.pos_label
                    )
                )
                self.found = frozenset(
                    learner.learners[1].env.get_subset_by_labels(
                        learner.learners[1].env.labeled, self.pos_label
                    )
                )

    @property
    def stop_criterion(self) -> bool:
        return self.phase == 1 and self.targetset.issubset(self.found)
