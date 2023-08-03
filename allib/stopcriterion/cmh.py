from typing import Any, FrozenSet, Generic
import numpy as np
from scipy.stats import hypergeom

from allib.activelearning.base import ActiveLearner

from ..stopcriterion.base import AbstractStopCriterion
from ..typehints.typevars import KT, LT
from ..activelearning.cmh import CMHMethod
from ..activelearning.base import ActiveLearner
from .others import CMH_HeuristicStoppingRule


class CMH_CertificationRule(AbstractStopCriterion[LT], Generic[LT]):
    phase: int
    test_result: bool
    pos_label: LT
    target_recall: float

    def __init__(
        self, pos_label: LT, target_recall: float, alpha: float = 0.05
    ) -> None:
        self.phase = 0
        self.pos_label = pos_label
        self.target_recall = target_recall
        self.alpha = alpha
        self.test_result = False

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, CMHMethod):
            self.phase = learner.current_learner
            if self.phase == 1:
                phase_0 = learner.learners[0]
                pos_count_all = len(
                    learner.env.get_subset_by_labels(
                        learner.env.labeled, self.pos_label
                    )
                )
                pos_count_phase_0 = len(
                    phase_0.env.get_subset_by_labels(
                        phase_0.env.labeled, self.pos_label
                    )
                )
                effort_phase_0 = len(phase_0.env.labeled)
                effort_all = len(learner.env.labeled)
                n_docs = len(learner.env.dataset)
                self.test_result = bool(
                    hypergeom.cdf(
                        pos_count_all - pos_count_phase_0,  # k
                        n_docs - effort_phase_0,  # N
                        int(
                            pos_count_all / self.target_recall - pos_count_phase_0
                        ),  # K_tar
                        effort_all - effort_phase_0,  # n
                    )
                    < self.alpha
                )

    @property
    def stop_criterion(self) -> bool:
        return self.phase == 1 and self.test_result


class CMH_HeuristicMethodRuleTwoPhase(
    CMH_HeuristicStoppingRule[KT, LT], Generic[KT, LT]
):
    phase: int

    def __init__(self, pos_label: LT, target_recall: float, alpha: float) -> None:
        super().__init__(pos_label, target_recall, alpha)
        self.phase = 0

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, CMHMethod):
            self.phase = learner.current_learner
            if self.phase == 0:
                super().update(learner.learners[0])
        else:
            self.stats.update(learner)
