from typing import Callable, Sequence, Any
from typing_extensions import Self

from ..stopcriterion.base import AbstractStopCriterion
from .learnersequence import LearnerSequence

from ..typehints.typevars import IT, KT, DT, VT, RT, LT
from .base import ActiveLearner
from .random import RandomSampling
from ..stopcriterion.others import CMH_HeuristicStoppingRule
from ..environment.base import AbstractEnvironment
from .autotarensemble import add_doc


def transfer(
    a: ActiveLearner[IT, KT, DT, VT, RT, LT], b: ActiveLearner[IT, KT, DT, VT, RT, LT]
) -> None:
    for ins in a.env.labeled:
        add_doc(b, ins, *a.env.labels[ins])


class CMHMethod(LearnerSequence[IT, KT, DT, VT, RT, LT]):
    _name = "CMH_Hybrid"

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        learner = super()._choose_learner()
        if self.current_learner == 1 and not learner.env.labeled:
            transfer(self.learners[0], self.learners[1])
        return learner

    @classmethod
    def builder(
        cls,
        learner_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]],
        target_recall: float = 0.95,
        alpha: float = 0.05,
        *_: Any,
        **__: Any
    ) -> Callable[..., Self]:
        learner_builders = [learner_builder, RandomSampling.builder()]

        def wrap_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *args,
            **kwargs
        ):
            stop_criteria = [
                CMH_HeuristicStoppingRule(
                    pos_label, target_recall, 0.5
                )  ## See figure 3 CMH
            ]
            learners = [
                builder(
                    env.from_environment(env),
                    pos_label=pos_label,
                    neg_label=neg_label,
                    *args,
                    **kwargs
                )
                for builder in learner_builders
            ]
            return cls(env, learners, stop_criteria)

        return wrap_func
