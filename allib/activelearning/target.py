from typing import Callable, Sequence, Any
from typing_extensions import Self

from ..stopcriterion.base import AbstractStopCriterion
from .learnersequence import LearnerSequence

from ..typehints.typevars import IT, KT, DT, VT, RT, LT
from .base import ActiveLearner
from .random import RandomSampling
from ..stopcriterion.heuristic import LabelCount
from ..environment.base import AbstractEnvironment


class TargetMethod(LearnerSequence[IT, KT, DT, VT, RT, LT]):
    _name = "TargetMethod"

    @classmethod
    def builder(
        cls,
        learner_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]],
        nrel: int,
        *_: Any,
        **__: Any
    ) -> Callable[..., Self]:
        learner_builders = [RandomSampling.builder(), learner_builder]

        def wrap_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *args,
            **kwargs
        ):
            stop_criteria = [LabelCount(pos_label, nrel)]
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
