from typing import Callable, Optional, Sequence, Any
from typing_extensions import Self

from ..stopcriterion.base import AbstractStopCriterion
from .learnersequence import LearnerSequence

from ..typehints.typevars import IT, KT, DT, VT, RT, LT
from .base import ActiveLearner
from .random import RandomSampling
from ..stopcriterion.heuristic import LabelCount
from ..environment.base import AbstractEnvironment
from .estimator import Estimator
import random


def add_doc(
    learner: ActiveLearner[IT, KT, Any, Any, Any, LT], identifier: KT, *label: LT
):
    doc = learner.env.dataset[identifier]
    learner.env.labels.set_labels(doc, *label)
    learner.set_as_labeled(doc)


class AutoTARFirstMethod(LearnerSequence[IT, KT, DT, VT, RT, LT]):
    _name = "AutoTARFirst"

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        learners: Sequence[ActiveLearner[IT, KT, DT, VT, RT, LT],],
        stopcriteria: Sequence[AbstractStopCriterion[LT]],
        pos_label: LT,
        neg_label: LT,
        *_,
        identifier: Optional[str] = None,
        **__
    ) -> None:
        super().__init__(env, learners, stopcriteria, *_, identifier=identifier, **__)
        self.pos_label = pos_label
        self.neg_label = neg_label

    def _init_estimator(self) -> None:
        l0 = self.learners[0]
        l1 = self.learners[1]
        l2 = self.learners[2]
        assert isinstance(l2, Estimator)
        pos = list(l0.env.get_subset_by_labels(l0.env.labeled, self.pos_label))
        neg = list(l1.env.get_subset_by_labels(l1.env.labeled, self.neg_label))
        pos_sample = random.sample(pos, k=len(l2.learners))
        neg_sample = random.sample(neg, k=len(l2.learners))
        for sl, pos_doc, neg_doc in zip(l2.learners, pos_sample, neg_sample):
            add_doc(sl, pos_doc, self.pos_label)
            add_doc(sl, neg_doc, self.neg_label)

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        learner = super()._choose_learner()
        if self.current_learner == 2 and not learner.env.labeled:
            self._init_estimator()
        return learner

    @classmethod
    def builder(
        cls,
        tar_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]],
        ens_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]],
        tar_stop: Callable[[LT, LT], AbstractStopCriterion[LT]],
        nirel: int,
        *_: Any,
        **__: Any
    ) -> Callable[..., Self]:
        learner_builders = [
            tar_builder,
            RandomSampling.builder(identifier="PreRandom"),
            ens_builder,
        ]

        def wrap_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *args,
            **kwargs
        ):
            stop_criteria = [
                tar_stop(pos_label, neg_label),
                LabelCount(neg_label, nirel),
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
            return cls(env, learners, stop_criteria, pos_label, neg_label)

        return wrap_func
