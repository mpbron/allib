from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar

from instancelib import Instance
from typing_extensions import Self

from ..environment.base import AbstractEnvironment
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints.typevars import DT, IT, KT, LT, RT, VT
from .base import ActiveLearner
from .ensembles import AbstractEnsemble
from .poolbased import PoolBasedAL

LOGGER = logging.getLogger(__name__)


class LearnerSequence(
    AbstractEnsemble[IT, KT, DT, VT, RT, LT],
    PoolBasedAL[IT, KT, DT, VT, RT, LT],
    Generic[IT, KT, DT, VT, RT, LT],
):
    _name = "LearnerSequence"
    stopcriteria: Sequence[AbstractStopCriterion]
    _sample_dict: Dict[KT, int]
    current_learner: int
    stop_interval: int

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        learners: Sequence[ActiveLearner[IT, KT, DT, VT, RT, LT],],
        stopcriteria: Sequence[AbstractStopCriterion[LT]],
        *_,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        super().__init__(env, identifier=identifier)
        self.learners = learners
        self.stopcriteria = stopcriteria
        self._sample_dict = dict()
        self.current_learner: int = 0
        self.stop_interval: int = 1
        self.it = 0

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        """Internal functions that selects the next active learner for the next query

        Returns
        -------
        ActiveLearner[IT, KT, DT, VT, RT, LT]
            One of the learners from the ensemble
        """
        learner = self.learners[self.current_learner]
        if self.stop_interval % self.stop_interval == 0 and self.current_learner < len(
            self.stopcriteria
        ):
            self.stopcriteria[self.current_learner].update(learner)

        if (
            self.current_learner < len(self.learners) - 1
            and self.stopcriteria[self.current_learner].stop_criterion
        ):
            self.current_learner += 1
            return self._choose_learner()
        return learner

    def __next__(self) -> IT:
        # Select the learner
        learner = self._choose_learner()

        # Select the next instance from the learner
        ins = next(learner)

        # Check if the instance identifier has not been labeled already
        while ins.identifier in self.env.labeled:
            # This instance has already been labeled my another learner.
            # Skip it and mark as labeled
            learner.set_as_labeled(ins)
            learner.env.labels.set_labels(
                ins.identifier, *self.env.labels[ins.identifier]
            )
            LOGGER.info(
                "The document with key %s was already labeled. Skipping", ins.identifier
            )
            learner = self._choose_learner()
            ins = next(learner)

        # Set the instances as sampled by learner with key al_idx and return the instance
        self._sample_dict[ins.identifier] = self.learners.index(learner)

        self.it += 1
        return ins

    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.labeled.add(instance)
        self.env.unlabeled.discard(instance)
        if instance.identifier in self._sample_dict:
            learner = self.learners[self._sample_dict[instance.identifier]]
            learner.set_as_labeled(instance)
            learner.env.labels.set_labels(instance, *self.env.labels[instance])

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.unlabeled.add(instance)
        self.env.labeled.discard(instance)
        if instance.identifier in self._sample_dict:
            learner = self.learners[self._sample_dict[instance.identifier]]
            learner.set_as_unlabeled(instance)
            del self._sample_dict[instance.identifier]

    @classmethod
    def builder(
        cls,
        learner_builders: Sequence[
            Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]]
        ],
        stop_criteria: Sequence[AbstractStopCriterion],
        *_: Any,
        **__: Any,
    ) -> Callable[..., Self]:
        def wrap_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT], *args, **kwargs
        ):
            learners = [
                builder(env.from_environment(env), *args, **kwargs)
                for builder in learner_builders
            ]
            return cls(env, learners, stop_criteria)

        return wrap_func
