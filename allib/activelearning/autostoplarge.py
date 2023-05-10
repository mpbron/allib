from typing import Any, Callable, FrozenSet, Generic, Mapping, Optional, Sequence, Tuple

import numpy as np
from instancelib.typehints import DT, KT, LT, RT, VT
from instancelib.utils.chunks import divide_iterable_in_lists
from typing_extensions import Self

from ..activelearning.autostop import AutoStopLearner

from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..estimation.autostop import HorvitzThompsonVar2
from ..estimation.base import AbstractEstimator
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints import IT
from .base import ActiveLearner
from .learnersequence import LearnerSequence
from ..stopcriterion.estimation import Conservative
import instancelib as il
import numpy.typing as npt


def divide_dataset(
    env: AbstractEnvironment[IT, KT, Any, Any, Any, Any],
    size: int = 2000,
    rng: np.random.Generator = np.random.default_rng(),
) -> Sequence[Tuple[FrozenSet[KT], FrozenSet[KT]]]:
    keys = env.dataset.key_list
    rng.shuffle(keys)  # type: ignore
    chunks = divide_iterable_in_lists(keys, size)
    return [(frozenset(unl), frozenset()) for unl in chunks]


class AutoStopLarge(
    LearnerSequence[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        """Internal functions that selects the next active learner for the next query

        Returns
        -------
        ActiveLearner[IT, KT, DT, VT, RT, LT]
            One of the learners from the ensemble
        """
        learner = self.learners[self.current_learner]
        if self.stop_interval % self.stop_interval == 0:
            self.stopcriteria[self.current_learner].update(learner)

        if (
            self.current_learner < len(self.learners) - 1
            and self.stopcriteria[self.current_learner].stop_criterion
        ):
            self.current_learner += 1
            return self._choose_learner()
        return learner

    @classmethod
    def builder(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        k_sample: int,
        batch_size: int,
        estimator_builder: Callable[[], AbstractEstimator],
        stopcriterion_builder: Callable[
            [AbstractEstimator, float], Callable[[LT, LT], AbstractStopCriterion]
        ],
        target: float = 0.95,
        size: int = 2000,
        identifier: Optional[str] = None,
        **__: Any,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            assert isinstance(env, MemoryEnvironment)
            parts = divide_dataset(env, size)
            envs = MemoryEnvironment.divide_in_parts(env, parts)
            stopcriteria = [
                stopcriterion_builder(estimator_builder(), target)(pos_label, neg_label)
                for _ in envs
            ]
            learners = [
                AutoStopLearner.builder(classifier_builder, k_sample, batch_size)(
                    part_env, pos_label, neg_label
                )
                for part_env in envs
            ]

            return cls(env, learners, stopcriteria)

        return builder_func

    @classmethod
    def build_conservative(cls, threshold=0.95) -> Callable[..., Self]:
        return cls.builder(
            {}, HorvitzThompsonVar2, Conservative.builder, threshold, 2000
        )
