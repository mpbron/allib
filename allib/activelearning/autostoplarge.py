from typing import (
    Any,
    Callable,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import instancelib as il
import numpy as np
import numpy.typing as npt
from instancelib.typehints import DT, KT, LT, RT, VT
from typing_extensions import Self

from ..activelearning.random import RandomSampling
from ..stopcriterion.sequence import DummyFalse

from ..activelearning.autostop import AutoStopLearner
from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..estimation.base import AbstractEstimator
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints import IT
from .base import ActiveLearner
from .learnersequence import LearnerSequence
from typing import TypeVar, Iterator
from math import floor

_T = TypeVar("_T")


def divide_iterable_in_lists(
    iterable: Iterable[_T], batch_size: int
) -> Iterator[Sequence[_T]]:
    return map(list, divide_iterable(iterable, batch_size))  # type: ignore


def divide_dataset(
    env: AbstractEnvironment[IT, KT, Any, Any, Any, Any],
    size: int = 3000,
    rng: np.random.Generator = np.random.default_rng(),
) -> Sequence[Tuple[FrozenSet[KT], FrozenSet[KT]]]:
    keys = env.dataset.key_list
    rng.shuffle(keys)  # type: ignore
    lk = len(keys)
    n = floor(lk / size)
    k, m = divmod(lk, n)
    return [
        (frozenset(keys[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]), frozenset())
        for i in range(n)
    ]


class AutoStopLarge(
    LearnerSequence[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    _name = "AUTOSTOP_LARGE"

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
            stopcriteria.append(DummyFalse())
            random_sampling = RandomSampling.builder()(
                env.from_environment(env, shared_labels=False)
            )
            learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]] = [
                AutoStopLearner.builder(classifier_builder, k_sample, batch_size)(
                    part_env, pos_label, neg_label
                )
                for part_env in envs
            ]
            learners.append(random_sampling)
            return cls(env, learners, stopcriteria)

        return builder_func
