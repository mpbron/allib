from typing import (Any, Callable, FrozenSet, Generic, Mapping, Optional,
                    Sequence, Tuple)

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


def divide_dataset(env: AbstractEnvironment[IT, KT, Any, Any, Any, Any], size: int = 2000, rng: np.random.Generator = np.random.default_rng()) -> Sequence[Tuple[FrozenSet[KT], FrozenSet[KT]]]:
    keys = env.dataset.key_list
    rng.shuffle(keys) # type: ignore
    chunks = divide_iterable_in_lists(keys, size)
    return [(frozenset(unl), frozenset()) for unl in chunks]

class AutoStopLarge(LearnerSequence[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT , LT]):
       
        
    @classmethod
    def builder(
        cls,
        autostop_params: Mapping[str, Any],
        estimator_builder: Callable[[], AbstractEstimator],
        stopcriterion_builder: Callable[[AbstractEstimator, float],Callable[[LT, LT], AbstractStopCriterion]],
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
            stopcriteria = [stopcriterion_builder(estimator_builder(), target)(pos_label, neg_label) for _ in envs]
            learners = [
                AutoStopLearner.builder(**autostop_params)(part_env, pos_label, neg_label)
                for part_env in envs
            ]
            
            return cls(env, learners, stopcriteria)

        return builder_func
    
    @classmethod
    def build_conservative(cls, threshold=0.95) -> Callable[..., Self]:
        return cls.builder({}, HorvitzThompsonVar2, Conservative.builder, threshold, 2000)