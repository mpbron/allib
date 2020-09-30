import random
from typing import Generic, Iterator, List, Optional, TypeVar

import numpy as np # type: ignore

from ..instances import Instance

from .base import ActiveLearner
from .poolbased import PoolbasedAL

KT = TypeVar("KT")

class RandomSampling(PoolbasedAL, Generic[KT]):
    _name = "Random"

    def calculate_ordering(self) -> List[KT]:
        return np.random.permutation(self._unlabeled.key_list).tolist()
