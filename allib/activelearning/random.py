import collections
from typing import Generic, TypeVar, Sequence, Tuple

import random

from .poolbased import PoolBasedAL

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
class RandomSampling(PoolBasedAL[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    _name = "Random"

    def update_ordering(self) -> None:
        keys = list(self.env.unlabeled.key_list)
        random.shuffle(list(keys))
        self.ordering = collections.deque(keys)
