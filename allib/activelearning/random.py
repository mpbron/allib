import random
from typing import Generic, Sequence, Tuple, TypeVar

from .poolbased import PoolBasedAL

from ..typehints import KT, DT, VT, RT, LT, IT
class RandomSampling(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    _name = "Random"

    def update_ordering(self) -> bool:
        keys = list(self.env.unlabeled.key_list)
        random.shuffle(keys)
        self._set_ordering(keys)
        return True