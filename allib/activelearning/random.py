import random
from typing import Generic, Sequence, Tuple, TypeVar

from .poolbased import PoolBasedAL

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
class RandomSampling(PoolBasedAL[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    _name = "Random"

    def update_ordering(self) -> bool:
        keys = list(self.env.unlabeled.key_list)
        random.shuffle(keys)
        self._set_ordering(keys)
        return True