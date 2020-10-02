from typing import Generic, TypeVar, Sequence

import numpy as np # type: ignore

from .poolbased import PoolbasedAL

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")

class RandomSampling(PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT], Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = "Random"

    def calculate_ordering(self) -> Sequence[KT]:
        return np.random.permutation(self._unlabeled.key_list).tolist()
