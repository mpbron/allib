from typing import Generic, TypeVar, Sequence, Tuple

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

    def calculate_ordering(self) -> Tuple[Sequence[KT], Sequence[float]]:
        keys = self.env.unlabeled.key_list
        random_floats = [0.0] * len(keys)
        shuffled_keys = np.random.permutation(keys).tolist()
        return shuffled_keys, random_floats
