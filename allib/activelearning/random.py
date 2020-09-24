import random
from typing import KT, Generic, Iterator, List, Optional, TypeVar

import numpy as np

from ..instances import Instance

from .base import ActiveLearner
from .poolbased import PoolbasedAL

KT = TypeVar("KT")


class RandomSampling(PoolbasedAL, Generic[KT]):
    _name = "Random"

    @ActiveLearner.query_log
    def query(self) -> Optional[Instance]:
        """Return a random document from the unlabeled set

        Returns
        -------
        Optional[Instance]

        """
        if not self._unlabeled.empty:
            key = random.choice(self._unlabeled.key_list)
            ins = self._unlabeled[key]
            return ins
        return None

    @ActiveLearner.query_batch_log
    def query_batch(self, batch_size: int) -> List[Instance]:
        if not self._unlabeled.empty:
            keys = random.sample(self._unlabeled.key_list, batch_size)
            return [self._unlabeled[key] for key in keys]
        return []

    def calculate_ordering(self) -> List[KT]:
        return np.random.permutation(self._unlabeled.key_list).tolist()
