from typing import Dict, Generic, Optional, Sequence
from .random import PoolBasedAL

from ..typehints import KT, DT, VT, RT, LT

class FixedOrdering(PoolBasedAL[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self, *_, identifier: Optional[str] = None, **__) -> None:
        super().__init__(identifier)
        self.metrics: Dict[KT, float] = dict()

    def enter_ordering(self, ordering: Sequence[KT], metrics: Optional[Sequence[float]] = None):
        self._set_ordering(ordering)
        if metrics is not None:
            self.metrics = {o: m for (o,m) in zip(ordering, metrics)}