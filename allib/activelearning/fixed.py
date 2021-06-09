from typing import Dict, Generic, Optional, Sequence, Tuple
from .random import PoolBasedAL

from ..typehints import KT, DT, VT, RT, LT, IT

class FixedOrdering(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):

    _name = "FixedOrdering"

    
    def __init__(self, *_, identifier: Optional[str] = None, 
                           label: Optional[LT] = None, **__) -> None:
        super().__init__(identifier=identifier)
        self.metrics: Dict[KT, float] = dict()
        self.label = label

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        if self.identifier is not None:
            return f"{self.identifier}", self.label
        return f"{self._name}", self.label


    def enter_ordering(self, ordering: Sequence[KT], metrics: Optional[Sequence[float]] = None):
        self._set_ordering(ordering)
        if metrics is not None:
            self.metrics = {o: m for (o,m) in zip(ordering, metrics)}