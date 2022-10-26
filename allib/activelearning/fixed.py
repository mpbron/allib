from typing import Dict, Generic, Optional, Sequence, Tuple

from allib.environment.base import AbstractEnvironment

from ..typehints import DT, IT, KT, LT, RT, VT
from .random import PoolBasedAL


class FixedOrdering(
    PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):

    _name = "FixedOrdering"

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        *_,
        identifier: Optional[str] = None,
        label: Optional[LT] = None,
        **__,
    ) -> None:
        super().__init__(env, identifier=identifier)
        self.metrics: Dict[KT, float] = dict()
        self.label = label

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        if self.identifier is not None:
            return f"{self.identifier}", self.label
        return f"{self._name}", self.label

    def enter_ordering(
        self, ordering: Sequence[KT], metrics: Optional[Sequence[float]] = None
    ):
        self._set_ordering(ordering)
        if metrics is not None:
            self.metrics = {o: m for (o, m) in zip(ordering, metrics)}
