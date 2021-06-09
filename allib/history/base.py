from abc import ABC, abstractmethod
from typing import (Any, FrozenSet, Generic, Iterable, Sequence,
                    TypeVar, Union)

from instancelib.instances import Instance
from instancelib.labels.base import LabelProvider
from instancelib.typehints import KT, LT

ST = TypeVar("ST")
class BaseLogger(ABC, Generic[KT, LT, ST]):
    @abstractmethod
    def log_ordering(self, ordering: Sequence[KT], metrics: Sequence[float], labeled: Iterable[KT], labels: LabelProvider[KT, LT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_sample(self, x: Union[KT, Instance[KT, Any, Any, Any]], sample_method: ST) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_label(self, x: Union[KT, Instance[KT, Any, Any, Any]], sample_method: ST,  *labels: LT) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_sampled_info(self, x: Union[KT, Instance[KT, Any, Any, Any]]) -> FrozenSet[ST]:
        raise NotImplementedError

    @abstractmethod
    def get_instances_by_method(self, sample_method: ST) -> FrozenSet[KT]:
        raise NotImplementedError

    @abstractmethod
    def get_label_order(self, x: Union[KT, Instance[KT, Any, Any, Any]]) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def labelset(self) -> FrozenSet[LT]: 
        raise NotImplementedError
