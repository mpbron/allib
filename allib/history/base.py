from typing import Generic, TypeVar, Any, FrozenSet, Union

from abc import ABC, abstractmethod
from ..instances import Instance

KT = TypeVar("KT")
LT = TypeVar("LT")
MT = TypeVar("MT")
ST = TypeVar("ST")


class BaseLogger(ABC, Generic[KT, LT, ST]):
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
