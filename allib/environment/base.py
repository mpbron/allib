from __future__ import annotations

from typing import Generic, Sequence, TypeVar, Any
from abc import ABC, abstractmethod, abstractclassmethod
from ..history import BaseLogger
from ..instances import InstanceProvider
from ..labels import LabelProvider

KT = TypeVar("KT")
LT = TypeVar("LT")
VT = TypeVar("VT")
DT = TypeVar("DT")
RT = TypeVar("RT")

class AbstractEnvironment(ABC, Generic[KT, DT, VT, RT, LT]):
    @abstractmethod
    def create_empty_provider(self) -> InstanceProvider[KT, DT, VT, RT]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset(self) -> InstanceProvider[KT, DT, VT, RT]:
        raise NotImplementedError

    @property
    @abstractmethod
    def unlabeled(self) -> InstanceProvider[KT, DT, VT, RT]:
        raise NotImplementedError

    @property
    @abstractmethod
    def labeled(self) -> InstanceProvider[KT, DT, VT, RT]:
        raise NotImplementedError

    @property
    @abstractmethod
    def logger(self) -> BaseLogger[KT, LT, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> LabelProvider[KT, LT]:
        raise NotImplementedError

    @property
    @abstractmethod
    def truth(self) -> LabelProvider[KT, LT]:
        raise NotImplementedError

    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def storage_exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, key: str) -> Any:
        raise NotImplementedError

    def add_vectors(self, keys: Sequence[KT], vectors: Sequence[VT]) -> None:
        self.dataset.bulk_add_vectors(keys, vectors)

    @abstractclassmethod
    def from_environment(cls, environment: AbstractEnvironment[KT, DT, VT, RT, LT], *args, **kwargs) -> AbstractEnvironment[KT, DT, VT, RT, LT]:
        raise NotImplementedError

