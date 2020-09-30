from __future__ import annotations

from typing import Generic, TypeVar
from abc import ABC, abstractmethod, abstractclassmethod
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
    def labels(self) -> LabelProvider[KT, LT]:
        raise NotImplementedError

    @abstractclassmethod
    def from_environment(cls, provider: AbstractEnvironment[KT, DT, VT, RT, LT]) -> AbstractEnvironment[KT, DT, VT, RT, LT]:
        raise NotImplementedError
