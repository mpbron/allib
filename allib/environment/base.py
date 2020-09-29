from __future__ import annotations

from abc import ABC, abstractmethod, abstractclassmethod
from ..instances import InstanceProvider
from ..labels import LabelProvider


class AbstractEnvironment(ABC):
    @abstractmethod
    def create_empty_provider(self) -> InstanceProvider:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_provider(self) -> InstanceProvider:
        raise NotImplementedError

    @property
    @abstractmethod
    def unlabeled_provider(self) -> InstanceProvider:
        raise NotImplementedError

    @property
    @abstractmethod
    def labeled_provider(self) -> InstanceProvider:
        raise NotImplementedError

    @property
    @abstractmethod
    def label_provider(self) -> LabelProvider:
        raise NotImplementedError

    @abstractclassmethod
    def from_environment(cls, provider: AbstractEnvironment) -> AbstractEnvironment:
        raise NotImplementedError
