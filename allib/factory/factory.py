from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")


class ObjectFactory:
    def __init__(self):
        self.builders = {}

    def register_constructor(self, key: str, constructor: Any) -> None:
        builder = ObjectBuilder(constructor)
        self.register_builder(key, builder)

    def register_builder(self, key: str, builder: AbstractBuilder):
        builder.register_factory(self)
        self.builders[key] = builder
        

    def create(self, key: str, **kwargs):
        builder = self.builders.get(key)
        if not builder:
            raise NotImplementedError(
                "The module '{}' is not registered".format(key))
        return builder(**kwargs)

    def attach(self, factory: ObjectFactory):
        for key, builder in factory.builders.items():
            self.register_builder(key, builder)


class AbstractBuilder(ABC):
    def __init__(self):
        self._factory = None

    def register_factory(self, factory: ObjectFactory):
        self._factory = factory

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError


class ObjectBuilder(AbstractBuilder, Generic[_T]):
    def __init__(self, constructor: _T) -> None:
        super().__init__()
        self.constructor = constructor

    def __call__(self, **kwargs) -> _T:
        return self.constructor(**kwargs)
