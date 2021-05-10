from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Optional, Callable

_T = TypeVar("_T")


class ObjectFactory:
    def __init__(self):
        self.builders = {}

    def get_constructor(self, key: Any) -> Any:
        builder = self.builders[key]
        if isinstance(builder, ObjectBuilder):
            return builder.constructor            
        raise KeyError(
            f"The requested key '{key}' has no retrievable constructor")

    def register_constructor(self, key: Any, constructor: Any) -> None:
        builder = ObjectBuilder[Any](constructor)
        self.register_builder(key, builder)

    def register_builder(self, key: Any, builder: AbstractBuilder):
        builder.register_factory(self)
        self.builders[key] = builder

    def create(self, key: Any, **kwargs):
        builder = self.builders.get(key)
        if not builder:
            raise NotImplementedError(
                "The module '{}' is not registered".format(key))
        return builder(**kwargs)

    def attach(self, factory: ObjectFactory):
        for key, builder in factory.builders.items():
            self.register_builder(key, builder)


class AbstractBuilder(ABC):
    _factory: ObjectFactory
    def __init__(self):
        self._factory = ObjectFactory()

    def register_factory(self, factory: ObjectFactory):
        self._factory = factory

    @abstractmethod
    def __call__(self, **kwargs): # type: ignore
        raise NotImplementedError


class ObjectBuilder(AbstractBuilder, Generic[_T]):
    def __init__(self, constructor: Callable[..., _T]) -> None:
        super().__init__()
        self.constructor = constructor

    def __call__(self, **kwargs) -> _T:
        return self.constructor(**kwargs)
