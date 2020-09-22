from factory.component import Component
from factory.factory import AbstractBuilder, ObjectFactory

from .catalog import EnvironmentType

from .django import DjangoEnvironment
from .cter import CterEnvironment
from .elastic import ElasticEnvironment
from .memory import MemoryEnvironment




class EnvironmentBuilder(AbstractBuilder):
    def __call__(self, environment_type: EnvironmentType, **kwargs):
        return self._factory.create(environment_type, **kwargs)


class EnvironmentFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.register_builder(Component.ENVIRONMENT, EnvironmentBuilder())
        self.register_constructor(EnvironmentType.MEMORY, MemoryEnvironment)
        self.register_constructor(EnvironmentType.DJANGO, DjangoEnvironment)
        self.register_constructor(EnvironmentType.ELASTIC, ElasticEnvironment)
        self.register_constructor(EnvironmentType.ELASTIC_CHAT, CterEnvironment)
