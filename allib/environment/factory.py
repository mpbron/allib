from .catalog import EnvironmentCatalog as ENV
from .memory import MemoryEnvironment

from ..factory import AbstractBuilder, ObjectFactory
from allib import Component


class EnvironmentBuilder(AbstractBuilder):
    def __call__(self, environment_type: ENV.Type, **kwargs):
        return self._factory.create(environment_type, **kwargs)


class EnvironmentFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.register_builder(Component.ENVIRONMENT, EnvironmentBuilder())
        self.register_constructor(ENV.Type.MEMORY, MemoryEnvironment)