from enum import Enum
from typing import Dict

from ..factory import AbstractBuilder, ObjectFactory
from ..module.component import Component

from .base import BaseBalancer, IdentityBalancer
from .catalog import BalancerCatalog as BL
from .double import DoubleBalancer
from .randomoversampling import RandomOverSampler
from .undersample import UndersampleBalancer


class BalancerBuilder(AbstractBuilder):
    def __call__(self, # type: ignore
            type: BL.Type, 
            config: Dict, **kwargs) -> BaseBalancer:
        return self._factory.create(type, **config)

class BalancerFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.register_builder(
            Component.BALANCER, BalancerBuilder())
        self.register_constructor(
            BL.Type.IDENTITY, IdentityBalancer)
        self.register_constructor(
            BL.Type.UNDERSAMPLING, UndersampleBalancer)
        self.register_constructor(
            BL.Type.RANDOM_OVER_SAMPLING, RandomOverSampler)
        self.register_constructor(
            BL.Type.DOUBLE, DoubleBalancer)
