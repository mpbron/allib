from enum import Enum
from typing import Dict
from factory.component import Component
from factory.factory import AbstractBuilder, ObjectFactory

from .base import IdentityBalancer, BaseBalancer
from .double import DoubleBalancer
from .randomoversampling import RandomOverSampler
from .undersample import UndersampleBalancer

class BalancerType(Enum):
    IDENTITY = "Identity"
    RANDOM_OVER_SAMPLING = "RandomOverSampling"
    UNDERSAMPLING = "UnderSampling"
    DOUBLE = "DoubleBalancer"

class BalancerBuilder(AbstractBuilder):
    def __call__(self, 
            balancer_type: BalancerType, 
            balancer_config: Dict, **kwargs) -> BaseBalancer:
        return self._factory.create(balancer_type, **balancer_config)

class BalancerFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.register_builder(
            Component.BALANCER, BalancerBuilder())
        self.register_constructor(
            BalancerType.IDENTITY, IdentityBalancer)
        self.register_constructor(
            BalancerType.UNDERSAMPLING, UndersampleBalancer)
        self.register_constructor(
            BalancerType.RANDOM_OVER_SAMPLING, RandomOverSampler)
        self.register_constructor(
            BalancerType.DOUBLE, DoubleBalancer)
