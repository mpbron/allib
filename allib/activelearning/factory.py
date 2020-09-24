from abc import ABC
from typing import Dict

from allib import Component
from ..factory import AbstractBuilder, ObjectFactory
from ..machinelearning import AbstractClassifier
from ..machinelearning import MachineLearningFactory

from .catalog import ALCatalog as AL
from .random import RandomSampling
from .uncertainty import MarginSampling, NearDecisionBoundary, EntropySampling, LeastConfidence
from .interleave import InterleaveAL
from .mostcertain import LabelMaximizer, MostCertainSampling


class ALBuilder(AbstractBuilder):
    def __call__(self, paradigm, **kwargs):
        return self._factory.create(paradigm, **kwargs)

class PoolbasedBuilder(AbstractBuilder):
    def __call__(
            self,
            query_type: AL.QueryType,
            machinelearning: Dict,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        return self._factory.create(query_type,
            classifier=classifier,
            **kwargs)

class ActiveLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(MachineLearningFactory())
        self.register_builder(Component.ACTIVELEARNER, ALBuilder())
        self.register_builder(AL.Paradigm.POOLBASED, PoolbasedBuilder())
        self.register_constructor(AL.QueryType.RANDOM_SAMPLING, RandomSampling)
        self.register_constructor(AL.QueryType.LEAST_CONFIDENCE, LeastConfidence)
        self.register_constructor(AL.QueryType.MAX_ENTROPY, EntropySampling)
        self.register_constructor(AL.QueryType.MARGIN_SAMPLING, MarginSampling)
        self.register_constructor(AL.QueryType.NEAR_DECISION_BOUNDARY, NearDecisionBoundary)
        self.register_constructor(AL.QueryType.INTERLEAVE, InterleaveAL)
        self.register_constructor(AL.QueryType.LABELMAXIMIZER, LabelMaximizer)
        self.register_constructor(AL.QueryType.MOST_CERTAIN, MostCertainSampling)
