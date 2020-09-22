from abc import ABC

from factory.component import Component
from factory.factory import AbstractBuilder, ObjectFactory
from machinelearning import AbstractClassifier
from machinelearning import MachineLearningFactory

from .catalog import QueryType, ALParadigm
from .random import RandomSampling
from .uncertainty import MarginSampling, NearDecisionBoundary, EntropySampling, LeastConfidence
from .interleave import InterleaveAL


class ALBuilder(AbstractBuilder):
    def __call__(self, al_paradigm, **kwargs):
        return self._factory.create(al_paradigm, **kwargs)

class PoolbasedBuilder(AbstractBuilder):
    def __call__(
            self,
            query_type: QueryType,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **kwargs)
        return self._factory.create(query_type,
            classifier=classifier,
            **kwargs)

class ActiveLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(MachineLearningFactory())
        self.register_builder(Component.ACTIVELEARNER, ALBuilder())
        self.register_builder(ALParadigm.POOLBASED, PoolbasedBuilder())
        self.register_constructor(QueryType.RANDOM_SAMPLING, RandomSampling)
        self.register_constructor(QueryType.LEAST_CONFIDENCE, LeastConfidence)
        self.register_constructor(QueryType.MAX_ENTROPY, EntropySampling)
        self.register_constructor(QueryType.MARGIN_SAMPLING, MarginSampling)
        self.register_constructor(QueryType.NEAR_DECISION_BOUNDARY, NearDecisionBoundary)
        self.register_constructor(QueryType.INTERLEAVE, InterleaveAL)
