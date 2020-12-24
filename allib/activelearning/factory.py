import functools
from abc import ABC
from typing import Dict, List, Optional, TypeVar

from ..factory import AbstractBuilder, ObjectFactory
from ..machinelearning import AbstractClassifier, MachineLearningFactory
from ..module.component import Component

from .base import ActiveLearner
from .catalog import ALCatalog as AL
from .ensembles import StrategyEnsemble
from .estimator import CycleEstimator, Estimator, NewEstimator
from .labelmethods import LabelProbabilityBased
from .ml_based import AbstractSelectionCriterion, ProbabilityBased
from .mostcertain import LabelMaximizer, MostCertainSampling
from .random import RandomSampling
from .uncertainty import (EntropySampling, LabelUncertainty, LeastConfidence, MarginSampling,
                          NearDecisionBoundary)

LT = TypeVar("LT")
class FallbackBuilder(AbstractBuilder):
    def __call__(self, **kwargs) -> ActiveLearner:
        if kwargs:
            fallback = self._factory.create(Component.ACTIVELEARNER, **kwargs)
            return fallback
        return RandomSampling()
class ALBuilder(AbstractBuilder):
    def __call__(self, paradigm, **kwargs):
        return self._factory.create(paradigm, **kwargs)

class ProbabilityBasedBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            query_type: AL.QueryType,
            machinelearning: Dict,
            fallback: Dict = dict(),
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        selection_criterion: AbstractSelectionCriterion = self._factory.create(query_type, **kwargs)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return ProbabilityBased(classifier, selection_criterion, built_fallback)

class LabelProbabilityBasedBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            query_type: AL.QueryType,
            machinelearning: Dict,
            label: LT,
            fallback: Dict = dict(),
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        selection_criterion: AbstractSelectionCriterion = self._factory.create(query_type, **kwargs)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelProbabilityBased(classifier, selection_criterion, label, built_fallback)


class PoolbasedBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            query_type: AL.QueryType,
            machinelearning: Dict,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        return self._factory.create(query_type,
            classifier=classifier,
            **kwargs)
class StrategyEnsembleBuilder(AbstractBuilder):
    def build_learner(self, classifier: AbstractClassifier, config):
        query_type = config["query_type"]
        params = {k: v for k,v in config if k not in ["query_type"]}
        return self._factory.create(query_type, classifier=classifier, **params)

    def __call__( # type: ignore
            self,
            learners: List[Dict],
            machinelearning: Dict, 
            probabilities: List[float],**kwargs):
        assert len(learners) == len(probabilities)
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        config_function = functools.partial(self.build_learner, classifier)
        configured_learners = list(map(config_function, learners))
        return StrategyEnsemble(classifier, configured_learners, probabilities)
class EstimatorBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self, learners: List[Dict], 
            machinelearning: Dict, **kwargs) -> Estimator:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        return Estimator(classifier, configured_learners)

class CycleEstimatorBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self, learners: List[Dict], 
            machinelearning: Dict, **kwargs) -> Estimator:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        return CycleEstimator(classifier, configured_learners)

class NewEstimatorBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self, learners: List[Dict], **kwargs) -> NewEstimator:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        return NewEstimator(configured_learners)

class ActiveLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(MachineLearningFactory())
        self.register_builder(Component.ACTIVELEARNER, ALBuilder())
        self.register_builder(Component.FALLBACK, FallbackBuilder())
        self.register_builder(AL.Paradigm.POOLBASED, PoolbasedBuilder())
        self.register_builder(AL.Paradigm.PROBABILITY_BASED, ProbabilityBasedBuilder())
        self.register_builder(AL.Paradigm.ESTIMATOR, EstimatorBuilder())
        self.register_builder(AL.Paradigm.CYCLE, CycleEstimatorBuilder())
        self.register_builder(AL.Paradigm.ENSEMBLE, StrategyEnsembleBuilder())
        self.register_builder(AL.Paradigm.NEWESTIMATOR, NewEstimatorBuilder())
        self.register_builder(AL.Paradigm.LABEL_PROBABILITY_BASED, LabelProbabilityBasedBuilder())
        self.register_constructor(AL.QueryType.RANDOM_SAMPLING, RandomSampling)
        self.register_constructor(AL.QueryType.LEAST_CONFIDENCE, LeastConfidence)
        self.register_constructor(AL.QueryType.MAX_ENTROPY, EntropySampling)
        self.register_constructor(AL.QueryType.MARGIN_SAMPLING, MarginSampling)
        self.register_constructor(AL.QueryType.NEAR_DECISION_BOUNDARY, NearDecisionBoundary)
        self.register_constructor(AL.QueryType.LABELMAXIMIZER, LabelMaximizer)
        self.register_constructor(AL.QueryType.LABELUNCERTAINTY, LabelUncertainty)
        self.register_constructor(AL.QueryType.MOST_CERTAIN, MostCertainSampling)
