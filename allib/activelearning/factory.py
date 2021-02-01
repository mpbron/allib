import functools
from abc import ABC
from typing import Any, Dict, List, Optional, TypeVar

from ..factory import AbstractBuilder, ObjectFactory
from ..machinelearning import AbstractClassifier, MachineLearningFactory
from ..module.component import Component

from .base import ActiveLearner
from .catalog import ALCatalog as AL
from .ensembles import StrategyEnsemble
from .estimator import CycleEstimator, Estimator, RetryEstimator
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
            identifier: Optional[str] = None,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        selection_criterion: AbstractSelectionCriterion = self._factory.create(query_type, **kwargs)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return ProbabilityBased(classifier, 
                                selection_criterion, 
                                built_fallback,
                                identifier=identifier)

class LabelProbabilityBasedBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            query_type: AL.QueryType,
            machinelearning: Dict,
            label: LT,
            fallback: Dict = dict(),
            identifier: Optional[str] = None,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        selection_criterion: AbstractSelectionCriterion = self._factory.create(query_type, **kwargs)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelProbabilityBased(classifier, 
                                     selection_criterion, 
                                     label, 
                                     built_fallback,
                                     identifier=identifier)


class PoolbasedBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            query_type: AL.QueryType,
            machinelearning: Dict,
            identifier: Optional[str] = None,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        return self._factory.create(query_type,
            classifier=classifier,
            identifier=identifier,
            **kwargs)
class StrategyEnsembleBuilder(AbstractBuilder):
    def build_learner(self, 
                      classifier: AbstractClassifier, 
                      config):
        query_type = config["query_type"]
        params = {k: v for k, v in config if k not in ["query_type"]}
        return self._factory.create(query_type, classifier=classifier, **params)

    def __call__( # type: ignore
            self,
            learners: List[Dict],
            machinelearning: Dict, 
            probabilities: List[float],
            identifier: Optional[str] = None,
            **kwargs):
        assert len(learners) == len(probabilities)
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        config_function = functools.partial(self.build_learner, classifier)
        configured_learners = list(map(config_function, learners))
        return StrategyEnsemble(classifier, configured_learners, probabilities)
class CycleEstimatorBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self, learners: List[Dict],
            identifier: Optional[str] = None, 
            **kwargs) -> Estimator:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]       
        return CycleEstimator(configured_learners)

class EstimatorBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self, learners: List[Dict], 
            identifier: Optional[str] = None,
            **kwargs) -> Estimator:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        return Estimator(configured_learners)

class RetryEstimatorBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self, learners: List[Dict], 
            identifier: Optional[str] = None,
            **kwargs) -> RetryEstimator:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        return RetryEstimator(configured_learners)

class ActiveLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(MachineLearningFactory())
        self.register_builder(Component.ACTIVELEARNER, ALBuilder())
        self.register_builder(Component.FALLBACK, FallbackBuilder())
        self.register_builder(AL.Paradigm.POOLBASED, PoolbasedBuilder())
        self.register_builder(AL.Paradigm.PROBABILITY_BASED, ProbabilityBasedBuilder())
        self.register_builder(AL.Paradigm.ESTIMATOR, EstimatorBuilder())
        self.register_builder(AL.Paradigm.CYCLE_ESTIMATOR, CycleEstimatorBuilder())
        self.register_builder(AL.Paradigm.ENSEMBLE, StrategyEnsembleBuilder())
        self.register_builder(AL.Paradigm.LABEL_PROBABILITY_BASED, LabelProbabilityBasedBuilder())
        self.register_constructor(AL.QueryType.RANDOM_SAMPLING, RandomSampling)
        self.register_constructor(AL.QueryType.LEAST_CONFIDENCE, LeastConfidence)
        self.register_constructor(AL.QueryType.MAX_ENTROPY, EntropySampling)
        self.register_constructor(AL.QueryType.MARGIN_SAMPLING, MarginSampling)
        self.register_constructor(AL.QueryType.NEAR_DECISION_BOUNDARY, NearDecisionBoundary)
        self.register_constructor(AL.QueryType.LABELMAXIMIZER, LabelMaximizer)
        self.register_constructor(AL.QueryType.LABELUNCERTAINTY, LabelUncertainty)
        self.register_constructor(AL.QueryType.MOST_CERTAIN, MostCertainSampling)
