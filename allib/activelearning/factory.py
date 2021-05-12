import functools
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Callable

from ..factory import AbstractBuilder, ObjectFactory
from ..machinelearning import AbstractClassifier, MachineLearningFactory
from ..module.component import Component
from .base import ActiveLearner
from .catalog import ALCatalog as AL
from .ensembles import StrategyEnsemble
from .estimator import CycleEstimator, Estimator, RetryEstimator
from .labelmethods import LabelProbabilityBased
from .ml_based import AbstractSelectionCriterion, ProbabilityBased
from .mostcertain import LabelMaximizer, LabelMaximizerNew, MostCertainSampling, MostConfidence
from .prob_ensembles import LabelProbEnsemble, ProbabilityBasedEnsemble, LabelMinProbEnsemble
from .random import RandomSampling
from .uncertainty import (EntropySampling, LabelUncertainty, LeastConfidence,
                          MarginSampling, NearDecisionBoundary, LabelUncertaintyNew, RandomMLStrategy)

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
            label: Any,
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
            identifier: Optional[str] = None,
            **kwargs):
        return self._factory.create(
            query_type, identifier=identifier,**kwargs)
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

class SelectionCriterionBuilder(AbstractBuilder):
    def __call__(self, query_type: AL.QueryType, **kwargs):
        return self._factory.create(query_type, **kwargs)

class ProbabilityEnsembleBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            strategies: List[Dict],
            machinelearning: Dict,
            fallback: Dict = dict(),
            identifier: Optional[str] = None,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        built_strategies: Sequence[AbstractSelectionCriterion] = [
            self._factory.create(Component.SELECTION_CRITERION, **dic) for dic in strategies
        ]
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return ProbabilityBasedEnsemble(classifier, 
                                built_strategies, 
                                fallback=built_fallback,
                                identifier=identifier)

class LabelProbilityBasedEnsembleBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            strategy: AL.QueryType,
            machinelearning: Dict,
            fallback: Dict = dict(),
            identifier: Optional[str] = None,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        if strategy not in self._factory.builders:
            raise NotImplementedError(f"The selection strategy {strategy} is not available")
        chosen_strategy = self._factory.get_constructor(strategy)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelProbEnsemble(classifier, 
                                 chosen_strategy, 
                                 fallback=built_fallback,
                                 identifier=identifier)

class LabelMinProbilityBasedEnsembleBuilder(AbstractBuilder):
    def __call__( # type: ignore
            self,
            strategy: AL.QueryType,
            machinelearning: Dict,
            fallback: Dict = dict(),
            identifier: Optional[str] = None,
            **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        if strategy not in self._factory.builders:
            raise NotImplementedError(f"The selection strategy {strategy} is not available")
        chosen_strategy = self._factory.get_constructor(strategy)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelMinProbEnsemble(classifier, 
                                 chosen_strategy, 
                                 fallback=built_fallback,
                                 identifier=identifier)

class ActiveLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(MachineLearningFactory())
        
        self.register_builder(Component.ACTIVELEARNER, ALBuilder())
        self.register_builder(Component.FALLBACK, FallbackBuilder())
        self.register_builder(Component.SELECTION_CRITERION, SelectionCriterionBuilder())
        self.register_builder(AL.Paradigm.POOLBASED, PoolbasedBuilder())
        self.register_builder(AL.Paradigm.PROBABILITY_BASED, ProbabilityBasedBuilder())
        self.register_builder(AL.Paradigm.ESTIMATOR, EstimatorBuilder())
        self.register_builder(AL.Paradigm.CYCLE_ESTIMATOR, CycleEstimatorBuilder())
        self.register_builder(AL.Paradigm.ENSEMBLE, StrategyEnsembleBuilder())
        self.register_builder(AL.Paradigm.LABEL_PROBABILITY_BASED, LabelProbabilityBasedBuilder())
        self.register_builder(AL.Paradigm.PROBABILITY_BASED_ENSEMBLE, ProbabilityEnsembleBuilder())
        self.register_builder(AL.Paradigm.LABEL_PROBABILITY_BASED_ENSEMBLE, LabelProbilityBasedEnsembleBuilder())
        self.register_builder(AL.Paradigm.LABEL_MIN_PROB_ENSEMBLE, LabelMinProbilityBasedEnsembleBuilder())
        
        self.register_constructor(AL.QueryType.RANDOM_SAMPLING, RandomSampling)
        self.register_constructor(AL.QueryType.LEAST_CONFIDENCE, LeastConfidence)
        self.register_constructor(AL.QueryType.MAX_ENTROPY, EntropySampling)
        self.register_constructor(AL.QueryType.MARGIN_SAMPLING, MarginSampling)
        self.register_constructor(AL.QueryType.NEAR_DECISION_BOUNDARY, NearDecisionBoundary)
        self.register_constructor(AL.QueryType.LABELMAXIMIZER, LabelMaximizer)
        self.register_constructor(AL.QueryType.LABELUNCERTAINTY, LabelUncertainty)
        self.register_constructor(AL.QueryType.MOST_CERTAIN, MostCertainSampling)
        self.register_constructor(AL.QueryType.MOST_CONFIDENCE, MostConfidence)
        self.register_constructor(AL.QueryType.LABELMAXIMIZER_NEW, LabelMaximizerNew)
        self.register_constructor(AL.QueryType.LABELUNCERTAINTY_NEW, LabelUncertaintyNew)
        self.register_constructor(AL.QueryType.RANDOM_ML, RandomMLStrategy)