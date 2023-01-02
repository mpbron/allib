import functools
from abc import ABC
from distutils.command.build import build
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

import instancelib as il
from instancelib.machinelearning.sklearn import SkLearnClassifier
from instancelib.typehints.typevars import DT, KT, LMT, LT, PMT, RT, VT
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression

from ..environment.base import AbstractEnvironment
from ..factory import AbstractBuilder, ObjectFactory
from ..machinelearning import AbstractClassifier, MachineLearningFactory
from ..module.component import Component
from ..typehints.typevars import IT
from .autostop import AutoStopLearner
from .autotar import AutoTarLearner, BinaryTarLearner
from .base import ActiveLearner
from .catalog import ALCatalog as AL
from .ensembles import StrategyEnsemble
from .estimator import CycleEstimator, Estimator, RetryEstimator
from .labelmethods import LabelProbabilityBased
from .ml_based import ProbabilityBased
from .mostcertain import (LabelMaximizer, LabelMaximizerNew,
                          MostCertainSampling, MostConfidence)
from .prob_ensembles import (LabelMinProbEnsemble, LabelProbEnsemble,
                             ProbabilityBasedEnsemble)
from .random import RandomSampling
from .selectioncriterion import AbstractSelectionCriterion
from .uncertainty import (EntropySampling, LabelUncertainty,
                          LabelUncertaintyNew, LeastConfidence, MarginSampling,
                          NearDecisionBoundary, RandomMLStrategy)


class FallbackBuilder(AbstractBuilder):
    def __call__(self, **kwargs) -> Callable[[AbstractEnvironment], ActiveLearner]:
        if kwargs:
            fallback = self._factory.create(Component.ACTIVELEARNER, **kwargs)
            return fallback
        return RandomSampling.builder()


class ALBuilder(AbstractBuilder):
    def __call__(self, paradigm, **kwargs):
        return self._factory.create(paradigm, **kwargs)


class ProbabilityBasedBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self,
        query_type: AL.QueryType,
        machinelearning: Dict,
        fallback: Dict = dict(),
        batch_size: int = 200,
        identifier: Optional[str] = None,
        **kwargs,
    ):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        selection_criterion: AbstractSelectionCriterion = self._factory.create(
            query_type, **kwargs
        )
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return ProbabilityBased.builder(
            classifier,
            selection_criterion,
            built_fallback,
            batch_size=batch_size,
            identifier=identifier,
        )


class LabelProbabilityBasedBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self,
        query_type: AL.QueryType,
        machinelearning: Dict,
        fallback: Dict = dict(),
        identifier: Optional[str] = None,
        **kwargs,
    ):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        selection_criterion: AbstractSelectionCriterion = self._factory.create(
            query_type, **kwargs
        )
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelProbabilityBased.builder(
            classifier,
            selection_criterion,
            built_fallback,            
            identifier=identifier,
        )


class PoolbasedBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self, query_type: AL.QueryType, identifier: Optional[str] = None, **kwargs
    ):
        return self._factory.create(query_type, identifier=identifier, **kwargs)

class CustomBuilder(AbstractBuilder):
    def __call__(self, method: AL.CustomMethods, **kwargs):
        return self._factory.create(method, **kwargs)


class StrategyEnsembleBuilder(AbstractBuilder):
    def build_learner(self, classifier: AbstractClassifier, config):
        query_type = config["query_type"]
        params = {k: v for k, v in config if k not in ["query_type"]}
        return self._factory.create(query_type, classifier=classifier, **params)

    def __call__(  # type: ignore
        self,
        learners: List[Dict],
        machinelearning: Dict,
        probabilities: List[float],
        identifier: Optional[str] = None,
        **kwargs,
    ):
        assert len(learners) == len(probabilities)
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        config_function = functools.partial(self.build_learner, classifier)
        configured_learners = list(map(config_function, learners))
        return StrategyEnsemble.builder(classifier, configured_learners, probabilities)


class CycleEstimatorBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self, learners: List[Dict], identifier: Optional[str] = None, **kwargs
    ) -> Callable[..., ActiveLearner]:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        return CycleEstimator.builder(configured_learners)


class EstimatorBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self,
        learners: List[Dict],
        identifier: Optional[str] = None,
        **kwargs,
    ) -> Callable[..., ActiveLearner]:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        return Estimator.builder(configured_learners)


class RetryEstimatorBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self, learners: List[Dict], identifier: Optional[str] = None, **kwargs
    ) -> Callable[..., ActiveLearner]:
        configured_learners = [
            self._factory.create(Component.ACTIVELEARNER, **learner_config)
            for learner_config in learners
        ]
        return RetryEstimator.builder(configured_learners)


class SelectionCriterionBuilder(AbstractBuilder):
    def __call__(self, query_type: AL.QueryType, **kwargs):
        return self._factory.create(query_type, **kwargs)


class ProbabilityEnsembleBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self,
        strategies: List[Dict],
        machinelearning: Dict,
        fallback: Dict = dict(),
        identifier: Optional[str] = None,
        **kwargs,
    ):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        built_strategies: Sequence[AbstractSelectionCriterion] = [
            self._factory.create(Component.SELECTION_CRITERION, **dic)
            for dic in strategies
        ]
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return ProbabilityBasedEnsemble(
            classifier, built_strategies, fallback=built_fallback, identifier=identifier
        )


class LabelProbilityBasedEnsembleBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self,
        strategy: AL.QueryType,
        machinelearning: Dict,
        fallback: Dict = dict(),
        identifier: Optional[str] = None,
        **kwargs,
    ):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        if strategy not in self._factory.builders:
            raise NotImplementedError(
                f"The selection strategy {strategy} is not available"
            )
        chosen_strategy = self._factory.get_constructor(strategy)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelProbEnsemble(
            classifier, chosen_strategy, fallback=built_fallback, identifier=identifier
        )


class LabelMinProbilityBasedEnsembleBuilder(AbstractBuilder):
    def __call__(  # type: ignore
        self,
        strategy: AL.QueryType,
        machinelearning: Dict,
        fallback: Dict = dict(),
        identifier: Optional[str] = None,
        **kwargs,
    ):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        if strategy not in self._factory.builders:
            raise NotImplementedError(
                f"The selection strategy {strategy} is not available"
            )
        chosen_strategy = self._factory.get_constructor(strategy)
        built_fallback = self._factory.create(Component.FALLBACK, **fallback)
        return LabelMinProbEnsemble(
            classifier, chosen_strategy, fallback=built_fallback, identifier=identifier
        )


MT = TypeVar("MT")


def classifier_builder(
    classifier: MT,
    build_method: Callable[
        [MT, il.AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
        il.AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT],
    ],
) -> Callable[
    [il.AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
    il.AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT],
]:
    def wrap_func(env: il.AbstractEnvironment[IT, KT, DT, VT, RT, LT]):
        return build_method(classifier, env)

    return wrap_func

class BinaryTarBuilder(AbstractBuilder):
    def __call__(self, 
                 machinelearning: Mapping[str, Any],
                 batch_size: int,
                 chunk_size: int = 2000, **kwargs):
        classifier = self._factory.create(Component.CLASSIFIER, **machinelearning)
        return BinaryTarLearner.builder(classifier, batch_size, chunk_size, **kwargs)


class AutoTARBuilder(AbstractBuilder):
    def __call__(self, 
                 machinelearning: Mapping[str, Any],
                 k_sample: int, 
                 batch_size: int, **kwargs):
        builder = self._factory.create(Component.CLASSIFIER, **machinelearning)
        at = AutoTarLearner.builder(builder, k_sample, batch_size, **kwargs)
        return at

class AutoSTOPBuilder(AbstractBuilder):
    def __call__(self, 
                 machinelearning: Mapping[str, Any],
                 k_sample: int, 
                 batch_size: int, 
                 **kwargs):
        builder = self._factory.create(Component.CLASSIFIER, **machinelearning)
        at = AutoStopLearner.builder(builder, k_sample, batch_size, **kwargs)
        return at


class ActiveLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(MachineLearningFactory())

        self.register_builder(Component.ACTIVELEARNER, ALBuilder())
        self.register_builder(Component.FALLBACK, FallbackBuilder())
        self.register_builder(
            Component.SELECTION_CRITERION, SelectionCriterionBuilder()
        )
        self.register_builder(AL.Paradigm.POOLBASED, PoolbasedBuilder())
        self.register_builder(AL.Paradigm.PROBABILITY_BASED, ProbabilityBasedBuilder())
        self.register_builder(AL.Paradigm.ESTIMATOR, EstimatorBuilder())
        self.register_builder(AL.Paradigm.CYCLE_ESTIMATOR, CycleEstimatorBuilder())
        self.register_builder(AL.Paradigm.CUSTOM, CustomBuilder())
        self.register_builder(AL.Paradigm.ENSEMBLE, StrategyEnsembleBuilder())
        self.register_builder(
            AL.Paradigm.LABEL_PROBABILITY_BASED, LabelProbabilityBasedBuilder()
        )
        self.register_builder(
            AL.Paradigm.PROBABILITY_BASED_ENSEMBLE, ProbabilityEnsembleBuilder()
        )
        self.register_builder(
            AL.Paradigm.LABEL_PROBABILITY_BASED_ENSEMBLE,
            LabelProbilityBasedEnsembleBuilder(),
        )
        self.register_builder(
            AL.Paradigm.LABEL_MIN_PROB_ENSEMBLE, LabelMinProbilityBasedEnsembleBuilder()
        )
        self.register_builder(AL.CustomMethods.AUTOTAR, AutoTARBuilder())
        self.register_builder(AL.CustomMethods.BINARYTAR, BinaryTarBuilder())
        self.register_builder(AL.CustomMethods.AUTOSTOP, AutoSTOPBuilder())
        self.register_constructor(AL.QueryType.RANDOM_SAMPLING, RandomSampling.builder)
        self.register_constructor(AL.QueryType.LEAST_CONFIDENCE, LeastConfidence)
        self.register_constructor(AL.QueryType.MAX_ENTROPY, EntropySampling)
        self.register_constructor(AL.QueryType.MARGIN_SAMPLING, MarginSampling)
        self.register_constructor(
            AL.QueryType.NEAR_DECISION_BOUNDARY, NearDecisionBoundary
        )
        self.register_constructor(AL.QueryType.LABELMAXIMIZER, LabelMaximizer)
        self.register_constructor(AL.QueryType.LABELUNCERTAINTY, LabelUncertainty)
        self.register_constructor(AL.QueryType.MOST_CERTAIN, MostCertainSampling)
        self.register_constructor(AL.QueryType.MOST_CONFIDENCE, MostConfidence)
        self.register_constructor(AL.QueryType.LABELMAXIMIZER_NEW, LabelMaximizerNew)
        self.register_constructor(
            AL.QueryType.LABELUNCERTAINTY_NEW, LabelUncertaintyNew
        )
        self.register_constructor(AL.QueryType.RANDOM_ML, RandomMLStrategy)
  
