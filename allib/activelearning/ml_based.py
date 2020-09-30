from __future__ import annotations

import functools
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generic, Iterator, List, Optional, TypeVar, Callable

import numpy as np # type: ignore
from sklearn.exceptions import NotFittedError# type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance
from ..machinelearning import AbstractClassifier

from .base import ActiveLearner
from .poolbased import PoolbasedAL
from .random import RandomSampling

LT = TypeVar("LT")

LOGGER = logging.getLogger(__name__)

class MLBased(RandomSampling):
    def __init__(self,
                 classifier: AbstractClassifier,
                 fallback = RandomSampling
                 ) -> None:
        super().__init__(classifier)
        self.fallback = fallback(classifier)

    def __call__(self, environment: AbstractEnvironment) -> PoolbasedAL:
        super().__call__(environment)
        self.fallback = self.fallback(environment)
        return self

    @staticmethod
    def iterator_fallback(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.fitted:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError) as ex:
                    LOGGER.warning("[%s] Falling back to model %s, because of:", self.name, self.fallback.name, exc_info=ex)
            return next(self.fallback)
        return wrapper

    @staticmethod
    def query_fallback(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.fitted:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError) as ex:
                    LOGGER.warning("[%s] Falling back to model %s, because of:", self.name, self.fallback.name, exc_info=ex)
            return self.fallback.query()
        return wrapper

    @staticmethod
    def query_batch_fallback(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.classifier.fitted:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError) as ex:
                    LOGGER.warning("[%s] Falling back to model %s, because of:", self.name, self.fallback.name, exc_info=ex)
            return self.fallback.query_batch(*args, **kwargs)
        return wrapper

class ProbabiltyBased(MLBased, ABC):
    def __init__(self, classifier: AbstractClassifier, fallback=RandomSampling):
        super().__init__(classifier, fallback)
        self._metric_result = None

    @staticmethod
    @abstractmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _calculate_metric(self) -> np.ndarray:
        assert self._unlabeled is not None
        feature_matrix = self._unlabeled.feature_matrix
        if feature_matrix is None:
            raise ValueError("The feature matrix is empty")
        prob_vec: np.ndarray = self.classifier.predict_proba(feature_matrix.matrix)
        metric_result = self.selection_criterion(prob_vec)
        return metric_result

    def calculate_ordering(self):
        metric_result = self._calculate_metric()
        arg_sort = np.argsort(metric_result)
        metric_sort = metric_result[arg_sort]
        ordering = np.flip(arg_sort).tolist()
        return ordering
        
    @MLBased.iterator_fallback
    def __next__(self) -> Instance:
        return super().__next__()


class LabelProbabilityBased(ProbabiltyBased, ABC, Generic[LT]):
    def __init__(self, classifier, label: LT) -> None:
        super().__init__(classifier)
        self.label = label
        self.labelposition: Optional[int] = None
    
    def __call__(self, environment: AbstractEnvironment) -> LabelProbabilityBased:
        super().__call__(environment)
        self.labelposition = self.classifier.get_label_column_index(self.label)
        return self

    @property
    def name(self) -> str:
        return f"{self._name} :: {self.label}"

    @staticmethod
    @abstractmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _calculate_metric(self) -> np.ndarray:
        assert self._unlabeled is not None
        assert self.labelposition is not None
        feature_matrix = self._unlabeled.feature_matrix
        if feature_matrix is None:
            raise ValueError("The feature matrix is empty")
        prob_vec: np.ndarray = self.classifier.predict_proba(feature_matrix.matrix)
        sliced_prob_vec = prob_vec[:,self.labelposition]
        return self.selection_criterion(sliced_prob_vec)

class LabelEnsemble(PoolbasedAL):
    _name = "LabelEnsemble"

    def __init__(self, classifier: AbstractClassifier, al_method: LabelProbabilityBased) -> None:
        super().__init__(classifier)
        self._almethod: Callable[..., LabelProbabilityBased] = al_method
        self._project = None
        self._learners: Dict[str, LabelProbabilityBased] = dict()

    def __call__(self, environment: AbstractEnvironment) -> LabelEnsemble:
        super().__call__(environment)
        assert self._labelprovider is not None
        self._learners = {
            label: self._almethod(self.classifier, label)(environment)
            for label in self._labelprovider.labelset
        }
        return self

    def calculate_ordering(self) -> List[int]:
        raise NotImplementedError

    def __next__(self) -> Instance:
        assert self._labelprovider is not None
        labelcounts = [(self._labelprovider.document_count(label), label) for label in self._labelprovider.labelset]
        min_label = min(labelcounts)[1]
        return next(self._learners[min_label])
        
    def query(self) -> Optional[Instance]:
        return next(self)

    def query_batch(self, batch_size: int) -> List[Instance]:
        return list(itertools.islice(self, batch_size))
