from __future__ import annotations

import functools
import itertools
import logging
from abc import ABC, abstractstaticmethod
from typing import Dict, Generic, Iterator, List, Optional, TypeVar

import numpy as np
from sklearn.exceptions import NotFittedError

from environment import AbstractEnvironment
from instances import Instance
from machinelearning import AbstractClassifier

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

    @abstractstaticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _calculate_metric(self) -> np.ndarray:
        feature_matrix = self._unlabeled.feature_matrix
        if feature_matrix is None:
            raise ValueError("The feature matrix is empty")
        prob_vec = self.classifier.predict_proba(feature_matrix.matrix)
        metric_result = self.selection_criterion(prob_vec)
        return self

    def calculate_ordering(self):
        metric_result = self._calculate_metric()
        arg_sort = np.argsort(metric_result)
        metric_sort = metric_result[arg_sort]
        ordering = np.flip(arg_sort).tolist()
        return ordering
        
    @MLBased.iterator_fallback
    def __next__(self) -> Instance:
        return super().__next__()

    @MLBased.query_fallback
    @ActiveLearner.query_log
    def query(self) -> Optional[Instance]:
        """Select the instance whose posterior probability is most near 0.5

        Returns
        -------
        Optional[Instance]
            An object containing:
                - The document identifier
                - The document vector
                - The document data
        """
        metric_result = self._calculate_metric()
        np_idx = np.argmax(metric_result)
        doc_id = self._unlabeled.feature_matrix.get_instance_id(np_idx)
        if doc_id is not None:
            return self._unlabeled[doc_id]
        return None
    
    @MLBased.query_batch_fallback
    @ActiveLearner.query_batch_log
    def query_batch(self, batch_size: int) -> List[Instance]:
        metric_result = self._calculate_metric()
        np_idxs = np.flip(np.argsort(metric_result[np.argpartition(metric_result, -batch_size)])).tolist()
        results = []
        for np_idx in np_idxs[0:batch_size]:
            doc_id = self._unlabeled.feature_matrix.get_instance_id(np_idx)
            if doc_id is not None:
                results.append(self._unlabeled[doc_id])
        return results


class LabelProbabilityBased(ProbabiltyBased, ABC, Generic[LT]):
    def __init__(self, classifier, label: LT) -> None:
        super().__init__(classifier)
        self.label = label
        self.labelposition = None
    
    def __call__(self, environment: AbstractEnvironment) -> LabelProbabilityBased:
        super().__call__(environment)
        self.labelposition = self.classifier.get_label_column_index(self.label)
        return self

    @property
    def name(self) -> str:
        return f"{self._name} :: {self.label}"

    @abstractstaticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _calculate_metric(self) -> np.ndarray:
        feature_matrix = self._unlabeled.feature_matrix
        if feature_matrix is None:
            raise ValueError("The feature matrix is empty")
        prob_vec = self.classifier.predict_proba(feature_matrix.matrix)
        sliced_prob_vec = prob_vec[:,self.labelposition]
        return self.selection_criterion(sliced_prob_vec)

class LabelEnsemble(PoolbasedAL):
    _name = "LabelEnsemble"

    def __init__(self, classifier: AbstractClassifier, al_method: LabelProbabilityBased) -> None:
        super().__init__(classifier)
        self._almethod = al_method
        self._project = None
        self._learners: Dict[str, LabelProbabilityBased] = dict()

    def __call__(self, environment: AbstractEnvironment) -> LabelEnsemble:
        super().__call__(environment)
        self._learners = {
            label: self._almethod(self.classifier, label)(environment)
            for label in self._labelprovider.labelset
        }
        return self

    def calculate_ordering(self) -> List[int]:
        raise NotImplementedError

    def __next__(self) -> Instance:
        labelcounts = [(self._labelprovider.document_count(label), label) for label in self._labelprovider.labelset]
        min_label = min(labelcounts)[1]
        return next(self._learners[min_label])
        
    def query(self) -> Optional[Instance]:
        return next(self)

    def query_batch(self, batch_size: int) -> List[Instance]:
        return list(itertools.islice(self, batch_size))
