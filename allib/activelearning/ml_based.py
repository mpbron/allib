from __future__ import annotations
from allib.instances.base import InstanceProvider

import functools
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar, Callable, Any, Sequence, Tuple

import numpy as np # type: ignore
from sklearn.exceptions import NotFittedError# type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance
from ..machinelearning import AbstractClassifier

from .poolbased import PoolbasedAL
from .random import RandomSampling

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
FT = TypeVar("FT")

LOGGER = logging.getLogger(__name__)

class MLBased(RandomSampling[KT, DT, VT, RT, LT, LVT, PVT], Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 fallback: Callable[..., PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT]] = RandomSampling[KT, DT, VT, RT, LT, LVT, PVT]
                 ) -> None:
        super().__init__(classifier)
        self.fallback = fallback(classifier)

    def __call__(self, 
                 environment: AbstractEnvironment[KT, DT, VT, RT, LT]
                 ) -> MLBased[KT, DT, VT, RT, LT, LVT, PVT]:
        super().__call__(environment)
        self.fallback = self.fallback(environment)
        return self

    @staticmethod
    def iterator_fallback(func: Callable[..., Instance[KT, DT, VT, RT]]) -> Callable[..., Instance[KT, DT, VT, RT]]:
        @functools.wraps(func)
        def wrapper(self: MLBased[KT, DT, VT, RT, LT, LVT, PVT], 
                    *args: Any, 
                    **kwargs: Dict[str, Any]) -> Instance[KT, DT, VT, RT]:
            if self.fitted:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError) as ex:
                    LOGGER.warning("[%s] Falling back to model %s, because of:", self.name, self.fallback.name, exc_info=ex)
            fallback_value = next(self.fallback)
            return fallback_value
        return wrapper

class ProbabiltyBased(MLBased[KT, DT, VT, RT, LT, LVT, PVT], ABC, Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    @staticmethod
    @abstractmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _calculate_metric(self) -> np.ndarray:
        assert self._unlabeled is not None
        feature_matrix = self._unlabeled.feature_matrix
        if feature_matrix is None:
            raise ValueError("The feature matrix is empty")
        prob_vec: np.ndarray = self.classifier.predict_proba(feature_matrix.matrix) # type: ignore
        metric_result = self.selection_criterion(prob_vec)
        return metric_result

    def calculate_ordering(self) -> Sequence[KT]:
        metric_result = self._calculate_metric()
        arg_sort = np.argsort(metric_result)
        metric_sort = metric_result[arg_sort] #type: ignore
        ordering = np.flip(arg_sort).tolist() #type: ignore 
        return ordering # type: ignore
        
    @MLBased.iterator_fallback
    def __next__(self) -> Instance[KT, DT, VT, RT]:
        value: Instance[KT, DT, VT, RT] = super(ProbabiltyBased, self).__next__()
        return value

class LabelProbabilityBased(ProbabiltyBased[KT, DT, VT, RT, LT, LVT, PVT], ABC, Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    def __init__(self, classifier: AbstractClassifier[KT, VT, LT, LVT, PVT], label: LT) -> None:
        super().__init__(classifier)
        self.label = label
        self.labelposition: Optional[int] = None
    
    def __call__(self, environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> LabelProbabilityBased[KT, DT, VT, RT, LT, LVT, PVT]:
        super().__call__(environment)
        self.labelposition = self.classifier.get_label_column_index(self.label)
        return self

    @property
    def name(self) -> Tuple[str, LT]:
        return self._name, self.label

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
        prob_vec: np.ndarray = self.classifier.predict_proba(feature_matrix.matrix) # type: ignore
        sliced_prob_vec = prob_vec[:,self.labelposition]
        return self.selection_criterion(sliced_prob_vec)

class LabelEnsemble(PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT], Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = "LabelEnsemble"

    def __init__(self, 
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 al_method: LabelProbabilityBased[KT, DT, VT, RT, LT, LVT, PVT]
                 ) -> None:
        super().__init__(classifier)
        self._almethod: Callable[..., LabelProbabilityBased[KT, DT, VT, RT, LT, LVT, PVT]] = al_method
        self._project = None
        self._learners: Dict[LT, LabelProbabilityBased[KT, DT, VT, RT, LT, LVT, PVT]] = dict()

    def __call__(self, environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> LabelEnsemble[KT, DT, VT, RT, LT, LVT, PVT]:
        super().__call__(environment)
        assert self._labelprovider is not None
        self._learners = {
            label: self._almethod(self.classifier, label)(environment)
            for label in self._labelprovider.labelset
        }
        return self

    def calculate_ordering(self) -> Sequence[KT]:
        raise NotImplementedError

    def __next__(self) -> Instance[KT, DT, VT, RT]:
        assert self._labelprovider is not None
        labelcounts = [(self._labelprovider.document_count(label), label) for label in self._labelprovider.labelset]
        min_label = min(labelcounts)[1]
        return next(self._learners[min_label])

