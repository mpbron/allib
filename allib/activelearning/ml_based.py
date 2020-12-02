from __future__ import annotations

import collections
import functools
import itertools
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from queue import Queue
from typing import (Any, Callable, Dict, Generic, Iterator, Optional, Sequence,
                    Tuple, TypeVar)

import numpy as np  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore

from ..environment import AbstractEnvironment
from ..instances.base import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier
from ..utils import divide_sequence, mapsnd
from ..utils.func import filter_snd_none, list_unzip, sort_on
from .base import ActiveLearner, NotInitializedException
from .poolbased import PoolBasedAL
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


class FeatureMatrix(Generic[KT]):
    def __init__(self, keys: Sequence[KT], vectors: Sequence[Optional[np.ndarray]]):
        # Filter all rows with None as Vector
        filtered_keys, filtered_vecs = filter_snd_none(keys, vectors)
        self.matrix = np.vstack(filtered_vecs)
        self.indices: Sequence[KT] = filtered_keys

    def get_instance_id(self, row_idx: int) -> KT:
        return self.indices[row_idx]

    @classmethod
    def generator_from_provider_mp(cls, provider: InstanceProvider[KT, Any, np.ndarray, Any], batch_size: int = 100) -> Iterator[FeatureMatrix[KT]]:
        for key_batch in divide_sequence(provider.key_list, batch_size):
            ret_keys, vectors = provider.bulk_get_vectors(key_batch)
            matrix = cls(ret_keys, vectors)
            yield matrix

    @classmethod
    def generator_from_provider(cls,
                                provider: InstanceProvider[KT, Any, np.ndarray, Any],
                                batch_size: int = 100) -> Iterator[FeatureMatrix[KT]]:
        for tuple_batch in provider.vector_chunker(batch_size):
            keys, vectors = list_unzip(tuple_batch)
            matrix = cls(keys, vectors)
            yield matrix


class MLBased(PoolBasedAL[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 fallback: PoolBasedAL[KT, DT, VT, RT, LT] = RandomSampling(),
                 batch_size = 200,
                 *_, **__
                 ) -> None:
        super().__init__()
        self.fitted = False
        self.classifier = classifier
        self.fallback = fallback
        self.batch_size = batch_size

    def __call__(self, 
            environment: AbstractEnvironment[KT, DT, VT, RT, LT]
        ) -> MLBased[KT, DT, VT, RT, LT, LVT, PVT]:
        super().__call__(environment)
        self.fallback = self.fallback(self.env)
        self.classifier = self.classifier(self.env)
        return self

    def retrain(self) -> None:
        if not self.initialized:
            raise NotInitializedException
        key_vector_pairs = itertools.chain.from_iterable(self.env.labeled.vector_chunker(self.batch_size))
        keys, vectors = list_unzip(key_vector_pairs)
        labelings = list(map(self.env.labels.get_labels, keys))
        self.classifier.fit_vectors(vectors, labelings)
        self.fitted = True

    def predict(self, instances: Sequence[Instance[KT, DT, VT, RT]]):
        if not self.initialized:
            raise NotInitializedException
        return self.classifier.predict_instances(instances)

    def predict_proba(self, instances: Sequence[Instance[KT, DT, VT, RT]]):
        if not self.initialized:
            raise NotInitializedException
        return self.classifier.predict_proba_instances(instances)

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        return f"{self._name} :: {self.classifier.name}", None

    @staticmethod
    def iterator_fallback(func: Callable[..., Instance[KT, DT, VT, RT]]) -> Callable[..., Instance[KT, DT, VT, RT]]:
        @functools.wraps(func)
        def wrapper(self: MLBased[KT, DT, VT, RT, LT, LVT, PVT],
                    *args: Any,
                    **kwargs: Dict[str, Any]) -> Instance[KT, DT, VT, RT]:
            if self.classifier.fitted:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError) as ex:
                    LOGGER.error("[%s] Falling back to model %s, because of: %s",
                                 self.name, self.fallback.name, ex, exc_info=ex)
            LOGGER.warn("[%s] Falling back to model %s, because it is not fitted", self.name, self.fallback.name)
            fallback_value = next(self.fallback)
            return fallback_value
        return wrapper


class ProbabiltyBased(MLBased[KT, DT, np.ndarray, RT, LT, np.ndarray, np.ndarray], ABC, Generic[KT, DT, RT, LT]):
    @staticmethod
    @abstractmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_predictions(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        """Calculate the probability matrix for the current feature matrix

        Args:
            matrix (FeatureMatrix[KT]): The matrix for which we want to know the predictions

        Returns:
            Tuple[Sequence[KT], np.ndarray]: A list of keys and the probability predictions belonging to it
        """
        prob_vec: np.ndarray = self.classifier.predict_proba(
            matrix.matrix)  # type: ignore
        keys = matrix.indices
        return keys, prob_vec

    @ActiveLearner.ordering_log
    def calculate_ordering(self) -> Tuple[Sequence[KT], Sequence[float]]:
        def get_metric_tuples(keys: Sequence[KT], vec: np.ndarray) -> Sequence[Tuple[KT, float]]:
            floats: Sequence[float] = vec.tolist()
            return list(zip(keys, floats))
        # Get a generator with that generates feature matrices from data
        matrices = FeatureMatrix[KT].generator_from_provider(
            self.env.unlabeled, self.batch_size)
        # Get the predictions for each matrix
        predictions = map(self._get_predictions, matrices)
        # Transfrorm the selection criterion function into a function that works on tuples and
        # applies the id :: a -> a function on the first element of the tuple and selection_criterion
        # on the second
        sel_func = mapsnd(self.selection_criterion)
        # Apply sel_func on the predictions
        metric_results = itertools.starmap(sel_func, predictions)
        # Transform the metric np.ndarray to a python List[float] and flatten the iterable
        # to a list of Tuple[KT, float] where float is the metric for the instance with
        # key KT
        metric_tuples = list(
            itertools.chain.from_iterable(
                itertools.starmap(
                    get_metric_tuples, metric_results)))
        # Sort the tuples in descending order, so that the key with the highest score
        # is on the first position of the list
        sorted_tuples = sort_on(1, metric_tuples)
        # Retrieve the keys from the tuples
        ordered_keys, ordered_metrics = list_unzip(sorted_tuples)
        return ordered_keys, ordered_metrics

    def update_ordering(self) -> None:
        self.retrain()
        ordering, _ = self.calculate_ordering()
        self.ordering = collections.deque(ordering)

    @MLBased.iterator_fallback
    def __next__(self) -> Instance[KT, DT, np.ndarray, RT]:
        value: Instance[KT, DT, np.ndarray, RT] = super(
            ProbabiltyBased, self).__next__()
        return value
