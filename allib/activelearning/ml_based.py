from __future__ import annotations
from allib.instances.base import InstanceProvider

import functools
import itertools
import logging
from multiprocessing import Pool
from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar, Callable, Any, Sequence, Tuple, Iterator, Iterable

import numpy as np # type: ignore
from sklearn.exceptions import NotFittedError# type: ignore

from ..environment import AbstractEnvironment
from ..instances.base import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier
from ..utils import mapsnd, divide_sequence


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

class FeatureMatrix(Generic[KT]):
    def __init__(self, keys: Sequence[KT], vectors: Sequence[Optional[np.ndarray]]):
        key_vecs = filter(lambda x: x[1] is not None, zip(keys, vectors))
        filtered_keys, filtered_vecs = map(list, zip(*key_vecs)) 
        self.matrix = np.vstack(filtered_vecs) 
        self.indices: Sequence[KT] = filtered_keys

    def get_instance_id(self, row_idx: int) -> KT:
        return self.indices[row_idx]

    @classmethod
    def generator_from_provider_mp(cls, provider: InstanceProvider[KT, Any, np.ndarray, Any], batch_size: int = 100) -> Iterator[FeatureMatrix[KT]]:
        for key_batch in divide_sequence(provider.key_list, batch_size):
            vectors = provider.bulk_get_vectors(key_batch)
            matrix = cls(key_batch, vectors)
            yield matrix

    @classmethod
    def generator_from_provider(cls, 
                                provider: InstanceProvider[KT, Any, np.ndarray, Any], 
                                batch_size: int = 100) -> Iterator[FeatureMatrix[KT]]:
        for tuple_batch in provider.vector_chunker(batch_size):
            keys, vectors = map(list, zip(*tuple_batch))
            matrix = cls(keys, vectors)
            yield matrix


class MLBased(RandomSampling[KT, DT, VT, RT, LT, LVT, PVT], Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 fallback: Callable[..., PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT]] = RandomSampling[KT, DT, VT, RT, LT, LVT, PVT],
                 *_, **__
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
            if self.classifier.fitted:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError) as ex:
                    LOGGER.error("[%s] Falling back to model %s, because of: %s", self.name, self.fallback.name, ex, exc_info=ex)
            fallback_value = next(self.fallback)
            return fallback_value
        return wrapper

class ProbabiltyBased(MLBased[KT, DT, np.ndarray, RT, LT, np.ndarray, np.ndarray], ABC, Generic[KT, DT, RT, LT]):
    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 fallback: Callable[..., PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT]] = RandomSampling[KT, DT, VT, RT, LT, LVT, PVT],
                 batch_size: int = 128, n_cores = 4, *_, **__) -> None:
        super().__init__(classifier, fallback)
        self.batch_size = batch_size
        self.n_cores = n_cores
    
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
        prob_vec: np.ndarray = self.classifier.predict_proba(matrix.matrix) # type: ignore
        keys = matrix.indices
        return keys, prob_vec

    def calculate_ordering(self) -> Sequence[KT]:
        def get_metric_tuples(keys: Sequence[KT], vec: np.ndarray) -> Sequence[Tuple[KT, float]]:
            floats: Sequence[float] = vec.tolist()
            return list(zip(keys, floats))
        # Get a generator with that generates feature matrices from data
        predictions: Iterable[Tuple[Sequence[KT], np.ndarray]] = []
        if self.n_cores > 1:
            matrices = FeatureMatrix[KT].generator_from_provider_mp(self.env.unlabeled, self.batch_size)
            with Pool(self.n_cores) as p:
                # Get the predictions for each matrix
                predictions = p.map_async(self._get_predictions, matrices).get()
        else:
            # Get the predictions for each matrix
            matrices = FeatureMatrix[KT].generator_from_provider(self.env.unlabeled, self.batch_size)
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
        sorted_tuples = sorted(metric_tuples, key= lambda x: x[1], reverse = True)
        # Retrieve the keys from the tuples
        ordered_keys, _ = zip(*sorted_tuples)
        return list(ordered_keys)

    @MLBased.iterator_fallback
    def __next__(self) -> Instance[KT, DT, np.ndarray, RT]:
        value: Instance[KT, DT, np.ndarray, RT] = super(ProbabiltyBased, self).__next__()
        return value

class LabelProbabilityBased(ProbabiltyBased[KT, DT, RT, LT], ABC, Generic[KT, DT, RT, LT]):
    def __init__(self, classifier: AbstractClassifier[KT, VT, LT, LVT, PVT], label: LT, *_, **__) -> None:
        super().__init__(classifier)
        self.label = label
        self.labelposition: Optional[int] = None
    
    def __call__(self, environment: AbstractEnvironment[KT, DT, np.ndarray, RT, LT]) -> LabelProbabilityBased[KT, DT, RT, LT]:
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

    def _get_predictions(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        prob_vec: np.ndarray = self.classifier.predict_proba(matrix.matrix) # type: ignore
        sliced_prob_vec: np.ndarray = prob_vec[:,self.labelposition] #type: ignore
        keys = matrix.indices
        return keys, sliced_prob_vec

class LabelEnsemble(PoolbasedAL[KT, DT, np.ndarray, RT, LT, np.ndarray, np.ndarray], Generic[KT, DT, RT, LT]):
    _name = "LabelEnsemble"

    def __init__(self, 
                 classifier: AbstractClassifier[KT, np.ndarray, LT, np.ndarray, np.ndarray],
                 al_method: LabelProbabilityBased[KT, DT, RT, LT],
                 *_, **__
                 ) -> None:
        super().__init__(classifier)
        self._almethod: Callable[..., LabelProbabilityBased[KT, DT, RT, LT]] = al_method
        self._project = None
        self._learners: Dict[LT, LabelProbabilityBased[KT, DT, RT, LT]] = dict()

    def __call__(self, environment: AbstractEnvironment[KT, DT, np.ndarray, RT, LT]) -> LabelEnsemble[KT, DT,RT, LT]:
        super().__call__(environment)
        self._learners = {
            label: self._almethod(self.classifier, label)(environment)
            for label in self.env.labels.labelset
        }
        return self

    def calculate_ordering(self) -> Sequence[KT]:
        raise NotImplementedError

    def __next__(self) -> Instance[KT, DT, np.ndarray, RT]:
        labelcounts = [(self.env.labels.document_count(label), label) for label in self.env.labels.labelset]
        min_label = min(labelcounts)[1]
        return next(self._learners[min_label])

