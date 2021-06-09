from __future__ import annotations

import collections
import functools
import itertools
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from queue import Queue
from typing import (Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator, List,
                    Optional, Sequence, Set, Tuple, TypeVar)

import numpy as np  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore

from ..environment import AbstractEnvironment
from ..exceptions import NoOrderingException, NotInitializedException
from ..exceptions.base import NoLabeledDataException, NoVectorsException
from instancelib.instances.base import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier
from ..utils import divide_sequence, mapsnd
from ..utils.func import filter_snd_none, fst, list_unzip, multisort, sort_on
from ..utils.random import get_random_generator
from .base import ActiveLearner, ProbabilityPrediction
from .ensembles import AbstractEnsemble
from .ml_based import (AbstractSelectionCriterion, FeatureMatrix, MLBased,
                       ProbabilityBased)
from .fixed import FixedOrdering
from .random import RandomSampling

from ..typehints import KT, DT, VT, RT, LT, LVT, PVT, IT

FT = TypeVar("FT")


LOGGER = logging.getLogger(__name__)

def get_probabilities(probabilities: Optional[Sequence[float]], learners: Sequence[Any]) -> Sequence[float]:
    if len(learners) == 0:
        return []
    if probabilities is not None and probabilities:
        return probabilities
    equal_prob = 1.0 / len(learners)
    probs = [equal_prob] * len(learners)
    return probs
    



class ProbabilityBasedEnsemble(AbstractEnsemble[IT, KT, DT, np.ndarray, RT, LT], 
                       ProbabilityBased[IT, KT, DT, RT, LT], 
                       Generic[IT, KT, DT, RT, LT]):
    _name = "ProbabilityBasedEnsemble"

    def __init__(self,
                 classifier: AbstractClassifier[KT, np.ndarray, LT, np.ndarray, np.ndarray],
                 strategies: Sequence[AbstractSelectionCriterion],
                 probabilities: Optional[Sequence[float]] = None, rng: Any = None,
                 batch_size = 200,
                 identifier: Optional[str] = None, 
                 fallback = RandomSampling[IT, KT, DT, np.ndarray, RT, LT](), *_, **__) -> None:
        self.classifier = classifier
        self.strategies = strategies
        self.learners = [FixedOrdering(identifier=strategy.name) for strategy in self.strategies]
        self.probabilities = get_probabilities(probabilities, strategies)
        self._rng: Any = get_random_generator(rng)
        self.fallback = fallback
        self.sampled: Set[KT] = set()
        self._has_ordering: bool = False
        self.batch_size = batch_size
        self.identifier = identifier
    
    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, np.ndarray, RT, LT]):
        """Initialize the learner with an environment

        Args:
            environment (AbstractEnvironment[KT, DT, VT, RT, LT]): the chosen environment

        Returns:
            StrategyEnsemble: The initialized environment
        """        
        super().__call__(environment)
        self.classifier = self.classifier(self.env)
        for learner in self.learners:
            learner(self.env)
        self.initialized = True
        return self

    def process_predictions(self, 
                            predictions: Tuple[Sequence[KT], np.ndarray]
                            ) -> Tuple[Sequence[KT], np.ndarray]:
        keys, matrix = predictions
        results = np.transpose(np.array([sel_criterion(matrix) for sel_criterion in self.strategies]))
        return keys, results

    def calculate_ordering(self) -> Sequence[Tuple[Sequence[KT], Sequence[float]]]:
        """Calculate the ordering for the unlabeled set of instances according
        to their predicted label probabilities by the learner's classifier.
        The provided selection criterion calculates a metric on the probabilities.

        Returns
        -------
        Tuple[Sequence[KT], Sequence[float]]
            A tuple of two lists of equal length and ordered descendingly
            according to the second list.

            - A list of instance keys with type KT
            - A list of scores, matching with the instance keys
        """        
               
        def to_tuples(pair: Tuple[Sequence[KT], np.ndarray]
                     ) -> Sequence[Tuple[KT, Sequence[float]]]:
            keys, matrix = pair
            results: Sequence[Sequence[float]] = [[]]
            if len(matrix.shape) < 2:
                results = matrix.reshape((len(keys), 1)).tolist()
            if len(matrix.shape) == 2:
                results = matrix.tolist()
            else:
                raise ValueError(
                    "The result matrix is incorrectly shaped. "
                    f"It should have shape ({len(keys)}, {len(self.strategies)})"
                    f"while the matrix has shape {matrix.shape}")            
            if len(keys) != len(results):
                raise ValueError(
                    "The results are incorrectly shaped." 
                    "The length of the list of keys and the list of scores should be equal.")
            zipped = list(zip(keys, results))
            return zipped

        LOGGER.info("[%s] Start calculating the prediction matrix for all unlabeled documents", self.name)
        # Get a generator with that generates feature matrices from data
        matrices = FeatureMatrix[KT].generator_from_provider(
            self.env.unlabeled, self.batch_size)
        # Get the predictions for each matrix
        predictions = map(self._get_predictions, matrices)
        # Apply selections strategies on the predictions
        metric_results = map(self.process_predictions, predictions)
    
        # Convert each Tuple[Sequence[KT], np.ndarray] with the metric results to a list 
        metric_tuples = map(to_tuples, metric_results)

        # Chain them together
        metrics_chained = list(itertools.chain.from_iterable(metric_tuples))
        LOGGER.info("[%s] Calculated all metrics", self.name)
        # Sort the tuples in descending order, so that the key with the highest score
        # is on the first position of the list
        sorted_tuples = multisort(metrics_chained)
        LOGGER.info("[%s] Sorted all metrics", self.name)
        # Retrieve the keys from the tuples
        ordered_key_metric_pairs = [list_unzip(sublist) for sublist in sorted_tuples]
        return ordered_key_metric_pairs

    def update_ordering(self) -> bool:
        """Calculates the ordering for Machine Learning based AL methods
        
        Returns
        -------
        bool
            True if updating succeeded
        """
        
        try:
            self.retrain()
            orderings = self.calculate_ordering()
        except (NotFittedError, IndexError, 
                ValueError, NoLabeledDataException, 
                NoVectorsException) as ex:
            self._uses_fallback = True
            LOGGER.error("[%s] Falling back to model %s, because of: %s",
                                 self.name, self.fallback.name, ex, exc_info=ex)
            self.fallback.update_ordering()
            self._set_ordering([])
            return False
        else:
            self._uses_fallback = False
            # Update ordering for sublearners
            for (ord, metrics) , learner in zip(orderings, self.learners):
                assert isinstance(learner, FixedOrdering)
                learner.enter_ordering(ord, metrics)
            self._has_ordering = True
            return True


    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, np.ndarray, RT, LT]:
        """Internal functions that selects the next active learner for the next query

        Returns:
            ActiveLearner[IT, KT, DT, VT, RT, LT]: One of the learners from the ensemble
        """        
        idxs = np.arange(len(self.learners))
        al_idx: int = self._rng.choice(idxs, size=1, p=self.probabilities)[0]
        learner = self.learners[al_idx]
        return learner
    
    def __next__(self) -> IT:
        if not self._uses_fallback:
            result = super().__next__()
            while result.identifier in self.sampled:
                result = super().__next__()
            self.sampled.add(result.identifier)
            return result
        result = next(self.fallback)
        return result



class LabelProbEnsemble(ProbabilityBasedEnsemble[IT, KT, DT, RT, LT], 
                        Generic[IT, KT, DT, RT, LT]):

    def __init__(self,
                 classifier: AbstractClassifier[KT, np.ndarray, LT, np.ndarray, np.ndarray],
                 strategy: Callable[..., AbstractSelectionCriterion],
                 batch_size = 200,
                 rng: Any = None,
                 identifier: Optional[str] = None, 
                 fallback = RandomSampling[IT, KT, DT, np.ndarray, RT, LT](), *_, **__) -> None:
        super().__init__(classifier, [], [], 
                         batch_size=batch_size, 
                         identifier=identifier,
                         rng=rng, 
                         fallback=fallback)
        self._strategy_builder = strategy

    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, np.ndarray, RT, LT]) -> LabelProbEnsemble[IT, KT, DT, RT, LT]:
        super().__call__(environment)
        labelset = list(self.env.labels.labelset)
        self.label_dict = {label: idx for idx, label in enumerate(labelset)}
        label_columns = list(map(self.classifier.get_label_column_index, labelset))
        self.strategies = list(map(self._strategy_builder, label_columns))
        self.probabilities = get_probabilities(None, self.strategies)
        zipped = zip(labelset, self.strategies)
        self.learners = [
            FixedOrdering[IT, KT,DT, np.ndarray, RT, LT](
                identifier=strategy.name, label=label)(self.env) 
            for label, strategy in zipped
        ]
        return self

   

class LabelMinProbEnsemble(LabelProbEnsemble[IT, KT, DT, RT, LT], 
                             Generic[IT, KT, DT, RT, LT]):
    
    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, np.ndarray, RT, LT]:
        labelcounts = [(self.env.labels.document_count(label), label)
                       for label in self.env.labels.labelset]
        min_label = min(labelcounts)[1]
        al_idx = self.label_dict[min_label]
        learner = self.learners[al_idx]
        return learner
    
