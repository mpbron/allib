from __future__ import annotations

import collections
import functools
import itertools
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from queue import Queue
from typing import (Any, Callable, Dict, FrozenSet, Generic, Iterator,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
from instancelib.instances.base import Instance, InstanceProvider
from sklearn.exceptions import NotFittedError  # type: ignore

from ..environment import AbstractEnvironment
from ..exceptions import NoOrderingException, NotInitializedException
from ..exceptions.base import NoLabeledDataException, NoVectorsException
from instancelib.machinelearning import AbstractClassifier
from ..utils import divide_sequence, mapsnd
from ..utils.func import filter_snd_none, list_unzip, sort_on
from .base import ActiveLearner
from .poolbased import PoolBasedAL
from .random import RandomSampling
from ..typehints import IT

from instancelib.typehints import KT, DT, VT, RT, LT, LMT, PMT

FT = TypeVar("FT")

LOGGER = logging.getLogger(__name__)


class ILMLBased(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT, LMT, PMT]):
    classifier: AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT]

    def __init__(self,
                 classifier: AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT],
                 fallback: PoolBasedAL[IT, KT, DT, VT, RT, LT] = RandomSampling[IT, KT, DT, VT, RT, LT](),
                 batch_size = 200,
                 *_, identifier: Optional[str] = None, **__
                 ) -> None:
        """Initialize an Machine Learning Based Active Learning method

        Parameters
        ----------
        classifier : AbstractClassifier[KT, VT, LT, LVT, PVT]
            The classifier
        fallback : PoolBasedAL[KT, DT, VT, RT, LT], optional
            If fitting a classifier is not possible choose 
            this ActiveLearner for sampling, 
            by default ``RandomSampling[KT, DT, VT, RT, LT]()``
        batch_size : int, optional
            The batch size for the feature matrix, by default 200
        """        
        super().__init__(identifier=identifier)
        self.fitted = False
        self.classifier = classifier
        self.fallback = fallback
        self.batch_size = batch_size
        self._uses_fallback = False

    @property
    def uses_fallback(self) -> bool:
        """Returns true if the fallback method is used 
        instead of the ML-based method

        Returns
        -------
        bool
            True if the model uses the fallback method
        """        
        return self._uses_fallback

    def __call__(self, 
            environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]
        ) -> ILMLBased[IT, KT, DT, VT, RT, LT, LMT, PMT]:
        super().__call__(environment)
        self.fallback = self.fallback(self.env)
        self.classifier.set_target_labels(self.env.labels.labelset)
        return self

    def retrain(self) -> None:
        """Retrain the classifier using the information in the
        labeled set

        Raises
        ------
        NotInitializedException
            If the AL method has no attached Environment
        """        
        self.classifier.fit_provider(self.env.labeled, self.env.labels)

    def predict(self, 
                instances: Sequence[Instance[KT, DT, VT, RT]]
               ) -> Sequence[FrozenSet[LT]]:
        """Predict the labels for the provided instances

        Parameters
        ----------
        instances : Sequence[Instance[KT, DT, VT, RT]]
            A sequence of instances. The instances should all
            have a vector of type `VT`; they should not be None

        Returns
        -------
        Sequence[FrozenSet[LT]]
            A list of labelings, matching the order of the `instances` parameters

        Raises
        ------
        NotInitializedException
            If the model has no attached Environment
        """        
        if not self.initialized:
            raise NotInitializedException
        _, labels = list_unzip(self.classifier.predict(instances))
        return labels
    
    def predict_proba(self, 
                      instances: Sequence[Instance[KT, DT, VT, RT]]
                      ) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        """Predict the labels and their probability 
        for the provided instances

        Parameters
        ----------
        instances : Sequence[Instance[KT, DT, VT, RT]]
            A sequence of instances. The instances should all
            have a vector of type `VT`; they should not be None

        Returns
        -------
        Sequence[FrozenSet[Tuple[LT, float]]]
            A list of labelings and probabilities, 
            matching the order of the `instances` parameters

        Raises
        ------
        NotInitializedException
            If the model has no attached Environment
        """        
        if not self.initialized:
            raise NotInitializedException
        _, labels = list_unzip(self.classifier.predict_proba(instances))
        return labels

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        return f"{self._name} :: {self.classifier.name}", None

    @staticmethod
    def iterator_fallback(func: Callable[..., IT]
                        ) -> Callable[..., IT]:
        """A decorator for fallback. If the ``__next__`` function
        fails, the fallback model's ``__next__`` will be used instead.

        The model catches the following exceptions:

        - :class:`sklearn.exceptions.NotFittedError` the model is not fitted
        - :class:`IndexError` e.g., there is no train data
        - :class:`ValueError` e.g., the train data has an incorrect format
        - :class:`StopIteration` e.g., there are no unlabeled documents
        - :class:`~allib.exceptions.base.NoOrderingException` an ordering 
            could not be established

        Parameters
        ----------
        func : Callable[..., Instance[KT, DT, VT, RT]]
            The :meth:`__next__` method of a learner, that may fail

        Returns
        -------
        Callable[..., Instance[KT, DT, VT, RT]]
            The same method with an fallback
        """        
        @functools.wraps(func)
        def wrapper(self: ILMLBased[IT, KT, DT, VT, RT, LT, LMT, PMT],
                    *args: Any,
                    **kwargs: Dict[str, Any]) -> IT:
            if not self.uses_fallback:
                try:
                    return func(self, *args, **kwargs)
                except (NotFittedError, IndexError, ValueError, 
                        StopIteration, NoOrderingException, NoVectorsException) as ex:
                    LOGGER.error("[%s] Falling back to model %s, because of: %s",
                                 self.name, self.fallback.name, ex, exc_info=ex)
            LOGGER.warn("[%s] Falling back to model %s, because it is not fitted", self.name, self.fallback.name)
            fallback_value = next(self.fallback)
            return fallback_value
        return wrapper


class AbstractSelectionCriterion(ABC):
    name: str = "AbstractSelectionCriterion"

    def __init__(self, *_, **__) -> None:
        pass

    @abstractmethod
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        """Calculates the selection metric given a probability matrix

        Parameters
        ----------
        prob_mat : np.ndarray
            The probability matrix with rows of class probabilities. 
            Shape should be ``(n_instances, n_classes)``

        Returns
        -------
        np.ndarray
            The result of the selection metrix. This has as shape
            ``(n_instances, )``

        """
        raise NotImplementedError

class ILProbabilityBased(ILMLBased[IT, KT, DT, np.ndarray, RT, LT, np.ndarray, np.ndarray], ABC, Generic[IT, KT, DT, RT, LT]):
    """
    An Active Learner that uses information in the probability matrix (i.e., the
    results of the classifier on the *unlabeled*  set of instances).

    Examples
    --------
    Usage:

    >>> classifier = SklearnClassifier(MultinomialNB(), LabelBinarizer())
    >>> sel_crit = EntropySampling() # Uncertainty Sampling
    >>> al = ProbabilityBased(classifier, sel_crit)
    >>> # Assume env contains your environment
    >>> al = al(env)
    >>> # Update the ordering
    >>> al.update_ordering()
    >>> # Retrieve the next document
    >>> instance = next(al)
    >>> # Label the document as Positive
    >>> label = "Positive"
    >>> al.env.labels.set_labels(instance, doc_label)
    >>> al.set_as_labeled(instance)
    """
    _name  = "ProbabilityBased"
    
    def __init__(self,
                 classifier: AbstractClassifier[IT, KT, DT, np.ndarray, RT, LT, np.ndarray, np.ndarray],
                 selection_criterion: AbstractSelectionCriterion,
                 fallback: PoolBasedAL[IT, KT, DT, np.ndarray, RT, LT] = RandomSampling[IT, KT, DT, np.ndarray, RT, LT](),
                 batch_size: int = 200,
                 *_, identifier: Optional[str] = None, **__
                 ) -> None:
        super().__init__(classifier, fallback, batch_size, identifier=identifier)
        self._selection_criterion = selection_criterion

    def selection_criterion(self, prob_mat: np.ndarray) -> np.ndarray:
        """Calculate the internal selection criterion for the given 
        probability matrix

        Parameters
        ----------
        prob_mat : np.ndarray
            The probability matrix with rows of class probabilities. 
            Shape should be ``(n_instances, n_classes)``

        Returns
        -------
        np.ndarray
            The result of the selection metrix. This has as shape
            ``(n_instances, )``
        """        
        return self._selection_criterion(prob_mat)

    #@ActiveLearner.ordering_log
    def calculate_ordering(self) -> Tuple[Sequence[KT], Sequence[float]]:
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
        def get_metric_tuples(keys: Sequence[KT], vec: np.ndarray) -> Sequence[Tuple[KT, float]]:
            floats: Sequence[float] = vec.tolist()
            return list(zip(keys, floats))        
        # Get the predictions for each matrix
        predictions = self.classifier.predict_proba_raw(self.env.unlabeled, self.batch_size)
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
        LOGGER.info("[%s] Calculated all metrics", self.name)
        # Sort the tuples in descending order, so that the key with the highest score
        # is on the first position of the list
        sorted_tuples = sort_on(1, metric_tuples)
        LOGGER.info("[%s] Sorted all metrics", self.name)
        # Retrieve the keys from the tuples
        ordered_keys, ordered_metrics = list_unzip(sorted_tuples)
        return ordered_keys, ordered_metrics

    def update_ordering(self) -> bool:
        """Calculates the ordering for Machine Learning based AL methods
        
        Returns
        -------
        bool
            True if updating succeeded
        """        
        try:
            self.retrain()
            ordering, _ = self.calculate_ordering()
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
            self._set_ordering(ordering)
            return True

    @MLBased.iterator_fallback # type: ignore
    def __next__(self) -> IT:
        value = super().__next__()
        return value
