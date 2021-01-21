from __future__ import annotations
import collections

from allib.machinelearning.base import AbstractClassifier

import functools
import itertools
import logging
from abc import ABC, abstractmethod
from typing import (Any, Callable, Deque, Dict, FrozenSet, Generic, Iterator,
                    List, Optional, Sequence, Tuple, TypeVar, Union, Set)

from ..environment import AbstractEnvironment
from ..instances import Instance, InstanceProvider
from ..labels.base import LabelProvider

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
FT = TypeVar("FT")
F = TypeVar("F", bound=Callable[..., Any])

ProbabilityPrediction = FrozenSet[Tuple[LT, float]]
LabelPrediction = FrozenSet[LT]

LOGGER = logging.getLogger(__name__)

class NotInitializedException(Exception):
    pass

class NoOrderingException(Exception):
    pass


class ActiveLearner(ABC, Iterator[Instance[KT, DT, VT, RT]], Generic[KT, DT, VT, RT, LT]):
    
    _name = "ActiveLearner"
    ordering: Optional[Deque[KT]]
    _env: Optional[AbstractEnvironment[KT, DT, VT, RT, LT]]

    
    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        """Return the name of the Active Learner

        Returns
        -------
        Tuple[str, Optional[LT]]
            The tuple contains a name and optionally the label if it the learner
            optimizes for a specific label
        """          
        return self._name, None
   
    def __iter__(self) -> ActiveLearner[KT, DT, VT, RT, LT]:
        """The Active Learning class is an iterator, iterating
        over instances

        Returns
        -------
        ActiveLearner[KT, DT, VT, RT, LT]
            The same Active Learner is an iterator, so `iter(al) == al`
        """        
        return self

    @property
    def env(self) -> AbstractEnvironment[KT, DT, VT, RT, LT]:
        """Every ActiveLearner has an Environment that is based on the
        `allib.environment.base.AbstractEnvironment`. The Environment 
        contains the dataset, and the current label state (i.e., which 
        documents are labeled, and which labels do they have)

        Returns
        -------
        AbstractEnvironment[KT, DT, VT, RT, LT]
            The environment that is attached to this Active Learner

        Raises
        ------
        NotInitializedException
            If there is no environment attached
        """
        if self._env is None:
            raise NotInitializedException
        return self._env

    @abstractmethod
    def update_ordering(self) -> bool:
        """Update the ordering of the Active Learning method

        Returns
        -------
        bool
            True if updating the ordering succeeded
        """             
        raise NotImplementedError

    @property
    @abstractmethod
    def has_ordering(self) -> bool:
        """Returns true if an ordering has been established for this Active Learner

        Returns
        -------
            bool: True if an ordering has been established
        
        See Also
        --------
        update_ordering : The method to create or update the ordering
        """        
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Instance[KT, DT, VT, RT]:
        """Return the next instance based on the ordering

        Returns
        -------
        Instance[KT, DT, VT, RT]
            The most informative instance based on the learners ordering

        See Also
        --------
        __iter__() : Optional function for iterating over instances
        """        
        raise NotImplementedError
       
    @staticmethod
    def iterator_log(func: F) -> F:
        """A decorator that logs iterator calls

        Parameters
        ----------
        func : F
            The `__next__()` function that iterates over Instances

        Returns
        -------
        F
            The same function with a logger wrapped around it
        """        
        @functools.wraps(func)
        def wrapper(
                self: ActiveLearner[KT, DT, VT, RT, LT], 
                *args: Any, **kwargs: Dict[str, Any]) -> FT:
            result: Union[Any,Instance[KT, DT, VT, RT]] = func(self, *args, **kwargs)
            if isinstance(result, Instance):
                LOGGER.info("Sampled document %i with method %s",
                            result.identifier, self.name) # type: ignore
                self.env.logger.log_sample(result, self.name)
            return result # type: ignore
        return wrapper # type: ignore
    
    @staticmethod
    def label_log(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: ActiveLearner[KT, DT, VT, RT, LT], 
                    instance: Instance[KT, DT, VT, RT], 
                    *args: Any, **kwargs: Any):
            labels = self.env.labels.get_labels(instance)
            self.env.logger.log_label(instance, self.name,  *labels)
            return func(self, instance, *args, **kwargs)
        return wrapper # type: ignore

    @staticmethod
    def ordering_log(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: ActiveLearner, *args: Any, **kwargs: Dict[str, Any]) -> FT:
            ordering, ordering_metric = func(self, *args, **kwargs)
            self.env.logger.log_ordering(ordering, ordering_metric, 
                                         self.env.labeled.key_list,
                                         self.env.labels)
            return ordering, ordering_metric # type: ignore
        return wrapper # type: ignore

    @abstractmethod
    def __call__(
            self, 
            environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> ActiveLearner[KT, DT, VT, RT, LT]:
        raise NotImplementedError

    def query(self) -> Optional[Instance[KT, DT, VT, RT]]:
        """Query the most informative instance
        Returns
        -------
        Optional[Instance]
            The most informative instance
        """
        return next(self, None)

   
    def query_batch(self, batch_size: int) -> List[Instance[KT, DT, VT, RT]]:
        """Query the `batch_size` most informative instances

        Parameters
        ----------
        batch_size : int
            The size of the batch

        Returns
        -------
        List[Instance]
            A batch with `len(batch) <= batch_size` 
        """
        return list(itertools.islice(self, batch_size))

    @abstractmethod
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as labeled

        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        raise NotImplementedError

    @abstractmethod
    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as unlabeled

        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        raise NotImplementedError    

    @property
    def len_unlabeled(self) -> int:
        """Return the number of unlabeled documents

        Returns
        -------
        int
            The number of labeled documents
        """
        return len(self.env.unlabeled)
    
    @property
    def len_labeled(self) -> int:
        """Return the number of labeled documents

        Returns
        -------
        int
            The number of labeled documents
        """
        return len(self.env.labeled)

    @property
    def size(self) -> int:
        """The number of initial unlabeled documents

        Returns
        -------
        int
            The number of unlabeled documents
        """
        return self.len_labeled + self.len_unlabeled


    @property
    def ratio_learned(self) -> float:
        """The labeling progress sofar

        Returns
        -------
        float
            the ratio
        """
        return self.len_labeled / self.size