from __future__ import annotations
import collections

from allib.machinelearning.base import AbstractClassifier

import logging
from abc import ABC, abstractmethod
from typing import (Any, Callable, Deque, Dict, FrozenSet, Generic, Iterator,
                    List, Optional, Sequence, Tuple, TypeVar, Union, Set)

from .base import ActiveLearner, NoOrderingException
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

LOGGER = logging.getLogger(__name__)

class PoolBasedAL(ActiveLearner[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self, *_, **__) -> None:
        self.initialized = False
        self._env: Optional[AbstractEnvironment[KT, DT, VT, RT, LT]] = None
        self.ordering = None
        self.sampled: Set[KT] = set()

    def __call__(self, environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> PoolBasedAL[KT, DT, VT, RT, LT]:
        """Initialze the active learner with an environment. After the environment is attached, the learner is 
        usable for sampling.
        
        Parameters
        ----------
        environment : AbstractEnvironment[KT, DT, VT, RT, LT]
            The environment that should be attached
        
        Returns
        -------
        PoolBasedAL[KT, DT, VT, RT, LT]
            A PoolBased Active Learning object with an attached environment
        """        
        self._env = environment
        self.initialized = True
        return self

    def _set_ordering(self, ordering: Sequence[KT]) -> None:
        """Set the ordering of the learner and clear the sampled set
        
        Parameters
        ----------
        ordering : Sequence[KT]
            The new ordering
        """        
        LOGGER.info("Setting a new ordering for %s", self.name)   
        self.ordering = collections.deque(ordering) # type: ignore
        self.sampled.clear()

    def update_ordering(self) -> bool:
        """Update the ordering of the active learner
        """
        ordering = self.env.unlabeled.key_list
        self._set_ordering(ordering)
        return True

    @property
    def has_ordering(self) -> bool:
        """Check if the learner has an ordering
        
        Returns
        -------
        bool
            True if it has an ordering
        """        
        return self.ordering is not None

    @ActiveLearner.iterator_log
    def __next__(self) -> Instance[KT, DT, VT, RT]:
        """Query the next instance according to the ordering
        
        Returns
        -------
        Instance[KT, DT, VT, RT]
            The next instance that should be labeled
        
        Raises
        ------
        StopIteration
            If all instances are labeled, we throw a stop iteration
        NoOrderingException
            If the ordering cannot be determined
        """        
        if self.ordering is not None:
            try:
                # Pop the most left item from the deque
                key = self.ordering.popleft()
                # If it has already been sampled, repeat
                while key in self.sampled:
                    key = self.ordering.popleft()
            except IndexError:
                # There are no more items left, raise StopIteration so iterator knows it should stop
                raise StopIteration()
            else:
                # Mark the instance as sampled
                self.sampled.add(key)
                # Retrieve the instance from the dataset
                result = self.env.dataset[key]
                return result
        raise NoOrderingException()
            


    @ActiveLearner.label_log
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as labeled
        
        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be marked as labeled
        """        
        self.env.unlabeled.discard(instance)
        self.env.labeled.add(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as unlabeled
        
        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be marked as unlabeled
        """        
        self.env.labeled.discard(instance)
        self.env.unlabeled.add(instance)