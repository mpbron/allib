from __future__ import annotations
import collections

from allib.machinelearning.base import AbstractClassifier

import logging
from abc import ABC, abstractmethod
from typing import (Any, Callable, Deque, Dict, FrozenSet, Generic, Iterator,
                    List, Optional, Sequence, Tuple, TypeVar, Union, Set)

from .base import ActiveLearner
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
class PoolBasedAL(ActiveLearner[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self, *_, **__) -> None:
        self.initialized = False
        self._env: Optional[AbstractEnvironment[KT, DT, VT, RT, LT]] = None
        self.ordering = None
        self.sampled: Set[KT] = set()

    def __call__(self, environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> PoolBasedAL[KT, DT, VT, RT, LT]:
        self._env = environment
        self.initialized = True
        return self

    def update_ordering(self) -> None:
        self.ordering = collections.deque(self.env.unlabeled.key_list)

    @ActiveLearner.iterator_log
    def __next__(self) -> Instance[KT, DT, VT, RT]:
        if self.ordering is None:
            self.update_ordering()
        try:
            key = self.ordering.popleft()
            while key not in self.env.unlabeled or key in self.sampled:
                key = self.ordering.popleft()
            self.sampled.add(key)
            return self.env.unlabeled[key]
        except IndexError:
            raise StopIteration()

    @ActiveLearner.label_log
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.unlabeled.discard(instance)
        self.env.labeled.add(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as unlabeled
        
        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        self.env.labeled.discard(instance)
        self.env.unlabeled.add(instance)