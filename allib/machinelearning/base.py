from __future__ import annotations

from typing import Iterable, Iterator, TypeVar, Generic, Set, List, Tuple, Optional, Sequence, FrozenSet
from abc import ABC, abstractmethod

import functools
import uuid
import os

from ..environment import AbstractEnvironment
from ..instances import Instance

VT = TypeVar("VT")
LVT = TypeVar("LVT")
LT = TypeVar("LT")
PVT = TypeVar("PVT")

class AbstractClassifier(ABC, Generic[VT, LVT, LT]):
    name = "AbstractClassifier"

    @abstractmethod
    def __call__(self, environment: AbstractEnvironment) -> AbstractClassifier:
        """Initialize the classifier by supplying the target labels
        
        Parameters
        ----------
        target_labels : Set[LT]
            A set with number of labels
        
        Returns
        -------
        AbstractClassifier
            [description]
        """        
        raise NotImplementedError

    @abstractmethod
    def fit(self, x_data: Sequence[VT], y_data: Sequence[LVT]):
        pass

    @abstractmethod
    def predict_proba(self, x_data: Sequence[VT]) -> PVT:
        pass

    @abstractmethod
    def predict(self, x_data: Sequence[VT]) -> Sequence[LVT]:
        pass

    @abstractmethod
    def encode_labels(self, labels: Iterable[LT]) -> LVT:
        pass

    @abstractmethod
    def predict_instances(self, instances: Sequence[Instance]) -> Sequence[LT]:
        pass

    @abstractmethod
    def fit_instances(self, instances: Sequence[Instance], labels: Sequence[Set[LT]]):
        pass

    @abstractmethod
    def predict_proba_instances(self, instances: Sequence[Instance]) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        pass

    @property
    @abstractmethod
    def fitted(self) -> bool:
        pass

    @abstractmethod
    def get_label_column_index(self, label: LT) -> int:
        raise NotImplementedError