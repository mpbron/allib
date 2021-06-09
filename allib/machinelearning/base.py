from __future__ import annotations

import functools
import os
import uuid
from abc import ABC, abstractmethod
from typing import (FrozenSet, Generic, Iterable, Iterator, List, Optional,
                    Sequence, Set, Tuple, TypeVar, Any)

from ..environment import AbstractEnvironment
from instancelib.instances import Instance

from ..typehints import KT, VT, LT, LVT, PVT


class AbstractClassifier(ABC, Generic[KT, VT, LT, LVT, PVT]):
    _name = "AbstractClassifier"

    @abstractmethod
    def __call__(self, 
            environment: AbstractEnvironment[Any, KT, Any, VT, Any, LT]
        ) -> AbstractClassifier[KT, VT, LT, LVT, PVT]:
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
    def predict_proba(self, x_data: VT) -> PVT:
        pass

    @abstractmethod
    def predict(self, x_data: VT) -> LVT:
        pass

    @abstractmethod
    def encode_labels(self, labels: Iterable[LT]) -> LVT:
        pass

    @abstractmethod
    def predict_instances(self, instances: Sequence[Instance[KT, Any, VT, Any]]) -> Sequence[FrozenSet[LT]]:
        pass

    @abstractmethod
    def fit_vectors(self, x_data: Sequence[VT], labels: Sequence[FrozenSet[LT]]):
        pass

    @abstractmethod
    def fit_instances(self, instances: Sequence[Instance[KT, Any, VT, Any]], labels: Sequence[FrozenSet[LT]]):
        pass

    @abstractmethod
    def predict_proba_instances(self, instances: Sequence[Instance[KT, Any, VT, Any]]) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        pass

    @property
    def name(self) -> str:
        return self._name
        
    @property
    @abstractmethod
    def fitted(self) -> bool:
        pass

    @abstractmethod
    def get_label_column_index(self, label: LT) -> int:
        raise NotImplementedError
