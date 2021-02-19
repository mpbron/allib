from __future__ import annotations

from typing import Generic, Sequence, TypeVar, Iterable, Dict, Any, Set

import numpy as np # type: ignore

from ..instances import DataPointProvider, DataBucketProvider
from ..labels.memory import MemoryLabelProvider
from ..history import MemoryLogger

from .base import AbstractEnvironment

KT = TypeVar("KT")
LT = TypeVar("LT")
VT = TypeVar("VT")
DT = TypeVar("DT")

# TODO Adjust MemoryEnvironment Generic Type (ADD ST)

class MemoryEnvironment(AbstractEnvironment[KT, DT, VT, DT, LT], Generic[KT, DT, VT, LT]):
    def __init__(
            self,
            dataset: DataPointProvider[KT, DT, VT],
            unlabeled: DataPointProvider[KT, DT, VT],
            labeled: DataPointProvider[KT, DT, VT],
            labelprovider: MemoryLabelProvider[KT, LT],
            logger: MemoryLogger[KT, LT, Any],
            truth: MemoryLabelProvider[KT, LT]
        ):
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._named_providers: Dict[str, DataPointProvider[KT, DT, VT]] = dict()
        self._logger = logger
        self._storage: Dict[str, Any] = dict()
        self._truth = truth

    @classmethod
    def from_data(cls, 
            target_labels: Iterable[LT], 
            indices: Sequence[KT], 
            data: Sequence[DT], 
            ground_truth: Sequence[Iterable[LT]],
            vectors: Sequence[VT]) -> MemoryEnvironment[KT, DT, VT, LT]:
        dataset = DataPointProvider[KT, DT, VT].from_data_and_indices(indices, data, vectors)
        unlabeled = DataBucketProvider[KT, DT, VT](dataset, dataset.key_list)
        labeled = DataBucketProvider[KT, DT, VT](dataset, [])
        labels = MemoryLabelProvider[KT, LT].from_data(target_labels, indices, [])
        logger = MemoryLogger[KT, LT, Any](labels)
        truth = MemoryLabelProvider[KT, LT].from_data(target_labels, indices, ground_truth)
        return cls(dataset, unlabeled, labeled, labels, logger, truth)

    @classmethod
    def from_environment(cls, environment: AbstractEnvironment[KT, DT, VT, Any, LT], shared_labels: bool = True, *args, **kwargs) -> MemoryEnvironment[KT, DT, VT, LT]:
        if isinstance(environment.dataset, DataPointProvider):
            dataset: DataPointProvider[KT, DT, VT] = environment.dataset
        else:
            dataset = DataPointProvider[KT, DT, VT].from_provider(environment.dataset)
        unlabeled = DataBucketProvider[KT, DT, VT](dataset, environment.unlabeled.key_list)
        labeled = DataBucketProvider[KT, DT, VT](dataset, environment.labeled.key_list)
        if isinstance(environment.labels, MemoryLabelProvider) and shared_labels:
            labels: MemoryLabelProvider[KT, LT] = environment.labels
        else:
            labels = MemoryLabelProvider[KT, LT](environment.labels.labelset, {}, {}) # type: ignore
        if isinstance(environment.logger, MemoryLogger):
            logger: MemoryLogger[KT, LT, Any] = environment.logger
        else:
            logger = MemoryLogger[KT, LT, Any](labels)
        if isinstance(environment, MemoryEnvironment):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[KT, LT](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, logger, truth)

    @classmethod
    def from_environment_only_data(cls, environment: AbstractEnvironment[KT, DT, VT, Any, LT]) -> MemoryEnvironment[KT, DT, VT, LT]:
        if isinstance(environment.dataset, DataPointProvider):
            dataset: DataPointProvider[KT, DT, VT] = environment.dataset
        else:
            dataset = DataPointProvider[KT, DT, VT].from_provider(environment.dataset)
        unlabeled = DataBucketProvider[KT, DT, VT](dataset, environment.dataset.key_list)
        labeled = DataBucketProvider[KT, DT, VT](dataset, [])
        labels = MemoryLabelProvider[KT, LT](environment.labels.labelset, {}, {})
        logger = MemoryLogger[KT, LT, Any](labels)
        if isinstance(environment, MemoryEnvironment):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[KT, LT](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, logger, truth)

    def create_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT]:
        self._named_providers[name] = DataBucketProvider[KT, DT, VT](self._dataset, [])
        return self._named_providers[name]

    def get_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT]:
        if name in self._named_providers:
            self.create_named_provider(name)
        return self._named_providers[name]

    def create_empty_provider(self) -> DataPointProvider[KT, DT, VT]:
        return DataPointProvider([])

    @property
    def dataset(self) -> DataPointProvider[KT, DT, VT]:
        return self._dataset

    @property
    def unlabeled(self) -> DataPointProvider[KT, DT, VT]:
        return self._unlabeled

    @property
    def labeled(self) -> DataPointProvider[KT, DT, VT]:
        return self._labeled

    @property
    def labels(self) -> MemoryLabelProvider[KT, LT]:
        return self._labelprovider

    @property
    def truth(self) -> MemoryLabelProvider[KT, LT]:
        return self._truth

    @property
    def logger(self) -> MemoryLogger[KT, LT, Any]: # TODO Replace Any Type
        return self._logger

    def store(self, key: str, value: Any) -> None:
        self._storage[key] = value

    
    def storage_exists(self, key: str) -> bool:
        return key in self._storage

    
    def retrieve(self, key: str) -> Any:
        return self._storage[key]

    
    
    



        

