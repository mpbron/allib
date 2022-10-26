from __future__ import annotations

from typing import Generic, Sequence, TypeVar, Iterable, Dict, Any, Set
from joblib import Memory

import numpy as np
import numpy.typing as npt

from instancelib.instances.memory import MemoryBucketProvider
from instancelib.instances.hdf5 import HDF5Instance, HDF5Provider
from instancelib.labels.memory import MemoryLabelProvider
from ..history import MemoryLogger

from .base import AbstractEnvironment

# TODO Adjust MemoryEnvironment Generic Type (ADD ST)

class HDF5Environment(AbstractEnvironment[int, str, npt.NDArray[Any], str, str]):
    def __init__(
            self,
            dataset: HDF5Provider,
            unlabeled: HDF5Provider,
            labeled: HDF5Provider,
            labelprovider: MemoryLabelProvider[int, str],
            logger: MemoryLogger[int, str, Any],
            truth: MemoryLabelProvider[int, str]
        ):
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._named_providers: Dict[str, HDF5Provider] = dict()
        self._logger = logger
        self._storage: Dict[str, Any] = dict()
        self._truth = truth

    @classmethod
    def from_data(cls, 
            target_labels: Iterable[str], 
            indices: Sequence[int], 
            data: Sequence[str], 
            ground_truth: Sequence[Set[str]],
            data_location: str,
            vector_location: str) -> HDF5Environment:
        dataset = HDF5Provider.from_data_and_indices(indices, data, data_location, vector_location)
        unlabeled = MemoryBucketProvider(dataset, dataset.key_list)
        labeled = MemoryBucketProvider(dataset, [])
        labels = MemoryLabelProvider[int, str].from_data(target_labels, indices, [])
        logger = MemoryLogger[int, str, Any](labels)
        truth = MemoryLabelProvider[int, str].from_data(target_labels, indices, ground_truth)
        return cls(dataset, unlabeled, labeled, labels, logger, truth)

    @classmethod
    def from_environment(cls, 
                         environment: AbstractEnvironment[int, str, npt.NDArray[Any], str, str], 
                         data_location: str = "", vector_location: str = "", 
                         shared_labels: bool = True, *args, **kwargs) -> HDF5Environment:
        if isinstance(environment.dataset, HDF5Provider):
            dataset = environment.dataset
        else:
            dataset = HDF5Provider.from_provider(environment.dataset, data_location, vector_location)
        unlabeled = MemoryBucketProvider(dataset, environment.unlabeled.key_list)
        labeled = MemoryBucketProvider(dataset, environment.labeled.key_list)
        if isinstance(environment.labels, MemoryLabelProvider) and shared_labels:
            labels: MemoryLabelProvider[int, str] = environment.labels
        else:
            labels = MemoryLabelProvider[int, str](environment.labels.labelset, {}, {}) # type: ignore
        if isinstance(environment.logger, MemoryLogger):
            logger: MemoryLogger[int, str, Any] = environment.logger
        else:
            logger = MemoryLogger[int, str, Any](labels)
        if isinstance(environment.truth, MemoryLabelProvider):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[int, str](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, logger, truth)

    @classmethod
    def from_environment_only_data(cls, 
                                   environment: AbstractEnvironment[int, str, npt.NDArray[Any], str, str],
                                   data_location: str, vector_location: str) -> HDF5Environment:
        if isinstance(environment.dataset, HDF5Provider):
            dataset = environment.dataset
        else:
            dataset = HDF5Provider.from_provider(environment.dataset, data_location, vector_location)
        unlabeled = MemoryBucketProvider(dataset, environment.dataset.key_list)
        labeled = MemoryBucketProvider(dataset, [])
        labels = MemoryLabelProvider[int, str](environment.labels.labelset, {}, {}) # type: ignore
        if isinstance(environment.logger, MemoryLogger):
            logger: MemoryLogger[int, str, Any] = environment.logger
        else:
            logger = MemoryLogger[int, str, Any](labels)
        if isinstance(environment.truth, MemoryLabelProvider):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[int, str](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, logger, truth)

    def create_named_provider(self, name: str) -> HDF5Provider:
        self._named_providers[name] = MemoryBucketProvider(self._dataset, [])
        return self._named_providers[name]

    def get_named_provider(self, name: str) -> HDF5Provider:
        if name in self._named_providers:
            self.create_named_provider(name)
        return self._named_providers[name]

    def create_empty_provider(self) -> MemoryBucketProvider:
        return MemoryBucketProvider(self._dataset, [])

    @property
    def dataset(self):
        return self._dataset

    @property
    def unlabeled(self):
        return self._unlabeled

    @property
    def labeled(self):
        return self._labeled

    @property
    def labels(self):
        return self._labelprovider

    @property
    def truth(self):
        return self._truth

    @property
    def logger(self): # TODO Replace Any Type
        return self._logger

    def store(self, key: str, value: Any) -> None:
        self._storage[key] = value

    
    def storage_exists(self, key: str) -> bool:
        return key in self._storage

    def retrieve(self, key: str) -> Any:
        return self._storage[key]

    
    
    



        

