from __future__ import annotations
from allib import labels

from typing import Generic, Sequence, Set, TypeVar, Iterable, Dict, Any

import numpy as np # type: ignore

from ..instances import DataPoint, DataPointProvider, Instance
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
            logger: MemoryLogger[KT, LT, Any]
        ):
        super().__init__()
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._providers = [self._dataset, self._unlabeled, self._labeled]
        self._named_providers: Dict[str, DataPointProvider[KT, DT, VT]] = dict()
        self._logger = logger

    @classmethod
    def from_data(cls, 
            target_labels: Iterable[LT], 
            indices: Sequence[KT], 
            data: Sequence[DT], 
            vectors: Sequence[VT]) -> MemoryEnvironment[KT, DT, VT, LT]:
        dataset = DataPointProvider.from_data_and_indices(indices, data, vectors)
        unlabeled = DataPointProvider.copy(dataset)
        labeled = DataPointProvider[KT, DT, VT]([])
        labelprovider = MemoryLabelProvider.from_data(target_labels, indices, [])
        logger = MemoryLogger[KT, LT, Any]()
        return cls(dataset, unlabeled, labeled, labelprovider, logger)

    @classmethod
    def from_environment(cls, environment: AbstractEnvironment[KT, DT, VT, DT, LT], shared_labels: bool = True) -> MemoryEnvironment[KT, DT, VT, LT]:
        dataset = DataPointProvider[KT, DT, VT].from_provider(environment.dataset)
        unlabeled = DataPointProvider[KT, DT, VT].from_provider(environment.unlabeled)
        labeled = DataPointProvider[KT, DT, VT].from_provider(environment.labeled)
        if isinstance(environment.labels, MemoryLabelProvider) and shared_labels:
            labels: MemoryLabelProvider[KT, LT] = environment.labels
        else:
            labels = MemoryLabelProvider[KT, LT].from_provider(environment.labels) # type: ignore
        if isinstance(environment.logger, MemoryLogger):
            logger: MemoryLogger[KT, LT, Any] = environment.logger
        else:
            logger = MemoryLogger[KT, LT, Any]()
        return cls(dataset, unlabeled, labeled, labels, logger)

    def create_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT]:
        self._named_providers[name] = DataPointProvider[KT, DT, VT]([])
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
    def logger(self) -> MemoryLogger[KT, LT, Any]: # TODO Replace Any Type
        return self._logger

    def set_vectors(self,
        instances: Sequence[Instance[KT, DT, VT, DT]], matrix: np.ndarray):
        vectors = matrix.tolist()
        for i, instance in enumerate(instances):
            key = instance.identifier
            instance.vector = np.array(vectors[i]).reshape(1, -1) # type: ignore
            for provider in self._providers:
                if key in provider and isinstance(instance, DataPoint):
                    provider[key] = instance


        

