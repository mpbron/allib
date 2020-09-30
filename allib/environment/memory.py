from __future__ import annotations
from allib import labels

from typing import Generic, Sequence, Set, TypeVar, Iterable, Dict

import numpy as np # type: ignore

from ..instances import DataPoint, DataPointProvider, Instance
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

KT = TypeVar("KT")
LT = TypeVar("LT")
VT = TypeVar("VT")
DT = TypeVar("DT")


class MemoryEnvironment(AbstractEnvironment[KT, DT, VT, DT, LT], Generic[KT, DT, VT, LT]):
    def __init__(
            self,
            dataset: DataPointProvider[KT, DT, VT],
            unlabeled: DataPointProvider[KT, DT, VT],
            labeled: DataPointProvider[KT, DT, VT],
            labelprovider: MemoryLabelProvider[KT, LT]
        ):
        super().__init__()
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._providers = [self._dataset, self._unlabeled, self._labeled]
        self._named_providers: Dict[str, DataPointProvider] = dict()
    
    @classmethod
    def from_data(cls, 
            target_labels: Iterable[LT], 
            indices: Sequence[KT], 
            data: Sequence[DT], 
            vectors: Sequence[VT]) -> MemoryEnvironment[KT, DT, VT, LT]:
        dataset = DataPointProvider.from_data_and_indices(indices, data, vectors)
        unlabeled = DataPointProvider.from_data_and_indices(indices, data, vectors)
        labeled = DataPointProvider[KT, DT, VT]([])
        labelprovider = MemoryLabelProvider.from_data(target_labels, indices, [])
        return cls(dataset, unlabeled, labeled, labelprovider)

    @classmethod
    def from_environment(cls, environment: AbstractEnvironment) -> AbstractEnvironment:
        dataset = DataPointProvider[KT, DT, VT].from_provider(environment.dataset)
        unlabeled = DataPointProvider[KT, DT, VT].from_provider(environment.unlabeled)
        labeled = DataPointProvider[KT, DT, VT].from_provider(environment.labeled)
        if isinstance(environment.labels, MemoryLabelProvider):
            labels = environment.labels
        else:
            labels = MemoryLabelProvider.from_provider(environment.labels)
        return cls(dataset, unlabeled, labeled, labels)

    def create_named_provider(self, name) -> DataPointProvider[KT, DT, VT]:
        self._named_providers[name] = DataPointProvider[KT, DT, VT]([])
        return self._named_providers[name]

    def get_named_provider(self, name) -> DataPointProvider[KT, DT, VT]:
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

    def set_vectors(self,
        instances: Sequence[Instance[KT, DT, VT, DT]], matrix: np.ndarray):
        vectors = matrix.tolist()
        for i, instance in enumerate(instances):
            key = instance.identifier
            instance.vector = np.array(vectors[i]).reshape(1, -1)
            for provider in self._providers:
                if key in provider and isinstance(instance, DataPoint):
                    provider[key] = instance


        

