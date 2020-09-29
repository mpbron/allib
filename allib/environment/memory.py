from __future__ import annotations

from typing import Generic, Sequence, Set, TypeVar

import numpy as np

from ..instances import Instance, DataPointProvider
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

KT = TypeVar("KT")
LT = TypeVar("LT")
VT = TypeVar("VT")
DT = TypeVar("DT")


class MemoryEnvironment(AbstractEnvironment, Generic[KT, LT, VT, DT]):
    def __init__(
            self,
            dataset: DataPointProvider,
            unlabeled: DataPointProvider,
            labeled: DataPointProvider,
            labelprovider: MemoryLabelProvider
        ):
        super().__init__()
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._providers = [self._dataset, self._unlabeled, self._labeled]
        self._named_providers = dict()
    
    @classmethod
    def from_data(cls, indices, data, vectors, labels):
        dataset = DataPointProvider.from_data_and_indices(indices, data, vectors)
        unlabeled = DataPointProvider.from_data_and_indices(indices, data, vectors)
        labeled = DataPointProvider([])
        labelprovider = MemoryLabelProvider(labels, indices, [])
        return cls(dataset, unlabeled, labeled, labelprovider)

    @classmethod
    def from_environment(cls, environment: AbstractEnvironment) -> AbstractEnvironment:
        dataset = DataPointProvider.from_provider(environment.dataset_provider)
        unlabeled = DataPointProvider.from_provider(environment.unlabeled_provider)
        labeled = DataPointProvider.from_provider(environment.labeled_provider)
        labelprovider = MemoryLabelProvider.from_provider(environment.label_provider)
        return cls(dataset, unlabeled, labeled, labelprovider)

    def create_named_provider(self, name) -> DataPointProvider:
        self._named_providers[name] = DataPointProvider([])

    def get_named_provider(self, name) -> DataPointProvider:
        if name in self._named_providers:
            self.create_named_provider(name)
        return self._named_providers[name]

    def create_empty_provider(self) -> DataPointProvider:
        return DataPointProvider([])

    @property
    def dataset_provider(self) -> DataPointProvider:
        return self._dataset

    @property
    def unlabeled_provider(self) -> DataPointProvider:
        return self._unlabeled

    @property
    def labeled_provider(self) -> DataPointProvider:
        return self._labeled

    @property
    def label_provider(self) -> MemoryLabelProvider:
        return self._labelprovider

    def set_vectors(self,
        instances: Sequence[Instance], matrix: VT):
        vectors = matrix.tolist()
        for i, instance in enumerate(instances):
            key = instance.identifier
            instance.vector = np.array(vectors[i]).reshape(1, -1)
            for provider in self._providers:
                if key in provider:
                    provider[key] = instance


        

