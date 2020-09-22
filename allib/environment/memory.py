from typing import Generic, Sequence, Set, TypeVar

import numpy as np

from instances import Instance
from instances.memory import DataPointProvider
from labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

KT = TypeVar("KT")
LT = TypeVar("LT")
VT = TypeVar("VT")
DT = TypeVar("DT")


class MemoryEnvironment(AbstractEnvironment, Generic[KT, LT, VT, DT]):
    def __init__(
            self,
            indices: Sequence[KT],
            data: Sequence[DT],
            vectors: Sequence[VT],
            labelset: Set[LT]):
        super().__init__()
        self.indices = indices
        self.data = data
        self.vectors = vectors
        self.labelset = labelset
        self._dataset = DataPointProvider(self.indices, self.data, self.vectors)
        self._unlabeled = DataPointProvider(self.indices, self.data, self.vectors)
        self._labeled = self.create_empty_provider()
        self._labelprovider = MemoryLabelProvider(self.labelset, [], [])
        self._providers = [self._dataset, self._unlabeled, self._labeled]

    def create_empty_provider(self) -> DataPointProvider:
        return DataPointProvider([], [], [])

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
