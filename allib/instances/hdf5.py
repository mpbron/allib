from __future__ import annotations

import itertools
from typing import (Any, Dict, Generic, Iterable, Iterator, Optional, Sequence, Tuple,
                    TypeVar)

from .base import Instance, InstanceProvider
from .memory import DataPoint
from .hdf5storage import HDF5VectorStorage

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
LVT = TypeVar("LVT")



class HDF5Provider(InstanceProvider[KT, DT, VT, DT], Generic[KT, DT, VT]):

    def __init__(self, data_storage: Dict[KT, DT], vector_storage_location: str) -> None:
        self.data_storage = data_storage
        self.vector_storage_location = vector_storage_location
        self.vectors = HDF5VectorStorage[KT](vector_storage_location)

    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[KT],
                              raw_data: Sequence[DT],
                              vector_storage_location: str):
        datapoints = {key: data for key, data in zip(indices, raw_data)}
        return cls(datapoints, vector_storage_location)

    @classmethod
    def from_data(cls, raw_data: Sequence[DT], vector_storage_location) -> HDF5Provider[KT, DT, VT]:
        indices = range(len(raw_data))
        return cls.from_data_and_indices(indices, raw_data, vector_storage_location)
    @classmethod
    def from_provider(cls, provider: InstanceProvider[KT, DT, VT, Any]) -> DataPointProvider[KT, DT, VT]:
        if isinstance(provider, DataPointProvider):
            return cls.copy(provider)
        instances = provider.bulk_get_all()
        datapoints = [DataPoint[KT, DT, VT](
            ins.identifier, ins.data, ins.vector) for ins in instances]
        return cls(datapoints)

    @classmethod
    def copy(cls, provider: DataPointProvider[KT, DT, VT]) -> DataPointProvider[KT, DT, VT]:
        instances = provider.bulk_get_all()
        return cls(instances)  # type: ignore

    def __iter__(self) -> Iterator[KT]:
        yield from self.dictionary.keys()

    def __getitem__(self, key: KT) -> DataPoint[KT, DT, VT]:
        return self.dictionary[key]

    def __setitem__(self, key: KT, value: Instance[KT, DT, VT, Any]) -> None:
        self.dictionary[key] = value  # type: ignore

    def __delitem__(self, key: KT) -> None:
        del self.dictionary[key]

    def __len__(self) -> int:
        return len(self.dictionary)

    def __contains__(self, key: object) -> bool:
        return key in self.dictionary

    @property
    def empty(self) -> bool:
        return not self.dictionary

    def get_all(self) -> Iterator[Instance[KT, DT, VT, DT]]:
        yield from list(self.values())

    def clear(self) -> None:
        self.dictionary = {}


class HDF5BucketProvider(InstanceProvider[KT, DT, VT, DT], Generic[KT, DT, VT]):
    def __init__(self, dataset: HDF5Provider[KT, DT, VT], instances: Iterable[KT]):
        self._elements = set(instances)
        self.dataset = dataset

    def __iter__(self) -> Iterator[KT]:
        yield from self._elements

    def __getitem__(self, key: KT) -> DataPoint[KT, DT, VT]:
        if key in self._elements:
            return self.dataset[key]
        raise KeyError(
            f"This datapoint with key {key} does not exist in this provider")

    def __setitem__(self, key: KT, value: Instance[KT, DT, VT, Any]) -> None:
        self._elements.add(key)
        self.dataset[key] = value  # type: ignore

    def __delitem__(self, key: KT) -> None:
        self._elements.discard(key)

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, key: object) -> bool:
        return key in self._elements

    @property
    def empty(self) -> bool:
        return not self._elements

    @classmethod
    def from_provider(cls, dataset: DataPointProvider[KT, DT, VT], provider: InstanceProvider[KT, DT, VT, Any]) -> DataBucketProvider[KT, DT, VT]:
        return cls(dataset, provider.key_list)

    @classmethod
    def copy(cls, provider: DataBucketProvider[KT, DT, VT]) -> DataBucketProvider[KT, DT, VT]:
        return cls(provider.dataset, provider.key_list)
