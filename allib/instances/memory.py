from __future__ import annotations
from ..utils.chunks import divide_iterable_in_lists

import itertools
from typing import (Any, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple,
                    TypeVar)

from .base import Instance, InstanceProvider

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
LVT = TypeVar("LVT")


class DataPoint(Instance[KT, DT, VT, DT], Generic[KT, DT, VT]):

    def __init__(self, identifier: KT, data: DT, vector: Optional[VT]) -> None:
        self._identifier = identifier
        self._data = data
        self._vector = vector

    @property
    def data(self) -> DT:
        return self._data

    @property
    def representation(self) -> DT:
        return self._data

    @property
    def identifier(self) -> KT:
        return self._identifier

    @property
    def vector(self) -> Optional[VT]:
        return self._vector

    @vector.setter
    def vector(self, value: Optional[VT]) -> None:  # type: ignore
        self._vector = value


class DataPointProvider(InstanceProvider[KT, DT, VT, DT], Generic[KT, DT, VT]):

    def __init__(self, datapoints: Iterable[DataPoint[KT, DT, VT]]) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}

    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[KT],
                              raw_data: Sequence[DT],
                              vectors: Optional[Sequence[Optional[VT]]] = None):
        if vectors is None or len(vectors) != len(indices):
            vectors = [None] * len(indices)
        datapoints = itertools.starmap(
            DataPoint[KT, DT, VT], zip(indices, raw_data, vectors))
        return cls(datapoints)

    @classmethod
    def from_data(cls, raw_data: Sequence[DT]) -> DataPointProvider[KT, DT, VT]:
        indices = range(len(raw_data))
        vectors = [None] * len(raw_data)
        datapoints = itertools.starmap(
            DataPoint[KT, DT, VT], zip(indices, raw_data, vectors))
        return cls(datapoints)

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
       
    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[Optional[VT]]]:
        vectors = [self[key].vector  for key in keys]
        return keys, vectors

    def bulk_get_all(self) -> List[Instance[KT, DT, VT, RT]]:
        return list(self.get_all())

    


class DataBucketProvider(DataPointProvider[KT, DT, VT], Generic[KT, DT, VT]):
    def __init__(self, dataset: DataPointProvider[KT, DT, VT], instances: Iterable[KT]):
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
