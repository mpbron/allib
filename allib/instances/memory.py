from __future__ import annotations

import itertools

from typing import Optional, Iterator, Generic, TypeVar, Sequence, Iterable

from .base import Instance, InstanceProvider

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
LVT = TypeVar("SKT")

class DataPoint(Instance, Generic[KT, VT, DT]):

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
    def vector(self, value: Optional[VT]) -> None:
        self._vector = value

class DataPointProvider(InstanceProvider, Generic[KT, VT, DT]):

    def __init__(self, datapoints: Iterable[DataPoint]) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}
        self._feature_matrix = None

    @classmethod
    def from_data_and_indices(cls, 
                  indices: Sequence[KT], 
                  raw_data: Sequence[DT], 
                  vectors: Optional[Sequence[VT]] = None):
        if vectors is None or len(vectors) != len(indices):
            vectors = [None] * len(indices)
        datapoints = itertools.starmap(DataPoint, zip(indices, raw_data, vectors))
        return cls(datapoints)

    @classmethod
    def from_data(cls, raw_data: Sequence[DT]):
        indices = range(len(raw_data))
        vectors = [None] * len(raw_data)
        datapoints = itertools.starmap(DataPoint, zip(indices, raw_data, vectors))
        return cls(datapoints)

    @classmethod
    def from_provider(cls, provider: InstanceProvider):
        instances = provider.bulk_get_all()
        return cls(instances)
        

    def __iter__(self) -> Iterator[KT]:
        yield from self.dictionary.keys()

    def __getitem__(self, key: KT) -> DataPoint:
        return self.dictionary[key]
    
    def __setitem__(self, key: KT, value: DataPoint) -> None:
        self.dictionary[key] = value

    def __delitem__(self, key: KT) -> None:
        del self.dictionary[key]

    def __len__(self) -> int:
        return len(self.dictionary)

    def __contains__(self, key: KT) -> bool:
        return key in self.dictionary

    @property
    def empty(self) -> bool:
        return not self.dictionary

    def get_all(self) -> Iterator[Instance]:
        yield from list(self.values())

    def clear(self) -> None:
        self.dictionary = {}

    