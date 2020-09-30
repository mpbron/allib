from __future__ import annotations

import itertools

from typing import Optional, Iterator, Generic, TypeVar, Sequence, Iterable, Any

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
    def vector(self, value: Optional[VT]) -> None:
        self._vector = value

class DataPointProvider(InstanceProvider[KT, DT, VT, DT], Generic[KT, DT, VT]):

    def __init__(self, datapoints: Iterable[DataPoint]) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}
        self._feature_matrix = None

    @classmethod
    def from_data_and_indices(cls, 
                  indices: Sequence[KT], 
                  raw_data: Sequence[DT], 
                  vectors: Optional[Sequence[Optional[VT]]] = None):
        if vectors is None or len(vectors) != len(indices):
            vectors = [None] * len(indices)
        datapoints = itertools.starmap(DataPoint, zip(indices, raw_data, vectors))
        return cls(datapoints)

    @classmethod
    def from_data(cls, raw_data: Sequence[DT]) -> DataPointProvider[KT, DT, VT]:
        indices = range(len(raw_data))
        vectors = [None] * len(raw_data)
        datapoints = itertools.starmap(DataPoint, zip(indices, raw_data, vectors))
        return cls(datapoints)

    @classmethod
    def from_provider(cls, provider: InstanceProvider[KT, VT, DT, Any]) -> DataPointProvider[KT, DT, VT]:
        instances = provider.bulk_get_all()
        datapoints = [DataPoint(ins.identifier, ins.data, ins.vector) for ins in instances]
        return cls(datapoints)
        

    def __iter__(self) -> Iterator[KT]:
        yield from self.dictionary.keys()

    def __getitem__(self, key: KT) -> DataPoint[KT, DT, VT]:
        return self.dictionary[key]
    
    def __setitem__(self, key: KT, value: DataPoint[KT, DT, VT]) -> None:
        self.dictionary[key] = value

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

    