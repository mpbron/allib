from __future__ import annotations

import functools
import itertools
from typing import (Any, Dict, Generic, Iterable, Iterator, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

import numpy as np # type: ignore
import pandas as pd # type: ignore

from ..utils.chunks import divide_iterable_in_lists

from ..utils.func import list_unzip
from .base import Instance, InstanceProvider
from .hdf5vector import HDF5VectorStorage

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
LVT = TypeVar("LVT")


class HDF5Instance(Instance[int, str, np.ndarray, str]):
    def __init__(self, identifier: int, data: str, vector: Optional[np.ndarray], 
                 vector_storage: HDF5VectorStorage[int]) -> None:
        self._identifier = identifier
        self._data = data
        self._vectorstorage = vector_storage
        self._vector = vector

    @property
    def data(self) -> str:
        return self._data

    @property
    def representation(self) -> str:
        return self._data

    @property
    def identifier(self) -> int:
        return int(self._identifier)

    @property
    def vector(self) -> Optional[np.ndarray]:
        if self._vector is None:
            if self._identifier in self._vectorstorage:
                self._vector = self._vectorstorage[self._identifier]
        return self._vector

    @vector.setter
    def vector(self, value: Optional[np.ndarray]) -> None:  # type: ignore
        if value is not None:
            self._vector = value
            with HDF5VectorStorage[int](self._vectorstorage.h5path, "a") as writeable_storage:
                writeable_storage[self.identifier] = value

    @classmethod
    def from_row(cls, vectorstorage: HDF5VectorStorage[int], 
                 row: Sequence[Union[int, str]]) -> HDF5Instance:
        return cls(row[0], row[1], None, vectorstorage)  # type: ignore


class HDF5Provider(InstanceProvider[int, str, np.ndarray, str]):
    def __init__(self, data_storage: str, vector_storage_location: str) -> None:
        self.data_storage = data_storage
        self.vector_storage_location = vector_storage_location
        self.vectors = HDF5VectorStorage[int](vector_storage_location)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.read_hdf(self.data_storage, "datastorage")  # type: ignore

    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[int],
                              raw_data: Sequence[str],
                              data_storage: str,
                              vector_storage_location: str):
        assert len(indices) == len(raw_data)
        datapoints = zip(indices, raw_data)
        dataframe = pd.DataFrame(datapoints, columns=["key", "data"])  # type: ignore
        dataframe.to_hdf(data_storage, "datastorage")  # type: ignore
        return cls(data_storage, vector_storage_location)

    @classmethod
    def from_data(cls, raw_data: Sequence[str], data_storage_location, 
                  vector_storage_location) -> HDF5Provider:
        indices = list(range(len(raw_data)))
        return cls.from_data_and_indices(indices, raw_data, data_storage_location, vector_storage_location)

    @classmethod
    def from_provider(cls, provider: InstanceProvider[int, str, np.ndarray, str], 
                      data_storage: str, vector_storage_location: str, *args, **kwargs) -> HDF5Provider:
        instances = provider.bulk_get_all()
        datapoints = [(ins.identifier, ins.data) for ins in instances]
        ins_vectors = [(ins.identifier, ins.vector) for ins in instances]
        identifiers, vectors = list_unzip(ins_vectors)
        dataframe = pd.DataFrame(datapoints, columns=["key", "data"])
        dataframe.to_hdf(data_storage, "datastorage")  # type: ignore
        with HDF5VectorStorage[int](vector_storage_location, "a") as storage:
            storage.add_bulk(identifiers, vectors)
        return cls(data_storage, vector_storage_location)

    def __iter__(self) -> Iterator[int]:
        key_col = self.dataframe["key"]
        for _, key in key_col.items():  # type: ignore
            yield int(key)

    def __getitem__(self, key: int) -> HDF5Instance:
        df = self.dataframe
        row = df[df.key == key]
        vector: Optional[np.ndarray] = None
        if key in self.vectors:
            vector = self.vectors[key]
        return HDF5Instance(key, row["data"].values[0], vector, self.vectors)

    def __setitem__(self, key: int, value: Instance[int, str, np.ndarray, str]) -> None:
        pass

    def __delitem__(self, key: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.dataframe)

    def __contains__(self, key: object) -> bool:
        df = self.dataframe
        return len(df[df.key == key]) > 0

    @property
    def empty(self) -> bool:
        return not self.dataframe

    def get_all(self):
        yield from list(self.values())

    def clear(self) -> None:
        pass

    def data_chunker(self, batch_size: int):
        constructor = functools.partial(HDF5Instance.from_row, self.vectors)
        df = self.dataframe
        instance_df = df.apply(constructor, axis=1)  # type: ignore
        instance_list: List[HDF5Instance] = instance_df.tolist()
        return divide_iterable_in_lists(instance_list, batch_size)

    def bulk_add_vectors(self, keys: Sequence[int], values: Sequence[np.ndarray]) -> None:
        with HDF5VectorStorage[int](self.vector_storage_location, "a") as writeable_storage:
            writeable_storage.add_bulk(keys, values)
        self.vectors.reload()

    def bulk_get_vectors(self, keys: Sequence[int]):
        ret_keys, vectors = self.vectors.get_vectors(keys)
        return ret_keys, vectors

    def vector_chunker(self, batch_size: int):
        self.vectors.reload()
        results = self.vectors.vectors_chunker(batch_size)
        yield from results


class HDF5BucketProvider(HDF5Provider):
    def __init__(self, dataset: HDF5Provider, identifiers: Iterable[int]):
        self._elements = set(identifiers)
        self.dataset = dataset
        self.vectors = dataset.vectors

    def __iter__(self):
        yield from self._elements

    def __getitem__(self, key: int):
        if key in self._elements:
            return self.dataset[key]
        raise KeyError(
            f"This datapoint with key {key} does not exist in this provider")

    def __setitem__(self, key: int, _: Instance[int, str, np.ndarray, str]) -> None:
        self._elements.add(key)

    def __delitem__(self, key: int) -> None:
        self._elements.discard(key)

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, key: object) -> bool:
        return key in self._elements

    @property
    def empty(self) -> bool:
        return not self._elements

    @classmethod
    def from_provider(cls, provider: InstanceProvider[int, str, np.ndarray, str], 
                      data_storage: str = "", vector_storage_location: str = "", *args, **kwargs) -> HDF5Provider:
        if isinstance(provider, HDF5Provider):
            return cls(provider, provider.key_list)
        hdf5_provider = HDF5Provider.from_provider(provider, data_storage, vector_storage_location, *args, **kwargs)
        return cls(hdf5_provider, hdf5_provider.key_list)

    @classmethod
    def copy(cls, provider: HDF5BucketProvider) -> HDF5BucketProvider:
        return cls(provider.dataset, provider.key_list)

    def vector_chunker(self, batch_size: int):
        results = self.dataset.vectors.get_vectors_zipped(self.key_list, batch_size)
        yield from results

    def data_chunker(self, batch_size: int) -> Iterator[Sequence[HDF5Instance]]:
        results = self.dataset.data_chunker(batch_size)
        in_set = functools.partial(
            filter, lambda ins: ins.identifier in self._elements)
        filtered = map(list, map(in_set, results)) # type: ignore
        return filtered # type: ignore
