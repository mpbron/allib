from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Generic, Iterator, List, Optional, TypeVar, Any, Mapping

import numpy as np #type: ignore

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
CT = TypeVar("CT")
LVT = TypeVar("LVT")


class Matrix(Generic[KT]):
    def __init__(self, matrix: np.ndarray, keys: List[KT]):
        self.matrix = matrix
        self.index_list = keys
        self.index_map = {
            key: np_key for (np_key, key) in enumerate(keys)
        }
        self.size = len(keys)

    def get_instance_id(self, row_idx: int) -> KT:
        return self.index_list[row_idx]

    def discard(self, identifier: KT) -> None:
        # Find the row_idx that belongs to the instance
        row_idx = self.index_map[identifier]

        # Delete data from the matrix and book keeping lists
        self.matrix = np.delete(self.matrix, row_idx, axis=0) # type: ignore
        del self.index_list[row_idx]
        del self.index_map[identifier]

        # Update keys after the removed key
        for key in self.index_list[row_idx:]:
            self.index_map[key] -= 1

        # Update size
        self.size -= 1

    def add(self, instance: Instance[KT, Any, Any, Any]) -> None:
        if instance.vector is not None:
            self.matrix = np.concatenate((self.matrix), instance.vector) # type: ignore
            self.index_list.append(instance.identifier)
            self.index_map[instance.identifier] = self.size  
            self.size += 1
        
class Instance(ABC, Generic[KT, DT, VT, RT]):

    @property
    @abstractmethod
    def data(self) -> DT:
        """Return the raw data of this instance


        Returns
        -------
        DT
            The Raw Data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def representation(self) -> RT:
        """Return a representation for annotation


        Returns
        -------
        RT
            A representation of the raw data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vector(self) -> Optional[VT]:
        """Get the vector represenation of the raw data

        Returns
        -------
        Optional[VT]
            The Vector
        """
        raise NotImplementedError

    @vector.setter
    def vector(self, value: Optional[VT]) -> None: # type: ignore
        raise NotImplementedError

    @property
    @abstractmethod
    def identifier(self) -> KT:
        """Get the identifier of the instance

        Returns
        -------
        KT
            The identifier key of the instance
        """
        raise NotImplementedError

class ContextInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT, CT]):
    @property
    @abstractmethod
    def context(self) -> CT:
        raise NotImplementedError


class ChildInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT]):
    @property
    @abstractmethod
    def parent(self) -> Instance[KT, DT, VT, RT]:
        raise NotImplementedError


class ParentInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT]):
    @property
    @abstractmethod
    def children(self) -> List[ChildInstance[KT, DT, VT, RT]]:
        raise NotImplementedError


class InstanceProvider(ABC, MutableMapping, Mapping[KT, Instance[KT, DT, VT, RT]], Generic[KT, DT, VT, RT]):
    _feature_matrix: Optional[Matrix[KT]]

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        raise NotImplementedError

    def add(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.__setitem__(instance.identifier, instance)
        if self._feature_matrix is not None:
            self._feature_matrix.add(instance)

    def discard(self, instance: Instance[KT, DT, VT, RT]) -> None:
        try:
            self.__delitem__(instance.identifier)
            if self._feature_matrix is not None:
                self._feature_matrix.discard(instance.identifier)
        except KeyError:
            pass  # To adhere to Set.discard(...) behavior

    @property
    def key_list(self) -> List[KT]:
        return list(self.keys())

    @property
    @abstractmethod
    def empty(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> Iterator[Instance[KT, DT, VT, RT]]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @property
    def feature_matrix(self) -> Matrix[KT]:
        if self._feature_matrix is None:
            vectors = [ins.vector for ins in self.values()]
            self._feature_matrix = Matrix(np.vstack(vectors), self.key_list) # type: ignore
        return self._feature_matrix

    def bulk_get_all(self) -> List[Instance[KT, DT, VT, RT]]:
        return list(self.get_all())