from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Generic, Iterator, Sequence, List, Optional, TypeVar, Any, Mapping

import numpy as np #type: ignore

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
CT = TypeVar("CT")
LVT = TypeVar("LVT")

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
    @abstractmethod
    def __contains__(self, item: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        raise NotImplementedError

    def add(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.__setitem__(instance.identifier, instance)

    def discard(self, instance: Instance[KT, DT, VT, RT]) -> None:
        try:
            self.__delitem__(instance.identifier)
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

    def bulk_get_vectors(self, keys: Sequence[KT]) -> Sequence[Optional[VT]]:
        vectors = [self[key].vector  for key in keys]
        return vectors

    def bulk_get_all(self) -> List[Instance[KT, DT, VT, RT]]:
        return list(self.get_all())