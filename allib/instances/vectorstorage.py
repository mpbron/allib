from abc import abstractmethod
from typing import (Generic, Iterator, MutableMapping,
                    Sequence, Tuple, TypeVar, Union)
KT = TypeVar("KT")
VT = TypeVar("VT")

class VectorStorage(MutableMapping[KT, VT], Generic[KT, VT]):
    @abstractmethod
    def __getitem__(self, k: KT) -> VT:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, k: KT, value: VT) -> None:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        raise NotImplementedError

    @abstractmethod
    def add_bulk(self, keys: Sequence[KT], values: Union[Sequence[VT]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], VT]:
        raise NotImplementedError

    @abstractmethod
    def matrices_chunker(self) -> Iterator[Tuple[KT, VT]]:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError
    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError