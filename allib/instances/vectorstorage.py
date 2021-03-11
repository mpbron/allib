import functools
from io import UnsupportedOperation
from abc import abstractmethod
from typing import (Any, Callable, Generic, Iterator, MutableMapping, Sequence,
                    Tuple, TypeVar, Union)

KT = TypeVar("KT")
VT = TypeVar("VT")
F = TypeVar("F", bound=Callable[..., Any])

class VectorStorage(MutableMapping[KT, VT], Generic[KT, VT]):
    @abstractmethod
    def writeable(self) -> bool:
        raise NotImplementedError

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
    def matrices_chunker(self) -> Iterator[Tuple[Sequence[KT], VT]]:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError
    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError

def ensure_writeable(func: F) -> F:
    """A decorator that ensures that the wrapped method is only
    executed when the object is opened in writable mode

    Parameters
    ----------
    func : F
        The method that should only be executed if the method

    Returns
    -------
    F
        The same method with a check wrapped around it
    """        
    @functools.wraps(func)
    def wrapper(
            self: VectorStorage[Any, Any], 
            *args: Any, **kwargs: Any) -> F:
        if not self.writeable:
            raise UnsupportedOperation(
                f"The object {self} is not writeable,"
                f" so the operation {func} is not supported.")
        return func(self, *args, **kwargs)
    return wrapper # type: ignore
