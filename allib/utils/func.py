import functools
import itertools

from typing import Iterable, TypeVar, Callable,  Tuple

_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")

def mapsnd(func: Callable[[_U], _V]) -> Callable[[_T, _U], Tuple[_T, _V]]:
    @functools.wraps(func)
    def wrap(fst: _T, snd: _U) -> Tuple[_T, _V]:
        result = func(snd)
        return (fst, result)
    return wrap

def all_equal(iterable: Iterable[_T]) -> bool:
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)
