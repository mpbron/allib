import functools

from typing import TypeVar, Callable,  Tuple

_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")

def mapsnd(func: Callable[[_U], _V]) -> Callable[[_T, _U], Tuple[_T, _V]]:
    @functools.wraps(func)
    def wrap(fst: _T, snd: _U) -> Tuple[_T, _V]:
        result = func(snd)
        return (fst, result)
    return wrap