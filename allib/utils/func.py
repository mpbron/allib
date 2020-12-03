import functools
import itertools

from typing import Any, Iterable, TypeVar, Callable,  Tuple, Optional, Sequence

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

def list_unzip(iterable: Iterable[Tuple[_T, _U]]) -> Tuple[Sequence[_T], Sequence[_U]]:
    try:
        fst_sequence, snd_sequence = map(list, zip(*iterable))
    except ValueError:
        return [], [] # type: ignore
    return fst_sequence, snd_sequence # type: ignore

def list_unzip3(iterable: Iterable[Tuple[_T, _U, _V]]) -> Tuple[Sequence[_T], Sequence[_U], Sequence[_V]]:
    try:
        fst_sequence, snd_sequence, trd_sequence = map(list, zip(*iterable))
    except ValueError:
        return [], [], [] #type: ignore
    return fst_sequence, snd_sequence, trd_sequence # type: ignore

def filter_snd_none_zipped(iter: Iterable[Tuple[_T, Optional[_U]]]) -> Tuple[Sequence[_T], Sequence[_U]]:
    filtered = filter(lambda x: x[1] is not None, iter)
    return list_unzip(filtered) # type: ignore

def filter_snd_none(fst_iter: Iterable[_T], 
                snd_iter: Iterable[Optional[_U]]
                ) -> Tuple[Sequence[_T], Sequence[_U]]:
    zipped = filter(lambda x: x[1] is not None, zip(fst_iter, snd_iter))
    return list_unzip(zipped) # type: ignore

def sort_on(index: int, seq: Sequence[Tuple[_T, _U]]) -> Sequence[Tuple[_T, _U]]:
    return sorted(seq, key=lambda k: k[index], reverse=True) # type: ignore

