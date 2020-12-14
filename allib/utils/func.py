import functools
import itertools

from typing import Any, Iterable, List, TypeVar, Callable,  Tuple, Optional, Sequence

_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")
_Z = TypeVar("_Z")

def clist(iter: Iterable[_Z]) -> List[_Z]:
    ret_list = list(iter)
    return ret_list

def mapsnd(func: Callable[[_U], _V]) -> Callable[[_T, _U], Tuple[_T, _V]]:
    @functools.wraps(func)
    def wrap(fst: _T, snd: _U) -> Tuple[_T, _V]:
        result = func(snd)
        return (fst, result)
    return wrap

def all_equal(iterable: Iterable[_T]) -> bool:
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False) # type: ignore

def list_unzip(iterable: Iterable[Tuple[_T, _U]]) -> Tuple[Sequence[_T], Sequence[_U]]:
    try:
        fst_iter, snd_iter = zip(*iterable)
        fst_sequence, snd_sequence = list(fst_iter), list(snd_iter)
    except ValueError:
        return [], [] # type: ignore
    return fst_sequence, snd_sequence # type: ignore

def list_unzip3(iterable: Iterable[Tuple[_T, _U, _V]]) -> Tuple[Sequence[_T], Sequence[_U], Sequence[_V]]:
    try:
        fst_sequence, snd_sequence, trd_sequence = map(list, zip(*iterable))  # type ignore
    except ValueError:
        return [], [], [] #type: ignore
    return fst_sequence, snd_sequence, trd_sequence # type: ignore

def filter_snd_none_zipped(iter: Iterable[Tuple[_T, Optional[_U]]]) -> Tuple[Sequence[_T], Sequence[_U]]:
    filtered = filter(lambda x: x[1] is not None, iter)
    return list_unzip(filtered) # type: ignore

def filter_snd_none(fst_iter: Iterable[_T], 
                snd_iter: Iterable[Optional[_U]]
                ) -> Tuple[Sequence[_T], Sequence[_U]]:
    zipped: Iterable[Tuple[_T, _U]] = filter(lambda x: x[1] is not None, zip(fst_iter, snd_iter)) # type: ignore
    return list_unzip(zipped)

def sort_on(index: int, seq: Sequence[Tuple[_T, _U]]) -> Sequence[Tuple[_T, _U]]:
    return sorted(seq, key=lambda k: k[index], reverse=True) # type: ignore

