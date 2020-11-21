import itertools

from typing import Optional, Tuple, TypeVar, Sequence, Iterator, List, Iterable
from operator import itemgetter
_T = TypeVar("_T")

def divide_sequence(full_list: Sequence[_T], batch_size: int) -> Iterator[Sequence[_T]]: 
    # looping till length l 
    for i in range(0, len(full_list), batch_size):  
        yield full_list[i:i + batch_size] 

def divide_iterable(iterable: Iterable[_T], batch_size: int) -> Iterator[Iterator[_T]]:
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, batch_size - 1))

def divide_iterable_in_lists(iterable: Iterable[_T], batch_size: int) -> Iterator[Sequence[_T]]:
    return map(list, divide_iterable(iterable, batch_size))


def get_consecutive(iterable: Sequence[int]) -> Iterable[Sequence[int]]:
    results = [list(map(itemgetter(1), g)) for k, g in itertools.groupby(enumerate(iterable), lambda x: x[0]-x[1])]
    yield from results # type: ignore

def get_range(iterable: Sequence[int]) -> Iterable[Tuple[int, Optional[int]]]:
    def minmax(sorted_iterable: Iterable[int]) -> Tuple[int, Optional[int]]:
        int_list = list(sorted_iterable)
        assert len(int_list) > 0
        min_int = int_list[0]
        max_int = int_list[-1]
        if min_int == max_int:
            return min_int, None
        return min_int, max_int + 1
    map_result = map(minmax, get_consecutive(iterable))
    yield from map_result