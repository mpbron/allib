import itertools

from typing import TypeVar, Sequence, Iterator, List, Iterable

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
