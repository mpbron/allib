from typing import TypeVar, Sequence, Iterator

_T = TypeVar("_T")

def divide_sequence(full_list: Sequence[_T], batch_size: int) -> Iterator[Sequence[_T]]: 
    # looping till length l 
    for i in range(0, len(full_list), batch_size):  
        yield full_list[i:i + batch_size] 