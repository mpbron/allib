from datetime import datetime

import collections
from typing import (Any, Deque, Dict, Generic, Optional, Set, Tuple,
                    TypeVar)

from allib.history.base import BaseLogger

from ..instances import Instance

KT = TypeVar("KT")
LT = TypeVar("LT")
MT = TypeVar("MT")
ST = TypeVar("ST")

SampleMethod = Tuple[ST, Optional[LT]]

class SampleEvent(Generic[KT, LT, ST]):
    def __init__(self, key: KT, method: SampleMethod):
        self.key = key
        self.method = method[0]
        self.label = method[1]
        self.timestamp = datetime.now()
    
    def __str__(self):
        if self.label is not None:
            return f"{self.timestamp} - SampleEvent - {self.key} - {self.method} - {self.label}"
        return f"{self.timestamp} - SampleEvent - {self.key} - {self.method}"

    def __repr__(self):
        return str(self)


class MemoryLogger(BaseLogger[KT, LT, SampleMethod], Generic[KT, LT, ST]):
    def __init__(self):
        self.sample_dict: Dict[KT, Set[SampleMethod]] = dict()
        self.sample_dict_inv: Dict[SampleMethod, Set[KT]] = dict()
        self.sample_history: Deque[SampleEvent[KT, LT, SampleMethod]] = collections.deque()

    def log_sample(self, x: Instance[KT, Any, Any, Any], sample_method: SampleMethod) -> None:
        self.sample_dict.setdefault(x.identifier, set()).add(sample_method)
        self.sample_dict_inv.setdefault(sample_method, set()).add(x.identifier)
        self.sample_history.append(SampleEvent[KT, LT, ST](x.identifier, sample_method))
    
    def get_sampled_info(self, x: Instance[KT, Any, Any, Any]):
        return frozenset(self.sample_dict.setdefault(x.identifier, set()))

    def get_instances_by_method(self, sample_method: SampleMethod):
        return frozenset(self.sample_dict_inv.setdefault(sample_method, set()))


 
