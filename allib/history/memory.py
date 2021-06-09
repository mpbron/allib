from __future__ import annotations

import collections
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, Optional,
                    Sequence, Set, Tuple, TypeVar, Union)

import pandas as pd  # type: ignore
from instancelib import Instance
from instancelib.labels import LabelProvider
from instancelib.labels.memory import MemoryLabelProvider
from instancelib.utils.to_key import to_key
from instancelib.typehints import KT, LT

from .base import BaseLogger

MT = TypeVar("MT")
ST = TypeVar("ST")

SampleMethod = Tuple[ST, Optional[LT]]

class Event(ABC, Generic[KT, LT, ST]):
    timestamp: datetime
    name: str = "Event"

    def __init__(self):
        self.taboo = ["timestamp", "name", "__orig_class__", "taboo"]
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        def kvpair(x: Any, y: Any) -> str:
            return f"{x} => {y}"
        dict_filter = [(k, v) for (k, v) in self.__dict__.items() if k not in self.taboo]
        dict_tuples = itertools.starmap(kvpair, dict_filter)
        dict_str = ", ".join(dict_tuples)
        return f"{self.timestamp} :: {self.name} :: {dict_str}"
    
    def __repr__(self) -> str:
        return str(self)


class SampleEvent(Event[KT, LT, ST], Generic[KT, LT, ST]):
    name = "Sampling"

    def __init__(self, key: KT, method: SampleMethod[ST, LT]):
        super(SampleEvent, self).__init__()
        self.key = key
        self.method = method[0]
        self.method_label = method[1]
        


class LabelEvent(SampleEvent[KT, LT, ST], Generic[KT, LT, ST]):
    name = "Label"
    def __init__(self, key: KT, method: SampleMethod[ST, LT], *labels: LT):
        super().__init__(key, method)
        self.labels = frozenset(labels)
        self.taboo.append("labels")

    def __str__(self):
        label_str = ", ".join(map(str, self.labels))
        super_str = super(LabelEvent, self).__str__()
        return f"{super_str}, labels => [{label_str}]"

    def __repr__(self):
        return str(self)

@dataclass
class Snapshot(Generic[KT, LT]):
    ordering: Sequence[KT]
    metrics: Sequence[float]
    labeled: FrozenSet[KT]
    labels: LabelProvider[KT, LT]

class MemoryLogger(BaseLogger[KT, LT, SampleMethod[ST, LT]], Generic[KT, LT,ST]):
    def __init__(self, label_provider: LabelProvider[KT, LT]):
        self._labelset = label_provider.labelset

        self.sample_dict: Dict[KT, Set[SampleMethod[ST, LT]]] = dict()
        self.sample_dict_inv: Dict[SampleMethod[ST, LT], Set[KT]] = dict()
        
        self.snapshots: Deque[Snapshot[KT, LT]] = collections.deque()
        self.sample_history: Deque[SampleEvent[KT, LT, ST]] = collections.deque()
        self.event_history: Deque[Event[KT, LT, ST]] = collections.deque()
        self.label_history: Deque[LabelEvent[KT, LT, ST]] = collections.deque()
        self.label_order: Dict[KT, int] = dict()
        self.label_it = 0

    @property
    def labelset(self) -> FrozenSet[LT]: 
        return self._labelset
        
    def log_ordering(self, ordering: Sequence[KT], metrics: Sequence[float], labeled: Iterable[KT], labels: LabelProvider[KT, LT]):
        labeled = frozenset(labeled)
        labels_copy = MemoryLabelProvider[KT, LT].from_provider(labels)
        self.snapshots.append(Snapshot[KT, LT](ordering, metrics, labeled, labels_copy))

    def log_sample(self, x: Union[KT, Instance[KT, Any, Any, Any]], sample_method: SampleMethod[ST, LT]) -> None:
        key = to_key(x)
        self.sample_dict.setdefault(key, set()).add(sample_method)
        self.sample_dict_inv.setdefault(sample_method, set()).add(key)
        event = SampleEvent[KT, LT, ST](key, sample_method)
        self.sample_history.append(event)
        self.event_history.append(event)

    def log_label(self, x: Union[KT, Instance[KT, Any, Any, Any]], sample_method: SampleMethod[ST, LT], *labels: LT):
        key = to_key(x)
        event = LabelEvent[KT, LT, ST](key, sample_method, *labels)
        self.event_history.append(event)
        self.label_history.append(event)
        self.label_order[key] = self.label_it
        self.label_it = self.label_it + 1

    def get_label_order(self, x: Union[KT, Instance[KT, Any, Any, Any]]) -> int:
        key = to_key(x)
        return self.label_order[key]
    
    def get_sampled_info(self, x: Union[KT, Instance[KT, Any, Any, Any]]):
        key = to_key(x)
        return frozenset(self.sample_dict.setdefault(key, set()))

    def get_instances_by_method(self, sample_method: SampleMethod[ST, LT]):
        return frozenset(self.sample_dict_inv.setdefault(sample_method, set()))

    def get_label_table(self) -> pd.DataFrame:
        def row_generator():
            for event in self.label_history:
                doc_labels = event.labels
                label_dict = {
                    label: (label in doc_labels) for label in self.labelset}
                event_dict = {
                    "timestamp": event.timestamp,
                    "instance_id": event.key,
                    "method": event.method,
                    "method_label": event.method_label,
                }
                yield {**event_dict, **label_dict}
        dataframe = pd.DataFrame(list(row_generator()))
        return dataframe

    def get_label_cumsum_table(self) -> pd.DataFrame:
        def row_generator():
            label_sum_dict: Dict[Tuple[LT, ST, Optional[LT]], int] = dict()
            for event in self.label_history:
                for label in event.labels:
                    dict_key = (label, event.method, event.method_label)
                    label_sum_dict.setdefault(dict_key, 0)
                    label_sum_dict[dict_key] += 1
                event_dict = {
                    "timestamp": event.timestamp,
                    "instance_id": event.key,
                }
                yield {**event_dict, **label_sum_dict}
        dataframe = pd.DataFrame(list(row_generator()))
        return dataframe
