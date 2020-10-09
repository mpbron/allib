from __future__ import annotations

import collections
import itertools
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (Any, Deque, Dict, Generic, List, Optional, Set, Tuple,
                    TypeVar, Union)

import pandas as pd

from ..instances import Instance
from ..labels import LabelProvider
from .base import BaseLogger

KT = TypeVar("KT")
LT = TypeVar("LT")
MT = TypeVar("MT")
ST = TypeVar("ST")

SampleMethod = Tuple[ST, Optional[LT]]

class Event(ABC, Generic[KT, LT, ST]):
    timestamp: datetime
    name: str = "Event"

    def __init__(self):
        self.taboo = ["timestamp", "name", "__orig_class__", "taboo"]
    
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

    def __init__(self, key: KT, method: SampleMethod):
        super(SampleEvent, self).__init__()
        self.key = key
        self.method = method[0]
        self.label = method[1]
        self.timestamp = datetime.now()


class LabelEvent(Event[KT, LT, Any], Generic[KT, LT]):
    name = "Label"
    def __init__(self, key: KT, *labels: LT):
        super(LabelEvent, self).__init__()
        self.key = key
        self.labels = frozenset(labels)
        self.timestamp = datetime.now()
        self.taboo.append("labels")

    def __str__(self):
        label_str = ", ".join(self.labels)
        super_str = super(LabelEvent, self).__str__()
        return f"{super_str}, labels => [{label_str}]"

    def __repr__(self):
        return str(self)

class MemoryLogger(BaseLogger[KT, LT, SampleMethod], Generic[KT, LT, ST]):
    def __init__(self, label_provider: LabelProvider[KT, LT]):
        self.sample_dict: Dict[KT, Set[SampleMethod]] = dict()
        self.sample_dict_inv: Dict[SampleMethod, Set[KT]] = dict()
        self.sample_history: Deque[SampleEvent[KT, LT, SampleMethod]] = collections.deque()
        self.event_history: Deque[Event] = collections.deque()
        self.label_history: Deque[LabelEvent[KT, LT]] = collections.deque()
        self.labels = label_provider

    def log_sample(self, x: Instance[KT, Any, Any, Any], sample_method: SampleMethod) -> None:
        self.sample_dict.setdefault(x.identifier, set()).add(sample_method)
        self.sample_dict_inv.setdefault(sample_method, set()).add(x.identifier)
        event = SampleEvent[KT, LT, ST](x.identifier, sample_method)
        self.sample_history.append(event)
        self.event_history.append(event)

    def log_label(self, x: Instance[KT, Any, Any, Any], *labels: LT):
        event = LabelEvent[KT, LT](x.identifier, *labels)
        self.event_history.append(event)
        self.label_history.append(event)
    
    def get_sampled_info(self, x: Instance[KT, Any, Any, Any]):
        return frozenset(self.sample_dict.setdefault(x.identifier, set()))

    def get_instances_by_method(self, sample_method: SampleMethod):
        return frozenset(self.sample_dict_inv.setdefault(sample_method, set()))

    def get_label_table(self) -> pd.DataFrame:
        def row_generator():
            for event in self.label_history:
                doc_labels = event.labels
                label_dict = {
                    label: (label in doc_labels) for label in self.labels.labelset}
                event_dict = {
                    "timestamp": event.timestamp,
                    "instance_id": event.key,
                }
                yield {**event_dict, **label_dict}
        dataframe = pd.DataFrame(list(row_generator()))
        return dataframe

    def get_label_cumsum_table(self) -> pd.DataFrame:
        def row_generator():
            label_sum_dict = {label: 0 for label in self.labels.labelset}
            for event in self.label_history:
                for label in event.labels:
                    label_sum_dict[label] += 1
                event_dict = {
                    "timestamp": event.timestamp,
                    "instance_id": event.key,
                }
                yield {**event_dict, **label_sum_dict}
        dataframe = pd.DataFrame(list(row_generator()))
        return dataframe