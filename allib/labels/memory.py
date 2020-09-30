from __future__ import annotations

from typing import (Dict, FrozenSet, Generic, Iterable, Iterator, Optional,
                    Sequence, Set, TypeVar, Union)

from ..instances import Instance, InstanceProvider
from .base import LabelProvider, to_key

LT = TypeVar("LT")
KT = TypeVar("KT")



class MemoryLabelProvider(LabelProvider[KT, LT]):
    """A Memory based implementation to test and benchmark AL algorithms
    """
    _labelset: FrozenSet[LT]
    _labeldict: Dict[KT, Set[LT]]
    _labeldict_inv: Dict[LT, Set[KT]]

    def __init__(self, 
            labelset: Iterable[LT], 
            labeldict: Dict[KT, Set[LT]], 
            labeldict_inv: Optional[Dict[LT, Set[KT]]] = None) -> None:
        self._labelset = frozenset(labelset)
        self._labeldict = labeldict
        if labeldict_inv is None:
            self._labeldict_inv = {label: set() for label in self._labelset}
            for key in self._labeldict.keys():
                for label in self._labeldict[key]:
                    self._labeldict_inv[label].add(key)
        else:
            self._labeldict_inv = labeldict_inv

    @classmethod
    def from_data(
            cls, 
            labelset: Iterable[LT], 
            indices: Sequence[KT], 
            labels: Sequence[Set[LT]]) -> MemoryLabelProvider[KT, LT]:
        labelset = frozenset(labelset)
        labeldict = {
            idx: labellist for (idx, labellist) in zip(indices, labels)
        }
        labeldict_inv: Dict[LT, Set[KT]] = {label: set() for label in labelset}
        # Store all instances in a Dictionary<LT, Set[ID]>
        for key, labellist in labeldict.items():
            for label in labellist:
                labeldict_inv[label].add(key)
        return cls(labelset, labeldict, labeldict_inv)

    @classmethod
    def from_provider(cls, provider: LabelProvider[KT, LT]) -> MemoryLabelProvider[KT, LT]:
        labelset = provider.labelset
        labeldict_inv = {label: provider.get_instances_by_label(label) for label in labelset}
        labeldict: Dict[KT, Set[LT]]= {}
        for label, key_list in labeldict_inv.items():
            for key in key_list:
                labeldict.setdefault(key, set()).add(label)
        return cls(labelset, labeldict, labeldict_inv)

    @property
    def labelset(self) -> FrozenSet[LT]:
        return self._labelset

    def remove_labels(self, instance: Union[KT, Instance], *labels: LT):
        key = to_key(instance)
        if key not in self._labeldict:
            raise KeyError("Key {} is not found".format(key))
        for label in labels:
            self._labeldict[key].discard(label)
            self._labeldict_inv[label].discard(key)

    def set_labels(self, instance: Union[KT, Instance], *labels: LT):
        key = to_key(instance)
        for label in labels:
            self._labeldict.setdefault(key, set()).add(label)
            self._labeldict_inv.setdefault(label, set()).add(key)

    def get_labels(self, instance: Union[KT, Instance]) -> Set[LT]:
        key = to_key(instance)
        return self._labeldict.setdefault(key, set())

    def get_instances_by_label(self, label: LT) -> Set[KT]:
        return self._labeldict_inv.setdefault(label, set())

    def document_count(self, label: LT) -> int:
        return len(self.get_instances_by_label(label))
            
