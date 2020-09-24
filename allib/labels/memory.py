from typing import Iterator, Generic, TypeVar, Set, Iterable, Sequence, Union

from ..instances import Instance, InstanceProvider
from .base import LabelProvider, to_key

LT = TypeVar("LT")
KT = TypeVar("KT")

class MemoryLabelProvider(LabelProvider, Generic[KT, LT]):
    """A Memory based implementation to test and benchmark AL algorithms
    """        
    def __init__(self, 
                 labelset: Iterable[LT], 
                 indices: Sequence[KT], 
                 labels: Sequence[Set[LT]]) -> None:
        self._labelset = frozenset(labelset)
        if len(labels) == len(indices):
            # Build a Dictionary<ID, Set[LT]>
            self._labeldict = {
                key: labels[i] for i, key in enumerate(indices)
            }
        else:
            self._labeldict = {
                key: set() for i, key in enumerate(indices)
            }
        self._labeldict_inv = {label: set() for label in self._labelset}
        # Store all instances in a Dictionary<LT, Set[ID]>
        for key in self._labeldict.keys():
            for label in self._labeldict[key]:
                self._labeldict_inv[label].add(key)

    @property
    def labelset(self) -> Set[LT]:
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
        if key not in self._labeldict:
            self._labeldict[key] = set()
        for label in labels:
            self._labeldict[key].add(label)
            self._labeldict_inv[label].add(key)

    def get_labels(self, instance: Union[KT, Instance]) -> Set[LT]:
        key = to_key(instance)
        return self._labeldict[key]

    def get_instances_by_label(self, label: LT) -> Set[KT]:
        return self._labeldict_inv[label]

    def document_count(self, label: LT) -> int:
        return len(self.get_instances_by_label(label))

    @classmethod
    def from_instance_provider(cls, labelset: Iterable[LT], provider: InstanceProvider):
        return cls(labelset, provider.keys(), [])
            
