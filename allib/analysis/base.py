import collections
from abc import ABC, abstractmethod
from typing import (
    Any,
    Deque,
    FrozenSet,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from instancelib import Instance
from instancelib.utils.func import union, value_map

from ..activelearning.base import ActiveLearner
from ..environment import AbstractEnvironment
from ..environment.base import AbstractEnvironment
from ..typehints import DT, IT, KT, LT, RT, VT
from .base import ActiveLearner


class AbstractStatistics(ABC, Generic[KT, LT]):
    per_round: List[Mapping[LT, int]]

    @abstractmethod
    def update(self, learner: ActiveLearner[Any, KT, Any, Any, Any, LT]):
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def rounds(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def current_label_count(self, label: LT) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_annotated(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def label_annotated(self, label: LT, it: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def label_per_round(self, label: LT) -> Sequence[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def annotations_per_round(self) -> Sequence[int]:
        raise NotImplementedError

    def annotations_since_last(self, label: LT) -> int:
        label_enumerator = list(enumerate(self.label_per_round(label)))
        pos_iterations = [it for it, count in label_enumerator if count > 0]
        if pos_iterations:
            last_pos_it = max(pos_iterations)
            annotated_at = np.array(self.annotations_per_round).cumsum()[last_pos_it]
            return self.current_annotated - annotated_at
        return 0


class StatsMixin(ABC, Generic[KT, LT]):
    @property
    @abstractmethod
    def stats(self) -> AbstractStatistics[KT, LT]:
        raise NotImplementedError


class StatsWrapper(StatsMixin[KT, LT], ActiveLearner[IT, KT, DT, VT, RT, LT]):
    learner: ActiveLearner[IT, KT, DT, VT, RT, LT]

    def __init__(
        self,
        learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
        stats: AbstractStatistics[KT, LT],
        batch_size: int = 20,
    ):
        self.learner = learner
        self._stats = stats
        self.batch_it = batch_size
        self.batch_size = batch_size

    @property
    def stats(self) -> AbstractStatistics[KT, LT]:
        return self._stats

    def __next__(self) -> IT:
        if self.batch_it >= self.batch_size:
            self.stats.update(self.learner)
            self.learner.update_ordering()
            self.batch_it = 0
        elem = self.learner.__next__()
        self.batch_it += 1
        return elem

    @property
    def env(self) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
        return self.learner.env

    def update_ordering(self) -> bool:
        return self.learner.update_ordering()

    @property
    def has_ordering(self) -> bool:
        return self.learner.has_ordering

    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.learner.set_as_labeled(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.learner.set_as_unlabeled(instance)

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        return self.learner.name


class AnnotationStatistics(AbstractStatistics[KT, LT]):
    labelwise: List[Mapping[LT, FrozenSet[KT]]]
    per_round: List[Mapping[LT, FrozenSet[KT]]]
    annotated_per_round: List[FrozenSet[KT]]
    annotated: List[FrozenSet[KT]]
    unlabeled: List[FrozenSet[KT]]
    dataset: List[FrozenSet[KT]]

    def __init__(self) -> None:
        self.labelwise = list()
        self.per_round = list()
        self.annotated = list()
        self.annotated_per_round = list()
        self.unlabeled = list()
        self.dataset = list()

    def update(self, learner: ActiveLearner[Any, KT, Any, Any, Any, LT]):
        previous_round = frozenset() if not self.annotated else self.annotated[-1]
        annotated = frozenset(learner.env.labeled)
        unlabeled = frozenset(learner.env.unlabeled)
        current_round = {
            label: (
                frozenset(
                    learner.env.get_subset_by_labels(
                        learner.env.labeled, label, labelprovider=learner.env.labels
                    )
                )
            )
            for label in learner.env.labels.labelset
        }
        current_round_new = {
            label: keys.difference(previous_round)
            for label, keys in current_round.items()
        }
        annotated_new = annotated.difference(previous_round)
        self.annotated_per_round.append(annotated_new)
        self.labelwise.append(current_round)
        self.annotated.append(annotated)
        self.unlabeled.append(unlabeled)
        self.per_round.append(current_round_new)
        self.dataset.append(frozenset(learner.env.dataset))

    @property
    def dataset_size(self) -> int:
        return 0 if not self.dataset else len(self.dataset[-1])

    @property
    def rounds(self) -> int:
        return len(self.annotated)

    def current_label_count(self, label: LT) -> int:
        return 0 if not self.labelwise else len(self.labelwise[-1][label])

    @property
    def current_annotated(self) -> int:
        return 0 if not self.annotated else len(self.annotated[-1])

    def label_annotated(self, label: LT, it: int) -> int:
        return 0 if not self.per_round else len(self.per_round[it][label])

    def label_per_round(self, label: LT) -> Sequence[int]:
        return [len(dc[label]) for dc in self.per_round]

    @property
    def annotations_per_round(self) -> Sequence[int]:
        return [len(an) for an in self.annotated_per_round]

    def batch_at_round(self, it: int) -> FrozenSet[KT]:
        return self.annotated_per_round[it]

    @property
    def last_batch(self) -> FrozenSet[KT]:
        return (
            frozenset()
            if not self.annotated_per_round
            else self.annotated_per_round[-1]
        )


class AnnotationStatisticsSlim(AbstractStatistics[KT, LT]):
    labelwise: List[Mapping[LT, int]]
    per_round: List[Mapping[LT, int]]
    previous_round: FrozenSet[KT]
    annotated_per_round: List[int]
    annotated: List[int]
    unlabeled: List[int]
    dataset: List[int]
    previous_rounds: FrozenSet[KT]

    def __init__(self) -> None:
        self.labelwise = list()
        self.per_round = list()
        self.annotated = list()
        self.annotated_per_round = list()
        self.unlabeled = list()
        self.dataset = list()
        self.previous_round = frozenset()

    def update(self, learner: ActiveLearner[Any, KT, Any, Any, Any, LT]):
        annotated = frozenset(learner.env.labeled)
        unlabeled = frozenset(learner.env.unlabeled)
        current_round = {
            label: (
                frozenset(
                    learner.env.get_subset_by_labels(
                        learner.env.labeled, label, labelprovider=learner.env.labels
                    )
                )
            )
            for label in learner.env.labels.labelset
        }
        current_round_new = {
            label: keys.difference(self.previous_round)
            for label, keys in current_round.items()
        }
        current_round_counts = value_map(len, current_round)
        current_round_new_counts = value_map(len, current_round_new)
        annotated_new = annotated.difference(self.previous_round)
        self.annotated_per_round.append(len(annotated_new))
        self.labelwise.append(current_round_counts)
        self.annotated.append(len(annotated))
        self.unlabeled.append(len(unlabeled))
        self.per_round.append(current_round_new_counts)
        self.dataset.append(len(learner.env.dataset))
        self.previous_round = annotated

    @property
    def dataset_size(self) -> int:
        return 0 if not self.dataset else self.dataset[-1]

    @property
    def rounds(self) -> int:
        return len(self.annotated)

    def current_label_count(self, label: LT) -> int:
        return 0 if not self.labelwise else self.labelwise[-1][label]

    @property
    def current_annotated(self) -> int:
        return 0 if not self.annotated else self.annotated[-1]

    def label_annotated(self, label: LT, it: int) -> int:
        return 0 if not self.per_round else self.per_round[it][label]

    def label_per_round(self, label: LT) -> Sequence[int]:
        return [dc[label] for dc in self.per_round]

    @property
    def annotations_per_round(self) -> Sequence[int]:
        return tuple(self.annotated_per_round)
