from __future__ import annotations

import collections
import random
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from typing_extensions import Self

import instancelib as il
import numpy as np
import numpy.typing as npt
from instancelib.typehints import DT, KT, LT, RT, VT

from ..environment.base import AbstractEnvironment
from ..machinelearning.transductive.il_tsvm import TSVM
from ..typehints import IT
from ..utils.func import list_unzip, sort_on
from ..utils.numpy import raw_proba_chainer
from .poolbased import PoolBasedAL


class TSVMLearner(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    rank_history: Dict[int, Mapping[KT, int]]
    sampled_sets: Dict[int, Sequence[KT]]
    current_sample: Deque[KT]
    classifier: il.AbstractClassifier[
        IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
    ]

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        pos_label: LT,
        neg_label: LT,
        k_sample: int,
        batch_size: int,
        *_,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        super().__init__(env, *_, identifier=identifier, **__)
        # Problem definition
        self.classifier = TSVM()
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.classifier.set_target_labels([neg_label, pos_label])

        # Batch and sample sizes
        self.k_sample = k_sample
        self.batch_size = batch_size

        # Record keeping for the current sample
        self.it = 0
        self.current_sample = collections.deque()

        # Record keeping for recall analysis
        self.rank_history = dict()
        self.sampled_sets = dict()

    def __call__(
        self, environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]
    ) -> PoolBasedAL[IT, KT, DT, VT, RT, LT]:
        super().__call__(environment)
        return self

    def update_ordering(self) -> bool:
        return True

    def _provider_sample(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        k_sample = min(self.k_sample, len(provider))
        sampled_keys = random.sample(provider.key_list, k_sample)
        sampled_provider = self.env.create_bucket(sampled_keys)
        return sampled_provider

    def _temp_augment_and_train(self):
        self.classifier.fit_provider(self.env.dataset, self.env.labels)

    def _predict(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Tuple[Sequence[KT], npt.NDArray[Any]]:
        raw_probas = self.classifier.predict_proba_provider_raw(provider)
        keys, matrix = raw_proba_chainer(raw_probas)
        return keys, matrix

    def _rank(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        keys, matrix = self._predict(provider)
        pos_column = self.classifier.get_label_column_index(self.pos_label)
        prob_vector = matrix[:, pos_column]
        floats: Sequence[float] = prob_vector.tolist()
        zipped = list(zip(keys, floats))
        ranking = sort_on(1, zipped)
        return ranking

    def _sample(self, distribution: Sequence[Tuple[KT, float]]) -> Sequence[KT]:
        keys, _ = list_unzip(distribution)
        sample = keys[0 : self.batch_size]
        return sample

    @classmethod
    def _to_history(cls, ranking: Sequence[Tuple[KT, float]]) -> Mapping[KT, int]:
        keys, _ = list_unzip(ranking)
        history = {k: i for i, k in enumerate(keys, start=1)}
        return history

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self._temp_augment_and_train()
            ranking = self._rank(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)

            # Increment batch_size for next for train iteration
            self.batch_size += round(self.batch_size / 10)
            self.it += 1
        return self.current_sample

    def __next__(self) -> IT:
        self.update_sample()
        while self.current_sample:
            ins_key = self.current_sample.popleft()
            if ins_key not in self.env.labeled:
                return self.env.dataset[ins_key]
        if not self.env.unlabeled.empty:
            return self.__next__()
        raise StopIteration()

    @classmethod
    def builder(cls, *args: Any, **kwargs: Any) -> Callable[..., Self]:
        return super().builder(*args, **kwargs)
