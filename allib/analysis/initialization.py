from __future__ import annotations
from abc import ABC, abstractmethod

import itertools
import logging
import random
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, List,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..environment import AbstractEnvironment
from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator

from ..typehints import KT, LT, IT, DT, VT, RT

LOGGER = logging.getLogger(__name__)


class Initializer(ABC, Generic[IT, KT, LT]):

    @abstractmethod
    def __call__(self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        raise NotImplementedError


class IdentityInitializer(Initializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        return learner

class RandomInitializer(Initializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, env: AbstractEnvironment[IT, KT, Any, Any, Any, LT], sample_size: int = 1, **kwargs) -> None:
        self.env = env
        self.sample_size = sample_size

    def get_random_sample_for_label(self, label: LT) -> Sequence[KT]:
        docs = random.sample(self.env.truth.get_instances_by_label(label), self.sample_size)
        return docs

    def get_initialization_sample(self) -> Sequence[KT]:
        docs = list(
            itertools.chain.from_iterable(
                map(
                    self.get_random_sample_for_label, 
                    self.env.labels.labelset)
                )
            )
        return docs

    def add_doc(self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT], identifier: KT):
        doc = learner.env.dataset[identifier]
        labels = self.env.truth.get_labels(doc)
        learner.env.labels.set_labels(doc, *labels)
        learner.set_as_labeled(doc)

    def __call__(self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        docs = self.get_initialization_sample()
        for doc in docs:
            self.add_doc(learner, doc)
        return learner

class UniformInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        docs = self.get_initialization_sample()
        for sublearner in learner.learners:
            for doc in docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner

class SeparateInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        for sublearner in learner.learners:
            docs = self.get_initialization_sample()
            for doc in docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner

class PositiveUniformInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, 
                 env: AbstractEnvironment[IT, KT, Any, Any, Any, LT], 
                 pos_label: LT, neg_label: LT, sample_size: int = 1, **kwargs) -> None:
        super().__init__(env, sample_size, **kwargs)
        self.pos_label = pos_label
        self.neg_label = neg_label


    def __call__(self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        pos_docs = self.get_random_sample_for_label(self.pos_label)
        for sublearner in learner.learners:
            for doc in pos_docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        for sublearner in learner.learners:
            neg_docs = self.get_random_sample_for_label(self.neg_label)
            for doc in neg_docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner