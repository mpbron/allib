from __future__ import annotations
from abc import ABC, abstractmethod

import itertools
import logging
import random
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
from typing_extensions import Self, TypeVar
from uuid import uuid4

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..environment import AbstractEnvironment
from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator
from ..activelearning.target import TargetMethod
from ..activelearning.autotarensemble import AutoTARFirstMethod

from ..typehints import KT, LT, IT, DT, VT, RT

LOGGER = logging.getLogger(__name__)

AT = TypeVar("AT", bound="ActiveLearner[Any, Any, Any, Any, Any, Any]", covariant=True)


class Initializer(ABC, Generic[IT, KT, LT]):
    @abstractmethod
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        raise NotImplementedError

    @classmethod
    def builder(cls, *args, **kwargs) -> Callable[..., Self]:
        def builder_func(*args, **kwargs) -> Self:
            return cls()

        return builder_func


class IdentityInitializer(Initializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        return learner


class RandomInitializer(Initializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, sample_size: int = 1) -> None:
        self.sample_size = sample_size

    def get_random_sample_for_label(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT], label: LT
    ) -> Sequence[KT]:
        docs = random.sample(
            learner.env.truth.get_instances_by_label(label), self.sample_size
        )
        return docs

    def get_initialization_sample(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT]
    ) -> Sequence[KT]:
        docs = list(
            itertools.chain.from_iterable(
                map(
                    lambda lbl: self.get_random_sample_for_label(learner, lbl),
                    sorted(learner.env.labels.labelset),
                )
            )
        )
        return docs

    def add_doc(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT], identifier: KT
    ):
        doc = learner.env.dataset[identifier]
        labels = learner.env.truth.get_labels(doc)
        learner.env.labels.set_labels(doc, *labels)
        learner.set_as_labeled(doc)
        LOGGER.info(f"Added {identifier} as prior knowledge with labels {list(labels)}")

    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        docs = self.get_initialization_sample(learner)
        for doc in docs:
            self.add_doc(learner, doc)
        return learner

    @classmethod
    def builder(cls, sample_size: int, *args, **kwargs) -> Callable[..., Self]:
        def builder_func(*args, **kwargs) -> Self:
            return cls(sample_size)

        return builder_func


class SeededRandomInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, sample_size: int = 1, seed: int = 42) -> None:
        super().__init__(sample_size)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_random_sample_for_label(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT], label: LT
    ) -> Sequence[KT]:
        docs = self.rng.choice(
            list(learner.env.truth.get_instances_by_label(label)), self.sample_size  # type: ignore
        )
        return docs

    @classmethod
    def builder(
        cls, sample_size: int, seed: int, *args, **kwargs
    ) -> Callable[..., Self]:
        def builder_func(*args, **kwargs) -> Self:
            return cls(sample_size, seed)

        return builder_func


class TargetInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, TargetMethod):
            return super().__call__(learner)
        sublearner = learner.learners[1]
        docs = self.get_initialization_sample(learner)
        for doc in docs:
            self.add_doc(sublearner, doc)
            self.add_doc(learner, doc)
        return learner


class PriorInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, AutoTARFirstMethod):
            return super().__call__(learner)
        sublearner = learner.learners[0]
        docs = self.get_initialization_sample(learner)
        for doc in docs:
            self.add_doc(sublearner, doc)
            self.add_doc(learner, doc)
        return learner


class UniformInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        docs = self.get_initialization_sample(learner)
        for sublearner in learner.learners:
            for doc in docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner


class SeparateInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        for sublearner in learner.learners:
            docs = self.get_initialization_sample(learner)
            for doc in docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner


class PositiveUniformInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, pos_label: LT, neg_label: LT, sample_size: int = 1) -> None:
        super().__init__(sample_size)
        self.pos_label = pos_label
        self.neg_label = neg_label

    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        pos_docs = self.get_random_sample_for_label(learner, self.pos_label)
        for sublearner in learner.learners:
            for doc in pos_docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        for sublearner in learner.learners:
            neg_docs = self.get_random_sample_for_label(learner, self.neg_label)
            for doc in neg_docs:
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner

    @classmethod
    def builder(cls, sample_size: int, *args, **kwargs) -> Callable[..., Self]:
        def builder_func(pos_label: LT, neg_label: LT, *args, **kwargs) -> Self:
            return cls(pos_label, neg_label, sample_size)

        return builder_func
