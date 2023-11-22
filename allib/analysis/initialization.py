from __future__ import annotations

import itertools
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Mapping, Optional, Sequence, TypeVar, Union
from uuid import uuid4

import numpy as np  # type: ignore
from typing_extensions import Self, TypeVar

from ..activelearning.autostop import AutoStopLearner
from ..activelearning.autotar import PSEUDO_INS_PROVIDER

from ..activelearning.learnersequence import LearnerSequence  # type: ignore
from ..activelearning.autotarensemble import AutoTARFirstMethod
from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator
from ..activelearning.target import TargetMethod
from ..activelearning.cmh import CMHMethod
from ..typehints import DT, IT, KT, LT, RT, VT
from instancelib.ingest.qrel import TrecDataset

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


class PseudoInstanceInitializer(Initializer[IT, KT, LT], Generic[IT, KT, DT, LT]):
    def __init__(self, pseudo_data: DT, pseudo_label: LT):
        self.pseudo_data = pseudo_data
        self.pseudo_label = pseudo_label
        self.pseudo_id = uuid4()

    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        ins = learner.env.create(data=self.pseudo_data, vector=None)
        learner.env.create_named_provider(PSEUDO_INS_PROVIDER, [ins.identifier])
        return learner

    @classmethod
    def from_trec(
        cls, trec: TrecDataset, topic_id: str
    ) -> PseudoInstanceInitializer[IT, str, str, str]:
        topic_df = trec.topics.set_index("id")
        topic_row = topic_df.xs(topic_id)
        pseudo_data: str = topic_row.title + " " + topic_row.query  # type: ignore
        return cls(pseudo_data, trec.pos_label)  # type: ignore


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
            list(learner.env.truth.get_instances_by_label(label)), self.sample_size
        )
        return docs

    def get_initialization_sample(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT]
    ) -> Sequence[KT]:
        docs = list(
            itertools.chain.from_iterable(
                map(
                    lambda lbl: self.get_random_sample_for_label(learner, lbl),
                    sorted(list(learner.env.labels.labelset)),
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
    rng: np.random.Generator
    seed: Optional[Union[int, np.random.BitGenerator, np.random.Generator]]

    def __init__(
        self,
        sample_size: int = 1,
        seed: Optional[Union[int, np.random.BitGenerator, np.random.Generator]] = None,
    ) -> None:
        super().__init__(sample_size)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_random_sample_for_label(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT], label: LT
    ) -> Sequence[KT]:
        docs = self.rng.choice(
            list(learner.env.truth.get_instances_by_label(label)), self.sample_size, replace=False  # type: ignore
        )
        return docs

    @classmethod
    def builder(
        cls,
        sample_size: int,
        *args,
        **kwargs,
    ) -> Callable[..., Self]:
        def builder_func(
            *args,
            seed: Optional[
                Union[int, np.random.BitGenerator, np.random.Generator]
            ] = None,
            **kwargs,
        ) -> Self:
            return cls(sample_size, seed)

        return builder_func


class SeededEnsembleInitializer(
    SeededRandomInitializer[IT, KT, LT], Generic[IT, KT, LT]
):
    def get_random_sample_for_label(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT], label: LT, size: int
    ) -> Sequence[KT]:
        docs = self.rng.choice(
            list(learner.env.truth.get_instances_by_label(label)), size, replace=False  # type: ignore
        )
        return docs

    def get_sample(
        self, learner: ActiveLearner[IT, KT, Any, Any, Any, LT]
    ) -> Mapping[LT, Sequence[KT]]:
        assert isinstance(learner, Estimator)
        learner_n = len(learner.learners)
        docs = {
            lbl: self.get_random_sample_for_label(
                learner, lbl, self.sample_size * learner_n
            )
            for lbl in sorted(list(learner.env.labels.labelset))  # type: ignore
        }  # type: ignore

        return docs

    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, Estimator):
            return super().__call__(learner)
        docmap = self.get_sample(learner)
        for _, docs in docmap.items():
            assert len(docs) == len(learner.learners)
            for sublearner, doc in zip(learner.learners, docs):
                self.add_doc(sublearner, doc)
                self.add_doc(learner, doc)
        return learner


class AutoStopLargeInitializer(
    SeededRandomInitializer[IT, KT, LT], Generic[IT, KT, LT]
):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, LearnerSequence):
            return super().__call__(learner)
        docs = self.get_initialization_sample(learner)
        for sublearner in learner.learners:
            if isinstance(sublearner, AutoStopLearner):
                for doc in docs:
                    if doc not in sublearner.env.dataset:
                        sublearner.env.dataset.add(sublearner.env.all_instances[doc])
                    self.add_doc(sublearner, doc)
                    self.add_doc(learner, doc)
                sublearner.key_seq = tuple(sublearner.env.dataset)
        return learner


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


class CMHInitializer(RandomInitializer[IT, KT, LT], Generic[IT, KT, LT]):
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    ) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        if not isinstance(learner, CMHMethod):
            return super().__call__(learner)
        sublearner = learner.learners[0]
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
