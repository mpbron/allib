from __future__ import annotations
from abc import ABC, abstractmethod
from allib.activelearning.ensembles import ManualEnsemble
from allib.activelearning.ml_based import MLBased

import collections
import itertools
import logging
import math
import os
import random
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, List,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance
from ..machinelearning import AbstractClassifier
from ..utils import get_random_generator
from .base import ActiveLearner, NotInitializedException

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
_T = TypeVar("_T")

LOGGER = logging.getLogger(__name__)


def intersection(first: FrozenSet[_T], *others: FrozenSet[_T]) -> FrozenSet[_T]:
    return first.intersection(*others)


def powerset(iterable: Iterable[_T]) -> FrozenSet[FrozenSet[_T]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    result = itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s)+1))
    return frozenset(map(frozenset, result))  # type: ignore


def _add_doc(learner: ActiveLearner[KT, DT, VT, RT, LT], key: KT):
    doc = learner.env.dataset[key]
    labels = learner.env.truth.get_labels(doc)
    learner.env.labels.set_labels(doc, *labels)
    learner.set_as_labeled(doc)


class Estimator(ManualEnsemble[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, 
                 rng: Any = None, *_, **__) -> None:
        probs = [1.0 / len(learners)] * \
            len(learners) if probabilities is None else probabilities
        super().__init__(learners, probs, rng)
    


class RetryEstimator(Estimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):

    def __next__(self) -> Instance[KT, DT, VT, RT]:
        # Choose the next random learners
        learner = self._choose_learner()

        # Select the next instance from the learner
        ins = next(learner)

        # Check if the instance identifier has not been labeled already
        while ins.identifier in self.env.labeled:
            # This instance has already been labeled my another learner.
            # Skip it and mark as labeled
            learner.set_as_labeled(ins)
            LOGGER.info(
                "The document with key %s was already labeled. Skipping", ins.identifier)
            ins = next(learner)

        # Set the instances as sampled by learner with key al_idx and return the instance
        self._sample_dict[ins.identifier] = self.learners.index(learner)
        return ins


class CycleEstimator(Estimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, rng: Any = None, *_, **__) -> None:
        super().__init__(learners, probabilities, rng)
        self.learnercycle = itertools.cycle(self.learners)

    def _choose_learner(self) -> ActiveLearner[KT, DT, VT, RT, LT]:
        return next(self.learnercycle)


class MultipleEstimator(ManualEnsemble[KT, DT, VT, RT, LT],  Generic[KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, rng: Any = None, *_, **__) -> None:
        probs = [1.0 / len(learners)] * \
            len(learners) if probabilities is None else probabilities
        super().__init__(learners, probs, rng)


