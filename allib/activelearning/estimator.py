from __future__ import annotations

import collections
import itertools
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, List,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from instancelib.instances import Instance
from instancelib.typehints import DT, KT, LT, LVT, PVT, RT, VT

from ..activelearning.ensembles import ManualEnsemble
from ..activelearning.ml_based import MLBased
from ..environment.base import IT, AbstractEnvironment
from ..machinelearning import AbstractClassifier
from ..utils import get_random_generator
from .base import ActiveLearner, NotInitializedException

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

class Estimator(ManualEnsemble[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, 
                 rng: Any = None, *_, **__) -> None:
        probs = [1.0 / len(learners)] * \
            len(learners) if probabilities is None else probabilities
        super().__init__(learners, probs, rng)
    


class RetryEstimator(Estimator[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):

    def __next__(self) -> IT:
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


class CycleEstimator(Estimator[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, rng: Any = None, *_, **__) -> None:
        super().__init__(learners, probabilities, rng)
        self.learnercycle = itertools.cycle(self.learners)

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        return next(self.learnercycle)


class MultipleEstimator(ManualEnsemble[IT, KT, DT, VT, RT, LT],  Generic[IT, KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, rng: Any = None, *_, **__) -> None:
        probs = [1.0 / len(learners)] * \
            len(learners) if probabilities is None else probabilities
        super().__init__(learners, probs, rng)


