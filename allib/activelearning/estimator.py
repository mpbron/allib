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
try:
    import rpy2.robjects as ro  # type: ignore
    from rpy2.robjects import pandas2ri  # type: ignore
    from rpy2.robjects.conversion import localconverter  # type: ignore
    from rpy2.robjects.packages import importr  # type: ignore
except ImportError:
    R_AVAILABLE = False
else:
    R_AVAILABLE = True


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


def _check_R():
    if not R_AVAILABLE:
        raise ImportError("Install rpy2 interop")





class Estimator(ManualEnsemble[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self,
                 learners: List[ActiveLearner[KT, DT, VT, RT, LT]],
                 probabilities: Optional[List[float]] = None, rng: Any = None, *_, **__) -> None:
        probs = [1.0 / len(learners)] * \
            len(learners) if probabilities is None else probabilities
        super().__init__(learners, probs, rng)
        _check_R()
        self.matrix_history: Deque[pd.DataFrame] = collections.deque()
        self.contingency_history: Deque[Dict[FrozenSet[int], int]] = collections.deque(
        )

        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "estimate.R")
        R["source"](r_script_file)

    def initialize_uniform(self, pos_label: LT, neg_label: LT) -> None:
        pos_docs = random.sample(
            self.env.truth.get_instances_by_label(pos_label), 1)
        neg_docs = random.sample(
            self.env.truth.get_instances_by_label(neg_label), 1)
        docs = pos_docs + neg_docs
        for learner in self.learners:
            for doc in docs:
                _add_doc(learner, doc)
                _add_doc(self, doc)

    def initialize_separate(self, pos_label: LT, neg_label: LT) -> None:
        n = len(self.learners)
        pos_docs = random.sample(
            self.env.truth.get_instances_by_label(pos_label), n)
        neg_docs = random.sample(
            self.env.truth.get_instances_by_label(neg_label), n)
        for i, learner in enumerate(self.learners):
            docs = [pos_docs[i], neg_docs[i]]
            for doc in docs:
                _add_doc(learner, doc)
                _add_doc(self, doc)

    def get_label_matrix(self, label: LT) -> pd.DataFrame:
        rows = {ins_key: {
            l_key: ins_key in learner.env.labeled
            for l_key, learner in enumerate(self.learners)}
            for ins_key in self.env.labels.get_instances_by_label(label)
        }
        dataframe = pd.DataFrame.from_dict(  # type: ignore
            rows, orient="index")
        self.matrix_history.append(dataframe)
        return dataframe

    def get_contingency_list(self, label: LT) -> Dict[FrozenSet[int], int]:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(
                label).intersection(learner.env.labeled)
            for learner_key, learner in enumerate(self.learners)
        }
        key_combinations = powerset(range(len(self.learners)))
        result = {
            combination: len(intersection(
                *[learner_sets[key] for key in combination]))
            for combination in key_combinations
            if len(combination) >= 1
        }
        self.contingency_history.append(result)
        return result

    def get_matrix(self, label: LT) -> np.ndarray:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(
                label).intersection(learner.env.labeled)
            for learner_key, learner in enumerate(self.learners)
        }
        n_learners = len(learner_sets)
        matrix = np.zeros(shape=(n_learners, n_learners))
        for i, key_a in enumerate(learner_sets):
            instances_a = learner_sets[key_a]
            for j, key_b in enumerate(learner_sets):
                if i != j:
                    instances_b = learner_sets[key_b]
                    intersection = instances_a.intersection(instances_b)
                    matrix[i, j] = len(intersection)
        return matrix

    def get_abundance(self, label: LT) -> Optional[Tuple[float, float]]:
        df = self.get_label_matrix(label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["get_abundance"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        ok_fit = res_df[res_df.infoFit == 0]
        if len(ok_fit) == 0:
            ok_fit = res_df
        best_result = ok_fit[ok_fit.BIC == ok_fit.BIC.min()]
        best_result = best_result[["abundance", "stderr"]]
        best_np = best_result.values
        return best_np[0, 0], best_np[0, 1]


class RetryEstimator(Estimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):

    def __next__(self) -> Instance[KT, DT, VT, RT]:
        # Choose the next random learners
        indices = np.arange(len(self.learners))
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


