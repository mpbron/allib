from __future__ import annotations
from allib.utils.func import all_subsets, intersection, list_unzip3, union
from allib.activelearning.base import ActiveLearner

import collections
import itertools
import logging
import os
from abc import ABC, abstractmethod
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, List,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..activelearning.estimator import Estimator

from .base import AbstractEstimator

try:
    import rpy2.robjects as ro  # type: ignore
    from rpy2.robjects import pandas2ri  # type: ignore
    from rpy2.robjects.conversion import localconverter  # type: ignore
    from rpy2.robjects.packages import importr  # type: ignore
except ImportError:
    R_AVAILABLE = False
else:
    R_AVAILABLE = True

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
_T = TypeVar("_T")
_U = TypeVar("_U")

LOGGER = logging.getLogger(__name__)

def powerset(iterable: Iterable[_T]) -> FrozenSet[FrozenSet[_T]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    result = itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s)+1))
    return frozenset(map(frozenset, result))  # type: ignore



def not_in_supersets(
        contingency: Dict[FrozenSet[_T], FrozenSet[_U]]
        ) -> Dict[FrozenSet[_T], FrozenSet[_U]]:
    ret_dict: Dict[FrozenSet[_T], FrozenSet[_U]] = {}
    sets = frozenset(contingency.keys())
    for key_set in sets:
        strict_supersets = frozenset(filter(
            lambda s: s.issuperset(key_set) and s != key_set,
            sets))
        in_supersets: FrozenSet[_U] = frozenset()
        if len(strict_supersets) > 0:
            in_supersets = union(*map(lambda k: contingency[k], strict_supersets))
        ret_dict[key_set] = contingency[key_set].difference(in_supersets)
    return ret_dict

def _check_R():
    if not R_AVAILABLE:
        raise ImportError("Install rpy2 interop")

class AbundanceEstimator(AbstractEstimator, Generic[KT, DT, VT, RT, LT]):
    def __init__(self):
        self.matrix_history: Deque[pd.DataFrame] = collections.deque()
        self.contingency_history: Deque[Dict[FrozenSet[int], int]] = collections.deque()
        self._start_r()

    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "multiple_estimate.R")
        R["source"](r_script_file)
       
    def get_label_matrix(self, 
                         estimator: Estimator[KT, DT, VT, RT, LT], 
                         label: LT) -> pd.DataFrame:
        rows = {ins_key: {
            l_key: ins_key in learner.env.labeled
            for l_key, learner in enumerate(estimator.learners)}
            for ins_key in estimator.env.labels.get_instances_by_label(label)
        }
        dataframe = pd.DataFrame.from_dict(  # type: ignore
            rows, orient="index")
        self.matrix_history.append(dataframe)
        return dataframe

    def get_contingency_list(self, 
                         estimator: Estimator[KT, DT, VT, RT, LT], 
                         label: LT) -> Dict[FrozenSet[int], int]:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(
                label).intersection(learner.env.labeled)
            for learner_key, learner in enumerate(estimator.learners)
        }
        key_combinations = powerset(range(len(estimator.learners)))
        result = {
            combination: len(intersection(
                *[learner_sets[key] for key in combination]))
            for combination in key_combinations
            if len(combination) >= 1
        }
        self.contingency_history.append(result)
        return result

    def get_matrix(self, 
                   estimator: Estimator[KT, DT, VT, RT, LT], 
                   label: LT) -> np.ndarray:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(
                label).intersection(learner.env.labeled)
            for learner_key, learner in enumerate(estimator.learners)
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

    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_label_matrix(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["get_abundance"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_abundance(self, 
                            estimator: Estimator[KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        ok_fit = res_df[res_df.infoFit == 0]
        if len(ok_fit) == 0:
            ok_fit = res_df
        best_result = ok_fit[ok_fit.BIC == ok_fit.BIC.min()]
        best_result = best_result[["abundance", "stderr"]]
        best_np = best_result.values
        return best_np[0, 0], best_np[0, 1]

    def __call__(self, learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0
        abundance, error = self.calculate_abundance(learner, label)
        return abundance, error

    def all_estimations(self, 
                        estimator: Estimator[KT, DT, VT, RT, LT], 
                        label: LT) -> Sequence[Tuple[str, float, float]]:
        res_df = self.calculate_abundance_R(estimator, label)
        ok_fit = res_df[res_df.infoFit == 0]
        if len(ok_fit) == 0:
            ok_fit = res_df
        results = ok_fit.values
        names = list(ok_fit.index)
        estimations = list(results[:,0])
        errors = list(results[:,1])
        tuples = list(zip(names, estimations, errors))
        return tuples
    
    def get_contingency_sets(self, 
                         estimator: Estimator[KT, DT, VT, RT, LT], 
                         label: LT) -> Dict[FrozenSet[int], FrozenSet[KT]]:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(
                label).intersection(learner.env.labeled)
            for learner_key, learner in enumerate(estimator.learners)
        }
        key_combinations = powerset(range(len(estimator.learners)))
        result = {
            combination: intersection(*[learner_sets[key] for key in combination])
            for combination in key_combinations
            if len(combination) >= 1
        }
        filtered_result = not_in_supersets(result)
        return filtered_result

    def get_occasion_history(self, 
                             estimator: Estimator[KT, DT, VT, RT, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets = self.get_contingency_sets(estimator, label)
        learner_keys = union(*contingency_sets.keys())
        rows = {i:
            {
                **{
                    f"learner_{learner_key}": int(learner_key in combination) 
                    for learner_key in learner_keys
                },  
                **{
                    "count": len(instances)
                }
            }
            for (i, (combination, instances)) in enumerate(contingency_sets.items())
        }
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        return df

class MeanAbundanceEstimator(AbundanceEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __call__(self, learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0
        _, estimations, errors = list_unzip3(self.all_estimations(learner, label))
        estimation = float(np.mean(estimations)) #type: ignore
        error = float(np.mean(errors)) # type: ignore
        return estimation, error

class MedianAbundanceEstimator(AbundanceEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __call__(self, learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0
        _, estimations, errors = list_unzip3(self.all_estimations(learner, label))
        estimation = float(np.median(estimations)) # type: ignore
        error = float(np.median(errors)) # type: ignore
        return estimation, error

class NegativeAbundanceEstimator(AbundanceEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(self, neg_label: LT):
        super().__init__()
        self.neg_label = neg_label
    
    def __call__(self, learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0
        neg_docs = learner.env.labels.document_count(self.neg_label)
        neg_abundance, error = self.calculate_abundance(learner, self.neg_label)
        pos_docs = learner.env.labels.document_count(label)
        estimate = pos_docs * (1 + (neg_docs /neg_abundance))
        error_estimate = (neg_docs / error) * pos_docs
        return estimate, error_estimate



class RaschEstimator(AbundanceEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_estimate.R")
        R["source"](r_script_file)

    def get_occasion_history(self, 
                             estimator: Estimator[KT, DT, VT, RT, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets = self.get_contingency_sets(estimator, label)
        learner_keys = union(*contingency_sets.keys())
        max_len = len(learner_keys)
        rows = {i:
            {
                **{
                    f"learner_{learner_key}": int(learner_key in combination) 
                    for learner_key in learner_keys
                },  
                **{
                    "count": len(instances),
                    "h1": len(all_subsets(combination, 2, max_len - 1))
                }
            }
            for (i, (combination, instances)) in enumerate(contingency_sets.items())
        }
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        return df

    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["get_abundance"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    
    def calculate_abundance(self, 
                            estimator: Estimator[KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        estimate_missing = res_df.values[0,0]
        estimate_error = res_df.values[0,1]
        total_found = estimator.env.labels.document_count(label)
        return (total_found + estimate_missing), estimate_error

    
    