from __future__ import annotations

import functools
import os
from abc import ABC, abstractmethod
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, List,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator
from ..utils.func import (all_subsets, intersection, list_unzip3,
                          not_in_supersets, powerset, union)
from .base import AbstractEstimator
from .rcapture import AbundanceEstimator, _check_R

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


class RaschEstimator(AbundanceEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "RaschEstimator"
    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_estimate_bootstrap.R")
        R["source"](r_script_file)

    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]], 
              all_learners: FrozenSet[int],
              positive: bool) -> Dict[str, int]:
        learner_set, instances = combination
        learner_cols = {
            f"learner_{learner_key}": int(learner_key in learner_set) 
            for learner_key in all_learners
        }
        count_col = {"count": len(instances)}
        interaction_cols = {
            f"h{i-1}": len(all_subsets(learner_set, i, i)) 
            for i in range(2, len(all_learners))
        }
        final_row = {
            **learner_cols,
            **interaction_cols,
            **count_col
        }
        return final_row

    def get_occasion_history(self, 
                             estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets = self.get_contingency_sets(estimator, label)
        learner_keys = union(*contingency_sets.keys())
        rows = {i:
            self.rasch_row(pair, learner_keys, True)
            for (i, pair) in enumerate(contingency_sets.items())
        }
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        self.matrix_history.append(df)
        return df

    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["rasch.single"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    
    def calculate_abundance(self, 
                            estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        estimate_missing = res_df.values[0,0]
        estimate_error = res_df.values[0,1]
        total_found = estimator.env.labels.document_count(label)
        return (total_found + estimate_missing), estimate_error * 2

    def all_estimations(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT) -> Sequence[Tuple[str, float, float]]:
        return []

    def __call__(self, learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        abundance, error = self.calculate_abundance(learner, label)
        return abundance, abundance-error, abundance + error

class NonParametricRasch(RaschEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "RaschNonParamatric"
    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_estimate_bootstrap.R")
        R["source"](r_script_file)

    

    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["rasch.nonparametric"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        horizon_lowerbound = res_df.values[0,1]
        horizon_upperbound = res_df.values[0,2]
        return horizon, horizon_lowerbound, horizon_upperbound
    
    def __call__(self, 
        learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        estimate, lower_bound, upper_bound = self.calculate_estimate(learner, label)
        return estimate, lower_bound, upper_bound

class ParametricRasch(NonParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "RaschParametric"
    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["rasch.parametric"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

class SingleInteractionRaschParametric(ParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "ParametericRaschInteractionsCombined"
    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]], 
              all_learners: FrozenSet[int],
              positive: bool) -> Dict[str, int]:
        learner_set, instances = combination
        learner_cols = {
            f"learner_{learner_key}": int(learner_key in learner_set) 
            for learner_key in all_learners
        }
        count_col = {"count": len(instances)}
        interaction_col = {"h1": len(
            all_subsets(
                learner_set, 
                2, len(all_learners) - 1)) 
        }
        final_row = {
            **learner_cols,
            **interaction_col,
            **count_col
        }
        return final_row

class SingleInteractionRaschParametric2(
        ParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "ParametricRaschOnly2ndOrder"
    
    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]], 
                  all_learners: FrozenSet[int],
                  positive: bool) -> Dict[str, int]:
        learner_set, instances = combination
        learner_cols = {
            f"learner_{learner_key}": int(learner_key in learner_set) 
            for learner_key in all_learners
        }
        count_col = {"count": len(instances)}
        interaction_col = {"h1": len(
            all_subsets(
                learner_set, 
                2, 2)) 
        }
        final_row = {
            **learner_cols,
            **interaction_col,
            **count_col
        }
        return final_row

class EMRasch(NonParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "EMRasch"
    def __init__(self, neg_label: LT) -> None:
        super().__init__()
        self.neg_label = neg_label

    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_em.R")
        R["source"](r_script_file)

    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        df_neg = self.get_occasion_history(estimator, self.neg_label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            df_neg_r = ro.conversion.py2rpy(df_neg)
            abundance_r = ro.globalenv["rasch.em"]
            r_df = abundance_r(df_pos_r, df_neg_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        return horizon, horizon, horizon
    
    def __call__(self, 
        learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        estimate, lower_bound, upper_bound = self.calculate_estimate(learner, label)
        return estimate, lower_bound, upper_bound

class EMRaschCombined(NonParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "EMRaschCombined"
    def get_contingency_sets_negative(self, 
                                      estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                                      label: LT) -> Dict[FrozenSet[int], FrozenSet[KT]]:
        learner_sets = {
                learner_key: (
                    frozenset(learner.env.labeled).
                        difference(
                            learner.env.labels.get_instances_by_label(label))
                )
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

    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]], 
              all_learners: FrozenSet[int],
              positive: bool) -> Dict[str, int]:
        learner_set, instances = combination
        learner_cols = {
            f"learner_{learner_key}": int(learner_key in learner_set) 
            for learner_key in all_learners
        }
        count_col = {"count": len(instances)}
        positive_col = {"positive": int(positive)}
        interaction_cols = {
            f"h{i-1}": len(all_subsets(learner_set, i, i)) 
            for i in range(2, len(all_learners))
        }
        pos_learner_cols = {
            f"learner_{learner_key}-positive": (
                int(learner_key in learner_set) if positive else 0)
            for learner_key in all_learners
        }
        interaction_pos_cols = {
            f"h{i-1}-positive": (
                len(all_subsets(learner_set, i, i)) if positive else 0)
            for i in range(2, len(all_learners))
        }
        final_row = {
            **learner_cols,
            **positive_col,
            **pos_learner_cols,
            **interaction_cols,
            **interaction_pos_cols,
            **count_col
        }
        return final_row

    def get_occasion_history(self, 
                             estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets_pos = self.get_contingency_sets(estimator, label)
        contingency_sets_neg = self.get_contingency_sets_negative(estimator, label)
        learner_keys = union(*contingency_sets_pos.keys())
        
        rasch_pos_func = functools.partial(
            self.rasch_row, all_learners=learner_keys, positive = True)
        rasch_neg_func = functools.partial(
            self.rasch_row, all_learners=learner_keys, positive = False)
        
        
        rows_pos = list(map(rasch_pos_func, contingency_sets_pos.items()))
        rows_neg = list(map(rasch_neg_func, contingency_sets_neg.items()))

        rows = dict(enumerate(rows_pos + rows_neg))
        
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        self.matrix_history.append(df)
        return df

    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_em_comb.R")
        R["source"](r_script_file)

    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            abundance_r = ro.globalenv["rasch.em.horizon"]
            r_df = abundance_r(df_pos_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        return horizon, horizon, horizon
    
    def __call__(self, 
        learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        estimate, lower_bound, upper_bound = self.calculate_estimate(learner, label)
        return estimate, lower_bound, upper_bound

class EMAlternative(EMRaschCombined[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]], 
              all_learners: FrozenSet[int],
              positive: bool) -> Dict[str, int]:
        learner_set, instances = combination
        learner_cols = {
            f"learner_{learner_key}": int(learner_key in learner_set) 
            for learner_key in all_learners
        }
        count_col = {"count": len(instances)}
        positive_col = {"positive": int(positive)}
        interaction_cols = {
            f"h{i-1}": len(all_subsets(learner_set, i, i)) 
            for i in range(2, len(all_learners))
        }
        interaction_pos_cols = {
            f"h{i-1}-positive": (
                len(all_subsets(learner_set, i, i)) if positive else 0)
            for i in range(2, len(all_learners))
        }
        final_row = {
            **learner_cols,
            **positive_col,
            **interaction_cols,
            **interaction_pos_cols,
            **count_col
        }
        return final_row

class EMRaschRidge(EMRaschCombined[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            abundance_r = ro.globalenv["rasch.ridge.em.horizon"]
            r_df = abundance_r(df_pos_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

class EMRaschCombinedBootstrap(EMRaschCombined[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            abundance_r = ro.globalenv["rasch.em.bootstrap.horizon"]
            r_df = abundance_r(df_pos_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        lowerbound = res_df.values[0,1]
        upperbound = res_df.values[0,2]
        return horizon, lowerbound, upperbound

class EMRaschRidgeCombinedBootstrap(EMRaschCombined[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def calculate_abundance_R(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            abundance_r = ro.globalenv["rasch.ridge.em.horizon.parametric"]
            r_df = abundance_r(df_pos_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        lowerbound = res_df.values[0,1]
        upperbound = res_df.values[0,2]
        return horizon, lowerbound, upperbound
